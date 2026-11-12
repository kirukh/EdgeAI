"""USB camera (UVC, e.g. Logitech C310) for the inspection.

* The camera records in ``camera.usb_capture`` (C310: its full 1280 × 960, MJPG) and every frame is
  scaled to the working size ``width × height``. The difference is the room for the digital zoom of
  the automatic camera setup: it crops the recorded frame in software.
* A grabber thread reads the camera all the time and keeps only the NEWEST frame. USB cameras queue
  frames; without this, the inspection could get an image that is several frames old.
* Exposure, gain, white balance, mains frequency and the automatic frame-rate reduction are set with
  ``v4l2-ctl`` (package v4l-utils) under the names the camera reports – OpenCV's own exposure
  properties mean different things in different OpenCV versions. Without v4l2-ctl the camera runs
  with OpenCV's basic settings and cannot be adjusted automatically.

* Many UVC cameras (the C310 among them) pad their MJPG frames with a few bytes. libjpeg decodes the
  frame correctly but prints "Corrupt JPEG data: N extraneous bytes before marker" for EVERY frame
  straight to the process's stderr. ``quiet_jpeg_warnings`` filters exactly these lines and counts them.

Video files and stream URLs (``usb:<file or URL>``) are read the same way, without camera controls.
"""

from __future__ import annotations

import os
import re
import subprocess
import threading
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any

import cv2
import numpy as np

from .camera import CameraSource
from .config import CameraConfig

EXPOSURE_UNIT_US = 100  # UVC "exposure_time_absolute" is given in units of 100 µs
MANUAL_EXPOSURE = 1  # UVC auto_exposure menu: 1 = manual, 3 = aperture priority (automatic)
POWER_LINE = {0: 0, 50: 1, 60: 2}  # power_line_frequency menu: disabled, 50 Hz, 60 Hz
FULL = (0.0, 0.0, 1.0, 1.0)

# The same control has different names depending on the kernel version
ALIASES: dict[str, tuple[str, ...]] = {
    "auto_exposure": ("auto_exposure", "exposure_auto"),
    "exposure": ("exposure_time_absolute", "exposure_absolute"),
    "dynamic_fps": ("exposure_dynamic_framerate", "exposure_auto_priority"),
    "gain": ("gain",),
    "wb_auto": ("white_balance_automatic", "white_balance_temperature_auto"),
    "wb_temp": ("white_balance_temperature",),
    "power_line": ("power_line_frequency",),
    "focus_auto": ("focus_automatic_continuous", "focus_auto"),
}
_CTRL = re.compile(r"^\s*(\w+)\s+0x[0-9a-fA-F]+\s+\((\w+)\)\s*:\s*(.*)$")

Runner = Callable[..., Any]  # subprocess.run or a stand-in (tests)


@dataclass
class Control:
    """One camera control as ``v4l2-ctl --list-ctrls`` reports it."""

    name: str
    kind: str  # int, bool, menu
    minimum: int = 0
    maximum: int = 1
    default: int = 0
    value: int = 0


def parse_controls(text: str) -> dict[str, Control]:
    """Controls from the output of ``v4l2-ctl --list-ctrls``."""
    out: dict[str, Control] = {}
    for line in text.splitlines():
        m = _CTRL.match(line)
        if not m:
            continue
        fields = dict(re.findall(r"(\w+)=(-?\d+)", m.group(3)))
        lo, hi = int(fields.get("min", 0)), int(fields.get("max", 1))
        out[m.group(1)] = Control(
            m.group(1), m.group(2), lo, hi, int(fields.get("default", lo)), int(fields.get("value", lo))
        )
    return out


class V4L2Controls:
    """Camera controls of a V4L2 device via ``v4l2-ctl`` (empty if the tool is missing)."""

    def __init__(self, device: str, run: Runner = subprocess.run):
        self.device, self.run = device, run
        self.available = False
        self.ctrls: dict[str, Control] = {}
        try:
            res = run(["v4l2-ctl", "-d", device, "--list-ctrls"], capture_output=True, text=True, timeout=5)
        except (OSError, subprocess.SubprocessError):
            return  # v4l2-ctl not installed
        if res.returncode == 0:
            self.available = True
            self.ctrls = parse_controls(res.stdout)

    def control(self, key: str) -> Control | None:
        """The camera's control for a key of ``ALIASES`` (or None)."""
        return next((self.ctrls[n] for n in ALIASES.get(key, (key,)) if n in self.ctrls), None)

    def has(self, key: str) -> bool:
        """Does the camera have this control?"""
        return self.control(key) is not None

    def set(self, key: str, value: int) -> int | None:
        """Sets a control (clamped to its range); returns the value set, or None if there is no such control."""
        c = self.control(key)
        if c is None:
            return None
        v = int(min(c.maximum, max(c.minimum, value)))
        self.run(["v4l2-ctl", "-d", self.device, "--set-ctrl", f"{c.name}={v}"], capture_output=True, timeout=5)
        c.value = v
        return v


class JpegWarningFilter:
    """Routes file descriptor 2 (stderr) through a pipe and drops libjpeg's "Corrupt JPEG data"
    lines – they come from C code, so Python's logging cannot filter them. Everything else is
    passed on unchanged. Installed once per process (``quiet_jpeg_warnings``)."""

    PATTERN = b"Corrupt JPEG data"

    def __init__(self) -> None:
        self.dropped = 0
        self._out = os.dup(2)
        read_fd, write_fd = os.pipe()
        os.dup2(write_fd, 2)
        os.close(write_fd)
        self._in = os.fdopen(read_fd, "rb", buffering=0)
        threading.Thread(target=self._pump, name="qc-stderr-filter", daemon=True).start()

    def _pump(self) -> None:
        buf = b""
        try:
            while chunk := self._in.read(4096):
                buf += chunk
                *lines, buf = buf.split(b"\n")
                for line in lines:
                    if self.PATTERN in line:
                        self.dropped += 1
                    else:
                        os.write(self._out, line + b"\n")
        finally:  # never leave stderr pointing at a pipe nobody reads
            os.dup2(self._out, 2)
            if buf:
                os.write(self._out, buf)


_jpeg_filter: JpegWarningFilter | None = None


def quiet_jpeg_warnings() -> JpegWarningFilter | None:
    """Hides libjpeg's harmless per-frame warnings for MJPG cameras (once per process)."""
    global _jpeg_filter
    if _jpeg_filter is None:
        try:
            _jpeg_filter = JpegWarningFilter()
        except OSError:
            return None
    return _jpeg_filter


class USBCameraSource(CameraSource):
    """USB camera (UVC) or a video file/stream: newest frame, software zoom, camera controls via v4l2-ctl."""

    name = "USB camera"

    def __init__(
        self,
        cfg: CameraConfig,
        device: str,
        open_capture: Callable[..., Any] = cv2.VideoCapture,
        controls: Callable[[str], V4L2Controls] = V4L2Controls,
    ):
        self.cfg = cfg
        self.is_device = device.isdigit() or device.startswith("/dev/video")
        self.path = f"/dev/video{device}" if device.isdigit() else device
        if self.is_device:
            self.cap = open_capture(int(device) if device.isdigit() else device, cv2.CAP_V4L2)
        else:
            self.cap = open_capture(device)
        if not self.cap.isOpened():
            raise RuntimeError(f"Could not open camera '{device}'")
        self.out_size = (int(cfg.width), int(cfg.height))
        if self.is_device:
            cw, ch = cfg.usb_capture or self.out_size
            self.cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter.fourcc(*cfg.usb_format))
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, cw)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, ch)
            self.cap.set(cv2.CAP_PROP_FPS, cfg.fps)
            self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        self.ctl = controls(self.path) if self.is_device else None
        self.can_control = bool(self.ctl and self.ctl.has("exposure"))
        self.crop: tuple[float, ...] = FULL
        self.exposure: tuple[int, float] | None = None
        self._setup_controls()
        self.info = self._describe()

        self._cv = threading.Condition()
        self._frame: np.ndarray | None = None
        self._t_frame, self._seq, self._taken = 0.0, 0, 0
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._grab, name="qc-usb-camera", daemon=True)
        self._thread.start()

    # ---- set-up ----
    def _setup_controls(self) -> None:
        cfg, ctl = self.cfg, self.ctl
        if ctl is None:
            return
        if not ctl.available:  # without v4l2-ctl: OpenCV's basic settings (meaning depends on the version)
            if cfg.exposure_us > 0:
                self.cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, MANUAL_EXPOSURE)
                self.cap.set(cv2.CAP_PROP_EXPOSURE, cfg.exposure_us / EXPOSURE_UNIT_US)
            if cfg.lock_white_balance:
                self.cap.set(cv2.CAP_PROP_AUTO_WB, 0)
            return
        ctl.set("power_line", POWER_LINE.get(cfg.power_line_hz, 1))  # no flicker stripes under mains lighting
        ctl.set("focus_auto", 0)  # never refocus between parts
        if cfg.exposure_us > 0:
            self.set_exposure(cfg.exposure_us, cfg.analogue_gain)
        if cfg.lock_white_balance and ctl.has("wb_temp"):
            ctl.set("wb_auto", 0)
            ctl.set("wb_temp", cfg.usb_white_balance_k)

    def _describe(self) -> str:
        w, h = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = self.cap.get(cv2.CAP_PROP_FPS)
        text = f"{self.path}: {w}×{h}" + (f" {self.cfg.usb_format}" if self.is_device else "")
        text += (f" @ {fps:.0f} fps" if fps and fps > 0 else "") + f" → {self.out_size[0]}×{self.out_size[1]}"
        if self.ctl is not None and not self.ctl.available:
            text += ", camera controls not available (install v4l-utils: sudo apt install v4l-utils)"
        elif self.ctl is not None:
            text += ", fixed focus" if not self.ctl.has("focus_auto") else ", autofocus switched off"
        return text

    # ---- frames ----
    def _grab(self) -> None:
        """Reads the camera continuously and keeps only the newest frame."""
        delay = 0.0 if self.is_device else 1.0 / max(1, self.cfg.fps)  # files: play back in real time
        while not self._stop.is_set():
            ok, frame = self.cap.read()
            if not ok:
                if not self.is_device:  # end of a video file: start again
                    self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                time.sleep(0.05)
                continue
            with self._cv:
                self._frame, self._t_frame = frame, time.monotonic()
                self._seq += 1
                self._cv.notify_all()
            if delay:
                time.sleep(delay)

    def read(self) -> np.ndarray | None:
        """USB camera: the next NEW frame (waits up to 1 s), zoomed and scaled to the working size."""
        with self._cv:
            if not self._cv.wait_for(lambda: self._seq > self._taken or self._stop.is_set(), timeout=1.0):
                return None
            if self._frame is None:
                return None
            frame, t, self._taken = self._frame, self._t_frame, self._seq
        self.last_age_s = max(0.0, time.monotonic() - t)
        return self._output(frame)

    def _output(self, frame: np.ndarray) -> np.ndarray:
        """Software zoom (crop) and scaling to ``width × height``."""
        h, w = frame.shape[:2]
        if tuple(self.crop) != FULL:
            x, y, cw, ch = self.crop
            x0, y0 = round(x * w), round(y * h)
            x1, y1 = max(x0 + 1, round((x + cw) * w)), max(y0 + 1, round((y + ch) * h))
            frame = frame[y0:y1, x0:x1]
        if (frame.shape[1], frame.shape[0]) != self.out_size:
            shrink = frame.shape[1] > self.out_size[0]
            frame = cv2.resize(frame, self.out_size, interpolation=cv2.INTER_AREA if shrink else cv2.INTER_LINEAR)
        return frame

    def close(self) -> None:
        """USB camera: stops the grabber thread and releases the device."""
        self._stop.set()
        with self._cv:
            self._cv.notify_all()
        self._thread.join(timeout=2)
        self.cap.release()

    # ---- controls for the automatic camera setup ----
    def set_exposure(self, exposure_us: int, gain: float) -> None:
        """USB camera: manual exposure in 100 µs steps, no automatic frame-rate reduction; gain as a
        factor of the camera's default gain (approximate – the sensor's gain curve is not known)."""
        if self.ctl is None or not self.ctl.available:
            return
        self.ctl.set("auto_exposure", MANUAL_EXPOSURE)
        self.ctl.set("dynamic_fps", 0)  # otherwise the camera lowers the frame rate in dim light
        units = self.ctl.set("exposure", max(1, round(exposure_us / EXPOSURE_UNIT_US)))
        g = self.ctl.control("gain")
        if g is not None:
            self.ctl.set("gain", round((g.default or (g.minimum + g.maximum) // 4) * max(gain, 0.0)))
        self.exposure = (int(units or 0) * EXPOSURE_UNIT_US, float(gain))

    def set_crop(self, crop: Sequence[float]) -> None:
        """USB camera: digital zoom on the recorded frame (x, y, w, h as fractions)."""
        self.crop = tuple(float(v) for v in crop)
