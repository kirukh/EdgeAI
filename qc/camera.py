"""Camera abstraction.

Supported sources (``CameraConfig.source``):

* ``pi``            – Raspberry Pi camera via picamera2 (libcamera), fixed exposure
* ``usb:<index>``   – USB camera via OpenCV (also ``usb:/dev/video0`` or an RTSP URL)
* ``folder:<path>`` – plays back images from a folder (offline tests with real photos)
* ``sim``           – simulated conveyor with synthetic parts
"""
from __future__ import annotations

import time
from pathlib import Path

import cv2
import numpy as np

from .config import CameraConfig

IMAGE_EXT = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}


class CameraSource:
    """Base class: ``read()`` returns the next BGR image (or ``None``)."""

    name = "base"
    # Sources with real parts moving on a conveyor → the trigger decides when to capture
    continuous = True

    def read(self) -> np.ndarray | None:  # pragma: no cover - interface
        raise NotImplementedError

    def close(self) -> None:
        pass


class PiCameraSource(CameraSource):
    name = "Raspberry Pi camera"

    def __init__(self, cfg: CameraConfig):
        from picamera2 import Picamera2  # only available on the Pi

        self.cam = Picamera2()
        config = self.cam.create_video_configuration(
            main={"size": (cfg.width, cfg.height), "format": "RGB888"},
            controls={"FrameRate": cfg.fps},
        )
        self.cam.configure(config)
        controls: dict = {}
        if cfg.exposure_us > 0:
            # Fixed exposure + gain → reproducible brightness.
            # A short exposure time reduces motion blur on the conveyor.
            controls.update({"AeEnable": False, "ExposureTime": int(cfg.exposure_us), "AnalogueGain": float(cfg.analogue_gain)})
        if cfg.lock_white_balance:
            controls.update({"AwbEnable": False, "ColourGains": tuple(cfg.colour_gains)})
        if controls:
            self.cam.set_controls(controls)
        self.cam.start()
        time.sleep(0.5)

    def read(self):
        # picamera2 "RGB888" delivers BGR byte order – matches OpenCV directly
        return self.cam.capture_array("main")

    def close(self):
        self.cam.stop()
        self.cam.close()


class OpenCVSource(CameraSource):
    name = "USB camera"

    def __init__(self, cfg: CameraConfig, device: str):
        dev: int | str = int(device) if device.isdigit() else device
        self.cap = cv2.VideoCapture(dev)
        if not self.cap.isOpened():
            raise RuntimeError(f"Could not open camera '{device}'")
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, cfg.width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, cfg.height)
        self.cap.set(cv2.CAP_PROP_FPS, cfg.fps)
        if cfg.exposure_us > 0:
            # V4L2: 1 = manual exposure; unit depends on the driver (usually 100 µs)
            self.cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 1)
            self.cap.set(cv2.CAP_PROP_EXPOSURE, cfg.exposure_us / 100.0)
        if cfg.lock_white_balance:
            self.cap.set(cv2.CAP_PROP_AUTO_WB, 0)

    def read(self):
        ok, frame = self.cap.read()
        return frame if ok else None

    def close(self):
        self.cap.release()


class FolderSource(CameraSource):
    """Each image in the folder corresponds to one part (one call = one part)."""

    name = "Image folder"
    continuous = False

    def __init__(self, folder: str, loop: bool = True):
        self.files = sorted(p for p in Path(folder).rglob("*") if p.suffix.lower() in IMAGE_EXT)
        if not self.files:
            raise RuntimeError(f"No images found in '{folder}'")
        self.loop = loop
        self.idx = 0
        self.current = cv2.imread(str(self.files[0]))

    @property
    def current_file(self) -> Path:
        return self.files[(self.idx - 1) % len(self.files)] if self.idx else self.files[0]

    def next_part(self):
        if self.idx >= len(self.files):
            if not self.loop:
                return None
            self.idx = 0
        self.current = cv2.imread(str(self.files[self.idx]))
        self.idx += 1
        return self.current

    def read(self):
        time.sleep(0.05)
        return self.current


class SimulatorSource(CameraSource):
    """Simulated conveyor: parts move through the image from left to right."""

    name = "Simulator"

    def __init__(self, cfg: CameraConfig, seed: int | None = None):
        from . import synthetic

        self.syn = synthetic
        self.cfg = cfg
        self.rng = np.random.default_rng(seed)
        self.size = (min(cfg.width, 960), int(min(cfg.width, 960) * cfg.height / cfg.width))
        self.background = synthetic.belt_background(self.size, cfg.sim_lighting, self.rng)
        self.part_type = cfg.sim_part_type
        self.defect_rate = cfg.sim_defect_rate
        self.force_good = False
        self._spawn()
        self.last_time = 0.0

    def _spawn(self):
        defect = None
        if not self.force_good and self.rng.random() < self.defect_rate:
            defect = str(self.rng.choice(self.syn.DEFECTS))
        self.part = self.syn.make_part(self.part_type, defect, self.rng)
        scale = self.size[0] / 640.0
        self.layers = self.syn.render_layers(self.part, self.cfg.sim_lighting, scale)
        self.x = -200.0
        self.y = 240 + self.rng.uniform(-25, 25)
        if self.cfg.sim_any_angle:
            self.angle = self.rng.uniform(0, 360)
        else:
            self.angle = self.rng.uniform(-6, 6) + (180 if self.rng.random() < 0.5 else 0)
        self.gap = self.rng.integers(0, 20)

    @property
    def current_defect(self) -> str | None:
        return self.part.defect

    def set_lighting(self, lighting: str):
        self.cfg.sim_lighting = lighting
        self.background = self.syn.belt_background(self.size, lighting, self.rng)
        self._spawn()

    def read(self):
        # Limit the frame rate so the simulator does not max out the CPU
        wait = 1.0 / max(1, self.cfg.fps) - (time.time() - self.last_time)
        if wait > 0:
            time.sleep(wait)
        self.last_time = time.time()
        self.x += self.cfg.sim_speed_px
        if self.x > 840 + self.gap * self.cfg.sim_speed_px:
            self._spawn()
        return self.syn.compose(
            self.part, (self.x, self.y, self.angle), self.cfg.sim_lighting,
            self.size, self.rng, layers=self.layers, background=self.background,
        )


def open_camera(cfg: CameraConfig) -> CameraSource:
    src = cfg.source
    if src == "sim":
        return SimulatorSource(cfg)
    if src == "pi":
        return PiCameraSource(cfg)
    if src.startswith("usb:"):
        return OpenCVSource(cfg, src[4:])
    if src.startswith("folder:"):
        return FolderSource(src[7:])
    raise ValueError(f"Unknown camera source '{src}'")
