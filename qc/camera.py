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
    info = ""
    last_age_s = 0.0          # age of the last frame returned by read() (sensor exposure → now)
    # Sources with real parts moving on a conveyor → the trigger decides when to capture
    continuous = True

    def read(self) -> np.ndarray | None:  # pragma: no cover - interface
        raise NotImplementedError

    def close(self) -> None:
        pass

    # ---- camera controls for the automatic camera setup (qc/autosetup.py) ----
    # Sources that cannot be controlled (USB, image folder) keep these defaults.
    can_control = False          # exposure / gain / crop can be set by software
    has_autofocus = False        # focus motor (e.g. Camera Module 3)
    crop = (0.0, 0.0, 1.0, 1.0)  # current sensor crop (x, y, w, h as fractions of the sensor)
    last_metadata: dict = {}

    def set_exposure(self, exposure_us: int, gain: float) -> None:
        raise NotImplementedError

    def set_colour_gains(self, gains) -> None:
        pass

    def set_crop(self, crop) -> None:
        raise NotImplementedError

    def awb_start(self) -> None:
        pass

    def awb_result(self):
        return None

    def set_lens(self, position: float) -> None:
        pass

    def af_start(self, window) -> None:
        pass

    def af_state(self) -> tuple[str, float | None]:
        """("scanning" | "focused" | "failed", lens position in dioptres)."""
        return "failed", None


class PiCameraSource(CameraSource):
    name = "Raspberry Pi camera"

    def __init__(self, cfg: CameraConfig):
        from picamera2 import Picamera2  # only available on the Pi

        self.cam = Picamera2()
        kwargs = {}
        if cfg.sensor_mode:
            # fixed sensor readout, e.g. IMX219 1640×1232 (binned, full field of view)
            kwargs["sensor"] = {"output_size": tuple(int(v) for v in cfg.sensor_mode)}
        try:
            config = self.cam.create_video_configuration(
                main={"size": (cfg.width, cfg.height), "format": "RGB888"},
                controls={"FrameRate": cfg.fps}, queue=False, **kwargs)
        except TypeError:          # older picamera2 without the "sensor" argument
            config = self.cam.create_video_configuration(
                main={"size": (cfg.width, cfg.height), "format": "RGB888"}, controls={"FrameRate": cfg.fps})
        self.cam.configure(config)
        self.has_autofocus = "AfMode" in self.cam.camera_controls
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
        self.sensor_info = self._describe(cfg)
        self.info = self.sensor_info + ("" if self.has_autofocus else
                                        ", fixed-focus lens (focus by turning the lens – use the focus assistant)")
        if self.has_autofocus:        # camera with autofocus (e.g. Camera Module 3)
            try:
                if cfg.lens_position is None:
                    self.cam.set_controls({"AfMode": 1})         # auto: one focus run, then the lens stays
                    self.cam.autofocus_cycle()
                else:
                    self.cam.set_controls({"AfMode": 0, "LensPosition": float(cfg.lens_position)})   # manual
                    time.sleep(0.3)
                pos = self.cam.capture_metadata().get("LensPosition")
                if pos:
                    self.info += f", focus fixed at {pos:.2f} dioptres (≈ {100 / pos:.0f} cm)"
            except Exception as e:  # noqa: BLE001 - focus problems must not stop the system
                self.info += f", focus could not be set: {e}"

    def _describe(self, cfg: CameraConfig) -> str:
        """Sensor model and readout mode for the log (purely informative, never fails)."""
        try:
            props = self.cam.camera_properties
            model = props.get("Model", "?")
            conf = self.cam.camera_configuration()
            sensor = (conf.get("sensor") if isinstance(conf, dict) else getattr(conf, "sensor", None)) or {}
            mode = sensor.get("output_size") if isinstance(sensor, dict) else getattr(sensor, "output_size", None)
            full = props.get("PixelArraySize")
            text = f"sensor {model}" + (f", readout {mode[0]}×{mode[1]}" if mode else "")
            crop = self.cam.capture_metadata().get("ScalerCrop")       # (x, y, w, h) on the pixel array
            if crop and full and crop[2] < 0.9 * full[0]:
                text += (f" – field of view CROPPED to {crop[2]}×{crop[3]} of {full[0]}×{full[1]} "
                         "(set camera.sensor_mode, e.g. [1640, 1232] for the IMX219)")
            return text + f", output {cfg.width}×{cfg.height}"
        except Exception as e:  # noqa: BLE001
            return f"sensor info unavailable ({e})"

    # ---- controls (applied by libcamera a few frames later; the caller waits for new frames) ----
    can_control = True

    def _crop_max(self):
        try:
            return tuple(self.cam.camera_controls["ScalerCrop"][1])          # (min, max, default)
        except Exception:  # noqa: BLE001
            w, h = self.cam.camera_properties.get("PixelArraySize", (0, 0))
            return 0, 0, w, h

    def set_exposure(self, exposure_us: int, gain: float) -> None:
        self.cam.set_controls({"AeEnable": False, "ExposureTime": int(exposure_us), "AnalogueGain": float(gain)})

    def set_colour_gains(self, gains) -> None:
        if gains:
            self.cam.set_controls({"AwbEnable": False, "ColourGains": (float(gains[0]), float(gains[1]))})

    def set_crop(self, crop) -> None:
        """Digital zoom: the ISP reads only this part of the sensor and scales it to the output size."""
        mx, my, mw, mh = self._crop_max()
        x, y, w, h = crop
        rect = (int(mx + x * mw), int(my + y * mh), max(64, int(w * mw)), max(48, int(h * mh)))
        self.cam.set_controls({"ScalerCrop": rect})
        self.crop = tuple(float(v) for v in crop)

    def awb_start(self) -> None:
        self.cam.set_controls({"AwbEnable": True})

    def awb_result(self):
        g = self.last_metadata.get("ColourGains")
        return (round(float(g[0]), 3), round(float(g[1]), 3)) if g else None

    def set_lens(self, position: float) -> None:
        if self.has_autofocus and position is not None:
            self.cam.set_controls({"AfMode": 0, "LensPosition": float(position)})       # 0 = manual

    def af_start(self, window) -> None:
        """Autofocus run on the part only (window = x, y, w, h as fractions of the current crop)."""
        if not self.has_autofocus:
            return
        cx, cy, cw, ch = self.crop
        mx, my, mw, mh = self._crop_max()
        # AfWindows are given relative to the ScalerCropMaximum rectangle
        x, y, w, h = window
        rect = (int((cx + x * cw) * mw), int((cy + y * ch) * mh), int(w * cw * mw), int(h * ch * mh))
        self.cam.set_controls({"AfMode": 1, "AfMetering": 1, "AfWindows": [rect]})   # auto, windows
        self.cam.set_controls({"AfTrigger": 0})                                       # start

    def af_state(self) -> tuple[str, float | None]:
        st = self.last_metadata.get("AfState")
        pos = self.last_metadata.get("LensPosition")
        return {2: "focused", 3: "failed"}.get(st, "scanning"), (float(pos) if pos is not None else None)

    def read(self):
        # picamera2 "RGB888" delivers BGR byte order – matches OpenCV directly.
        # queue=False: always a fresh frame, never one that waited in a buffer while the Pi was busy.
        # last_age_s = time since the sensor exposed it → exact timing of the reject pulse.
        req = self.cam.capture_request()
        try:
            frame = req.make_array("main")
            self.last_metadata = req.get_metadata()
            ts = self.last_metadata.get("SensorTimestamp")
        finally:
            req.release()
        self.last_age_s = max(0.0, (time.monotonic_ns() - ts) / 1e9) if ts else 0.0
        if self.last_age_s > 1.0:            # clock domains differ – do not trust it
            self.last_age_s = 0.0
        return frame

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
    """Simulated conveyor: parts move through the image from left to right.

    Also simulates switchable lighting (front light/backlight, for dual-light mode)
    and can show a checkerboard for testing the camera calibration.
    """

    name = "Simulator"
    MM_PER_PX = 0.25          # simulated scale at 640 px image width (plate A ≈ 75 × 42 mm)

    def __init__(self, cfg: CameraConfig, seed: int | None = None):
        from . import synthetic

        self.syn = synthetic
        self.cfg = cfg
        self.rng = np.random.default_rng(seed)
        self.size = (min(cfg.width, 960), int(min(cfg.width, 960) * cfg.height / cfg.width))
        self.light = cfg.sim_lighting             # lighting currently switched on
        self._backgrounds: dict[str, np.ndarray] = {}
        self.part_type = cfg.sim_part_type
        self.defect_rate = cfg.sim_defect_rate
        self.force_good = False
        self.force_defect = False                 # self-test: next parts are defective
        self.board: dict | None = None            # checkerboard for calibration tests
        # simulated camera controls (brightness ∝ exposure × gain, nominal = 4000 µs × 1.0)
        self.exposure = (cfg.exposure_us or 4000, cfg.analogue_gain)
        self.crop = (0.0, 0.0, 1.0, 1.0)
        self.has_autofocus = bool(cfg.sim_autofocus)
        self.defocus = float(cfg.sim_defocus)
        self.hold = False                          # part lies still in the middle (camera setup)
        self._af = None
        self.last_metadata = {}
        self._spawn()
        self.last_time = 0.0

    def _background(self, light: str) -> np.ndarray:
        if light not in self._backgrounds:
            self._backgrounds[light] = self.syn.belt_background(self.size, light, self.rng)
        return self._backgrounds[light]

    def _spawn(self):
        defect = None
        if self.force_defect or (not self.force_good and self.rng.random() < self.defect_rate):
            defect = str(self.rng.choice(self.syn.DEFECTS))
        self.part = self.syn.make_part(self.part_type, defect, self.rng)
        self._layers: dict[str, tuple] = {}
        self.x = -200.0
        self.y = 240 + self.rng.uniform(-25, 25)
        if self.cfg.sim_any_angle:
            self.angle = self.rng.uniform(0, 360)
        else:
            self.angle = self.rng.uniform(-6, 6) + (180 if self.rng.random() < 0.5 else 0)
        self.gap = self.rng.integers(0, 20)

    def _layer(self, light: str):
        key = (light, self.crop[2])
        if key not in self._layers:
            self._layers[key] = self.syn.render_layers(self.part, light,
                                                       self.size[0] / 640.0 / self.crop[2] * self.cfg.sim_part_scale)
        return self._layers[key]

    # ---- simulated camera controls ----
    can_control = True
    NOMINAL = 4000.0

    def set_exposure(self, exposure_us: int, gain: float) -> None:
        self.exposure = (float(exposure_us), float(gain))

    def set_crop(self, crop) -> None:
        self.crop = tuple(float(v) for v in crop)

    def awb_result(self):
        return tuple(self.cfg.colour_gains)

    def set_lens(self, position: float) -> None:
        if self.has_autofocus and position is not None:
            self.defocus = 0.0 if abs(position - 4.0) < 0.05 else 2.0

    def af_start(self, window) -> None:
        if self.has_autofocus:
            self._af = 4                           # "scanning" for a few frames

    def af_state(self) -> tuple[str, float | None]:
        if not self.has_autofocus:
            return "failed", None
        return ("scanning", None) if self._af else ("focused", 4.0)

    def hold_part(self, on: bool) -> None:
        """Camera setup in the simulator: a good part lies still under the camera."""
        self.hold = on
        if on:
            self.force_good = True
            self._spawn()
            self.x = 320.0
        else:
            self.force_good = False
            self._spawn()

    @property
    def current_defect(self) -> str | None:
        return self.part.defect

    def set_lighting(self, lighting: str):
        """Simulator setting (UI): lighting used in single-light mode."""
        self.cfg.sim_lighting = lighting
        self.light = lighting
        self._spawn()

    def switch_light(self, light: str):
        """Called by the light controller (dual-light mode): same part, other lighting."""
        if light in ("front", "back"):
            self.light = light

    def show_board(self, cols: int = 9, rows: int = 6, square_mm: float = 10.0):
        """Show a checkerboard at a new random pose (calibration test)."""
        # 1st board lies flat on the belt; later ones are tilted (like a user would hold it)
        tilt = 0.0 if self.board is None and not getattr(self, "_boards_shown", 0) else 0.22
        self._boards_shown = getattr(self, "_boards_shown", 0) + 1
        self.board = {"cols": cols, "rows": rows, "square_mm": square_mm, "seed": int(self.rng.integers(1 << 30)),
                      "tilt": tilt}

    def hide_board(self):
        self.board = None

    def _render_board(self) -> np.ndarray:
        b = self.board
        rng = np.random.default_rng(b["seed"])
        scale = self.size[0] / 640.0
        sq = b["square_mm"] / self.MM_PER_PX * scale
        nx, ny = b["cols"] + 1, b["rows"] + 1
        board = np.full((int(ny * sq) + 2, int(nx * sq) + 2), 255, np.uint8)
        for j in range(ny):
            for i in range(nx):
                if (i + j) % 2 == 0:
                    cv2.rectangle(board, (int(i * sq), int(j * sq)), (int((i + 1) * sq) - 1, int((j + 1) * sq) - 1), 0, -1)
        h, w = board.shape
        src = np.float32([[0, 0], [w, 0], [w, h], [0, h]])
        cx, cy = self.size[0] / 2 + rng.uniform(-30, 30) * scale, self.size[1] / 2 + rng.uniform(-20, 20) * scale
        ang = np.radians(rng.uniform(-12, 12))
        rot = np.array([[np.cos(ang), -np.sin(ang)], [np.sin(ang), np.cos(ang)]])
        corners = (src - [w / 2, h / 2]) @ rot.T + [cx, cy]
        # perspective of a tilted plane: one side nearer (larger) than the other
        t = b.get("tilt", 0.0)
        if t:
            ax = rng.uniform(0, 2 * np.pi)
            d = np.array([np.cos(ax), np.sin(ax)])
            rel = corners - corners.mean(0)
            k = (rel @ d) / (np.abs(rel @ d).max() + 1e-9)          # −1 … 1 along the tilt axis
            corners = corners.mean(0) + rel * (1 + t * k)[:, None]
        corners += rng.uniform(-2, 2, corners.shape) * scale
        H = cv2.getPerspectiveTransform(src, corners.astype(np.float32))
        img = cv2.warpPerspective(board, H, self.size, borderValue=90)
        img = cv2.GaussianBlur(img, (0, 0), 0.7)
        noise = self.rng.normal(0, 2, img.shape)
        return cv2.cvtColor(np.clip(img + noise, 0, 255).astype(np.uint8), cv2.COLOR_GRAY2BGR)

    def read(self):
        # Limit the frame rate so the simulator does not max out the CPU
        wait = 1.0 / max(1, self.cfg.fps) - (time.time() - self.last_time)
        if wait > 0:
            time.sleep(wait)
        self.last_time = time.time()
        if self.board is not None:
            return self._render_board()
        if self._af:
            self._af -= 1
            if not self._af:
                self.defocus = 0.0
        if not self.hold:
            self.x += self.cfg.sim_speed_px
        if self.x > 840 + self.gap * self.cfg.sim_speed_px:
            self._spawn()
        cx, cy, f = self.crop[0] * 640.0, self.crop[1] * 480.0, self.crop[2]
        pose = ((self.x - cx) / f, (self.y - cy) / f, self.angle)          # world → zoomed output
        img = self.syn.compose(self.part, pose, self.light, self.size, self.rng,
                               layers=self._layer(self.light), background=self._background(self.light))
        k = self.exposure[0] * self.exposure[1] / self.NOMINAL
        if abs(k - 1.0) > 1e-3:
            img = np.clip(img.astype(np.float32) * k, 0, 255).astype(np.uint8)
        if self.defocus > 0:
            img = cv2.GaussianBlur(img, (0, 0), self.defocus)
        return img


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
