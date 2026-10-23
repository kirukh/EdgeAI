"""Camera settings per part type: stored profiles, the automatic camera setup and the focus assistant.

The automatic setup has two parts:

1. **Static step** (:meth:`CameraSetup.run`): one good part lies still under the camera →
   white balance, exposure, autofocus (cameras with a focus motor), sharpness check, size and
   position of the part.
2. **Belt path** (:meth:`CameraSetup.track`): the first part that runs through the image during
   teach-in gives belt direction and speed → digital zoom along the path and an exposure time
   short enough for the motion blur limit.

The result (a *camera profile*) is stored with the model and applied whenever the model is selected.
"""

from __future__ import annotations

import time
from collections.abc import Callable
from datetime import datetime
from typing import Any

import cv2
import numpy as np

from . import autosetup as AS
from .alignment import PartDetection, detect_part, levels, resize_to_width, to_gray
from .camera import CameraSource
from .config import AppConfig
from .trigger import PartTrigger

LOW_CONTRAST = 40  # grey levels between part and belt
FILLS_IMAGE = 1.35  # part size × this > image size → no room for zoom or position scatter
FOCUS_MINUTES = 10
AF_MAX_FRAMES = 120
EXPOSURE_ROUNDS = 12


class CameraSetup:
    """Camera profiles per part type, the automatic camera setup and the focus assistant."""

    def __init__(
        self,
        camera: CameraSource,
        cfg: AppConfig,
        trigger: PartTrigger,
        log: Callable[[str, str], None],
        wait_frames: Callable[[int], None],
        latest_frame: Callable[[], np.ndarray | None],
    ):
        self.camera, self.cfg, self.trigger, self.log = camera, cfg, trigger, log
        self.wait_frames, self.latest_frame = wait_frames, latest_frame
        self.base_axis = cfg.trigger.axis
        self.applied: dict[str, Any] | None = None  # settings currently applied to the camera
        self.status: dict[str, Any] | None = None  # running/last static setup (shown in the UI)
        self.pending_profile: dict[str, Any] | None = None  # result of the static step, waiting for teach-in
        self.profile: dict[str, Any] | None = None  # profile of the part being taught in
        self.path: dict[str, Any] | None = None  # first moving part during teach-in: belt direction + zoom
        self.focus: dict[str, Any] | None = None  # focus assistant {"until", "best", "value"}

    @property
    def controllable(self) -> bool:
        """Can this camera be adjusted automatically (exposure, gain, zoom)?"""
        return bool(self.camera.can_control and self.cfg.autosetup.enabled)

    # ------------------------------------------------------------------ profiles
    def apply(self, profile: dict[str, Any] | None, force: bool = False) -> None:
        """Applies a part type's camera profile (or the base settings from config.json)."""
        cam, base, p = self.camera, self.cfg.camera, profile or {}
        exposure: tuple[int, float] | None
        if "exposure_us" in p:
            exposure = (p["exposure_us"], p.get("gain", 1.0))
        else:
            exposure = (base.exposure_us, base.analogue_gain) if base.exposure_us > 0 else None
        want: dict[str, Any] = {
            "exposure": exposure,
            "gains": tuple(p["colour_gains"])
            if p.get("colour_gains")
            else (tuple(base.colour_gains) if base.lock_white_balance else None),
            "lens": p.get("lens_position", base.lens_position),
            "crop": tuple(p.get("crop") or AS.FULL),
            "axis": p.get("axis") or self.base_axis,
        }
        if want == self.applied and not force:
            return
        if cam.can_control:
            try:
                if want["exposure"]:
                    cam.set_exposure(*want["exposure"])
                if want["gains"]:
                    cam.set_colour_gains(want["gains"])
                if want["lens"] is not None:
                    cam.set_lens(want["lens"])
                cam.set_crop(want["crop"])
            except Exception as e:
                self.log("error", f"Camera settings could not be applied: {e}")
        self.cfg.trigger.axis = want["axis"]
        self.trigger.armed, self.trigger.prev_pos = True, None
        self.applied = want

    def begin_teach_in(self) -> None:
        """Teach-in starts: with a fresh static setup the first moving part measures the belt path."""
        if self.pending_profile is not None:
            self.profile, self.pending_profile = self.pending_profile, None
            self.path = {"track": [], "seen": False}
        else:
            self.profile, self.path = None, None
            self.apply(None)  # never teach in with another part's zoom

    def end_teach_in(self) -> None:
        """Teach-in finished or cancelled: the belt path measurement ends."""
        self.profile, self.path = None, None

    # ------------------------------------------------------------------ static step
    def run(self) -> dict[str, Any]:
        """Static step of the automatic setup (blocking, run it in a thread). Returns the profile."""
        self.status = {
            "running": True,
            "step": "starting",
            "error": None,
            "result": None,
            "started": datetime.now().isoformat(timespec="seconds"),
        }
        self.pending_profile, self.applied = None, None  # camera state changes now
        try:
            gains = self._white_balance()
            exposure, gain, work, det = self._exposure()
            lens = self._autofocus(det, work)
            prof = self._measure(exposure, gain, gains, lens)
        except Exception as ex:
            self.status.update(running=False, step="failed", error=str(ex))
            raise
        self.pending_profile = prof
        self.status.update(
            running=False,
            step="done",
            result={
                "exposure_ms": round(exposure / 1000, 2),
                "gain": gain,
                "white_balance": bool(gains),
                "autofocus": self.camera.has_autofocus,
                "lens_position": lens,
                "contrast": prof["contrast"],
                "edge_width_px": prof["edge_width_full_px"],
                "notes": prof["notes"],
            },
        )
        return prof

    def _step(self, name: str) -> None:
        if self.status is not None:
            self.status["step"] = name

    def _measure_part(self) -> tuple[np.ndarray, np.ndarray, PartDetection | None]:
        """Latest frame → (frame, work image, complete detection or None)."""
        frame = self.latest_frame()
        if frame is None:
            raise ValueError("No camera image.")
        work = resize_to_width(frame, self.cfg.localization.work_width)
        det = detect_part(to_gray(work), self.cfg.localization)
        return frame, work, (det if det is not None and det.complete else None)

    def _white_balance(self) -> tuple[float, float] | None:
        self._step("white balance")
        cam = self.camera
        cam.set_crop(AS.FULL)
        cam.set_exposure(int(self.cfg.camera.exposure_us or 4000), 1.0)
        cam.awb_start()
        self.wait_frames(15)
        gains = cam.awb_result()
        if gains:
            cam.set_colour_gains(gains)
        return gains

    def _exposure(self) -> tuple[int, float, np.ndarray, PartDetection]:
        """Brightest pixels of part + surroundings → target level, without clipping."""
        self._step("exposure")
        e, g = int(self.cfg.camera.exposure_us or 4000), 1.0
        det = None
        for _ in range(EXPOSURE_ROUNDS):
            self.wait_frames(4)
            frame, work, det = self._measure_part()
            level = AS.bright_level(frame, det.contour if det else None, frame.shape[1] / work.shape[1])
            e, g, done = AS.next_exposure(e, g, level, self.cfg.autosetup)
            self.camera.set_exposure(e, g)
            if done and det is not None:
                break
        if det is None:
            raise ValueError(
                "No complete part found. Lay ONE good part flat under the camera, in the middle "
                "of the image, and try again."
            )
        return e, g, work, det

    def _autofocus(self, det: PartDetection, work: np.ndarray) -> float | None:
        """Focus run on the part (only cameras with a focus motor, e.g. Camera Module 3)."""
        cam = self.camera
        if not cam.has_autofocus:
            return None
        self._step("focus")
        x, y, w, h = cv2.boundingRect(det.contour.astype(np.int32))
        width, height = work.shape[1], work.shape[0]
        cam.af_start((x / width, y / height, w / width, h / height))
        state, pos = "scanning", None
        for _ in range(AF_MAX_FRAMES):
            self.wait_frames(1)
            state, pos = cam.af_state()
            if state != "scanning":
                break
        if state == "focused" and pos:
            lens = round(pos, 3)
            cam.set_lens(lens)
            return lens
        self.log("warn", "Autofocus did not find a sharp image – check that the part is in the middle.")
        return None

    def _measure(
        self, exposure: int, gain: float, gains: tuple[float, float] | None, lens: float | None
    ) -> dict[str, Any]:
        """Size, position, contrast and sharpness of the part → camera profile."""
        self._step("measuring")
        self.wait_frames(4)
        _, work, det = self._measure_part()
        if det is None:
            raise ValueError("The part is no longer visible – keep it still under the camera.")
        gray = to_gray(work)
        part_level, bg_level = levels(gray, det)
        edge = AS.edge_width(gray, det.contour)
        width, height = work.shape[1], work.shape[0]
        cx, cy, size = AS.part_extent(det.contour)
        center = AS.to_sensor(cx, cy, (width, height), AS.FULL)
        diameter = (size / width, size / height)
        prof = {
            "auto": True,
            "created": datetime.now().isoformat(timespec="seconds"),
            "exposure_us": int(exposure),
            "gain": float(gain),
            "colour_gains": list(gains) if gains else None,
            "lens_position": lens,
            "crop": list(AS.FULL),
            "zoom": 1.0,
            "axis": self.base_axis,
            "center": [round(center[0], 4), round(center[1], 4)],
            "diameter": [round(diameter[0], 4), round(diameter[1], 4)],
            "contrast": round(abs(part_level - bg_level), 1),
            "edge_width_full_px": round(edge, 2) if edge else None,
        }
        prof["notes"] = self._notes(prof, gain, diameter, edge)
        return prof

    def _notes(self, prof: dict[str, Any], gain: float, diameter: tuple[float, float], edge: float | None) -> list[str]:
        ac, notes = self.cfg.autosetup, []
        if prof["contrast"] < LOW_CONTRAST:
            notes.append(
                f"Low contrast between part and belt ({prof['contrast']:.0f} grey levels) – "
                "a darker/brighter belt or backlight would make detection more reliable."
            )
        if gain >= ac.max_gain - 1e-3:
            notes.append("Little light: the image needs the maximum gain (more noise). Brighter lighting helps.")
        if min(diameter) * FILLS_IMAGE > 1.0:
            notes.append(
                "The part almost fills the image – parts may be cut off at the edge. Mount the camera "
                "a little higher (no zoom is possible either way)."
            )
        if not self.camera.has_autofocus and edge and edge > ac.edge_width_max_px:
            notes.append(
                f"The image is blurry (edges {edge:.1f} px). This camera has no focus motor: "
                "turn the lens once by hand (focus assistant), then run the setup again."
            )
        return notes

    # ------------------------------------------------------------------ belt path
    def track(self, frame: np.ndarray, t: float) -> bool:
        """Follows the first part that runs through the image. True once the path is known."""
        lp, det = self.path, self.trigger.last_det
        if lp is None:
            return False
        height, width = frame.shape[:2]
        if det is not None and det.complete:
            s = self.trigger.scale
            lp["track"].append(
                (t, *AS.to_sensor(det.centroid[0] * s, det.centroid[1] * s, (width, height), self.camera.crop))
            )
            lp["seen"] = True
        elif lp["seen"] and det is None:  # the part has left the image
            path = AS.path_from_track(lp["track"])
            if path is None:  # it did not move (taken away by hand) → wait for the next
                lp["track"], lp["seen"] = [], False
                return False
            self.finish_path(path, (width, height))
            return True
        return False

    def finish_path(self, path: dict[str, Any] | None, out_size: tuple[int, int] | None = None) -> None:
        """Belt direction known (or a stopped belt: ``path`` None) → digital zoom along the path and
        an exposure limit for the belt speed."""
        ac, prof = self.cfg.autosetup, self.profile
        self.path = None
        if prof is None:
            return
        axis = path["axis"] if path else (prof.get("axis") or self.base_axis)
        cx, cy = prof["center"]
        if path:
            cx, cy = (cx, path["across"]) if axis == "x" else (path["across"], cy)
        crop, zoom = AS.crop_for_path((cx, cy), tuple(prof["diameter"]), axis, ac)
        prof.update(crop=list(crop), zoom=zoom, axis=axis)
        if path:
            frame = self.latest_frame()
            width, height = out_size or ((frame.shape[1], frame.shape[0]) if frame is not None else (640, 480))
            v_out = path["speed"] * (width if axis == "x" else height) / crop[2]  # output px per second
            e_max = ac.max_blur_px / max(v_out, 1e-6) * 1e6
            product = prof["exposure_us"] * prof["gain"]
            e, g = AS.split_exposure(product, min(ac.max_exposure_us, e_max), ac)
            prof.update(
                exposure_us=e, gain=g, belt_speed_px_s=round(v_out, 1), motion_blur_px=round(v_out * e / 1e6, 2)
            )
            if product / e > ac.max_gain + 1e-6:
                msg = (
                    f"Not enough light for this belt speed: motion blur {prof['motion_blur_px']:.1f} px "
                    f"(goal ≤ {ac.max_blur_px}). Brighter light or a slower belt improves accuracy."
                )
                prof.setdefault("notes", []).append(msg)
                self.log("warn", msg)
        self.apply(prof, force=True)
        self.trigger.armed, self.trigger.prev_pos = False, None  # ignore the part still in the image
        blur = (
            f", exposure {prof['exposure_us'] / 1000:.2f} ms (motion blur {prof['motion_blur_px']} px)" if path else ""
        )
        self.log("ok", f"Belt direction {axis}, digital zoom {zoom:.2f}×{blur} – now capturing the good parts.")

    # ------------------------------------------------------------------ focus assistant
    def set_focus_assist(self, on: bool) -> None:
        """Live sharpness value in the camera image – for lenses that are focused by hand
        (Camera Module v2/IMX219, HQ camera, Global Shutter camera)."""
        if not on and self.focus is None:
            return
        self.focus = {"until": time.time() + FOCUS_MINUTES * 60, "best": 0.0, "value": 0.0} if on else None
        self.log(
            "info",
            "Focus assistant " + ("on – turn the lens slowly until the value is at its maximum." if on else "off."),
        )

    @staticmethod
    def sharpness(gray: np.ndarray) -> float:
        """Focus measure: variance of the Laplacian (higher = sharper). Robust against noise
        through a light blur; only comparable for the same scene and lighting."""
        g = cv2.GaussianBlur(gray, (3, 3), 0)
        return float(cv2.Laplacian(g, cv2.CV_32F).var())

    def draw_focus(self, img: np.ndarray, frame: np.ndarray) -> bool:
        """Draws the focus bar into the preview. False once the assistant has timed out."""
        fa = self.focus
        if fa is None:
            return False
        if time.time() > fa["until"]:
            self.focus = None
            return False
        h, w = frame.shape[:2]
        det = self.trigger.last_det  # measure on the part if one is visible, else on the central 50 %
        on_part = det is not None and det.complete
        if det is not None and on_part:
            x, y, bw, bh = cv2.boundingRect((det.contour.astype(np.float32) * self.trigger.scale).astype(np.int32))
            x0, y0, x1, y1 = max(0, x), max(0, y), min(w, x + bw), min(h, y + bh)
        else:
            x0, y0, x1, y1 = w // 4, h // 4, 3 * w // 4, 3 * h // 4
        roi = frame[y0:y1, x0:x1]
        if roi.size == 0:
            return True
        v = self.sharpness(to_gray(roi))
        fa["value"] = 0.7 * fa["value"] + 0.3 * v if fa["value"] else v  # smooth sensor noise
        fa["best"] = max(fa["best"], fa["value"])
        s = img.shape[1] / w
        cv2.rectangle(img, (int(x0 * s), int(y0 * s)), (int(x1 * s), int(y1 * s)), (0, 200, 255), 1)
        rel = fa["value"] / fa["best"] if fa["best"] else 0.0
        col = (60, 200, 60) if rel > 0.95 else (0, 200, 255) if rel > 0.8 else (60, 60, 230)
        cv2.rectangle(img, (0, 0), (img.shape[1], 44), (0, 0, 0), cv2.FILLED)
        cv2.rectangle(img, (8, 30), (8 + int((img.shape[1] - 16) * min(1.0, rel)), 40), col, cv2.FILLED)
        cv2.putText(
            img,
            f"FOCUS {fa['value']:.0f}   best {fa['best']:.0f}   ({rel * 100:.0f} %)",
            (8, 22),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            col,
            1,
            cv2.LINE_AA,
        )
        if not on_part:
            cv2.putText(
                img,
                "no part in view - lay a part flat under the camera",
                (8, img.shape[0] - 36),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 200, 255),
                1,
                cv2.LINE_AA,
            )
        return True

    def state(self) -> dict[str, Any]:
        """Camera state for the UI."""
        exp = (self.applied or {}).get("exposure")
        return {
            "exposure_us": exp[0] if exp else None,
            "gain": exp[1] if exp else None,
            "zoom": round(1.0 / self.camera.crop[2], 2),
            "axis": self.cfg.trigger.axis,
        }
