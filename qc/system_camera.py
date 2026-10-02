"""Camera side of the QC system: camera profiles, automatic camera setup, focus assistant, calibration."""
from __future__ import annotations

import threading
import time
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np

from . import autosetup as AS
from . import calibration as qcal
from .alignment import _levels, detect_part, resize_to_width, to_gray



from .visualize import encode_jpg as _jpg


class CameraMixin:
    """Mixin of :class:`qc.system.QCSystem` (uses its state: camera, cfg, lock, model, …)."""

    FOCUS_MINUTES = 10

    def set_focus_assist(self, on: bool) -> None:
        """Live sharpness value in the camera image – for lenses that are focused by hand
        (Camera Module v2/IMX219, HQ camera, Global Shutter camera)."""
        self.focus_assist = {"until": time.time() + self.FOCUS_MINUTES * 60, "best": 0.0, "value": 0.0} if on else None
        self.log("info", "Focus assistant " + ("on – turn the lens slowly until the value is at its maximum."
                                               if on else "off."))

    @staticmethod
    def sharpness(gray: np.ndarray) -> float:
        """Focus measure: variance of the Laplacian (higher = sharper). Robust against noise
        through a light blur; only comparable for the same scene and lighting."""
        g = cv2.GaussianBlur(gray, (3, 3), 0)
        return float(cv2.Laplacian(g, cv2.CV_32F).var())

    def _draw_focus(self, img: np.ndarray, frame: np.ndarray, fa: dict) -> None:
        h, w = frame.shape[:2]
        # measure in full resolution on the central 50 % (or on the part, if one is visible)
        det = self.trigger.last_det
        on_part = det is not None and det.complete
        if on_part:
            x, y, bw, bh = cv2.boundingRect((det.contour.astype(np.float32) * self.trigger.scale).astype(np.int32))
            x0, y0, x1, y1 = max(0, x), max(0, y), min(w, x + bw), min(h, y + bh)
        else:
            x0, y0, x1, y1 = w // 4, h // 4, 3 * w // 4, 3 * h // 4
        roi = frame[y0:y1, x0:x1]
        if roi.size == 0:
            return
        v = self.sharpness(to_gray(roi))
        fa["value"] = 0.7 * fa["value"] + 0.3 * v if fa["value"] else v       # smooth sensor noise
        fa["best"] = max(fa["best"], fa["value"])
        s = img.shape[1] / w
        cv2.rectangle(img, (int(x0 * s), int(y0 * s)), (int(x1 * s), int(y1 * s)), (0, 200, 255), 1)
        rel = fa["value"] / fa["best"] if fa["best"] else 0.0
        col = (60, 200, 60) if rel > 0.95 else (0, 200, 255) if rel > 0.8 else (60, 60, 230)
        cv2.rectangle(img, (0, 0), (img.shape[1], 44), (0, 0, 0), cv2.FILLED)
        cv2.rectangle(img, (8, 30), (8 + int((img.shape[1] - 16) * min(1.0, rel)), 40), col, cv2.FILLED)
        cv2.putText(img, f"FOCUS {fa['value']:.0f}   best {fa['best']:.0f}   ({rel * 100:.0f} %)", (8, 22),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, col, 1, cv2.LINE_AA)
        if not on_part:
            cv2.putText(img, "no part in view - lay a part flat under the camera", (8, img.shape[0] - 36),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 200, 255), 1, cv2.LINE_AA)

    def _apply_camera(self, profile: dict | None) -> None:
        """Applies a part type's camera profile (or the base settings from config.json)."""
        cam, base, p = self.camera, self.cfg.camera, profile or {}
        want = {
            "exposure": (p["exposure_us"], p.get("gain", 1.0)) if "exposure_us" in p else
                        ((base.exposure_us, base.analogue_gain) if base.exposure_us > 0 else None),
            "gains": tuple(p["colour_gains"]) if p.get("colour_gains") else
                     (tuple(base.colour_gains) if base.lock_white_balance else None),
            "lens": p.get("lens_position", base.lens_position),
            "crop": tuple(p.get("crop") or AS.FULL),
            "axis": p.get("axis") or self._base_axis,
        }
        if want == self._applied:
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
            except Exception as e:  # noqa: BLE001 - a camera problem must not stop the system
                self.log("error", f"Camera settings could not be applied: {e}")
        self.cfg.trigger.axis = want["axis"]
        self.trigger.armed, self.trigger.prev_pos = True, None
        self._applied = want

    def _wait_frames(self, n: int, timeout: float = 5.0) -> None:
        with self._frame_cv:
            target = self._frame_seq + n
            self._frame_cv.wait_for(lambda: self._frame_seq >= target, timeout)

    def _measure_part(self):
        """Latest frame → (frame, work image, detection or None)."""
        frame = self.last_raw
        if frame is None:
            raise ValueError("No camera image.")
        work = resize_to_width(frame, self.cfg.localization.work_width)
        det = detect_part(to_gray(work), self.cfg.localization)
        return frame, work, (det if det is not None and det.complete else None)

    def start_camera_setup(self) -> None:
        """Step 1 of the automatic camera setup: ONE good part lies still under the camera."""
        with self.lock:
            if self.mode != "idle":
                raise ValueError("Stop the current mode first.")
            if not self.camera.can_control:
                raise ValueError("This camera source cannot be adjusted automatically (Pi camera or simulator only).")
            self.mode = "camera_setup"
            self.pending_profile = None
            self._applied = None                 # camera state changes now – re-apply everything afterwards
            self.cam_setup = {"running": True, "step": "starting", "error": None, "result": None,
                              "started": datetime.now().isoformat(timespec="seconds")}
        threading.Thread(target=self._camera_setup_run, name="qc-camera-setup", daemon=True).start()

    def cancel_camera_setup(self) -> None:
        with self.lock:
            self.pending_profile = None
            self.cam_setup = None
            if self.mode == "camera_setup":
                self.mode = "idle"
            self._applied = None
            self._apply_camera(self.model.camera_profile if self.model else None)

    def _camera_setup_run(self) -> None:
        cam, ac, st = self.camera, self.cfg.autosetup, self.cam_setup
        if self.is_sim:
            cam.hold_part(True)
        try:
            st["step"] = "white balance"
            cam.set_crop(AS.FULL)
            e, g = float(self.cfg.camera.exposure_us or 4000), 1.0
            cam.set_exposure(e, g)
            cam.awb_start()
            self._wait_frames(15)
            gains = cam.awb_result()
            if gains:
                cam.set_colour_gains(gains)

            st["step"] = "exposure"
            det = None
            for _ in range(12):
                self._wait_frames(4)
                levels = []
                for light in (self._lights() or [None]):
                    if light:
                        self.io.set_light(light)
                        self._wait_frames(max(2, self.cfg.lighting.settle_frames + 1))
                    frame, work, det = self._measure_part()
                    levels.append(AS.bright_level(frame, det.contour if det else None, frame.shape[1] / work.shape[1]))
                if self.dual:
                    self.io.set_light(self.cfg.lighting.idle_light)
                e, g, done = AS.next_exposure(e, g, max(levels), ac)
                cam.set_exposure(e, g)
                if done and det is not None:
                    break
            if det is None:
                raise ValueError("No complete part found. Lay ONE good part flat under the camera, in the middle "
                                 "of the image, and try again.")

            lens = None
            if cam.has_autofocus:
                st["step"] = "focus"
                x, y, w, h = cv2.boundingRect(det.contour.astype(np.int32))
                W, H = work.shape[1], work.shape[0]
                cam.af_start((x / W, y / H, w / W, h / H))
                for _ in range(120):
                    self._wait_frames(1)
                    state, pos = cam.af_state()
                    if state != "scanning":
                        break
                if state == "focused" and pos:
                    lens = round(pos, 3)
                    cam.set_lens(lens)
                else:
                    self.log("warn", "Autofocus did not find a sharp image – check that the part is in the middle.")

            st["step"] = "measuring"
            self._wait_frames(4)
            frame, work, det = self._measure_part()
            if det is None:
                raise ValueError("The part is no longer visible – keep it still under the camera.")
            gray = to_gray(work)
            part_level, bg_level = _levels(gray, det)
            edge = AS.edge_width(gray, det.contour)
            W, H = work.shape[1], work.shape[0]
            cx, cy, D = AS.part_extent(det.contour)
            center = AS.to_sensor(cx, cy, (W, H), AS.FULL)
            diameter = (D / W, D / H)
            prof = {"auto": True, "created": datetime.now().isoformat(timespec="seconds"),
                    "exposure_us": int(e), "gain": float(g), "colour_gains": list(gains) if gains else None,
                    "lens_position": lens, "crop": list(AS.FULL), "zoom": 1.0, "axis": self._base_axis,
                    "center": [round(center[0], 4), round(center[1], 4)],
                    "diameter": [round(diameter[0], 4), round(diameter[1], 4)],
                    "contrast": round(abs(part_level - bg_level), 1),
                    "edge_width_full_px": round(edge, 2) if edge else None}
            notes = []
            if prof["contrast"] < 40:
                notes.append(f"Low contrast between part and belt ({prof['contrast']:.0f} grey levels) – "
                             "a darker/brighter belt or backlight would make detection more reliable.")
            if g >= ac.max_gain - 1e-3:
                notes.append("Little light: the image needs the maximum gain (more noise). Brighter lighting helps.")
            if min(diameter) * 1.35 > 1.0:
                notes.append("The part almost fills the image – parts may be cut off at the edge. Mount the camera "
                             "a little higher (no zoom is possible either way).")
            if not cam.has_autofocus and edge and edge > ac.edge_width_max_px:
                notes.append(f"The image is blurry (edges {edge:.1f} px). This camera has no focus motor: "
                             "turn the lens once by hand (focus assistant), then run the setup again.")
            prof["notes"] = notes
            with self.lock:
                self.pending_profile = prof
                st.update(running=False, step="done", result={
                    "exposure_ms": round(e / 1000, 2), "gain": g, "white_balance": bool(gains),
                    "autofocus": cam.has_autofocus, "lens_position": lens, "contrast": prof["contrast"],
                    "edge_width_px": prof["edge_width_full_px"], "notes": notes})
            self.log("ok", f"Camera adjusted: exposure {e / 1000:.2f} ms, gain {g:.2f}"
                     + (f", focus {lens:.2f} dpt" if lens else "") + ". Next: let the parts run – the first one "
                     "measures the belt direction and sets the zoom.")
            for n in notes:
                self.log("warn", n)
        except Exception as ex:  # noqa: BLE001
            st.update(running=False, step="failed", error=str(ex))
            self.log("error", f"Automatic camera setup failed: {ex}")
            self._applied = None
            self._apply_camera(self.model.camera_profile if self.model else None)
        finally:
            if self.is_sim:
                cam.hold_part(False)
            with self.lock:
                if self.mode == "camera_setup":
                    self.mode = "idle"

    def _lights(self) -> list[str]:
        return ["back", "front"] if self.dual else []

    def _track_path(self, frame: np.ndarray, t: float) -> None:
        """Follows the first part that runs through the image (full field of view)."""
        lp = self.learn_path
        det = self.trigger.last_det
        H, W = frame.shape[:2]
        if det is not None and det.complete:
            s = self.trigger.scale
            lp["track"].append((t,) + AS.to_sensor(det.centroid[0] * s, det.centroid[1] * s, (W, H), self.camera.crop))
            lp["seen"] = True
        elif lp["seen"] and det is None:                 # the part has left the image
            path = AS.path_from_track(lp["track"])
            if path is None:                             # it did not move (taken away by hand) → wait for the next
                lp["track"], lp["seen"] = [], False
                return
            with self.lock:
                if self.learn_path is not None:
                    self._finish_path(path, (W, H))

    def _finish_path(self, path: dict | None, out_size: tuple[int, int] | None = None) -> None:
        """Belt direction known → digital zoom along the path and exposure limit for the belt speed."""
        ac, prof = self.cfg.autosetup, self.learn_profile
        self.learn_path = None
        if prof is None:
            return
        axis = path["axis"] if path else (prof.get("axis") or self._base_axis)
        cx, cy = prof["center"]
        if path:
            if axis == "x":
                cy = path["across"]
            else:
                cx = path["across"]
        crop, zoom = AS.crop_for_path((cx, cy), tuple(prof["diameter"]), axis, ac)
        prof.update(crop=list(crop), zoom=zoom, axis=axis)
        if path:
            W, H = out_size or (self.last_raw.shape[1], self.last_raw.shape[0])
            v_out = path["speed"] * (W if axis == "x" else H) / crop[2]       # output px per second
            e_max = ac.max_blur_px / max(v_out, 1e-6) * 1e6
            product = prof["exposure_us"] * prof["gain"]
            e, g = AS.split_exposure(product, min(ac.max_exposure_us, e_max), ac)
            prof.update(exposure_us=e, gain=g, belt_speed_px_s=round(v_out, 1),
                        motion_blur_px=round(v_out * e / 1e6, 2))
            if product / e > ac.max_gain + 1e-6:
                msg = (f"Not enough light for this belt speed: motion blur {prof['motion_blur_px']:.1f} px "
                       f"(goal ≤ {ac.max_blur_px}). Brighter light or a slower belt improves accuracy.")
                prof.setdefault("notes", []).append(msg)
                self.log("warn", msg)
        self._applied = None
        self._apply_camera(prof)
        self.trigger.armed, self.trigger.prev_pos = False, None     # ignore the part still in the image
        self.log("ok", f"Belt direction {axis}, digital zoom {zoom:.2f}×"
                       + (f", exposure {prof['exposure_us'] / 1000:.2f} ms (motion blur {prof['motion_blur_px']} px)"
                          if path else "") + " – now capturing the good parts.")

    def calib_start(self, cols: int | None = None, rows: int | None = None, square_mm: float | None = None) -> None:
        self._apply_camera(None)            # calibrate at the full field of view (zoom is accounted for later)
        c = self.cfg.calibration
        self.calib_session = {"cols": int(cols or c.board_cols), "rows": int(rows or c.board_rows),
                              "square_mm": float(square_mm or c.square_mm), "corners": [], "size": None,
                              "preview": None}

    def calib_capture(self) -> dict:
        if self.mode != "idle":
            raise ValueError("Stop learning/inspection before calibrating.")
        if self.calib_session is None:
            self.calib_start()
        s = self.calib_session
        frame = self.last_raw
        if frame is None:
            raise ValueError("No camera image yet.")
        corners = qcal.find_corners(frame, s["cols"], s["rows"])
        s["preview"] = _jpg(resize_to_width(qcal.draw_corners(frame, corners, s["cols"], s["rows"]), 640), 80)
        if corners is None:
            raise ValueError(f"Checkerboard with {s['cols']} × {s['rows']} inner corners not found – "
                             "check the corner count, lighting and that the whole board is visible.")
        size = (frame.shape[1], frame.shape[0])
        if s["size"] and s["size"] != size:
            raise ValueError("Camera resolution changed during the calibration – start again.")
        s["size"] = size
        s["corners"].append(corners)
        n = len(s["corners"])
        self.log("info", f"Calibration image {n} captured" + (" (scale reference – board flat on the belt)." if n == 1 else "."))
        return {"count": n}

    def calib_compute(self) -> dict:
        s = self.calib_session
        if not s or not s["corners"]:
            raise ValueError("Capture at least one checkerboard image first.")
        cal = qcal.compute(s["corners"], s["size"], s["cols"], s["rows"], s["square_mm"])
        cal.save(self.cfg.calibration.file)
        with self.lock:
            self.calibration = cal
            self.undistort = qcal.Undistorter(cal)
        self.calib_session = None
        self._apply_camera(self.model.camera_profile if self.model else None)
        msg = f"Calibration saved: {self.mm_per_px:.4f} mm/px"
        if cal.camera_matrix:
            msg += f", lens distortion corrected (RMS {cal.rms_px:.2f} px)"
        elif not cal.note:
            msg += f" (scale only – capture ≥ {qcal.MIN_IMAGES_DISTORTION} images to also correct lens distortion)"
        self.log("ok", msg + ". Retrain existing models.")
        if cal.note:
            self.log("warn", cal.note)
        return self.calibration_status()

    def calib_cancel(self) -> None:
        self.calib_session = None
        self._apply_camera(self.model.camera_profile if self.model else None)

    def calib_delete(self) -> None:
        Path(self.cfg.calibration.file).unlink(missing_ok=True)
        with self.lock:
            self.calibration, self.undistort = None, None
        self.log("info", "Calibration deleted – measuring in pixels.")

    def calibration_status(self) -> dict:
        c = self.calibration
        s = self.calib_session
        return {
            "calibrated": c is not None,
            "id": c.id if c else None,
            "mm_per_px": round(self.mm_per_px, 5) if c else None,
            "distortion": bool(c and c.camera_matrix),
            "rms_px": round(c.rms_px, 3) if c and c.rms_px is not None else None,
            "n_images": c.n_images if c else 0,
            "board": c.board if c else None,
            "note": c.note if c else "",
            "session": {"count": len(s["corners"]), "cols": s["cols"], "rows": s["rows"],
                        "square_mm": s["square_mm"], "has_preview": s["preview"] is not None} if s else None,
            "defaults": {"cols": self.cfg.calibration.board_cols, "rows": self.cfg.calibration.board_rows,
                         "square_mm": self.cfg.calibration.square_mm},
        }
