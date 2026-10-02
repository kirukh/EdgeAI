"""Status for the web UI and plain-language notices for the supervisor."""
from __future__ import annotations



from . import model as qm




class StatusMixin:
    """Mixin of :class:`qc.system.QCSystem` (uses its state: camera, cfg, lock, model, …)."""

    DRIFT_HINTS = {
        "contrast": "The image contrast has changed. Check the lighting (lamp off, dirty or ageing).",
        "background": "The belt / background brightness has changed. Check ambient light, clean the belt or backlight.",
        "area": "Parts look bigger or smaller than when taught. Has the camera been moved?",
        "sharpness": "The image is getting blurry. Clean the camera lens and check the focus.",
    }

    def supervisor_notices(self, drift: dict | None = None) -> list[dict]:
        """Short plain-language hints for the supervisor screen (no technical details)."""
        out: list[dict] = []
        if self.camera.continuous and self.fps == 0 and self.last_frame is None:
            out.append({"level": "error", "text": "No camera image. Check the camera cable and restart the device."})
        if self.model is None and self.mode == "idle":
            out.append({"level": "info", "text": "Select a part, or teach in a new part."})
        if self.model_warnings():
            out.append({"level": "error", "text": "This part has to be taught in again (the camera setup was "
                                                  "changed). Call the setup technician or teach it in again."})
        if self.model is not None:
            drift = drift or self.drift.state()
            keys = {r["key"] for r in drift.get("rows", []) if r.get("warning")}
            out += [{"level": "warn", "text": self.DRIFT_HINTS[k]} for k in ("sharpness", "contrast", "background", "area")
                    if k in keys]
        if self.skipped:
            out.append({"level": "error", "text": f"{self.skipped} part(s) were rejected without inspection because "
                                                  "the system could not keep up. Slow the belt down."})
        prof = (self.model.camera_profile or {}) if self.model else {}
        if prof.get("sharp") is False:
            out.append({"level": "warn", "text": "The camera image is not sharp – the lens has to be focused once "
                                                 "(setup technician). Detection is less precise until then."})
        st = self.last_selftest
        if st and not st.get("passed") and self.model is not None and st.get("model") == self.model.name:
            out.append({"level": "error", "text": "The last reference part check FAILED. Do not start production – "
                                                  "call the setup technician."})
        ls = self.last_self_learning
        if ls and ls.get("automatic") and not ls.get("accepted"):
            out.append({"level": "info", "text": "An automatic model update was rejected (safety check). "
                                                 "The model is unchanged – please inform the setup technician."})
        return out

    def status(self) -> dict:
        m = self.model
        drift = self.drift.state() if m else {"rows": [], "warnings": []}
        return {
            "mode": self.mode, "camera": self.camera.name, "fps": round(self.fps, 1),
            "armed": self.trigger.armed, "continuous": self.camera.continuous,
            "model": m.summary() if m else None,
            "model_warnings": self.model_warnings(),
            "learning": {"name": self.learn_name, "count": len(self.learn_captures),
                         "target": self.cfg.target_reference_count, "minimum": qm.MIN_REFERENCES,
                         "auto_finish": self.learn_auto_finish, "path_pending": self.learn_path is not None},
            "camera_setup": self.cam_setup,
            "camera_controls": bool(self.camera.can_control and self.cfg.autosetup.enabled),
            "camera_state": {"exposure_us": (self._applied or {}).get("exposure", (None,))[0] if (self._applied or {}).get("exposure") else None,
                             "gain": (self._applied or {}).get("exposure", (None, None))[1] if (self._applied or {}).get("exposure") else None,
                             "zoom": round(1.0 / self.camera.crop[2], 2), "axis": self.cfg.trigger.axis,
                             "auto": bool(m and (m.camera_profile or {}).get("auto"))},
            "self_learning_running": self._bg_learning is not None,
            "queue": self._jobs.qsize(), "skipped": self.skipped,
            "notices": self.supervisor_notices(drift),
            "setup_pin_required": bool(self.cfg.ui.setup_pin),
            "focus_assist": {"value": round(self.focus_assist["value"], 1), "best": round(self.focus_assist["best"], 1)}
            if self.focus_assist else None,
            "camera_info": getattr(self.camera, "info", ""),
            "counts": self.counts, "feedback": self.feedback_counts, "last_result": self.last_result,
            "messages": list(self.messages)[:12],
            "settings": {"shots_per_part": self.shots_per_part, "lighting_mode": self.cfg.lighting.mode,
                         "idle_light": self.cfg.lighting.idle_light,
                         "rotation_search": self.cfg.localization.rotation_search,
                         "self_learning": self.cfg.self_learning.enabled,
                         "self_learning_auto": self.cfg.self_learning.auto,
                         "self_learning_every": self.cfg.self_learning.auto_every,
                         "self_learning_margin": self.cfg.self_learning.margin,
                         "self_learning_max_pool": self.cfg.self_learning.max_pool},
            "last_self_learning": self.last_self_learning,
            "io": self.io.status(),
            "drift": drift,
            "calibration": self.calibration_status(),
            "selftest": {"active": self.selftest, "last": self.last_selftest},
            "sim": {"lighting": self.camera.cfg.sim_lighting, "defect_rate": self.camera.defect_rate,
                    "part_type": self.camera.part_type, "any_angle": self.camera.cfg.sim_any_angle,
                    "board": self.camera.board is not None}
            if self.is_sim else None,
        }
