"""Digital outputs: reject (ejector), signal lamps and lighting.

On the Raspberry Pi the outputs are driven via ``gpiozero`` (BCM pin numbers).
Without a Pi – or if no pin is configured – everything runs in *simulated* mode:
the actions are only logged, so the process can be tested without hardware.

Reject timing: the ejector sits downstream of the camera. ``reject_delay_ms`` is
the travel time of a part from the camera to the ejector; it is counted from the
moment the image was captured (not from the end of the inspection), so the
inspection time does not shift the ejection point.

Wiring note: never connect valves, relays or LED strips directly to a GPIO pin –
use a driver (transistor/MOSFET module or relay board with a free-wheeling diode).
"""
from __future__ import annotations

import threading
import time
from collections import deque
from datetime import datetime

from .config import IOConfig


class _Pin:
    def __init__(self, pin: int | None, active_high: bool, backend: str, name: str, log):
        self.pin, self.name, self.log = pin, name, log
        self.dev = None
        self.state = False
        if pin is not None and backend == "gpio":
            from gpiozero import DigitalOutputDevice

            self.dev = DigitalOutputDevice(pin, active_high=active_high, initial_value=False)

    def set(self, on: bool, quiet: bool = False) -> None:
        if self.pin is None:
            return
        self.state = on
        if self.dev is not None:
            self.dev.on() if on else self.dev.off()
        if not quiet:
            self.log(f"{self.name} (GPIO {self.pin}) {'ON' if on else 'OFF'}")

    def pulse(self, ms: float) -> None:
        if self.pin is None:
            return
        self.set(True)
        threading.Timer(ms / 1000.0, self.set, args=(False,)).start()

    def close(self) -> None:
        if self.dev is not None:
            self.dev.off()
            self.dev.close()


class IOController:
    def __init__(self, cfg: IOConfig, light_hook=None):
        """``light_hook(name)`` is additionally called when the light changes (used by the simulator)."""
        self.cfg = cfg
        self.events: deque[dict] = deque(maxlen=40)
        self.light_hook = light_hook
        self.backend = "simulated"
        pins = [cfg.reject_pin, cfg.ok_lamp_pin, cfg.nok_lamp_pin, cfg.front_light_pin, cfg.back_light_pin]
        if cfg.enabled and any(p is not None for p in pins):
            try:
                import gpiozero  # noqa: F401

                self.backend = "gpio"
            except Exception as e:  # noqa: BLE001 - no Pi / library missing → simulated
                self._log(f"GPIO not available ({e}) – running simulated")
        hi = cfg.active_high
        self.reject = _Pin(cfg.reject_pin, hi, self.backend, "Reject", self._log)
        self.ok_lamp = _Pin(cfg.ok_lamp_pin, hi, self.backend, "OK lamp", self._log)
        self.nok_lamp = _Pin(cfg.nok_lamp_pin, hi, self.backend, "NOK lamp", self._log)
        self.lights = {"front": _Pin(cfg.front_light_pin, hi, self.backend, "Front light", self._log),
                       "back": _Pin(cfg.back_light_pin, hi, self.backend, "Backlight", self._log)}
        self.current_light: str | None = None
        self.rejects = 0

    def _log(self, text: str) -> None:
        self.events.appendleft({"t": datetime.now().strftime("%H:%M:%S.%f")[:-3], "text": text})

    # -------------------------------------------------------------- lighting
    def set_light(self, name: str) -> None:
        """Switches to front light, backlight (or "off")."""
        if name == self.current_light:
            return
        for n, pin in self.lights.items():
            pin.set(n == name, quiet=True)
        self.current_light = name
        if self.light_hook:
            self.light_hook(name)

    # ------------------------------------------------------------- results
    def signal_result(self, status: str, t_capture: float | None = None) -> None:
        """Signal lamps + delayed reject pulse for NOK parts."""
        if status not in ("OK", "NOK"):
            return
        lamp = self.ok_lamp if status == "OK" else self.nok_lamp
        if lamp.pin is not None:
            other = self.nok_lamp if status == "OK" else self.ok_lamp
            other.set(False, quiet=True)
            lamp.set(True, quiet=True)
            threading.Timer(self.cfg.lamp_ms / 1000.0, lamp.set, args=(False, True)).start()
        if status == "NOK" and self.cfg.reject_pin is not None:
            elapsed = (time.time() - t_capture) * 1000 if t_capture else 0.0
            wait = max(0.0, self.cfg.reject_delay_ms - elapsed)
            if elapsed > self.cfg.reject_delay_ms:
                self._log(f"WARNING: inspection took {elapsed:.0f} ms – longer than the reject delay "
                          f"({self.cfg.reject_delay_ms} ms); part may have passed the ejector")
            self.rejects += 1
            threading.Timer(wait / 1000.0, self.reject.pulse, args=(self.cfg.reject_pulse_ms,)).start()
            self._log(f"NOK → reject scheduled in {wait:.0f} ms")

    def test_reject(self) -> None:
        if self.cfg.reject_pin is None:
            self._log("Test: no reject pin configured (io.reject_pin)")
            return
        self._log("Test reject pulse")
        self.reject.pulse(self.cfg.reject_pulse_ms)

    def status(self) -> dict:
        c = self.cfg
        return {"backend": self.backend, "reject_pin": c.reject_pin, "reject_delay_ms": c.reject_delay_ms,
                "ok_lamp_pin": c.ok_lamp_pin, "nok_lamp_pin": c.nok_lamp_pin,
                "front_light_pin": c.front_light_pin, "back_light_pin": c.back_light_pin,
                "light": self.current_light, "rejects": self.rejects, "events": list(self.events)[:12]}

    def close(self) -> None:
        for p in [self.reject, self.ok_lamp, self.nok_lamp, *self.lights.values()]:
            p.close()
