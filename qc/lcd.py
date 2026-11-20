"""Grove-LCD RGB Backlight (16 × 2 characters, I²C) next to the conveyor: piece counter and result.

    ┌────────────────┐
    │Teil 128        │   line 1: running number of the inspected part (OK + NOK since the last reset)
    │OK              │   line 2: result of that part – backlight green (OK) or red (NOT OK)
    └────────────────┘

Backlight: green = OK, red = NOT OK, blue = nothing happening (inspection stopped, waiting for the
first part, or no new part for ``lcd.result_seconds`` – then line 2 shows "Bereit" again).

The display sits on the Pi's I²C bus 1 (Grove pins SDA/SCL, 3.3 V or 5 V, GND). It has two
chips: the text controller (address 0x3e) and the backlight. The backlight chip differs between
the versions of the display, with different registers (as in Seeed's own library rgb_lcd.cpp):

* version 5, address 0x30: reset 0x00 ← 0x07, all LEDs on 0x04 ← 0x15; red/green/blue = 0x06/0x07/0x08
  (register 0x04 is the LED MODE there – writing a colour value into it makes the LEDs blink or go off)
* older versions, address 0x62 (PCA9633): 0x00 ← 0, 0x01 ← 0, 0x08 ← 0xAA; red/green/blue = 0x04/0x03/0x02
 I²C is slow (a few ms per character), so the display is written by its
own thread – the inspection never waits for it. If the display is missing or the bus fails, a
warning is logged once and the inspection carries on without it.

Needs the Python package ``smbus2`` or ``smbus`` (``sudo apt install python3-smbus``) and I²C
switched on (``sudo raspi-config`` → Interface Options → I2C).
"""

from __future__ import annotations

import importlib
import threading
import time
from collections.abc import Callable
from typing import Any, Protocol

from .config import LCDConfig

WIDTH = 16
RGB_V5 = 0x30  # backlight chip of display version 5 (older versions: 0x62)
COLOURS = {  # backlight (R, G, B)
    "idle": (0, 0, 255),
    "OK": (0, 160, 0),
    "NOK": (255, 0, 0),
    "NO_PART": (160, 110, 0),
}


class Bus(Protocol):
    """The part of ``smbus2.SMBus`` / ``smbus.SMBus`` that is used."""

    def write_byte_data(self, addr: int, register: int, value: int) -> None:
        """Writes one byte into a register of the device at ``addr``."""


def open_bus(number: int) -> Bus:
    """I²C bus ``number`` via smbus2 or (system package) smbus."""
    for name in ("smbus2", "smbus"):  # pip package / Raspberry Pi OS system package
        try:
            module = importlib.import_module(name)
        except ImportError:
            continue
        bus: Bus = module.SMBus(number)
        return bus
    raise RuntimeError("Python package for I2C missing – install it with:  sudo apt install python3-smbus")


def address(value: int | str) -> int:
    """I²C address from config.json: a number (62) or a hex string ("0x3e")."""
    return int(value, 0) if isinstance(value, str) else int(value)


def fit(text: str) -> str:
    """Text for one line: plain ASCII (the display has no umlauts), cut/padded to 16 characters."""
    plain = text.replace("ä", "ae").replace("ö", "oe").replace("ü", "ue").replace("ß", "ss")
    plain = plain.replace("Ä", "Ae").replace("Ö", "Oe").replace("Ü", "Ue")
    plain = "".join(c if 32 <= ord(c) < 127 else "?" for c in plain)
    return plain[:WIDTH].ljust(WIDTH)


class GroveLCD:
    """Text and backlight of the Grove-LCD RGB Backlight (direct I²C commands, no library needed)."""

    def __init__(self, bus: Bus, lcd_addr: int = 0x3E, rgb_addr: int = 0x30):
        self.bus, self.lcd_addr, self.rgb_addr = bus, lcd_addr, rgb_addr
        self._command(0x28)  # 2 lines, 5×8 dots
        self._command(0x0C)  # display on, cursor off
        self._command(0x06)  # cursor moves to the right
        self._command(0x01)  # clear
        time.sleep(0.002)
        self.v5 = rgb_addr == RGB_V5
        if self.v5:
            self.bus.write_byte_data(rgb_addr, 0x00, 0x07)  # reset the backlight chip
            time.sleep(0.001)
            self.bus.write_byte_data(rgb_addr, 0x04, 0x15)  # all three LEDs on (brightness from 0x06–0x08)
        else:  # PCA9633
            self.bus.write_byte_data(rgb_addr, 0x00, 0x00)  # normal mode
            self.bus.write_byte_data(rgb_addr, 0x01, 0x00)
            self.bus.write_byte_data(rgb_addr, 0x08, 0xAA)  # all LEDs PWM-controlled
        self.lines = ["", ""]

    def _command(self, cmd: int) -> None:
        self.bus.write_byte_data(self.lcd_addr, 0x80, cmd)
        time.sleep(0.001)

    def colour(self, rgb: tuple[int, int, int]) -> None:
        """Backlight colour (0–255 each)."""
        r, g, b = rgb
        if self.v5:
            for register, value in ((0x06, r), (0x07, g), (0x08, b)):
                self.bus.write_byte_data(self.rgb_addr, register, value)
            return
        self.bus.write_byte_data(self.rgb_addr, 0x04, r)
        self.bus.write_byte_data(self.rgb_addr, 0x03, g)
        self.bus.write_byte_data(self.rgb_addr, 0x02, b)

    def write(self, line1: str, line2: str) -> None:
        """Both lines; only lines whose text changed are sent (I²C is slow)."""
        for row, text in enumerate((fit(line1), fit(line2))):
            if text == self.lines[row]:
                continue
            self._command(0x80 if row == 0 else 0xC0)  # cursor to the start of the line
            for ch in text:
                self.bus.write_byte_data(self.lcd_addr, 0x40, ord(ch))
            self.lines[row] = text


class PartDisplay:
    """Piece counter + result on the LCD, written by a background thread (newest state wins)."""

    def __init__(
        self,
        cfg: LCDConfig,
        log: Callable[[str, str], None],
        connect: Callable[[LCDConfig], Any] | None = None,
    ):
        self.cfg, self.log = cfg, log
        self._connect = connect or (lambda c: GroveLCD(open_bus(c.bus), address(c.lcd_address), address(c.rgb_address)))
        self._cv = threading.Condition()
        self._want: tuple[str, str, str] | None = None
        self._stop = False
        self.error: str | None = None
        self.shown: tuple[str, str, str] | None = None  # last state written to the display
        self._thread: threading.Thread | None = None
        if cfg.enabled:
            self._thread = threading.Thread(target=self._run, name="qc-lcd", daemon=True)
            self._thread.start()

    # ---- what the system calls
    def ready(self, count: int) -> None:
        """No result yet (start, counter reset): number of parts so far and "ready"."""
        self._show(self.cfg.line1.format(n=count), self.cfg.ready_text, "idle")

    def stopped(self, count: int) -> None:
        """Inspection stopped / nothing running: blue, line 2 "Gestoppt"."""
        self._show(self.cfg.line1.format(n=count), self.cfg.stopped_text, "idle")

    def result(self, count: int, status: str) -> None:
        """Result of an inspected part: ``count`` = its running number, ``status`` OK / NOK / NO_PART."""
        text = {"OK": self.cfg.ok_text, "NOK": self.cfg.nok_text}.get(status, self.cfg.no_part_text)
        self._show(self.cfg.line1.format(n=count), text, status)

    def close(self) -> None:
        """Stops the display thread."""
        with self._cv:
            self._stop = True
            self._cv.notify_all()
        if self._thread is not None:
            self._thread.join(timeout=2)

    # ---- display thread
    def _show(self, line1: str, line2: str, state: str) -> None:
        if not self.cfg.enabled:
            return
        with self._cv:
            self._want = (line1, line2, state)
            self._cv.notify_all()

    def _run(self) -> None:
        lcd = None
        while True:
            with self._cv:
                shown = self.shown
                # a result stays green/red for result_seconds; without a new part the display turns blue
                hold = self.cfg.result_seconds if shown and shown[2] != "idle" and self.cfg.result_seconds > 0 else None
                if not self._cv.wait_for(lambda: self._stop or self._want is not None, timeout=hold):
                    assert shown is not None
                    self._want = (shown[0], self.cfg.ready_text, "idle")
                if self._stop:
                    return
                want, self._want = self._want, None
            assert want is not None
            try:
                if lcd is None:
                    lcd = self._connect(self.cfg)
                line1, line2, state = want
                lcd.write(line1, line2)
                lcd.colour(COLOURS.get(state, COLOURS["idle"]))
                self.shown, self.error = want, None
            except Exception as e:  # display unplugged, I²C off … – the inspection carries on
                lcd = None
                if self.error is None:
                    self.log(
                        "warn",
                        f"LCD display not reachable ({e}). Check the cable (SDA/SCL), that I2C is switched on "
                        "(sudo raspi-config → Interface Options → I2C) and the addresses in config.json (lcd).",
                    )
                self.error = str(e)
                time.sleep(1.0)
