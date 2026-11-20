"""Grove-LCD RGB Backlight: piece counter on line 1, result on line 2, backlight colour."""

from __future__ import annotations

import time

import pytest
from conftest import wait

from qc.config import AppConfig, ConfigError, LCDConfig
from qc.lcd import COLOURS, GroveLCD, PartDisplay, address, fit


class FakeBus:
    """Records the I²C writes; ``fail`` makes every write raise (display unplugged)."""

    def __init__(self, fail: bool = False):
        self.writes: list[tuple[int, int, int]] = []
        self.fail = fail

    def write_byte_data(self, addr: int, register: int, value: int) -> None:
        if self.fail:
            raise OSError("[Errno 121] Remote I/O error")
        self.writes.append((addr, register, value))

    def text(self) -> str:
        return "".join(chr(v) for a, r, v in self.writes if a == 0x3E and r == 0x40)

    def colour(self) -> tuple[int, int, int]:
        rgb = {r: v for a, r, v in self.writes if a == 0x30 and r in (2, 3, 4)}
        return rgb[4], rgb[3], rgb[2]


def test_lines_are_plain_ascii_and_16_characters():
    assert fit("Teil 7") == "Teil 7          "
    assert fit("Prüfung läuft – größer als sechzehn") == "Pruefung laeuft "
    assert fit("Grün → OK") == "Gruen ? OK      "
    assert address("0x3e") == 0x3E and address(48) == 0x30


def test_grove_lcd_writes_both_lines_and_only_what_changed():
    bus = FakeBus()
    lcd = GroveLCD(bus)
    assert (0x30, 0x08, 0xAA) in bus.writes  # backlight chip set up
    lcd.write("Teil 1", "OK")
    lcd.colour((0, 160, 0))
    assert bus.text() == fit("Teil 1") + fit("OK") and bus.colour() == (0, 160, 0)
    assert (0x3E, 0x80, 0xC0) in bus.writes  # cursor to line 2
    bus.writes.clear()
    lcd.write("Teil 1", "NOT OK")  # line 1 unchanged → not sent again
    assert bus.text() == fit("NOT OK")


def test_part_display_shows_counter_result_and_colour():
    bus = FakeBus()
    disp = PartDisplay(LCDConfig(enabled=True), lambda *_: None, lambda c: GroveLCD(bus))
    try:
        disp.result(12, "NOK")
        wait(lambda: disp.shown == ("Teil 12", "NOT OK", "NOK"), 5)
        assert bus.colour() == COLOURS["NOK"]
        disp.result(13, "OK")
        wait(lambda: disp.shown == ("Teil 13", "OK", "OK"), 5)
        assert bus.colour() == COLOURS["OK"]
        disp.ready(0)
        wait(lambda: disp.shown == ("Teil 0", "Bereit", "idle"), 5)
    finally:
        disp.close()


def test_result_turns_blue_when_nothing_happens():
    bus = FakeBus()
    disp = PartDisplay(LCDConfig(enabled=True, result_seconds=0.3), lambda *_: None, lambda c: GroveLCD(bus))
    try:
        disp.result(5, "OK")
        wait(lambda: disp.shown == ("Teil 5", "OK", "OK"), 5)
        assert bus.colour() == COLOURS["OK"]
        wait(lambda: disp.shown == ("Teil 5", "Bereit", "idle"), 5)  # no new part → blue
        assert bus.colour() == COLOURS["idle"] == (0, 0, 255)
        disp.stopped(5)
        wait(lambda: disp.shown == ("Teil 5", "Gestoppt", "idle"), 5)
    finally:
        disp.close()


def test_missing_display_is_reported_once_and_does_not_stop_anything():
    messages: list[tuple[str, str]] = []
    disp = PartDisplay(
        LCDConfig(enabled=True), lambda lvl, txt: messages.append((lvl, txt)), lambda c: GroveLCD(FakeBus(True))
    )
    try:
        disp.result(1, "OK")
        wait(lambda: disp.error is not None, 5)
        disp.result(2, "OK")
        time.sleep(0.2)
        assert len(messages) == 1 and messages[0][0] == "warn" and "I2C" in messages[0][1]
    finally:
        disp.close()


def test_disabled_display_does_nothing():
    calls: list[LCDConfig] = []
    disp = PartDisplay(LCDConfig(), lambda *_: None, lambda c: calls.append(c))
    disp.result(1, "OK")
    disp.close()
    assert not calls and disp.shown is None


def test_config_checks_the_addresses(tmp_path):
    p = tmp_path / "config.json"
    p.write_text('{"lcd": {"enabled": true, "rgb_address": "0x62"}}')
    assert AppConfig.load(p).lcd.rgb_address == "0x62"
    p.write_text('{"lcd": {"lcd_address": "hello"}}')
    with pytest.raises(ConfigError, match=r"lcd\.lcd_address"):
        AppConfig.load(p)


def test_inspection_counts_parts_on_the_display(make_system):
    s = make_system(sim_defect_rate=0.5)
    bus = FakeBus()
    s.lcd.close()
    s.lcd = PartDisplay(LCDConfig(enabled=True), s.log, lambda c: GroveLCD(bus))
    s.start()
    s.start_learning("Plate", auto_finish=True)
    s.teach.target = 4
    wait(lambda: s.mode == "idle" and s.model is not None, 120)
    s.start_inspection()
    wait(lambda: s.part_count >= 3, 120)
    with s.lock:  # no further result while checking
        last = s.results.history[0]
        want = (f"Teil {s.part_count}", "OK" if last["status"] == "OK" else "NOT OK", last["status"])
        wait(lambda: s.lcd.shown == want, 10)
        s.stop_inspection()
    wait(lambda: s.lcd.shown == (f"Teil {s.part_count}", "Gestoppt", "idle"), 10)
    s.reset_counts()
    wait(lambda: s.lcd.shown == ("Teil 0", "Bereit", "idle"), 10)
