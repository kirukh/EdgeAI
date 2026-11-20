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

    def colour(self, addr: int = 0x30) -> tuple[int, int, int]:
        """Last colour written – registers of display version 5 (0x30) or the older PCA9633 (0x62)."""
        regs = (0x06, 0x07, 0x08) if addr == 0x30 else (0x04, 0x03, 0x02)
        rgb = {r: v for a, r, v in self.writes if a == addr and r in regs}
        return rgb[regs[0]], rgb[regs[1]], rgb[regs[2]]


def test_lines_are_plain_ascii_and_16_characters():
    assert fit("Part 7") == "Part 7          "
    assert fit("Prüfung läuft – größer als sechzehn") == "Pruefung laeuft "
    assert fit("Grün → OK") == "Gruen ? OK      "
    assert address("0x3e") == 0x3E and address(48) == 0x30


def test_grove_lcd_writes_both_lines_and_only_what_changed():
    bus = FakeBus()
    lcd = GroveLCD(bus)
    assert bus.writes[-2:] == [(0x30, 0x00, 0x07), (0x30, 0x04, 0x15)]  # v5: reset, all LEDs on
    lcd.write("Part 1", "OK")
    lcd.colour((0, 160, 0))
    assert bus.text() == fit("Part 1") + fit("OK") and bus.colour() == (0, 160, 0)
    assert (0x3E, 0x80, 0xC0) in bus.writes  # cursor to line 2
    bus.writes.clear()
    lcd.write("Part 1", "NOT OK")  # line 1 unchanged → not sent again
    assert bus.text() == fit("NOT OK")


def test_backlight_registers_of_both_display_versions():
    """Version 5 (0x30) has the LED mode in register 0x04 – a colour value there switches the LEDs off or
    makes them blink. The colour goes to 0x06–0x08 there, to 0x04/0x03/0x02 on older displays (0x62)."""
    for addr, setup in ((0x30, (0x30, 0x04, 0x15)), (0x62, (0x62, 0x08, 0xAA))):
        bus = FakeBus()
        lcd = GroveLCD(bus, rgb_addr=addr)
        assert setup in bus.writes
        bus.writes.clear()
        lcd.colour((0, 160, 0))
        assert bus.colour(addr) == (0, 160, 0) and len(bus.writes) == 3
        if addr == 0x30:
            assert all(r != 0x04 for _, r, _ in bus.writes)  # the LED mode stays "on"


def test_part_display_shows_counter_result_and_colour():
    bus = FakeBus()
    disp = PartDisplay(LCDConfig(enabled=True), lambda *_: None, lambda c: GroveLCD(bus))
    try:
        disp.result(12, "NOK")
        wait(lambda: disp.shown == ("Part 12", "NOT OK", "NOK"), 5)
        assert bus.colour() == COLOURS["NOK"]
        disp.result(13, "OK")
        wait(lambda: disp.shown == ("Part 13", "OK", "OK"), 5)
        assert bus.colour() == COLOURS["OK"]
        disp.ready(0)
        wait(lambda: disp.shown == ("Part 0", "Ready", "idle"), 5)
    finally:
        disp.close()


def test_result_turns_blue_when_nothing_happens():
    bus = FakeBus()
    disp = PartDisplay(LCDConfig(enabled=True, result_seconds=0.3), lambda *_: None, lambda c: GroveLCD(bus))
    try:
        disp.result(5, "OK")
        wait(lambda: disp.shown == ("Part 5", "OK", "OK"), 5)
        assert bus.colour() == COLOURS["OK"]
        wait(lambda: disp.shown == ("Part 5", "Ready", "idle"), 5)  # no new part → blue
        assert bus.colour() == COLOURS["idle"] == (0, 0, 255)
        disp.stopped(5)
        wait(lambda: disp.shown == ("Part 5", "Stopped", "idle"), 5)
    finally:
        disp.close()


def test_display_is_reset_at_start_and_dark_at_the_end():
    """Whatever another program (e.g. lcd-test) left on the display is cleared when qc starts."""
    bus = FakeBus()
    disp = PartDisplay(LCDConfig(enabled=True), lambda *_: None, lambda c: GroveLCD(bus))
    disp.ready(0)
    wait(lambda: disp.shown == ("Part 0", "Ready", "idle"), 5)
    first = bus.writes[:6]
    assert [w for w in first if w[0] == 0x3E][:3] == [(0x3E, 0x80, 0x28)] * 3  # power-on initialisation
    assert (0x3E, 0x80, 0x01) in bus.writes  # cleared
    assert bus.writes.index((0x30, 0x00, 0x07)) < bus.writes.index((0x30, 0x04, 0x15))  # backlight reset, LEDs on
    disp.close()
    assert disp.shown == ("", "", "off") and bus.colour() == (0, 0, 0)
    assert bus.writes[-4] == (0x3E, 0x80, 0x01)  # text cleared at the end


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
        want = (f"Part {s.part_count}", "OK" if last["status"] == "OK" else "NOT OK", last["status"])
        wait(lambda: s.lcd.shown == want, 10)
        s.stop_inspection()
    wait(lambda: s.lcd.shown == (f"Part {s.part_count}", "Stopped", "idle"), 10)
    s.reset_counts()
    wait(lambda: s.lcd.shown == ("Part 0", "Ready", "idle"), 10)
