"""GPS reader for the gpyes 2.0 (u-blox 8) USB module.

The module enumerates as a CDC-ACM serial device (/dev/ttyACM0) speaking
NMEA 0183. We parse:
  RMC - position, speed over ground, course over ground, UTC, validity
  GGA - fix quality, satellite count, HDOP, altitude

Rate: u-blox 8 defaults to 1 Hz. For a moving car that makes ego_speed up
to 1 s stale, so on startup we optionally send UBX-CFG-RATE to raise the
measurement rate (5 Hz is reliable for multi-GNSS on M8). If the write
fails the reader still works at 1 Hz - check ``rate_configured`` in
diagnostics.

Permissions: the port is group ``dialout``. One-time fix on a new device:
    sudo usermod -aG dialout $USER   # then log out and back in
"""

from __future__ import annotations

import math
import threading
from dataclasses import dataclass, field, replace

import pynmea2
import serial

from sensors.time_sync import TimebaseStamp, GpsUtcOffsetTracker, now_mono, now_wall

KNOTS_TO_MPS = 0.514444


@dataclass(frozen=True)
class GpsFix:
    valid: bool = False
    lat: float = float("nan")
    lon: float = float("nan")
    speed_mps: float = float("nan")
    heading_deg: float = float("nan")
    fix_quality: int = 0
    num_sats: int = 0
    hdop: float = float("nan")
    altitude_m: float = float("nan")
    utc_epoch_s: float = float("nan")
    t_mono: float = 0.0          # when the sentence was read (monotonic)
    t_wall: float = 0.0
    # As on Frame: set only when the fix came from another device, in which case
    # t_mono above is converted or proxied rather than locally observed.
    timebase: "TimebaseStamp | None" = None

    def age_s(self, t_mono_now: float) -> float:
        return t_mono_now - self.t_mono if self.t_mono > 0 else float("inf")


@dataclass
class GpsDiagnostics:
    sentences_parsed: int = 0
    parse_errors: int = 0
    rate_configured: bool = False
    port_open: bool = False
    last_error: str = ""
    raw_log_path: str = ""
    recent_intervals_s: list = field(default_factory=list)

    def observed_rate_hz(self) -> float:
        if len(self.recent_intervals_s) < 2:
            return 0.0
        mean = sum(self.recent_intervals_s) / len(self.recent_intervals_s)
        return 1.0 / mean if mean > 0 else 0.0


def ubx_cfg_rate(meas_rate_ms: int) -> bytes:
    """Build a UBX-CFG-RATE frame (class 0x06, id 0x08).

    payload: measRate (u2, ms), navRate (u2, cycles), timeRef (u2, 1=GPS)
    """
    payload = (
        int(meas_rate_ms).to_bytes(2, "little")
        + (1).to_bytes(2, "little")
        + (1).to_bytes(2, "little")
    )
    body = bytes([0x06, 0x08]) + len(payload).to_bytes(2, "little") + payload
    ck_a = ck_b = 0
    for b in body:
        ck_a = (ck_a + b) & 0xFF
        ck_b = (ck_b + ck_a) & 0xFF
    return b"\xb5\x62" + body + bytes([ck_a, ck_b])


class GpsReader:
    def __init__(
        self,
        port: str = "/dev/ttyACM0",
        baud: int = 9600,
        configure_rate: bool = True,
        target_rate_hz: int = 5,
        stale_after_s: float = 2.0,
        raw_log_path: str | None = None,
    ) -> None:
        self.port = port
        self.baud = baud
        self.configure_rate = configure_rate
        self.target_rate_hz = max(1, int(target_rate_hz))
        self.stale_after_s = stale_after_s
        self.raw_log_path = raw_log_path

        self._serial: serial.Serial | None = None
        self._thread: threading.Thread | None = None
        self._lock = threading.Lock()
        self._fix = GpsFix()
        self._running = False
        self._raw_log = None
        self._last_rmc_mono: float | None = None
        self.diagnostics = GpsDiagnostics(raw_log_path=raw_log_path or "")
        self.utc_offset = GpsUtcOffsetTracker()

    # -- lifecycle -------------------------------------------------------

    def start(self) -> "GpsReader":
        try:
            self._serial = serial.Serial(self.port, self.baud, timeout=1.0)
        except (serial.SerialException, PermissionError) as exc:
            raise RuntimeError(
                f"cannot open GPS port {self.port}: {exc}\n"
                "If this is a permission error, run once:\n"
                "  sudo usermod -aG dialout $USER\n"
                "then log out and back in."
            ) from exc
        self.diagnostics.port_open = True
        if self.configure_rate:
            self._try_configure_rate()
        if self.raw_log_path:
            self._raw_log = open(self.raw_log_path, "a", buffering=1)
        self._running = True
        self._thread = threading.Thread(target=self._loop, name="gps", daemon=True)
        self._thread.start()
        return self

    def stop(self) -> None:
        self._running = False
        if self._thread is not None:
            self._thread.join(timeout=2.0)
        if self._serial is not None:
            self._serial.close()
        if self._raw_log is not None:
            self._raw_log.close()

    def _try_configure_rate(self) -> None:
        assert self._serial is not None
        try:
            frame = ubx_cfg_rate(int(1000 / self.target_rate_hz))
            self._serial.write(frame)
            self._serial.flush()
            self.diagnostics.rate_configured = True
        except serial.SerialException as exc:
            self.diagnostics.last_error = f"UBX rate config failed: {exc}"

    # -- reader thread ----------------------------------------------------

    def _loop(self) -> None:
        assert self._serial is not None
        while self._running:
            try:
                raw = self._serial.readline()
            except serial.SerialException as exc:
                self.diagnostics.last_error = str(exc)
                continue
            if not raw:
                continue
            t_mono, t_wall = now_mono(), now_wall()
            line = raw.decode("ascii", errors="replace").strip()
            if self._raw_log is not None and line.startswith("$"):
                self._raw_log.write(f"{t_wall:.3f} {line}\n")
            if not line.startswith("$"):
                continue
            try:
                msg = pynmea2.parse(line)
            except pynmea2.ParseError:
                self.diagnostics.parse_errors += 1
                continue
            self._ingest(msg, t_mono, t_wall)

    def _ingest(self, msg: pynmea2.NMEASentence, t_mono: float, t_wall: float) -> None:
        self.diagnostics.sentences_parsed += 1
        with self._lock:
            fix = self._fix
            if msg.sentence_type == "RMC":
                valid = getattr(msg, "status", "V") == "A"
                speed_knots = getattr(msg, "spd_over_grnd", None)
                course = getattr(msg, "true_course", None)
                utc = float("nan")
                if getattr(msg, "datetime", None) is not None:
                    try:
                        utc = msg.datetime.timestamp()
                        self.utc_offset.update(utc, t_wall)
                    except (ValueError, OSError, OverflowError):
                        pass
                fix = replace(
                    fix,
                    valid=valid,
                    lat=msg.latitude if valid else fix.lat,
                    lon=msg.longitude if valid else fix.lon,
                    speed_mps=(
                        float(speed_knots) * KNOTS_TO_MPS
                        if valid and speed_knots is not None
                        else fix.speed_mps
                    ),
                    heading_deg=(
                        float(course) if valid and course is not None else fix.heading_deg
                    ),
                    utc_epoch_s=utc if not math.isnan(utc) else fix.utc_epoch_s,
                    t_mono=t_mono,
                    t_wall=t_wall,
                )
                if self._last_rmc_mono is not None:
                    self.diagnostics.recent_intervals_s.append(t_mono - self._last_rmc_mono)
                    if len(self.diagnostics.recent_intervals_s) > 20:
                        self.diagnostics.recent_intervals_s.pop(0)
                self._last_rmc_mono = t_mono
            elif msg.sentence_type == "GGA":
                try:
                    fix = replace(
                        fix,
                        fix_quality=int(msg.gps_qual or 0),
                        num_sats=int(msg.num_sats or 0),
                        hdop=float(msg.horizontal_dil) if msg.horizontal_dil else float("nan"),
                        altitude_m=float(msg.altitude) if msg.altitude is not None else float("nan"),
                        t_mono=t_mono,
                        t_wall=t_wall,
                    )
                except (TypeError, ValueError):
                    pass
            self._fix = fix

    # -- consumers ---------------------------------------------------------

    def latest(self) -> GpsFix:
        with self._lock:
            return self._fix

    def is_stale(self, t_mono_now: float | None = None) -> bool:
        fix = self.latest()
        return fix.age_s(t_mono_now if t_mono_now is not None else now_mono()) > self.stale_after_s
