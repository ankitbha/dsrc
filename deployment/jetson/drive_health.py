"""Health of a drive while it is happening, for a reader who cannot see the car.

Why this exists. Every failure the 2026-09-08 drives produced was silent. The
session died from heat with `link_ms` at 135,359 ms against a p50 of 52 ms, and no
gate in this system looks at `link_ms`. The phone sat at `severe` for the whole
drive and nothing said so. 1,632 ticks recorded zero detections and the only way to
find out was to go and look afterwards. A drive is expensive and unrepeatable, and
the point of this file is that a broken one says so while there is still time to
stop.

THE RULE THIS FOLLOWS, and the reason it is written down. Three outcomes, never two:
something is wrong, everything checked is fine, and **the check could not be made**.
Collapsing the third into either of the others is how an instrument reports success
on its own failure -- and this project has a drive whose `usb2` said `connected`
through a total network outage, and a `camera.blind_ticks: quiet` on a drive with 40
blind episodes. So every field here is a tri-state, and a value that could not be
read is `None` and reported as UNKNOWN, never as OK.

Thresholds come from the drives, not from taste. Where a number is chosen the
measurement behind it is named, so a later reader can argue with the evidence rather
than the number.
"""
from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass, field
from typing import Any

# ---------------------------------------------------------------------------
# Verdicts. Ordered worst-first: `worst_of` takes the earliest in this tuple.
# UNKNOWN sits above OK deliberately -- not knowing is worse than knowing things
# are fine, and a summary that rounds UNKNOWN down to OK is the failure this
# module exists to avoid.
# ---------------------------------------------------------------------------
BROKEN = "BROKEN"
DEGRADED = "DEGRADED"
UNKNOWN = "UNKNOWN"
OK = "OK"
SEVERITY = (BROKEN, DEGRADED, UNKNOWN, OK)


def worst_of(verdicts: list[str]) -> str:
    for level in SEVERITY:
        if level in verdicts:
            return level
    return UNKNOWN


#: A tick stream is stalled after this long with no new tick. The drives ran at a
#: 203 ms median spacing with a largest legitimate gap of 2.6 s, so 10 s is roughly
#: four times the worst normal gap and cannot fire on ordinary jitter.
TICK_STALL_S = 10.0

#: `link_ms` above this is the phone failing to keep up. Measured p50 52 ms, p95
#: 110 ms, and 135,359 ms on the tick where the session died. 2,000 ms is ~18x the
#: p95 and two orders of magnitude below the death, so it fires early enough to be
#: worth acting on and never on jitter.
LINK_DEGRADED_MS = 2000.0

#: Thermal statuses that stalled a drive. `severe` is where the 2026-09-08 session
#: died; `moderate` is reported but not treated as a fault, because the backoff is
#: designed to handle it and did.
THERMAL_BROKEN = {"severe", "critical", "emergency", "shutdown"}

#: Below this, logging will stop mid-drive. 150 MB/hour of metadata plus MJPG video
#: at roughly 1.4 GB/hour measured on 2026-09-08.
DISK_FREE_GB_MIN = 5.0


@dataclass
class Check:
    """One named observation, its verdict, and why."""

    name: str
    verdict: str
    detail: str
    value: Any = None

    def to_record(self) -> dict[str, Any]:
        return {"verdict": self.verdict, "detail": self.detail, "value": self.value}


@dataclass
class Health:
    at_wall: float
    run: str | None
    verdict: str
    checks: list[Check] = field(default_factory=list)

    def to_record(self) -> dict[str, Any]:
        return {
            "at_wall": self.at_wall,
            "run": self.run,
            "verdict": self.verdict,
            "checks": {c.name: c.to_record() for c in self.checks},
            # The one-line form, so a reader who wants only the headline never has
            # to reconstruct it and cannot reconstruct it differently.
            "line": summarise(self),
        }


def summarise(health: Health) -> str:
    bad = [c for c in health.checks if c.verdict in (BROKEN, DEGRADED)]
    unknown = [c for c in health.checks if c.verdict == UNKNOWN]
    parts = [f"{health.verdict}"]
    if bad:
        parts.append("; ".join(f"{c.name}: {c.detail}" for c in bad))
    if unknown:
        parts.append("could not check: " + ", ".join(c.name for c in unknown))
    if not bad and not unknown:
        parts.append("all checks passed")
    return " | ".join(parts)


# ---------------------------------------------------------------------------
# The checks. Each takes already-gathered facts, so the verdict logic is pure and
# testable without a Jetson, a phone or a drive -- which is the only way a check
# that fires once a month gets exercised at all.
# ---------------------------------------------------------------------------
def check_ticks(tick_count: int | None, last_tick_age_s: float | None) -> Check:
    if tick_count is None or last_tick_age_s is None:
        return Check("ticks", UNKNOWN, "no tick records could be read")
    if tick_count == 0:
        return Check("ticks", BROKEN, "no ticks at all: the phone has not dialled in", 0)
    if last_tick_age_s > TICK_STALL_S:
        return Check("ticks", BROKEN,
                     f"stalled: last tick {last_tick_age_s:.0f}s ago", tick_count)
    return Check("ticks", OK, f"{tick_count} ticks, newest {last_tick_age_s:.1f}s ago",
                 tick_count)


def check_link(link_ms: float | None) -> Check:
    """The observable that caught the thermal death and that no gate watches."""
    if link_ms is None:
        return Check("link", UNKNOWN, "no link timing on the newest tick")
    if link_ms > LINK_DEGRADED_MS:
        return Check("link", BROKEN,
                     f"{link_ms:.0f}ms: the phone is not keeping up "
                     f"(p50 was 52ms; 135359ms was a dead session)", link_ms)
    return Check("link", OK, f"{link_ms:.0f}ms", link_ms)


def check_thermal(status: str | None, skin_c: float | None) -> Check:
    if status is None and skin_c is None:
        return Check("thermal", UNKNOWN, "no thermal telemetry from the phone")
    if status in THERMAL_BROKEN:
        return Check("thermal", BROKEN,
                     f"phone at {status}"
                     + (f", skin {skin_c:.1f}C" if skin_c is not None else "")
                     + ": a drive died here on 2026-09-08", status)
    if status == "moderate":
        return Check("thermal", DEGRADED,
                     f"phone at moderate"
                     + (f", skin {skin_c:.1f}C" if skin_c is not None else ""), status)
    return Check("thermal", OK,
                 f"{status}" + (f", skin {skin_c:.1f}C" if skin_c is not None else ""),
                 status)


def check_gps(gps_valid: bool | None, fix_fraction: float | None) -> Check:
    """A drive without a fix is worthless for anything positional, and every bench
    run in this project had `gps_hz 0.0` without anyone noticing until a drive."""
    if gps_valid is None:
        return Check("gps", UNKNOWN, "no gps field on the newest tick")
    if not gps_valid:
        return Check("gps", BROKEN, "no fix on the newest tick", False)
    if fix_fraction is not None and fix_fraction < 0.9:
        return Check("gps", DEGRADED,
                     f"fix on only {fix_fraction*100:.0f}% of recent ticks",
                     fix_fraction)
    return Check("gps", OK,
                 "fix present"
                 + (f", {fix_fraction*100:.0f}% of recent ticks" if fix_fraction else ""),
                 True)


def check_video(video_bytes: int | None, index_lines: int | None,
                video_enabled: bool) -> Check:
    """Whether this drive will be re-processable. A drive whose frames are discarded
    cannot explain its own perception failure -- which is what happened on
    2026-09-08 and cost the answer to the drive's central question."""
    if not video_enabled:
        return Check("video", BROKEN,
                     "frame logging is off: this drive cannot be re-processed", False)
    if video_bytes is None or index_lines is None:
        return Check("video", UNKNOWN, "video or its index could not be read")
    if video_bytes == 0 or index_lines == 0:
        return Check("video", BROKEN, "frame logging on but nothing written yet", 0)
    return Check("video", OK,
                 f"{video_bytes/1e6:.0f}MB, {index_lines} indexed frames", index_lines)


def check_disk(free_gb: float | None) -> Check:
    if free_gb is None:
        return Check("disk", UNKNOWN, "free space could not be read")
    if free_gb < DISK_FREE_GB_MIN:
        return Check("disk", BROKEN, f"{free_gb:.1f}GB left: logging will stop", free_gb)
    return Check("disk", OK, f"{free_gb:.0f}GB free", free_gb)


def check_clock(ntp_synchronised: bool | None) -> Check:
    """Task 65: a drive started before NTP syncs has `t_wall` stepping mid-run."""
    if ntp_synchronised is None:
        return Check("clock", UNKNOWN, "sync state could not be read")
    if not ntp_synchronised:
        return Check("clock", DEGRADED,
                     "clock not NTP-synced: t_wall may step mid-run", False)
    return Check("clock", OK, "NTP synced", True)


def assess(facts: dict[str, Any]) -> Health:
    """Every check, and the worst verdict among them.

    Total by construction: each `check_*` returns a named verdict on every path, and
    a fact that could not be gathered arrives as `None` and becomes UNKNOWN. There is
    no branch that falls through to silence.
    """
    checks = [
        check_ticks(facts.get("tick_count"), facts.get("last_tick_age_s")),
        check_link(facts.get("link_ms")),
        check_thermal(facts.get("thermal_status"), facts.get("skin_temp_c")),
        check_gps(facts.get("gps_valid"), facts.get("gps_fix_fraction")),
        check_video(facts.get("video_bytes"), facts.get("index_lines"),
                    bool(facts.get("video_enabled"))),
        check_disk(facts.get("disk_free_gb")),
        check_clock(facts.get("ntp_synchronised")),
    ]
    return Health(at_wall=facts.get("at_wall", time.time()),
                  run=facts.get("run"),
                  verdict=worst_of([c.verdict for c in checks]),
                  checks=checks)


# ---------------------------------------------------------------------------
# Gathering. Separated from the verdicts so a read failure produces `None` rather
# than an exception that takes the whole watcher down -- a watcher that dies on a
# transient is worse than no watcher, because its silence reads as calm.
# ---------------------------------------------------------------------------
def newest_run(log_dir: str) -> str | None:
    """The run being written right now, chosen by when its metadata was last
    written.

    Not by directory name, and not by directory mtime. A directory's mtime moves
    only when an entry is created or removed inside it, so a run that is actively
    appending to `metadata.jsonl` does not touch it, while a finished run bumps
    it at teardown by creating `summary.json`. On 2026-09-08 that ordered a dead
    run above the live one, and reading the dead run's frozen counters produced a
    "writers frozen" report on a drive that was healthy. Run names cannot break
    the tie either: a run directory is named when the process starts and the
    process may then wait minutes for the phone to dial, so the names sorted in
    the opposite order to the runs themselves on that same day.
    """
    try:
        runs = [os.path.join(log_dir, d) for d in os.listdir(log_dir)
                if d.startswith("run_")]
        runs = [r for r in runs if os.path.isdir(r)]
        if not runs:
            return None

        def last_written(run: str) -> float:
            meta = os.path.join(run, "metadata.jsonl")
            try:
                return os.path.getmtime(meta)
            except OSError:
                # No metadata yet: the run has just been created and has not been
                # written to. Fall back to the directory so a brand-new run is
                # still visible, but rank it below any run with real writes.
                try:
                    return os.path.getmtime(run)
                except OSError:
                    return 0.0

        return max(runs, key=last_written)
    except OSError:
        return None


def _tail_ticks(path: str, want: int = 200) -> list[dict]:
    """The last few tick records, read from the tail so cost does not grow with the
    drive. A 19 MB metadata file re-read every 15 s would make the watcher itself
    the load."""
    try:
        size = os.path.getsize(path)
        start = max(0, size - 400_000)
        with open(path, "rb") as fh:
            fh.seek(start)
            chunk = fh.read().decode("utf-8", "ignore")
    except OSError:
        return []
    lines = chunk.splitlines()
    if start > 0:
        # The seek landed mid-record, so the first line is a fragment. When the
        # whole file fits in the window `start` is 0 and that first line is a
        # complete record: dropping it unconditionally lost one tick from every
        # short run, which is exactly when a watcher has fewest to work with.
        lines = lines[1:]
    out = []
    for line in lines:
        try:
            r = json.loads(line)
        except Exception:
            continue
        if r.get("type") == "tick":
            out.append(r)
    return out[-want:]


def gather(log_dir: str, video_enabled: bool, ntp_synchronised: bool | None) -> dict:
    facts: dict[str, Any] = {"at_wall": time.time(), "video_enabled": video_enabled,
                             "ntp_synchronised": ntp_synchronised}
    run = newest_run(log_dir)
    facts["run"] = os.path.basename(run) if run else None
    if run is None:
        return facts

    ticks = _tail_ticks(os.path.join(run, "metadata.jsonl"))
    # `count` is the tail's count, not the drive's; named so nobody reads it as the
    # total. The watcher cares whether ticks are ARRIVING, not how many there are.
    facts["tick_count"] = len(ticks)
    if ticks:
        last = ticks[-1]
        facts["last_tick_age_s"] = max(0.0, time.time() - last.get("t_wall", 0))
        facts["link_ms"] = last.get("link_ms")
        gps = last.get("gps") or {}
        facts["gps_valid"] = gps.get("valid")
        valid = sum(1 for t in ticks if (t.get("gps") or {}).get("valid"))
        facts["gps_fix_fraction"] = valid / len(ticks)
        thermal = (((last.get("sensing") or {}).get("attribution") or {})
                   .get("rules") or {}).get("thermal_backoff") or {}
        facts["thermal_status"] = thermal.get("thermal_status")
        facts["skin_temp_c"] = thermal.get("skin_temp_c")

    for name, key in (("video.avi", "video_bytes"), ("video_index.jsonl", None)):
        path = os.path.join(run, name)
        try:
            size = os.path.getsize(path)
        except OSError:
            size = None
        if key:
            facts[key] = size
    try:
        with open(os.path.join(run, "video_index.jsonl")) as fh:
            facts["index_lines"] = sum(1 for _ in fh)
    except OSError:
        facts["index_lines"] = None

    try:
        st = os.statvfs(log_dir)
        facts["disk_free_gb"] = st.f_bavail * st.f_frsize / 1e9
    except OSError:
        facts["disk_free_gb"] = None
    return facts
