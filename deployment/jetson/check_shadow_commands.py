#!/usr/bin/env python3
"""Task 43: does a shadow log predict what live gating would have commanded?

`score_shadow.py` already proves the incumbent's *decisions* replay exactly
from the log (`_replay_incumbent`). What it does not check is the half this
task is about: whether the command built from a replayed decision would have
been identical in live mode except for the one boolean that names the mode.
`command_for` is the only reader of the mode anywhere in this codebase
(`policy/shadow_mode.py`), so on the Jetson side that property is close to
structural -- `test_shadow_mode.py:44-74` already proves it on 120 in-process
ticks. This module proves it again against a real drive's real logged
inputs, and against the real wire codec rather than Python objects (D17):
`shadowed.here == live.here` is two references to one `HereQuery` and would
still pass even if `command_for` had dropped the query on one branch, so
each command is round-tripped through `RateCommand.to_wire`/`from_wire`
before anything is compared.

Two checks, both per tick:

  command replay    Replay the incumbent (same mechanism as
                     `score_shadow._replay_incumbent`), then build BOTH the
                     shadow and the live command from that one replayed
                     `Decision`. Require equality on `rates`, `trigger`,
                     `here` and `t_capture_mono_ns`, and inequality on
                     `shadow` alone.
  logged shadow flag `sensing.shadow` is the one logged key the replay does
                     not check at all -- `score_shadow.py`'s own docstring
                     names this: a corrupted or flipped column still reads
                     `replay_identity.status: ok`. Checked directly here
                     against the drive's own recorded mode
                     (`summary.json`'s `sensing.mode.mode`), which is fixed
                     for the whole drive (`run_demo.py` never calls
                     `flip_to`).

The half this module cannot check is the load-bearing one: whether the PHONE
actually withheld a shadow command. `ConfigApplier`'s counters
(`applied`/`shadowed`) never cross the wire (`transport/messages.py`'s
`PhoneTelemetry` has no such field, and adding one is a wire change this
plan does not take -- D18), so they are read from `logcat` at teardown
instead: `parse_config_applier_stats` and `check_phone_applier` below.

    python3 deployment/jetson/check_shadow_commands.py <run_dir>
        [--serial S] [--no-json]

Exit code: 0 = every check that ran passed, 2 = the log refused to check at
all or a mismatch was found.
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from pathlib import Path
from typing import Any, Callable

JETSON_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(JETSON_DIR))

from eval_run import _read_summary, load_records  # noqa: E402
from policy.sensing_controller import Inputs, SensingController  # noqa: E402
from policy.shadow_mode import LIVE, MODES, SHADOW, command_for  # noqa: E402
from score_shadow import REFUSAL_NO_METADATA, ReplayClock, _log_refusal, _sensing_ticks  # noqa: E402
from transport.messages import RateCommand  # noqa: E402

#: A tick record this tool cannot check at all: `sensing` is present (so
#: `score_shadow`'s own refusals do not fire) but the top-level capture stamp
#: `command_for` needs is not. That stamp lives on `Tick.to_record()`, not on
#: `TickOutcome.to_record()` (`sensing`), so a fixture built from the sensing
#: block alone -- as `test_score_shadow.py`'s does -- does not carry it.
REFUSAL_CAPTURE_STAMP_ABSENT = "t_capture_mono_ns_absent"
#: The drive's own recorded mode is unreadable -- no `summary.json`, or a
#: `sensing.mode.mode` that is not one of `MODES` -- so there is nothing to
#: check the logged `shadow` flag against.
REFUSAL_DRIVE_MODE_UNKNOWN = "drive_mode_unknown"


def _decode(command: RateCommand) -> RateCommand:
    """Round-trip a command through the real `rate_cmd` codec.

    `to_wire`/`from_wire` is what a real link does to every command; a
    Python-level comparison of the objects `command_for` builds skips both
    the encoding this codec applies and the fresh objects decoding
    produces, so a bug only the codec would expose stays invisible.
    """
    extensions, payload = command.to_wire()
    return RateCommand.from_wire(extensions, payload)


def _here_tuple(here: Any) -> tuple | None:
    """A `HereQuery` reduced to its comparable fields, or `None`.

    A tuple rather than the dataclass itself: two decoded `HereQuery`
    instances from equal input already compare equal by value, so this adds
    nothing there -- it is here so a caller comparing `None` against a real
    query gets a clean unequal rather than an `AttributeError` reaching for
    a field on `None`.
    """
    if here is None:
        return None
    return (here.in_, here.location_ref, here.lat, here.lon, here.radius_m)


def compare_tick_commands(tick: dict[str, Any], decision: Any) -> dict[str, Any]:
    """Build both commands from ONE replayed decision and compare them decoded.

    `tick` supplies only `t_capture_mono_ns`; `decision` is what the
    replayed incumbent decided for this tick. Requires equality on `rates`,
    `trigger`, `here` and `t_capture_mono_ns`, and requires `shadow` to
    differ -- `True` on the shadow command, `False` on the live one. Any
    other outcome on `shadow` (mutated to agree, or not a bool) is itself a
    mismatch.
    """
    capture_ns = tick["t_capture_mono_ns"]
    shadow_cmd = _decode(command_for(decision, SHADOW, t_capture_mono_ns=capture_ns))
    live_cmd = _decode(command_for(decision, LIVE, t_capture_mono_ns=capture_ns))
    mismatches: list[str] = []
    if shadow_cmd.rates != live_cmd.rates:
        mismatches.append("rates")
    if shadow_cmd.trigger != live_cmd.trigger:
        mismatches.append("trigger")
    if _here_tuple(shadow_cmd.here) != _here_tuple(live_cmd.here):
        mismatches.append("here")
    if shadow_cmd.t_capture_mono_ns != live_cmd.t_capture_mono_ns:
        mismatches.append("t_capture_mono_ns")
    if shadow_cmd.shadow is not True or live_cmd.shadow is not False:
        mismatches.append("shadow")
    return {"tick_id": tick.get("tick_id"), "ok": not mismatches, "mismatches": mismatches}


def replay_and_compare(sensing_ticks: list[dict]) -> dict[str, Any]:
    """Replay the incumbent tick by tick (same mechanism as
    `score_shadow._replay_incumbent`) and compare the shadow/live commands
    built from each replayed decision.

    A fresh `SensingController`/`ReplayClock` pair, exactly as
    `_replay_incumbent` builds -- reusing its replayed records was
    considered and rejected: those are `Decision.to_record()` dicts, and
    `command_for` needs the `Decision` object itself (`.rates`, `.trigger`,
    `.here_query`), which the dict form does not carry.
    """
    clock = ReplayClock()
    controller = SensingController(clock=clock)
    mismatches: list[dict[str, Any]] = []
    for t in sensing_ticks:
        sensing = t["sensing"]
        inputs = Inputs.from_record(sensing["decision_inputs"])
        clock.set(sensing["decided_at_mono"])
        decision = controller.decide(inputs)
        result = compare_tick_commands(t, decision)
        if not result["ok"]:
            mismatches.append(result)
    return {"ticks": len(sensing_ticks), "mismatched": len(mismatches), "mismatches": mismatches}


def check_logged_shadow_flag(sensing_ticks: list[dict], drive_mode: str) -> dict[str, Any]:
    """`sensing.shadow` against the drive's own recorded mode, per tick.

    The one logged column no replay can check: `_replay_incumbent` compares
    `record` (what replay produced) against `sensing` (the log), and
    `shadow` is on every logged tick but never in what `Decision.to_record()`
    produces -- there is nothing on the replay side to compare it against.
    `score_shadow.py`'s own docstring names the consequence: flipping the
    whole column still reads `replay_identity.status: ok`.
    """
    expected = drive_mode != LIVE
    mismatched_tick_ids = [
        t.get("tick_id") for t in sensing_ticks if t["sensing"]["shadow"] != expected
    ]
    return {
        "drive_mode": drive_mode,
        "expected_shadow": expected,
        "ticks": len(sensing_ticks),
        "mismatched_tick_ids": mismatched_tick_ids,
        "ok": not mismatched_tick_ids,
    }


def check(run_dir: Path) -> dict[str, Any]:
    """Everything this tool reports, as a JSON-able dict. No file I/O beyond
    reading the run directory, matching `score_shadow.score`'s own contract.
    """
    metadata_path = run_dir / "metadata.jsonl"
    if not metadata_path.exists():
        return {"run": str(run_dir), "refused": REFUSAL_NO_METADATA}
    loaded = load_records(metadata_path)
    ticks = loaded.ticks
    sensing_ticks = _sensing_ticks(ticks)
    refusal, refusal_detail = _log_refusal(ticks, sensing_ticks)
    if refusal is not None:
        refused: dict[str, Any] = {"run": str(run_dir), "refused": refusal}
        if refusal_detail:
            refused.update(refusal_detail)
        return refused
    missing_capture = [t.get("tick_id") for t in sensing_ticks if "t_capture_mono_ns" not in t]
    if missing_capture:
        return {
            "run": str(run_dir), "refused": REFUSAL_CAPTURE_STAMP_ABSENT,
            "first_tick_id": missing_capture[0],
        }
    summary = _read_summary(run_dir)
    drive_mode = ((summary.get("sensing") or {}).get("mode") or {}).get("mode")
    if drive_mode not in MODES:
        return {"run": str(run_dir), "refused": REFUSAL_DRIVE_MODE_UNKNOWN}
    command_replay = replay_and_compare(sensing_ticks)
    shadow_flag = check_logged_shadow_flag(sensing_ticks, drive_mode)
    return {
        "run": str(run_dir),
        "drive_mode": drive_mode,
        "ticks": len(sensing_ticks),
        "command_replay": command_replay,
        "logged_shadow_flag": shadow_flag,
        "overall_ok": command_replay["mismatched"] == 0 and shadow_flag["ok"],
    }


# -- The phone-side half: applied/shadowed do not cross the wire (D18) ------

#: Kotlin's default data-class `toString` for `ConfigApplier.Stats`, e.g.:
#:   config applier stats Stats(applied=0, shadowed=7, lastTrigger=
#:   advisory_margin_narrow, currentRates={}, hereConfigured=false)
#: Measured verbatim off a real device (ZY227VV4XC, task 42's smoke run).
_CONFIG_APPLIER_STATS_RE = re.compile(
    r"config applier stats Stats\("
    r"applied=(?P<applied>\d+), "
    r"shadowed=(?P<shadowed>\d+), "
    r"lastTrigger=(?P<last_trigger>[^,]*), "
    r"currentRates=\{(?P<current_rates>[^}]*)\}, "
    r"hereConfigured=(?P<here_configured>true|false)\)"
)

#: `adb logcat -v epoch`'s own line format: leading whitespace (logcat pads
#: columns it does not have a value for), then `<seconds>.<millis>`, pid,
#: tid, level, tag, message. Measured verbatim off ZY227VV4XC:
#:   "         1788585230.408 20794 20794 I SensingService: config applier ..."
_EPOCH_LINE_RE = re.compile(r"^\s*(?P<t>\d+\.\d+)\s+\d+\s+\d+\s+\w\s+[^:]*:\s?(?P<message>.*)$")


def parse_config_applier_stats(
    logcat_text: str, *, window: tuple[float, float] | None = None,
) -> dict[str, Any] | None:
    """The phone's `ConfigApplier.Stats` teardown line, from a `logcat -d`
    dump. `None` when nothing qualifies -- the app never tore down (the
    drive is still running), the ring buffer already rotated the line out,
    or (with a `window`) every match found is outside it -- rather than a
    zeroed dict that would read as a real report of nothing shadowed.

    `window`, given as `(start, end)` wall-clock epoch seconds, restricts
    matches to lines whose OWN `-v epoch` timestamp falls inside it (A2,
    validation round 1): the device-global ring buffer holds a line for far
    longer than one run lasts -- measured: a line was still the only match
    57 minutes after it was written -- so an unwindowed pull silently reads
    a PREVIOUS run's teardown line into the artifact of a LATER one whenever
    the buffer still holds one and the counts happen to satisfy the check.
    `None` (the default) keeps the old, unscoped behaviour for a caller
    working from text with no epoch prefix at all -- a fixture, or text a
    caller has already sliced to one run itself.

    The LAST match inside the window (or in the whole text, unwindowed),
    not the first: `SensingService.onSensingDown` runs on every stop, so a
    phone that rebound mid-drive tears down more than once, and only the
    final one -- provided it is still inside this run's own window --
    describes the whole session.
    """
    match = None
    if window is None:
        for candidate in _CONFIG_APPLIER_STATS_RE.finditer(logcat_text):
            match = candidate
    else:
        start, end = window
        for line in logcat_text.splitlines():
            line_match = _EPOCH_LINE_RE.match(line)
            if line_match is None:
                continue
            t = float(line_match.group("t"))
            if not (start <= t <= end):
                continue
            candidate = _CONFIG_APPLIER_STATS_RE.search(line_match.group("message"))
            if candidate is not None:
                match = candidate
    if match is None:
        return None
    rates: dict[str, float] = {}
    raw_rates = match.group("current_rates").strip()
    if raw_rates:
        for pair in raw_rates.split(", "):
            key, _, value = pair.partition("=")
            rates[key] = float(value)
    last_trigger = match.group("last_trigger")
    return {
        "applied": int(match.group("applied")),
        "shadowed": int(match.group("shadowed")),
        "last_trigger": None if last_trigger == "null" else last_trigger,
        "current_rates": rates,
        "here_configured": match.group("here_configured") == "true",
    }


#: How much slop to allow around [run start, run end] when scoping a logcat
#: pull to one run's own window. The Jetson's `log_health.json.t_wall` and
#: the phone's own `-v epoch` timestamp are two different devices' clocks;
#: measured 0.75s apart on a real drive (the phone tears down last, after
#: the Jetson's own `close()`), so this covers ordinary two-device clock
#: drift without reaching anywhere near an adjacent run.
RUN_WINDOW_MARGIN_S = 30.0


def _run_window(run_dir: Path, *, margin_s: float = RUN_WINDOW_MARGIN_S) -> tuple[float, float] | None:
    """`[start, end]` wall-clock epoch seconds for this run, expressed on
    the PHONE's clock, or `None` when either bound is unavailable.

    `start` is the first TICK record's own `t_wall` (`Tick.to_record()`);
    the first LINE of `metadata.jsonl` is not necessarily a tick (a run
    that opened with a `failure_event` has one ahead of tick 0). `end` is
    `log_health.json`'s `t_wall` -- the metadata logger's own close()-time
    timestamp, the closest thing to "this run definitely ended" that is on
    disk -- plus `margin_s`.

    Both are on the JETSON's clock as recorded, but the logcat lines this
    window is matched against are timestamped on the PHONE's (B10,
    validation round 2) -- two different, unsynchronised clocks: measured
    on real hardware at +0.935 to +0.952 s across three samples minutes
    apart, and near zero for the same two devices earlier in the same
    session, so it is not even constant. Shifted here by
    `summary["phone"]["wall_clock_offset_s"]` (recorded from the handshake,
    `sensors/phone_link.py`) when available, which also fixes the margin
    being one-sided: an unshifted `[start, end + margin_s]` tolerates a
    phone running behind for the whole run's duration while tolerating one
    running more than `margin_s` ahead nowhere at all, and past that this
    is A2's own contamination one layer down -- run N's line falling
    outside window N while run N-1's falls inside it. `None` offset (a run
    from before this fix, or one with no phone session at all) falls back
    to the unshifted, Jetson-clock window rather than refusing outright.
    """
    metadata_path = run_dir / "metadata.jsonl"
    if not metadata_path.exists():
        return None
    start = None
    with open(metadata_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except ValueError:
                continue
            if record.get("type") == "tick" and "t_wall" in record:
                start = record["t_wall"]
                break
    if start is None:
        return None
    log_health_path = run_dir / "log_health.json"
    if not log_health_path.exists():
        return None
    try:
        log_health = json.loads(log_health_path.read_text())
    except ValueError:
        return None
    end = log_health.get("t_wall")
    if end is None:
        return None
    offset_s = ((_read_summary(run_dir).get("phone") or {}).get("wall_clock_offset_s")) or 0.0
    return start + offset_s, end + offset_s + margin_s


RunAdb = Callable[..., "subprocess.CompletedProcess"]


def pull_config_applier_stats(
    serial: str, window: tuple[float, float], *, run: RunAdb = subprocess.run, timeout_s: float = 30.0,
) -> dict[str, Any] | None:
    """Shell `adb -s <serial> logcat -d -v epoch -s SensingService:I` and
    parse the teardown line whose OWN timestamp falls inside `window` (A2,
    validation round 1): the ring buffer holds a line for far longer than
    one run lasts, so an unscoped pull can silently read a PREVIOUS run's
    counters into THIS run's artifact. `None` when `adb` could not be
    reached at all, or when nothing in the window matched -- the caller
    cannot tell these apart from this return value alone, which is why
    `apply_phone_applier_check`'s own detail message names the window it
    looked in rather than repeating "no line found".
    """
    try:
        result = run(
            ["adb", "-s", serial, "logcat", "-d", "-v", "epoch", "-s", "SensingService:I"],
            capture_output=True, text=True, timeout=timeout_s,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    if result.returncode != 0:
        return None
    return parse_config_applier_stats(result.stdout, window=window)


def check_phone_applier(
    applier_stats: dict[str, Any], *, drive_mode: str, commands_sent: int,
) -> dict[str, Any]:
    """43.6: the phone's counters against the Jetson's own record of what it sent.

    On a shadow drive every command sent has `shadow=True` (`command_for` is
    read once per tick from the drive's one fixed mode, D16), so the phone
    must have applied none of them and shadowed exactly as many as the
    Jetson's `rate_cmd` channel counter says it sent -- the phone is the
    only direct witness that a shadow command was not acted on
    (`ConfigApplier.kt` returns before `applied` is incremented).

    On a LIVE drive this returns `ok: None` (B3, validation round 1):
    whether the phone applied every commanded rate correctly is a live-mode
    correctness question, and this task's scope is the decision function a
    shadow log predicts, not whether live mode itself behaves -- that is
    task 44's territory (`shadow_mode.py`'s own module docstring: "a shadow
    log predicts the decision function... and does NOT predict the
    trajectory"). Asserting `applied == commands_sent` here would be a
    claim about live mode this task never gathered evidence for; `applied`
    is still reported, just not asserted on.
    """
    if drive_mode == SHADOW:
        ok = applier_stats["applied"] == 0 and applier_stats["shadowed"] == commands_sent
    else:
        ok = None
    return {
        "drive_mode": drive_mode,
        "commands_sent": commands_sent,
        "applied": applier_stats["applied"],
        "shadowed": applier_stats["shadowed"],
        "ok": ok,
    }


def rate_cmd_commands_sent(summary: dict[str, Any]) -> int | None:
    """The Jetson's own `rate_cmd` channel counter, from `summary["phone"]`.

    `wire.channels.rate_cmd.sent` (`session.stats()`, the transport layer's
    own count of frames it sent on that channel) rather than
    `sensing.rate_commands_sent` (`SensingLoop`'s tally of how often it
    *decided* to send) -- the two are usually equal but are different
    layers, and this check is about what the phone actually received, which
    the transport counter is one hop closer to.
    """
    channels = ((summary.get("phone") or {}).get("wire") or {}).get("channels") or {}
    rate_cmd = channels.get("rate_cmd")
    return None if rate_cmd is None else rate_cmd.get("sent")


def apply_phone_applier_check(
    result: dict[str, Any], *, serial: str | None, run_dir: Path,
) -> dict[str, Any]:
    """Adds `phone_applier` to `result` and folds its `ok` into
    `overall_ok` -- both in one place, so the fold cannot be skipped for
    one outcome and not another the way it was before this fix (A1,
    validation round 1): `overall_ok` used to stay untouched whenever
    `pull_config_applier_stats` returned `None`, because that branch set
    `phone_applier["ok"] = False` and returned without ever reaching the
    line that folds it in -- so `--serial NOSUCHSERIAL` against a real run
    reported `phone_applier ok=False` and still exited 0.

    `serial` absent means the phone check was never asked for, which is
    written as `{"ok": None, "detail": "not requested"}` rather than
    leaving the key out entirely -- a reader of the JSON must be able to
    tell "not asked" from "asked and passed" without also knowing whether
    `--serial` was on the command line, and `overall_ok` is left exactly as
    `check()` computed it.
    """
    if not serial:
        result["phone_applier"] = {"ok": None, "detail": "not requested"}
        return result
    window = _run_window(run_dir)
    if window is None:
        result["phone_applier"] = {
            "ok": False,
            "detail": "the run's own window could not be established (missing "
                      "metadata.jsonl's first tick t_wall or log_health.json)",
        }
    else:
        applier_stats = pull_config_applier_stats(serial, window)
        if applier_stats is None:
            result["phone_applier"] = {
                "ok": False,
                "detail": (
                    f"no ConfigApplier stats line inside the run window "
                    f"[{window[0]:.3f}, {window[1]:.3f}] (adb unreachable, the "
                    f"app has not torn down yet, or the ring buffer rotated the "
                    f"line out)"
                ),
            }
        else:
            commands_sent = rate_cmd_commands_sent(_read_summary(run_dir))
            result["phone_applier"] = (
                {"ok": False, "detail": "rate_cmd channel counter absent from summary.json"}
                if commands_sent is None
                else check_phone_applier(
                    applier_stats, drive_mode=result["drive_mode"], commands_sent=commands_sent,
                )
            )
    # Outside the if/else above so it runs for EVERY outcome once a check
    # was actually requested -- the bug this fixes was this line living
    # inside just one branch. `ok` may be None (B3's live-drive case, or a
    # window/adb failure that could not even be judged) -- `and` with None
    # would corrupt a True `overall_ok` into `None`, so only a definite
    # False is folded in.
    phone_ok = result["phone_applier"]["ok"]
    if phone_ok is False:
        result["overall_ok"] = False
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--serial", help="adb serial to pull the phone's ConfigApplier stats from")
    parser.add_argument("--no-json", action="store_true", help="skip writing shadow_command_check.json")
    args = parser.parse_args()

    result = check(args.run_dir)
    if "refused" in result:
        print(f"refused: {result['refused']}")
        if not args.no_json:
            (args.run_dir / "shadow_command_check.json").write_text(json.dumps(result, indent=2))
        return 2

    result = apply_phone_applier_check(result, serial=args.serial, run_dir=args.run_dir)

    print(
        f"drive_mode={result['drive_mode']} ticks={result['ticks']} "
        f"command_replay mismatched={result['command_replay']['mismatched']} "
        f"logged_shadow_flag ok={result['logged_shadow_flag']['ok']} "
        f"phone_applier ok={result['phone_applier']['ok']}"
    )
    if not args.no_json:
        (args.run_dir / "shadow_command_check.json").write_text(json.dumps(result, indent=2))
    return 0 if result["overall_ok"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
