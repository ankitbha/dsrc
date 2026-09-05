#!/usr/bin/env python3
"""Live advisory demo on the Jetson (plan_deployment.md Tasks P1/P6/P8).

Wires camera + GPS -> perception -> observation -> actor -> advisory,
with the dashboard on the main thread and the pipeline on a worker
thread (latest-frame-wins at every handoff).

Typical uses
  python3 run_demo.py --selfcheck                  # verify hardware/software
  python3 run_demo.py                              # live, display + logging
  python3 run_demo.py --headless --duration-s 600  # in-car logging run
  python3 run_demo.py --source file:clip.mp4       # desk run from a video
  python3 run_demo.py --scenario data/scenarios/i495_cruise.json
                                                   # simulated drive: dashcam
                                                   # video + scripted GPS
  python3 run_demo.py --source file:clip.mp4 --sim-gps const:25
                                                   # quick GPS sim, no file

The system is ADVISORY ONLY: it has no connection to vehicle controls,
and recommendations must not be followed while driving (see
plan_deployment.md, "Safety and demo constraints").
"""

from __future__ import annotations

import argparse
import atexit
import hashlib
import json
import math
import os
import signal
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import Any

JETSON_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(JETSON_DIR))

from transport.tcp import DEFAULT_PORT as PHONE_DEFAULT_PORT  # noqa: E402

import yaml  # noqa: E402

from dumpsys_util import parse_last_update_time  # noqa: E402
from perception.detector import TrtYoloDetector  # noqa: E402
from perception.distance import DistanceEstimator  # noqa: E402
from perception.observation_builder import BuilderConfig, ObservationBuilder  # noqa: E402
from perception.tracker import IouTracker  # noqa: E402
from pipeline import PerceptionPolicyPipeline  # noqa: E402
from policy.actor_runtime import ActorRuntime  # noqa: E402
from policy.advisory import AdvisoryDecoder  # noqa: E402
from sensors.camera_stream import CameraStream  # noqa: E402
from sensors.gps_reader import GpsFix, GpsReader  # noqa: E402


def load_config(path: str | Path) -> dict:
    with open(path) as f:
        config = yaml.safe_load(f)
    config["_config_path"] = str(path)
    return config


def resolve_model_path(config: dict, key_path: str) -> str:
    section, key = key_path.split(".")
    raw = Path(config[section][key])
    return str(raw if raw.is_absolute() else JETSON_DIR / raw)


def build_components(
    config: dict,
    source_override: str | None = None,
    use_gps: bool = True,
    gps_sim_spec: str | dict | None = None,
    phone: "PhoneLink | None" = None,
):
    cam_cfg = config["camera"]
    if phone is not None:
        # Both, or neither. The two backends are channels of one session, so a
        # run cannot take its camera from a phone and its GPS from somewhere
        # else -- and letting the flags express that would invite a run whose
        # two halves came from different devices and different clocks.
        camera = phone.camera
        gps = phone.gps
        return camera, gps, *_build_rest(config)

    camera = CameraStream(
        source=source_override or str(cam_cfg["source"]),
        width=cam_cfg["width"],
        height=cam_cfg["height"],
        fps=cam_cfg["fps"],
        fourcc=cam_cfg["fourcc"],
    )

    gps = None
    if gps_sim_spec is not None:
        from sensors.gps_sim import SimulatedGps

        gps = SimulatedGps(gps_sim_spec, stale_after_s=config["gps"]["stale_after_s"])
    elif use_gps:
        g = config["gps"]
        gps = GpsReader(
            port=g["port"],
            baud=g["baud"],
            configure_rate=g["configure_rate"],
            target_rate_hz=g["target_rate_hz"],
            stale_after_s=g["stale_after_s"],
        )

    return camera, gps, *_build_rest(config)


def _build_rest(config: dict):
    """Everything downstream of the two sensors, which does not care where they came from."""
    # Read here rather than inherited. Split out of `build_components`, this body
    # kept referring to that function's local `cam_cfg`, which made it a free
    # global that does not exist -- a NameError on every call, in live mode,
    # bench_latency and replay_demo alike. `--selfcheck` never calls it, which is
    # how it shipped.
    cam_cfg = config["camera"]
    d = config["detector"]
    detector = TrtYoloDetector(
        engine_path=resolve_model_path(config, "detector.engine"),
        input_size=d["input_size"],
        conf_threshold=d["conf_threshold"],
        iou_threshold=d["iou_threshold"],
        vehicle_classes=tuple(d["vehicle_classes"]),
        max_detections=d["max_detections"],
        hood_line_y_px=cam_cfg.get("hood_line_y_px"),
    )

    t = config["tracker"]
    tracker = IouTracker(
        iou_match_threshold=t["iou_match_threshold"],
        max_age_frames=t["max_age_frames"],
        min_hits=t["min_hits"],
    )

    dist_cfg = config["distance"]
    distance = DistanceEstimator(
        fx_px=cam_cfg["fx_px"],
        cx_px=cam_cfg["cx_px"],
        horizon_y_px=cam_cfg["horizon_y_px"],
        camera_height_m=cam_cfg["camera_height_m"],
        method=dist_cfg["method"],
        ema_alpha=dist_cfg["ema_alpha"],
        rel_speed_window=dist_cfg["rel_speed_window"],
        rel_speed_min_span_s=dist_cfg["rel_speed_min_span_s"],
        class_widths_m={int(k): float(v) for k, v in dist_cfg["class_widths_m"].items()},
        max_range_m=dist_cfg["max_range_m"],
        contact_cutoff_y_px=(
            cam_cfg.get("hood_line_y_px") or (cam_cfg["height"] - 3)
        ),
    )

    # B11 (validation round 2): the observation-section-plus-two-overrides
    # merge lives once, in BuilderConfig.from_full_config, so this and
    # src.analysis.observation_parity's harness build the identical
    # BuilderConfig from the identical config rather than each carrying its
    # own copy of the same three lines.
    builder = ObservationBuilder(BuilderConfig.from_full_config(config))

    p = config["policy"]
    actor = ActorRuntime(
        bundle_prefix=resolve_model_path(config, "policy.bundle"),
        deterministic=p["deterministic"],
    )
    decoder = AdvisoryDecoder(
        units=config["ui"]["units"],
        min_contextual_speed_mps=p["min_contextual_speed_mps"],
        confidence_low_below=p["confidence_low_below"],
        confidence_high_at=p["confidence_high_at"],
    )

    pipeline = PerceptionPolicyPipeline(detector, tracker, distance, builder, actor, decoder)
    return pipeline, actor


def telemetry_record(tick) -> dict:
    rec = tick.to_record()
    for heavy in ("obs", "encoded", "head_probs", "field_sources", "vehicles"):
        rec.pop(heavy, None)
    return rec


def tick_session_id(phone) -> Any:
    """Which peer clock this tick's phone-side stages were measured against."""
    if phone is None or phone.session is None:
        return None
    return phone.session.session_id


def _log_timebase_estimates(
    logger, phone, last_estimate_ids: dict[str, int | None], session_id: Any,
) -> None:
    """Write a `timebase_estimate` line whenever either estimator publishes a
    new one, so an offline reader can re-derive a conversion against the
    estimate that was actually current at the time rather than only the one
    live at the end of the run.

    `TimebaseEstimator.estimate` is a property; `OneWayEstimator.estimate` is a
    method -- the two classes are not otherwise identical, and this is the one
    place both are read side by side.

    `usable`/`why_not_usable` are read off the estimator, not the persisted
    `TimebaseEstimate`: the gate that decides them -- sample count, staleness,
    the RTT ceiling -- lives on the estimator, and the dataclass an estimate is
    handed as carries none of it. Without this an offline reader sees only
    `offset_ns` and `estimate_id` and has no way to tell an estimate the live
    adapter would have converted against from one it would have proxied
    around.

    `session_id` ties the line to the peer clock it measured, and is passed in
    rather than read off `phone.session` here: the caller already reads it once
    for the tick record itself, and a second, independent read of the same
    field is the shape of defect this file has already had to fix once. Both
    estimators are replaced whole on every redial (`phone_link._rebind`), so
    their `estimate_id` counters restart at 1 on the new session -- an offline
    reader matching on wall time alone could otherwise pick an estimate left
    over from the previous phone.
    """
    sources = (
        ("round_trip", phone.round_trip_estimator, phone.round_trip_estimator.estimate),
        ("one_way", phone.estimator, phone.estimator.estimate()),
    )
    for source, estimator, estimate in sources:
        if estimate is None or estimate.estimate_id == last_estimate_ids[source]:
            continue
        last_estimate_ids[source] = estimate.estimate_id
        logger.write({
            "type": "timebase_estimate",
            "source": source,
            "session_id": session_id,
            # This device's own wall clock, for an offline reader correlating
            # this estimate against a phone log's wall-stamped receipts -- both
            # devices are NTP-locked, so wall time is a valid matching key even
            # though it is never used for the latency arithmetic itself.
            "t_wall": time.time(),
            "usable": estimator.usable,
            "why_not_usable": estimator.why_not_usable(),
            **estimate.to_record(),
        })


#: Where the install step records what it put on the phone -- read here so
#: a run record can name the APK that produced it (A4, validation round 2).
#: Untracked, same as every other model artifact (models/.gitignore is "*").
INSTALLED_APK_RECORD = JETSON_DIR / "models" / "installed_apk.json"
#: Where the DEPLOY step (on the SOURCE machine, before the rsync copy that
#: strips `.git`) records the commit it deployed (B16, validation round 3):
#: `git rev-parse HEAD` below returns `commit: null` on the only machine a
#: run actually happens on -- `~/dsrc-task40` is an rsync copy, not a git
#: checkout -- so the other three provenance fields populate and this one
#: never does. `scripts/record_deployed_commit.py` writes this file on the
#: source machine, alongside the rsync, the same pattern
#: `record_installed_apk.py` established for the phone's APK. Untracked,
#: same as every other model artifact.
DEPLOYED_COMMIT_RECORD = JETSON_DIR / "models" / "deployed_commit.json"

#: Must match `scripts/record_deployed_commit.py`'s own `SOURCE_HASH_
#: EXCLUDE_DIRS`/`SOURCE_HASH_EXCLUDE_SUFFIXES` exactly (B21, validation
#: round 4): a difference here makes an unchanged tree hash differently on
#: each side, and every run would report a false mismatch.
SOURCE_HASH_EXCLUDE_DIRS = frozenset({"models", "__pycache__", ".pytest_cache", ".git"})
SOURCE_HASH_EXCLUDE_SUFFIXES = frozenset({".pyc", ".pyo"})

#: `apk_last_update_time_check`'s closed vocabulary (B20, validation round
#: 4): `null` used to mean two different things -- "not applicable, no
#: serial was given" and "applicable, but the device would not answer" --
#: read alike by anything downstream. A word from this set says which.
APK_TIMESTAMP_MATCHED = "matched"
APK_TIMESTAMP_MISMATCHED = "mismatched"
APK_TIMESTAMP_NO_SERIAL = "no_serial"
APK_TIMESTAMP_SIDECAR_HAS_NO_TIMESTAMP = "sidecar_has_no_timestamp"
APK_TIMESTAMP_DEVICE_DID_NOT_ANSWER = "device_did_not_answer"
APK_TIMESTAMP_VOCABULARY = frozenset({
    APK_TIMESTAMP_MATCHED, APK_TIMESTAMP_MISMATCHED, APK_TIMESTAMP_NO_SERIAL,
    APK_TIMESTAMP_SIDECAR_HAS_NO_TIMESTAMP, APK_TIMESTAMP_DEVICE_DID_NOT_ANSWER,
})

#: `source_tree_check`'s own closed vocabulary, same reasoning, for B21's
#: staleness signal on `commit` (validation round 4).
SOURCE_TREE_MATCHED = "matched"
SOURCE_TREE_MISMATCHED = "mismatched"
SOURCE_TREE_NOT_APPLICABLE_GIT_CHECKOUT = "not_applicable_git_checkout"
SOURCE_TREE_NO_SIDECAR_HASH = "no_sidecar_hash"
SOURCE_TREE_NO_COMMIT_RECORDED = "no_commit_recorded"
SOURCE_TREE_VOCABULARY = frozenset({
    SOURCE_TREE_MATCHED, SOURCE_TREE_MISMATCHED, SOURCE_TREE_NOT_APPLICABLE_GIT_CHECKOUT,
    SOURCE_TREE_NO_SIDECAR_HASH, SOURCE_TREE_NO_COMMIT_RECORDED,
})


def _live_source_tree_sha256(deploy_root: Path) -> str | None:
    """Recomputes `scripts/record_deployed_commit.py`'s own
    `source_tree_sha256`, against THIS tree, NOW (B21, validation round
    4) -- duplicated rather than imported, same as
    `_live_apk_last_update_time` duplicates `record_installed_apk.py`'s
    `_dumpsys_last_update_time` (CLI in /scripts, /nash for modules).
    `SOURCE_HASH_EXCLUDE_DIRS`/`_SUFFIXES` above must match that script's
    own constants exactly, or this disagrees with an unchanged tree.
    """
    if not deploy_root.is_dir():
        return None
    paths = [p for p in deploy_root.rglob("*") if p.is_file()]
    paths.sort(key=lambda p: p.relative_to(deploy_root).parts)
    hasher = hashlib.sha256()
    for path in paths:
        rel_parts = path.relative_to(deploy_root).parts
        if any(part in SOURCE_HASH_EXCLUDE_DIRS for part in rel_parts):
            continue
        if path.suffix in SOURCE_HASH_EXCLUDE_SUFFIXES:
            continue
        hasher.update("/".join(rel_parts).encode("utf-8"))
        hasher.update(b"\0")
        hasher.update(path.read_bytes())
        hasher.update(b"\0")
    return hasher.hexdigest()


def _live_apk_last_update_time(serial: str) -> str | None:
    """`lastUpdateTime` off `dumpsys package com.dsrc.phone`, queried NOW
    (A5, validation round 3) -- not the sidecar's recorded value, which
    `_build_provenance` reads separately and compares this against. `None`
    when adb cannot answer or the field is absent.

    The extraction itself is `dumpsys_util.parse_last_update_time`, shared
    with `scripts/record_installed_apk.py`'s own `_dumpsys_last_update_
    time` rather than duplicated (B23, validation round 4): these two
    values are COMPARED for equality, so a regex hardened in one copy and
    not the other would make that comparison assert a reinstall that
    never happened. Only the subprocess call itself (different timeout,
    same error handling) stays separate in each file.
    """
    try:
        result = subprocess.run(
            ["adb", "-s", serial, "shell", "dumpsys", "package", "com.dsrc.phone"],
            capture_output=True, text=True, timeout=10.0,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    return parse_last_update_time(result.stdout)


def _build_provenance(config: dict, *, serial: str | None = None) -> dict[str, Any]:
    """The four inputs a run's `advisory`/`sensing`/`perception` numbers
    depend on that nothing else in the record names (A4, validation round
    2): the code revision, the policy bundle, the detector engine, and the
    phone's APK.

    Given a run directory alone, none of these was previously recoverable:
    `summary.json`'s only model field was `policy_trained`, `models/
    .gitignore` is `*`, `metadata.jsonl` has no run-header record, and this
    function's own `git rev-parse` can itself return `commit: null` on a
    tree that is not a git checkout -- an ad-hoc rsync copy, same as this
    task's own `~/dsrc-task40` -- named rather than silently omitted, so a
    reader knows the absence is the tree's, not a bug here. `commit` falls
    back to `DEPLOYED_COMMIT_RECORD` when `git rev-parse` itself fails
    (B16, validation round 3): that file is written on the SOURCE machine,
    before the rsync that strips `.git`, so it is the one place the real
    tree's commit survives onto the ONLY machine a run happens on.

    `installed_apk_record`/`deployed_commit_record` carry those two
    sidecars WHOLE (B22, validation round 4), the same way `policy_bundle`
    already did: cherry-picking them down to `apk_sha256`/`commit` alone
    dropped `dirty`, `apk_name`, `version_code`, `version_name` and
    `local_apk_sha256` -- a bare 64-character hash that cannot be resolved
    to a build once the APK archive itself is gone. `apk_sha256`/`commit`
    stay as top-level fields too, since the two checks below need them as
    plain values to compare against a live re-read.

    `apk_last_update_time_check` (B20, validation round 4): a closed
    vocabulary (`APK_TIMESTAMP_VOCABULARY`) rather than a bare `bool |
    None`, because `None` used to mean two different things read alike --
    "not applicable, no serial was given" (the tailnet path) and
    "applicable, but the device would not answer" (a fault on a USB
    drive). `serial`, given, re-reads the phone's CURRENT `lastUpdateTime`
    and compares it to the value `apk_sha256`'s own sidecar was written
    against (A5, validation round 3): `apk_sha256` names an install that
    may since have been replaced by a later `adb install -r` nobody
    re-ran `record_installed_apk.py` for.

    `source_tree_check` (B21, validation round 4): `commit`'s own
    staleness signal. `DEPLOYED_COMMIT_RECORD` has no `--delete` behind
    it (`models/` is gitignored, and the deploy rsync does not pass it),
    so a re-deploy that forgets to run `record_deployed_commit.py` leaves
    deploy A's sidecar beside deploy B's code -- and this function always
    takes that sidecar path on the jetson, since `git rev-parse` always
    fails there, so it would report commit A with full confidence about a
    tree that is actually B's. This recomputes `record_deployed_commit.
    py`'s own `source_tree_sha256` against THIS tree and compares it to
    the sidecar's recorded value; `recorded_at` alone would not have
    caught this, since `rsync -a` preserves source mtimes and a freshly
    deployed file can be OLDER than a stale sidecar.
    """
    commit = None
    commit_from_git = False
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=str(JETSON_DIR),
            capture_output=True, text=True, timeout=5.0,
        )
        if result.returncode == 0:
            commit = result.stdout.strip()
            commit_from_git = True
    except (OSError, subprocess.SubprocessError):
        pass

    deployed_commit_record = None
    if commit is None and DEPLOYED_COMMIT_RECORD.exists():
        try:
            deployed_commit_record = json.loads(DEPLOYED_COMMIT_RECORD.read_text())
            commit = deployed_commit_record.get("commit")
        except ValueError:
            pass

    if commit is None:
        source_tree_check = SOURCE_TREE_NO_COMMIT_RECORDED
    elif commit_from_git:
        source_tree_check = SOURCE_TREE_NOT_APPLICABLE_GIT_CHECKOUT
    else:
        recorded_tree_sha256 = (deployed_commit_record or {}).get("source_tree_sha256")
        if recorded_tree_sha256 is None:
            source_tree_check = SOURCE_TREE_NO_SIDECAR_HASH
        else:
            live_tree_sha256 = _live_source_tree_sha256(JETSON_DIR)
            source_tree_check = (
                SOURCE_TREE_MATCHED if live_tree_sha256 == recorded_tree_sha256
                else SOURCE_TREE_MISMATCHED
            )

    # A copy, not a computation: the bundle sidecar is already JSON and
    # already carries sim_commit and contract_fingerprint.
    policy_bundle = None
    bundle_json_path = Path(resolve_model_path(config, "policy.bundle") + ".json")
    if bundle_json_path.exists():
        try:
            policy_bundle = json.loads(bundle_json_path.read_text())
        except ValueError:
            pass

    detector_engine_sha256 = None
    engine_path = Path(resolve_model_path(config, "detector.engine"))
    if engine_path.exists():
        detector_engine_sha256 = hashlib.sha256(engine_path.read_bytes()).hexdigest()

    installed_apk_record = None
    apk_sha256 = None
    apk_last_update_time = None
    if INSTALLED_APK_RECORD.exists():
        try:
            installed_apk_record = json.loads(INSTALLED_APK_RECORD.read_text())
            apk_sha256 = installed_apk_record.get("sha256")
            apk_last_update_time = installed_apk_record.get("last_update_time")
        except ValueError:
            pass

    if serial is None:
        apk_last_update_time_check = APK_TIMESTAMP_NO_SERIAL
    elif apk_last_update_time is None:
        apk_last_update_time_check = APK_TIMESTAMP_SIDECAR_HAS_NO_TIMESTAMP
    else:
        live_last_update_time = _live_apk_last_update_time(serial)
        if live_last_update_time is None:
            apk_last_update_time_check = APK_TIMESTAMP_DEVICE_DID_NOT_ANSWER
        elif live_last_update_time == apk_last_update_time:
            apk_last_update_time_check = APK_TIMESTAMP_MATCHED
        else:
            apk_last_update_time_check = APK_TIMESTAMP_MISMATCHED

    return {
        "commit": commit,
        "source_tree_check": source_tree_check,
        "deployed_commit_record": deployed_commit_record,
        "policy_bundle": policy_bundle,
        "detector_engine_sha256": detector_engine_sha256,
        "apk_sha256": apk_sha256,
        "apk_last_update_time": apk_last_update_time,
        "apk_last_update_time_check": apk_last_update_time_check,
        "installed_apk_record": installed_apk_record,
    }


def _thermal_summary(thermal_sampler, stats_sampler) -> dict[str, Any] | None:
    """`summary["thermal"]`, or `None` when no sampler was constructed this run.

    Stops the sampler before reading its own record, in that order: `stop()`
    joins the sampler's thread, and `_samples` is incremented inside the same
    lock as the rest of one pass but before `_write_sample` writes that pass's
    `thermal_sample` line to disk. Reading the record first can therefore race
    a pass that has already counted itself but not yet written -- `samples`
    would then be one more than the `thermal_sample` lines actually on disk,
    never one fewer. Stopping first, so the thread has finished its last write
    before `to_record()` runs, closes that window.
    """
    if thermal_sampler is None:
        return None
    thermal_sampler.stop()
    return thermal_sampler.to_record(
        jtop_available=stats_sampler.available if stats_sampler is not None else None
    )


def _write_log_health(logger, run_dir: Path) -> None:
    """`log_health.json`: the metadata logger's own final state, in a file of
    its own rather than a key in `summary.json`.

    `write_summary` opens `summary.json` with `"w"`, so folding this in would
    truncate a good summary if this write failed, and `dropped_records` is
    only finalised inside `close()`, which runs after `write_summary` -- so a
    summary can never carry the logger's own final drop count. Written after
    `close()` so `writer_failure` and `dropped_records` are whatever they will
    ever be, and `bytes_on_disk` is the flushed, closed file's real size.
    """
    health = {
        "t_wall": time.time(), "t_mono": time.monotonic(),
        "dropped_records": logger.dropped_records,
        "writer_failure": logger.writer_failure,
        "queue_depth": logger._queue.maxsize,
        "thread_alive_at_close": logger._thread.is_alive(),
        "path": logger.path.name,
        "bytes_on_disk": logger.path.stat().st_size if logger.path.exists() else 0,
    }
    (run_dir / "log_health.json").write_text(json.dumps(health, indent=2))


def apply_scenario(config: dict, args: argparse.Namespace) -> dict | None:
    """Load a simulated-drive scenario and fold it into config/args.

    Scenario JSON (paths relative to the scenario file):
      {
        "description": "...",
        "video": "../clips/i495_eastbound_480p.webm",
        "camera": {"fx_px": 360, "horizon_y_px": 205, ...},   // optional overrides
        "observation": {"free_flow_speed_mps": 26.8},         // optional overrides
        "gps": { ...gps_sim profile... }                       // optional
      }
    Explicit CLI flags (--source, --sim-gps) win over scenario fields.
    """
    if not args.scenario:
        return None
    path = Path(args.scenario).expanduser()
    with open(path) as f:
        scenario = json.load(f)
    base = path.resolve().parent
    if scenario.get("video") and not args.source:
        video = Path(scenario["video"])
        args.source = f"file:{video if video.is_absolute() else (base / video).resolve()}"
    for section in ("camera", "observation"):
        for key, value in (scenario.get(section) or {}).items():
            if key not in config[section]:
                raise KeyError(f"scenario {section}.{key} is not a known config key")
            config[section][key] = value
    scenario["_scenario_path"] = str(path)
    return scenario


# ---------------------------------------------------------------------------


def run_live(config: dict, args: argparse.Namespace, scenario: dict | None = None) -> int:
    from logio.metadata_logger import (
        MetadataLogger,
        SystemStatsSampler,
        TelemetrySender,
        make_run_dir,
    )
    from logio.video_logger import VideoLogger
    from ui.dashboard import DashboardWindow, TickSlot, render_dashboard

    gps_sim_spec = args.sim_gps or (scenario or {}).get("gps")

    phone = None
    usb_acceptor = None
    # Bound here, not just inside `if args.usb:`, so `_build_provenance`'s
    # A5 (validation round 3) live re-check has a name to read at teardown
    # regardless of which branch ran -- `None` on the tailnet path (there
    # is no adb serial to check with at all) rather than a `NameError`.
    serial = None
    if args.usb and not args.phone:
        print("[run] --usb has no effect without --phone", file=sys.stderr)
        return 2
    if args.phone:
        if gps_sim_spec is not None:
            # `--phone` takes BOTH sensors from the handset, so a simulated GPS
            # beside it is the very thing the one-flag design exists to prevent:
            # two halves of one run from two devices on two clocks. Refused here,
            # before the detector engine and policy bundle load, because the old
            # behaviour was to print "GPS: SIMULATED", wait for the phone, warm
            # everything up and then die on `gps.sim` -- which a PhoneGpsReader
            # does not have -- outside the try/finally, tearing nothing down.
            print("[run] --phone takes camera and gps from the handset; "
                  "--sim-gps and a scenario gps profile cannot apply to it",
                  file=sys.stderr)
            return 2
        if args.no_gps or args.require_gps:
            # Both are decisions about a local GPS this run does not have.
            # Ignoring them silently let a run ask for no GPS and get one.
            print("[run] --no-gps and --require-gps do not apply under --phone",
                  file=sys.stderr)
            return 2
        if args.source or (scenario or {}).get("camera"):
            # The camera half of the same door. Refusing only the GPS side left
            # this open, and it is the worse one: a scenario's `camera` block
            # rewrites fx_px, cx_px and horizon_y_px, which reach
            # `DistanceEstimator` -- so a clip's intrinsics get applied to the
            # phone's frames and every range comes out scaled by the ratio of the
            # two focal lengths, silently, into the observation builder and the
            # policy. The scenario's `video` would be discarded at the same time,
            # which is the inert-flag failure the GPS guard already refuses.
            print("[run] --phone supplies the camera; --source and a scenario "
                  "camera block cannot apply to it", file=sys.stderr)
            return 2
        from sensors.phone_link import PhoneLink

        if args.usb:
            if args.phone_host != "0.0.0.0":
                print("[run] --usb and --phone-host are mutually exclusive: a command "
                      "line cannot claim USB and name a tailnet address", file=sys.stderr)
                return 2
            from transport.usb import AdbError, UsbAcceptor, attached_serial

            serial = args.usb_serial
            if serial is None:
                serial, reason = attached_serial()
                if serial is None:
                    # Refuses rather than falling back to TCP: a fallback here
                    # would produce a run that looks like a USB run and is not.
                    print(f"[run] --usb: no device to establish a reverse against "
                          f"({reason})", file=sys.stderr)
                    return 2
            try:
                usb_acceptor = UsbAcceptor(serial, port=args.phone_port)
            except AdbError as exc:
                print(f"[run] --usb: could not establish the reverse mapping: {exc}",
                      file=sys.stderr)
                return 2
            # B2: the reverse mapping is external state on the device, not a
            # socket that dies with this process. Everything from here to the
            # teardown at the end of this function (build_components, camera
            # warmup, the worker thread, the display loop) can raise, and
            # none of it is behind the `finally` that eventually calls
            # `phone.stop()` -- which is what closes this acceptor -- because
            # that `finally` cannot exist until the objects it tears down
            # (camera, gps, ...) have themselves been built. Registered here,
            # at construction, rather than wrapping the downstream code in a
            # new try/finally: `atexit` runs on any exception that reaches
            # the top of the process, verified for both an uncaught
            # exception and a KeyboardInterrupt. `close()` is idempotent, so
            # a normal run that already closed it through `phone.stop()`
            # calls it again harmlessly.
            atexit.register(usb_acceptor.close)
            # C5 (validation round 2, upgraded to required): a bare `kill`
            # (SIGTERM) does not go through atexit at all -- Python's default
            # disposition for it terminates the process immediately, which
            # is a different path from an uncaught exception reaching the
            # top of the interpreter (that one does still run atexit,
            # confirmed above). Left unhandled, the leaked reverse mapping
            # holds a device-side listener on 127.0.0.1:47811 -- exactly
            # what `ImuWireTest.kt` binds a `ServerSocket` on
            # (`LinkConfig.DEFAULT_PORT`) -- so an instrumented run after a
            # SIGTERM'd drive fails with `EADDRINUSE` for a reason that has
            # nothing to do with it, and R2's own operational check
            # ("adb reverse --list before every run, assert non-empty")
            # passes on stale state. Raising SystemExit from the handler is
            # enough: it still reaches the top of the interpreter and still
            # runs the atexit callback just registered above (verified
            # empirically), so the handler does not call usb_acceptor.close()
            # itself.
            def _usb_sigterm_handler(signum: int, frame: Any) -> None:
                raise SystemExit(128 + signum)

            signal.signal(signal.SIGTERM, _usb_sigterm_handler)
            print(f"[run] usb: adb reverse tcp:{usb_acceptor.port} tcp:{usb_acceptor.port} "
                  f"on {serial}", flush=True)

        phone = PhoneLink(
            host=args.phone_host, port=args.phone_port,
            acceptor=usb_acceptor,
        )
        print(f"[run] waiting up to {args.phone_wait_s:.0f}s for a phone on "
              f"{phone.host}:{phone.port} (the phone dials; the Jetson cannot)", flush=True)
        if not phone.wait_for_phone(timeout_s=args.phone_wait_s):
            # Hard failure, deliberately. Falling back to the local camera and a
            # simulated GPS would produce a clean-looking run whose data never
            # went near a handset, and nothing downstream distinguishes the two.
            for refusal in phone.refusals:
                # A dialled-in phone we turned away is a different problem from a
                # phone that never called, and it is the likelier one during
                # bring-up. Sending the operator to look at the network when the
                # answer was a version mismatch costs an afternoon.
                print(f"[run] refused a connection -- {refusal}", file=sys.stderr)
            phone.stop()
            print("[run] no phone dialled in; refusing to run on local sources instead",
                  file=sys.stderr)
            return 2
        # Captured as soon as the phone is on, and again at the end. A relayed hop puts
        # tens of milliseconds into the link segment, and `link_ms` alone cannot say
        # whether it was a slow link or a relay. A previous measurement in this project
        # is unattributable for exactly that reason.
        from tailnet import path_for_address, peer_paths

        network_at_start = peer_paths()
        # Looked up by the SESSION's peer address, not by the phone's device id. The
        # device id is `Settings.Secure.ANDROID_ID` and the peer dict is keyed on
        # Tailscale's `HostName`: different namespaces, never equal, so the lookup
        # matched nothing and the line read `unknown` on every run.
        started_stats = phone.session.stats()
        session_path = path_for_address(network_at_start, started_stats.peer)
        # Which session this describes. After a redial the start and end entries are
        # two different sessions, and without an id the pair reads as one path
        # changing mid-drive -- which is what the block's own comment invites.
        session_path["session_id"] = started_stats.session_id
        print(f"[run] session peer {session_path['session_peer']}: "
              f"{session_path['path']}"
              f"{' via ' + session_path['relay'] if session_path.get('relay') else ''}"
              f"{'' if session_path['over_tailnet'] else ' -- NOT over the tailnet'}",
              flush=True)
        print(f"[run] phone {phone.peer_device_id} on session {phone.session.session_id}",
              flush=True)

    camera, gps, pipeline, actor = build_components(
        config, args.source, use_gps=not args.no_gps, gps_sim_spec=gps_sim_spec,
        phone=phone,
    )
    if gps_sim_spec is not None:
        print(f"[run] GPS: SIMULATED ({args.sim_gps or 'scenario profile'})")
    print(f"[run] detector warmup: {pipeline.detector.warmup():.1f} ms/frame")
    if not actor.is_trained:
        print("[run] WARNING: policy bundle is RANDOM-INIT (untrained); advisory values are placeholders")

    camera.start()

    logger = video_logger = stats_sampler = thermal_sampler = failures = None
    telemetry = None
    run_dir = None
    if config["telemetry"]["enabled"]:
        telemetry = TelemetrySender(config["telemetry"]["udp_host"], config["telemetry"]["udp_port"])
    if config["logio"]["metadata"] and not args.no_log:
        run_dir = make_run_dir(config["paths"]["log_dir"])
        logger = MetadataLogger(run_dir, config_path=config["_config_path"])
        if config["logio"]["video"]:
            video_logger = VideoLogger(run_dir, fps=config["camera"]["fps"])
        if config["logio"]["system_stats"]:
            stats_sampler = SystemStatsSampler(logger, config["logio"]["system_stats_interval_s"]).start()
        if config["logio"]["thermal"]:
            from sensors.thermal import ThermalSampler

            thermal_sampler = ThermalSampler(
                logger, phone=phone, interval_s=config["logio"]["thermal_interval_s"],
            ).start()
        if config["logio"]["failures"]:
            from logio.failure_log import FailureSampler

            failures = FailureSampler(
                logger, phone=phone, camera=camera, gps=gps, pipeline=pipeline,
                interval_s=config["logio"]["failure_interval_s"],
            ).start()
        print(f"[run] logging to {run_dir}")

    if gps is not None:
        if run_dir is not None and config["logio"]["nmea"]:
            gps.raw_log_path = str(run_dir / "nmea.log")
            gps.diagnostics.raw_log_path = gps.raw_log_path
        try:
            gps.start()
        except RuntimeError as exc:
            if args.require_gps:
                raise
            print(f"[run] GPS unavailable, continuing without it:\n      {exc}")
            gps = None

    if logger is not None and gps_sim_spec is not None:
        # ground truth + clock anchor for eval_run.py
        logger.write(
            {
                "type": "scenario",
                "scenario_path": (scenario or {}).get("_scenario_path"),
                "description": (scenario or {}).get("description"),
                "video_source": camera.source,
                "gps_profile": gps.sim.profile.to_dict(),
                "gps_start_wall": gps.start_wall,
                "gps_start_mono": gps.start_mono,
            }
        )

    v2v = None
    if config["v2v"]["enabled"]:
        from v2v.beacon import BeaconTransceiver

        v = config["v2v"]
        v2v = BeaconTransceiver(
            port=v["port"], beacon_hz=v["beacon_hz"], peer_ttl_s=v["peer_ttl_s"], range_m=v["range_m"]
        ).start()
        print(f"[run] V2V beacons on UDP :{v['port']} as {v2v.unit_id}")

    display = config["ui"]["display"] and not args.headless
    slot = TickSlot()
    stop = threading.Event()
    assumed_lane = config["observation"]["assumed_lane"]
    target_hz = float(config["loop"]["target_hz"])
    deadline = time.monotonic() + args.duration_s if args.duration_s else None
    last_print = 0.0
    # The estimate_id last written for each estimator, so a line goes out only
    # when the current estimate actually changed rather than once a tick. None
    # is a real prior state (no estimate published yet), distinct from having
    # written id 0, so this is not a plain int default.
    last_estimate_ids: dict[str, int | None] = {"round_trip": None, "one_way": None}

    # Constructed for a phone run only. Deciding rates for a local camera would
    # produce a command with nowhere to go, and the `here` query would describe a
    # road nobody is asking about.
    sensing = None
    if phone is not None:
        from policy.sensing_loop import SensingLoop
        from policy.shadow_mode import LIVE, SHADOW, ModeHolder

        sensing = SensingLoop(
            modes=ModeHolder(LIVE if args.live_rates else SHADOW),
            heartbeat_s=args.rate_heartbeat_s,
        )
        print("[run] sensing control: " + ("LIVE -- rates are applied on the phone"
              if args.live_rates else
              "shadow -- decisions recorded, nothing applied"), flush=True)

    def worker() -> None:
        nonlocal last_print
        try:
            _tick_loop()
        except BaseException as exc:  # noqa: BLE001
            # Recorded and re-raised, never swallowed. `BaseException` rather
            # than `Exception` because `KeyboardInterrupt` reaching this thread
            # is itself a fact worth recording, and catching it to record it
            # costs nothing when the `raise` below puts it straight back.
            if failures is not None:
                failures.note_pipeline_exception(exc)
            raise
        finally:
            # `stop.set()` used to be the last statement of the body, so anything
            # that raised took the thread out without it. The main loop is
            # `while not stop.is_set(): sleep(0.2)`, so the run hung with the whole
            # teardown unreached: no summary, no `phone.stop()`, and up to a
            # megabyte of buffered tick records lost, since MetadataLogger flushes
            # only in `close()`. `pipeline.step` could always raise; task 31 added a
            # second call site whose documented contract IS to raise.
            stop.set()

    def _tick_loop() -> None:
        nonlocal last_print
        while not stop.is_set():
            # At the TOP, so it governs every iteration rather than only the ones that
            # got a frame. It used to sit below the `continue`, and a camera that is
            # connected and silent therefore ran the drive forever: in headless mode
            # nothing else sets `stop`, so no summary was written and MetadataLogger,
            # which flushes only in `close()`, lost the whole tick log.
            #
            # Three states reach a silent camera. A JPEG that will not decode is
            # swallowed per frame, so a wrong `format` or a broken codec is permanent.
            # A handset whose camera pipeline has died while GPS and telemetry keep
            # the session healthy never trips the supervisor. And the redial gap lasts
            # up to `rebind_timeout_s`, which is 120 s -- that one is deliberate and
            # survivable, which is exactly why the run has to be able to reach its own
            # deadline while it lasts.
            if deadline and time.monotonic() >= deadline:
                break
            frame = camera.wait_for_fresh(timeout=1.0)
            if frame is None:
                if failures is not None:
                    failures.note_no_frame(end_of_stream=camera.end_of_stream)
                if camera.end_of_stream:
                    break
                continue
            if failures is not None:
                failures.note_frame()
            fix = gps.latest() if gps is not None else GpsFix()
            peers = None
            if v2v is not None:
                v2v.update_ego(fix, assumed_lane)
                peers = v2v.peers(fix)
            # Asked here, against this tick's own fix. `HereFeed.at` answers from
            # geometry against the position it is handed, not from a cached answer,
            # so asking anywhere else would describe a different piece of road.
            feed = None
            if phone is not None:
                feed = phone.here.at(fix, time.monotonic())
            tick = pipeline.step(frame, fix, peers, feed=feed)
            outcome = None
            if sensing is not None:
                # After the tick, because the decision reads the policy margin and
                # the density bin this tick produced -- and before the log, so the
                # record carries what was decided from it.
                outcome = sensing.on_tick(tick, phone)
            if logger is not None:
                record = tick.to_record()
                # Which peer clock this tick's phone-side stages, if any, were
                # measured or converted against -- the same key the persisted
                # `timebase_estimate` lines carry, so an offline reader can
                # refuse to convert this tick's `return` stage against an
                # estimate left over from a different session. Read once here,
                # rather than once for the record and again inside
                # `_log_timebase_estimates`, so the two cannot name different
                # sessions.
                session_id = tick_session_id(phone)
                if phone is not None:
                    record["session_id"] = session_id
                if outcome is not None:
                    record["sensing"] = outcome.to_record()
                if thermal_sampler is not None:
                    record["thermal"] = thermal_sampler.latest()
                if failures is not None:
                    record["failures"] = failures.latest()
                logger.write(record)
                if phone is not None:
                    _log_timebase_estimates(logger, phone, last_estimate_ids, session_id)
            if video_logger is not None:
                video_logger.write(frame.image)
            if telemetry is not None:
                telemetry.send(telemetry_record(tick))
            slot.publish(tick, frame.image)
            now = time.monotonic()
            if not display and now - last_print >= args.print_every:
                last_print = now
                link = "" if tick.link_ms is None else f" (link {tick.link_ms:5.1f})"
                print(
                    f"[{tick.tick_id:6d}] {tick.advisory.one_line()} | "
                    f"jetson {tick.jetson_ms:5.1f} ms{link}"
                )
            if args.max_ticks and tick.tick_id + 1 >= args.max_ticks:
                break
            if target_hz > 0:
                budget = 1.0 / target_hz - (time.monotonic() - frame.t_mono)
                if budget > 0:
                    time.sleep(budget)

    worker_thread = threading.Thread(target=worker, name="pipeline", daemon=True)
    worker_thread.start()

    window = None
    try:
        if display:
            window = DashboardWindow(fullscreen=config["ui"]["fullscreen"])
            horizon = config["camera"]["horizon_y_px"]
            shown = -1
            while not stop.is_set():
                tick, image = slot.latest()
                if tick is None or tick.tick_id == shown:
                    time.sleep(0.005)
                    continue
                shown = tick.tick_id
                canvas = render_dashboard(
                    image, tick, pipeline.stats.snapshot(), actor.is_trained, horizon_y=horizon
                )
                if window.show(canvas) == "quit":
                    stop.set()
        else:
            while not stop.is_set():
                time.sleep(0.2)
    except KeyboardInterrupt:
        stop.set()
    finally:
        stop.set()
        worker_thread.join(timeout=3.0)
        summary = {
            "ticks": pipeline._tick_counter,
            "stats": pipeline.stats.snapshot(),
            "camera_dropped_frames": camera.dropped_frames,
            "camera_file_recoveries": camera.file_recoveries,
            "policy_trained": actor.is_trained,
            # A4 (validation round 2): the code revision, policy bundle,
            # detector engine and phone APK that produced everything in
            # `advisory`/`sensing`/`perception` below -- none of which was
            # recoverable from a run directory alone before this. `serial`
            # is `None` off the tailnet path; `_build_provenance` reports
            # `apk_last_update_time_check: "no_serial"` rather than
            # checking anything in that case (A5/B20, validation rounds
            # 3/4).
            "build": _build_provenance(config, serial=serial),
        }
        if phone is not None:
            # Both ends of the run, because a path can change mid-drive: Tailscale
            # upgrades a relayed connection to a direct one when it manages to, and the
            # link segment measured before and after that are different quantities.
            end = peer_paths()
            ended = phone.session.stats() if phone.session is not None else None
            session_at_end = None
            if ended is not None:
                session_at_end = path_for_address(end, ended.peer)
                session_at_end["session_id"] = ended.session_id
            summary["network"] = {
                # What route the session's bytes actually took, at both ends of the
                # run. Without this the block recorded only how this machine WOULD
                # reach each online peer, which is the same whether or not the session
                # used that route -- so a run carried over USB wrote the same network
                # record as one carried over the tailnet.
                "session_at_start": session_path,
                "session_at_end": session_at_end,
                # Reachability, as context. Named separately so it is not read as the
                # session's path.
                "reachability_at_start": network_at_start,
                "reachability_at_end": end,
            }
        if usb_acceptor is not None:
            # Beside `network`, not folded into it: `network` answers "what
            # would this machine's tailnet reachability say", `usb` answers
            # "which cable, and did the reverse mapping survive the drive" --
            # task 32's lesson was that a record which cannot name its own
            # path is not evidence, and `path_for_address` already reports
            # `127.0.0.1` as USB, which says which path, not which cable or
            # how often it dropped.
            summary["usb"] = usb_acceptor.usb_record()
        if sensing is not None:
            # What was decided and how much of it reached the phone. A drive whose
            # commands were all refused and one where nothing needed sending write
            # the same tick log otherwise.
            summary["sensing"] = sensing.to_record()
        thermal_summary = _thermal_summary(thermal_sampler, stats_sampler)
        if thermal_summary is not None:
            summary["thermal"] = thermal_summary
        if failures is not None:
            # Stopped before it is read, for the reason `_thermal_summary`
            # gives for its own sampler: stopping joins the thread, so its
            # last records are on disk before `to_record()` reads its state.
            failures.stop()
            summary["failures"] = failures.to_record()
        if phone is not None:
            # Which clock produced the stamps and how the offset was obtained.
            # Without it a run where every stamp took the proxy path and one where
            # conversion worked write the same summary, and an offline reader
            # cannot tell a measured drive from a fictional one.
            summary["phone"] = phone.to_record()
        camera.stop()
        if gps is not None:
            gps.stop()
        if v2v is not None:
            v2v.stop()
        if video_logger is not None:
            video_logger.close()
        if stats_sampler is not None:
            stats_sampler.stop()
        if logger is not None:
            logger.write_summary(summary)
            logger.close()
            _write_log_health(logger, logger.run_dir)
            print(f"[run] logs: {logger.run_dir}")
        if telemetry is not None:
            telemetry.close()
        if window is not None:
            window.close()
        if phone is not None:
            # Last, after the sources that read from it. Stopping the link first
            # would close the session under two live reader threads. Only the
            # failure path used to do this at all, so a successful run left the
            # accept socket bound, the session open until the phone timed itself
            # out, and the responder thread running.
            phone.stop()
        print(summary_line(summary, camera.dropped_frames))
    return 0


# ---------------------------------------------------------------------------


def selfcheck(config: dict, args: argparse.Namespace) -> int:
    """Task P1 deliverable: verify every subsystem and report."""
    failures = 0

    def report(name: str, ok: bool, detail: str = "", warn: bool = False) -> None:
        nonlocal failures
        tag = "[ OK ]" if ok else ("[WARN]" if warn else "[FAIL]")
        if not ok and not warn:
            failures += 1
        print(f"{tag} {name}{': ' + detail if detail else ''}")

    report("config", True, config["_config_path"])

    engine_path = resolve_model_path(config, "detector.engine")
    if not Path(engine_path).exists():
        report("detector engine", False, f"{engine_path} missing - run ./export_detector.sh")
    else:
        try:
            det = TrtYoloDetector(engine_path, input_size=config["detector"]["input_size"])
            ms = det.warmup()
            report("detector engine", True, f"warmup {ms:.1f} ms/frame")
            det.close()
        except Exception as exc:
            report("detector engine", False, str(exc))

    try:
        actor = ActorRuntime(resolve_model_path(config, "policy.bundle"))
        import numpy as np

        out = actor.act(np.zeros(39, dtype=np.float32))
        detail = f"action {out.action['desired_speed_bin']}, {out.latency_ms:.2f} ms"
        if not actor.is_trained:
            detail += " (UNTRAINED random-init bundle)"
        report("actor policy", True, detail)
    except Exception as exc:
        report("actor policy", False, f"{exc}")

    try:
        cam = CameraStream(source=args.source or str(config["camera"]["source"]))
        cam.start()
        frame = cam.wait_for_fresh(timeout=3.0)
        cam.stop()
        if frame is not None:
            report("camera (hardware)", True, f"{frame.image.shape[1]}x{frame.image.shape[0]}")
        else:
            report("camera (hardware)", False, "opened but no frames in 3 s")
    except Exception as exc:
        report("camera (hardware)", False, str(exc))

    try:
        gps = GpsReader(
            port=config["gps"]["port"],
            baud=config["gps"]["baud"],
            configure_rate=config["gps"]["configure_rate"],
            target_rate_hz=config["gps"]["target_rate_hz"],
        )
        gps.start()
        time.sleep(3.0)
        fix = gps.latest()
        d = gps.diagnostics
        gps.stop()
        if d.sentences_parsed == 0:
            report("gps (hardware)", False, "port open but no NMEA sentences in 3 s")
        elif not fix.valid:
            report(
                "gps (hardware)", True,
                f"{d.sentences_parsed} sentences, NO FIX yet (normal indoors), "
                f"rate cfg {'sent' if d.rate_configured else 'failed'}",
            )
        else:
            report(
                "gps (hardware)", True,
                f"fix {fix.lat:.5f},{fix.lon:.5f} sats {fix.num_sats} "
                f"rate ~{d.observed_rate_hz():.1f} Hz",
            )
    except Exception as exc:
        report("gps (hardware)", False, str(exc))

    has_display = bool(os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY"))
    report(
        "display", True,
        "available" if has_display else "no DISPLAY - use --headless",
        warn=not has_display,
    )

    log_root = Path(config["paths"]["log_dir"]).expanduser()
    try:
        log_root.mkdir(parents=True, exist_ok=True)
        probe = log_root / ".write_probe"
        probe.write_text("ok")
        probe.unlink()
        report("log dir", True, str(log_root))
    except OSError as exc:
        report("log dir", False, f"{log_root}: {exc}")

    print(f"\nselfcheck: {'PASS' if failures == 0 else f'{failures} failure(s)'}")
    return 1 if failures else 0


def summary_line(summary: dict, dropped_frames: int) -> str:
    """The end-of-run line, extracted so it can be tested.

    It could not be before, and that mattered: an empty latency series now
    reports absence rather than zeros, and the first fallback here carried only
    `mean` while the format string reads `p50` and `p95` too -- so a zero-tick
    run raised KeyError inside the shutdown path and lost its summary. Nothing
    covered `run_demo` at all, so a test had to reimplement this line, and a
    test that reimplements the code cannot catch a change to it.
    """
    absent = dict.fromkeys(("mean", "p50", "p95"), float("nan"))
    latency = summary["stats"].get("jetson_ms") or absent
    return (
        f"[run] {summary['ticks']} ticks | jetson mean {latency['mean']:.1f} ms, "
        f"p50 {latency['p50']:.1f} ms, p95 {latency['p95']:.1f} ms | "
        f"dropped frames {dropped_frames}"
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", default=str(JETSON_DIR / "config.yaml"))
    parser.add_argument("--source", help="override camera.source (e.g. file:clip.mp4)")
    parser.add_argument("--scenario", help="simulated-drive JSON (video + camera + gps profile)")
    parser.add_argument(
        "--sim-gps",
        help="simulate GPS: const:<mps>[@lat,lon,heading] or a profile.json path",
    )
    parser.add_argument("--selfcheck", action="store_true", help="verify subsystems and exit")
    parser.add_argument("--headless", action="store_true", help="no display window")
    parser.add_argument("--no-log", action="store_true")
    parser.add_argument("--no-gps", action="store_true")
    # One flag for both sensors, because they are two channels of one session.
    # A `--phone-camera` that could be given without a `--phone-gps` would let a
    # run take its two sensors from different devices on different clocks.
    parser.add_argument(
        "--phone", action="store_true",
        help="take camera AND gps from a phone over the transport, instead of local devices",
    )
    parser.add_argument("--phone-host", default="0.0.0.0", help="address to accept the phone on")
    parser.add_argument("--phone-port", type=int, default=PHONE_DEFAULT_PORT)
    parser.add_argument(
        "--phone-wait-s", type=float, default=60.0,
        help="how long to wait for the phone to dial in before giving up",
    )
    # The USB path: an `adb reverse` mapping in place of the tailnet listener.
    # Mutually exclusive with --phone-host: a command line cannot claim USB
    # and name a tailnet address, because a `--phone-host 127.0.0.1` typed by
    # hand is indistinguishable from a genuine loopback test and establishes
    # no reverse at all.
    parser.add_argument(
        "--usb", action="store_true",
        help="accept the phone over adb reverse instead of the tailnet; "
             "mutually exclusive with --phone-host",
    )
    parser.add_argument(
        "--usb-serial",
        help="the adb serial to establish the reverse against; auto-detected "
             "when exactly one device is attached",
    )
    # Shadow is the default and live is chosen. A drive that gates for real because
    # nobody passed a flag is the wrong failure to have.
    parser.add_argument(
        "--live-rates", action="store_true",
        help="apply the sensing controller's rates on the phone; without it the "
             "decisions are recorded and nothing is applied",
    )
    parser.add_argument(
        "--rate-heartbeat-s", type=float, default=5.0,
        help="resend an unchanged rate command after this long, so a phone that "
             "reconnected mid-drive is not left on whatever it had",
    )
    parser.add_argument("--require-gps", action="store_true", help="fail instead of degrading without GPS")
    parser.add_argument("--duration-s", type=float, default=0.0)
    parser.add_argument("--max-ticks", type=int, default=0)
    parser.add_argument("--print-every", type=float, default=1.0, help="headless status period (s)")
    args = parser.parse_args()

    config = load_config(args.config)
    scenario = apply_scenario(config, args)
    if args.selfcheck:
        return selfcheck(config, args)
    return run_live(config, args, scenario)


if __name__ == "__main__":
    raise SystemExit(main())
