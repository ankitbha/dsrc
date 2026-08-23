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
import json
import math
import os
import sys
import threading
import time
from pathlib import Path

JETSON_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(JETSON_DIR))

import yaml  # noqa: E402

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
):
    cam_cfg = config["camera"]
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

    obs_cfg = dict(config["observation"])
    obs_cfg["gps_stale_after_s"] = config["gps"]["stale_after_s"]
    builder = ObservationBuilder(BuilderConfig.from_dict(obs_cfg))

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
    return camera, gps, pipeline, actor


def telemetry_record(tick) -> dict:
    rec = tick.to_record()
    for heavy in ("obs", "encoded", "head_probs", "field_sources", "vehicles"):
        rec.pop(heavy, None)
    return rec


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
    camera, gps, pipeline, actor = build_components(
        config, args.source, use_gps=not args.no_gps, gps_sim_spec=gps_sim_spec
    )
    if gps_sim_spec is not None:
        print(f"[run] GPS: SIMULATED ({args.sim_gps or 'scenario profile'})")
    print(f"[run] detector warmup: {pipeline.detector.warmup():.1f} ms/frame")
    if not actor.is_trained:
        print("[run] WARNING: policy bundle is RANDOM-INIT (untrained); advisory values are placeholders")

    camera.start()

    logger = video_logger = stats_sampler = None
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

    def worker() -> None:
        nonlocal last_print
        while not stop.is_set():
            frame = camera.wait_for_fresh(timeout=1.0)
            if frame is None:
                if camera.end_of_stream:
                    break
                continue
            fix = gps.latest() if gps is not None else GpsFix()
            peers = None
            if v2v is not None:
                v2v.update_ego(fix, assumed_lane)
                peers = v2v.peers(fix)
            tick = pipeline.step(frame, fix, peers)
            if logger is not None:
                logger.write(tick.to_record())
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
            if deadline and now >= deadline:
                break
            if target_hz > 0:
                budget = 1.0 / target_hz - (time.monotonic() - frame.t_mono)
                if budget > 0:
                    time.sleep(budget)
        stop.set()

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
        }
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
            print(f"[run] logs: {logger.run_dir}")
        if telemetry is not None:
            telemetry.close()
        if window is not None:
            window.close()
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
