# DSRC Edge Deployment — Jetson Orin Nano

Advisory-only, in-vehicle prototype of the self-regulating AV stack
(see `../../plans/plan_deployment.md`). A windshield camera and GPS feed an
edge perception pipeline that reconstructs the simulation actor's observation
vector; the simulation-trained policy then produces speed/headway/lane
recommendations on a display, in real time.

> **Safety:** this system is never connected to throttle, brake, or steering.
> Recommendations must not be followed while driving. Collect data with a
> passenger operating the device. The dashboard shows a permanent
> "ADVISORY ONLY" banner.

**Measured on this device (2026-06-11, full pipeline, YOLOv8n FP16 @640):**

| stage | mean (ms) | p50 (ms) | p95 (ms) |
|---|---|---|---|
| detection (TensorRT FP16, incl. pre/post) | 17.7 | 17.7 | 18.0 |
| tracking + distance | 1.1 | 1.1 | 1.2 |
| observation build + encode | 0.4 | 0.4 | 0.5 |
| actor + advisory decode | 0.5 | 0.5 | 0.6 |
| **end-to-end (capture → advisory)** | **19.8** | **19.8** | **20.2** |

48.5 FPS pipeline throughput; raw TensorRT GPU compute 3.9 ms (trtexec).
Regenerate with `python3 bench_latency.py` (writes `models/bench_results.{md,json}`).
The plan's target is < 200 ms end-to-end; live runs add one camera frame
interval (~33 ms at 30 fps) on top of the compute path above.

## Hardware

| component | this rig | notes |
|---|---|---|
| compute | Jetson Orin Nano Dev Kit (Super), JetPack 6 / L4T R36.4.7, CUDA 12.6, TensorRT 10.3 | |
| GPS | gpyes 2.0 (u-blox 8) on `/dev/ttyACM0` | NMEA @ 1 Hz default; we raise to 5 Hz at startup (UBX-CFG-RATE) |
| camera | USB UVC, 1280x720 MJPG 30 fps | none attached yet — pipeline fully testable without it (see below) |
| display | HDMI/touchscreen for the dashboard | optional; `--headless` otherwise |
| optional later | second (rear) camera, OBD-II speed | see ARCHITECTURE.md "Extension roadmap" |

## One-time setup on a fresh unit

```bash
# 1. serial-port permission for the GPS (then LOG OUT and back in)
sudo usermod -aG dialout $USER

# 2. python deps (system torch/tensorrt/opencv come from JetPack - don't pip them)
pip install --user -r requirements_jetson.txt

# 3. build the detection engine for THIS device (~10 min, device-specific)
./export_detector.sh                     # yolov8n @ 640 -> models/yolov8n_640_fp16.engine

# 4. actor policy bundle
python3 policy/export_policy.py --random --out models/actor_policy        # until training delivers a checkpoint
# python3 policy/export_policy.py --checkpoint /path/actor.pt --out models/actor_policy

# 5. verify everything
python3 run_demo.py --selfcheck
```

`--random` produces a correctly-shaped untrained actor so the entire pipeline
and all latency numbers work before training finishes; the dashboard shows an
`UNTRAINED POLICY` banner until a real checkpoint is exported.

## Calibration (do before trusting distances)

```bash
# fastest: focal length from the camera spec-sheet FOV
python3 calibration/camera_calibration.py fov --width 1280 --hfov-deg 78

# proper: checkerboard intrinsics (9x6 inner corners, ~20 photos)
python3 calibration/camera_calibration.py intrinsics --images '~/calib/*.jpg'

# horizon row (redo after every camera remount!)
python3 calibration/camera_calibration.py horizon --source 0
```

Put `fx_px / cx_px / cy_px / horizon_y_px` into `config.yaml` under `camera:`,
and tape-measure `camera_height_m` (lens center to road). Distance accuracy is
secondary to latency for this prototype, but the *relative* trends (closing
speeds, density) should be stable — see ARCHITECTURE.md for error behavior.

## Running

```bash
python3 run_demo.py                          # live: dashboard + logging
python3 run_demo.py --headless --duration-s 600   # in-car logging run
python3 run_demo.py --source file:clip.mp4 --no-gps   # desk run on a video
python3 replay_demo.py --log ~/dsrc_logs/run_20260611_120000   # Task P7 replay
python3 bench_latency.py                     # latency table for the paper
```

Dashboard keys: `q` quit, `f` fullscreen.
Each run writes `~/dsrc_logs/run_<timestamp>/` with `metadata.jsonl` (one
record per tick: observation, provenance, action, advisory, latencies, GPS),
`video.avi` (raw frames, if `logio.video: true` — required for replay),
`nmea.log`, `run_config.yaml`, and `summary.json`. Records use Python-JSON
`Infinity` literals (sim convention for "no vehicle"); read with Python/pandas.

## Simulated drives (no hardware needed)

A *scenario* bundles a dashcam clip, per-clip camera geometry, and a
scripted GPS profile; `SimulatedGps` (`sensors/gps_sim.py`) replaces the
serial reader and dead-reckons fixes at 5 Hz with optional noise and
scripted dropout windows. The whole real pipeline — TRT detector, tracker,
distances, observation, actor — runs paced in real time, exactly as live.

```bash
# full simulated drive (dashcam video + scripted GPS), then score it
python3 run_demo.py --headless --scenario data/scenarios/i495_cruise.json
python3 eval_run.py ~/dsrc_logs/run_<timestamp>   # gates: PASS/FAIL + report.md + plots

# quick GPS sim on any clip, no scenario file
python3 run_demo.py --source file:clip.mp4 --sim-gps const:25
```

Shipped scenarios (clips + licenses: `data/clips/SOURCES.md`):
- `i495_cruise.json` — 280 s beltway cruise, dense traffic, one scripted
  4 s GPS dropout (exercises hold-on-stale).
- `highway_decel.json` — 30 s approach to a pileup; leader-gap closing and
  the braking profile 24→5 m/s.

`eval_run.py` checks: e2e p95 < 200 ms, tick rate ≥ 25 Hz, GPS freshness,
ego-speed RMSE vs the scripted profile, perception coverage. Advisory
content is reported but never gated (meaningless until a trained bundle).

Preparing a new clip: `transcode_clip.py <clip>` (cleans decode glitches,
makes MJPG), then `calibration/auto_horizon.py <clip> [--hood-line N]` to
fit `horizon_y_px`/`camera_height_m` for the scenario's camera block.
Caveats for borrowed dashcam footage: fx comes from an assumed HFOV, so
*absolute* distances are only order-of-magnitude (lane assignment and
closing-speed signs are fx-invariant); if the ego hood is visible, set
`hood_line_y_px` or the detector reports the hood as a phantom leader.

## Troubleshooting

| symptom | fix |
|---|---|
| `cannot open GPS port ... permission` | `sudo usermod -aG dialout $USER`, re-login |
| `engine ... failed to load` / missing | rebuild for this device: `./export_detector.sh` (engines don't transfer across JetPack/TRT versions) |
| `actor bundle not found` | `python3 policy/export_policy.py --random --out models/actor_policy` |
| GPS opens but `NO FIX` | normal indoors; needs sky view, cold start can take ~1 min |
| no camera at `/dev/video0` | check `lsusb` / cable; desk-test with `--scenario data/scenarios/i495_cruise.json` |
| video file ends early / decode errors | OpenCV recovers automatically (`camera_file_recoveries` in summary.json); for a permanently clean copy: `python3 transcode_clip.py <clip>` |
| CSI camera wanted | pip OpenCV has no GStreamer; either use a USB camera or install system OpenCV and use `source: "csi:0"` |
| import errors in tests on a dev machine | run from `deployment/jetson/`: `python3 -m pytest tests/` (tests add paths via conftest) |

## Documentation map

- `ARCHITECTURE.md` — internals: dataflow, threading, latency budget, what one
  tick measures, deviations from the plan, extension roadmap.
- `PROGRESS.md` — status against plan tasks P1–P8, measured numbers, exact
  next steps. **Start here when resuming work.**
- `../../plans/plan_deployment.md` — the original deployment plan.
- `../../specs/observation_schema.md`, `action_schema.md` — the simulator's own
  contracts. This deployment no longer mirrors them: the observation carries
  the seven fields its readers read (`perception/observation_builder.py`,
  `OBS_FIELDS`) and the advisory is a speed. `specs/safety_contract_golden.json`
  is the one contract still vendored onto the device.
