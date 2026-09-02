# Architecture — Jetson Advisory Deployment

How the prototype works internally, why it is shaped this way, and how it
maps onto the simulation. Written to be sufficient for resuming work cold.

## 1. Design priorities

1. **Latency over accuracy** (project decision). Every choice below trades
   estimation quality for milliseconds; the budget table in §4 shows the result.
2. **Sim-contract fidelity.** The actor must see the same 39-dim observation
   it saw in training, including the spec's neutral fallbacks for unsensed
   fields. The contract is vendored and test-locked (§6).
3. **Degrade, never die.** Missing GPS, no camera, no display, no trained
   checkpoint - each has a defined degraded mode, because in-car debugging
   time is expensive.

## 2. Dataflow

```
 camera thread          GPS thread              V2V rx thread (optional)
 (latest frame slot)    (latest fix + NMEA log) (peer table, TTL 2s)
        |                     |                     |
        v                     v                     v
 +------------------- pipeline thread (pipeline.py) -------------------+
 |  TensorRT YOLOv8n FP16  ->  IoU tracker  ->  pinhole distance/      |
 |  (detector.py)              (tracker.py)     lateral + d/dt slope   |
 |                                              (distance.py)          |
 |                    -> observation_builder.py                        |
 |                       39-field sim obs + provenance tags            |
 |                    -> actor_runtime.py (numpy MLP mirror)           |
 |                    -> advisory.py (decode = sim wrappers)           |
 +------------+------------------+------------------+-----------------+
              |                  |                  |
              v                  v                  v
        TickSlot (UI)      metadata.jsonl      UDP telemetry
              |            video.avi           (port 47900)
              v
        main thread: dashboard render (ui/dashboard.py)
```

Handoffs are all **latest-value-wins** (no queues on the hot path): a frame
that was never processed is dropped, a dashboard that renders slowly skips
ticks, a stalled SD card drops log records. Stale data is worse than missing
data for a real-time advisory.

Threading: capture, GPS, logging, video encode, jtop sampling, V2V are
threads; the heavy stages (cv2 resize, TensorRT execute, numpy) release the
GIL, so a single process suffices. The OpenCV GUI must own the main thread,
so the pipeline runs on a worker and the dashboard polls a `TickSlot`.

## 3. Module map

```
run_demo.py            entry: live demo, --selfcheck, --headless, --scenario
replay_demo.py         re-run pipeline over a recorded run, compare outputs
bench_latency.py       latency table generator (synthetic or video input)
eval_run.py            score a logged run: PASS/FAIL gates, report.md, plots
transcode_clip.py      clean glitchy test clips -> MJPG AVI (no ffmpeg CLI here)
pipeline.py            per-tick orchestration, Tick record, rolling p50/p95
sensors/camera_stream.py   V4L2/file/CSI capture thread, latest-frame slot,
                           mid-file decode-error recovery (reopen + seek)
sensors/gps_reader.py      gpyes/u-blox NMEA thread, 5 Hz UBX config, GpsFix
sensors/gps_sim.py         scripted-profile GPS twin of GpsReader (sim drives)
sensors/time_sync.py       monotonic vs wall clock rules, GPS-UTC offset
perception/detector.py     TensorRT 10 wrapper (pinned buffers), letterbox, NMS
perception/tracker.py      SORT-lite: greedy IoU + constant-velocity predict
perception/distance.py     ground-plane / width-prior distance, lateral, dZ/dt
perception/observation_builder.py  sensors -> 39-field sim observation + provenance
policy/sim_contract.py     VENDORED sim contract (fields, scales, encode, bins)
policy/export_policy.py    sim checkpoint -> TorchScript bundle (+ --random)
policy/actor_runtime.py    bundle loader; numpy MLP fast path (verified vs TS)
policy/advisory.py         action -> driver-facing text; decode mirrors sim
ui/dashboard.py            HUD render + window (main thread)
logio/                     JSONL metadata, raw video, UDP telemetry, jtop stats
v2v/beacon.py              optional UDP-broadcast cooperation beacons
calibration/camera_calibration.py  fov / checkerboard / horizon helpers
calibration/auto_horizon.py  fit horizon row + camera height from detections
data/scenarios/, data/clips/  simulated-drive definitions + test footage
tests/                     contract-equality, geometry, builder, smoke tests
```

## 4. Latency budget (measured 2026-06-11, this device)

End-to-end p50 **19.8 ms** / p95 20.2 ms at 48.5 FPS (`models/bench_results.json`).

| stage | p50 | notes / applied optimizations |
|---|---|---|
| detection | 17.7 ms | GPU compute is only 3.9 ms (trtexec); the rest is letterbox resize + `cv2.dnn.blobFromImage` preprocessing (was 23 ms with numpy preprocessing) and CPU NMS. Pinned host buffers, single CUDA stream. |
| tracking + distance | 1.1 ms | greedy IoU, no Hungarian/Kalman |
| observation + encode | 0.4 ms | pure-python field assembly |
| actor + advisory | 0.5 ms | numpy mirror of the TorchScript MLP (was ~4 ms p50 / 10.7 ms p95 through the TorchScript interpreter); mirror is verified against TS at load |
| capture wait | + up to 1 frame interval | 33 ms at 30 fps camera; not in the table above (bench uses pre-stamped frames). Live e2e ≈ 25-55 ms expected. |

Remaining levers, in order of value: 448-px engine (`./export_detector.sh
yolov8n.pt 448`, roughly halves detection), INT8 calibration, GStreamer
zero-copy capture (needs system OpenCV). None needed for the 200 ms target.

**The 200 ms target is on `jetson_ms`, not on `e2e_ms`.** With a phone camera the
capture happens on another device, so end-to-end is the sum of a bounded link
segment and an exact on-Jetson one. The threshold is a claim about what this
hardware can do, and charging it for a link the Jetson does not control would
fail a run for the network's behaviour -- and would loosen silently whenever the
timebase cannot convert a capture stamp, because the link segment then drops out
of the sum. `eval_run` gates `latency_jetson_p95`.

## 5. Simulation ↔ prototype observation mapping

(paper table; provenance is logged per-tick in `field_sources`, from the
closed vocabulary in `perception/provenance.py`. The map covers all 39 slots
`sim_contract.encoded_slot_names()` lists, not just the 33 flat observation
fields below -- `cooperation.*` and `nearby_av_lane_distribution.<lane>` each
get their own entry, dotted, so the missingness metric is a statement about
the whole encoded vector.)

| sim observation field | prototype source | provenance |
|---|---|---|
| ego_speed | GPS RMC speed-over-ground (5 Hz); held during dropouts | measured / measured_converted / measured_arrival_proxy / fallback_neutral (held value during a dropout) |
| ego_acceleration | least-squares slope of GPS speed (~1 s window), refused whenever this tick's GPS fix is not fresh | derived / fallback_neutral |
| ego_lane | `observation.assumed_lane` (no lane detection in v0) | static_config |
| ego_headway_s | leader_gap / ego_speed (inf when no leader, as sim) | derived |
| target_headway_s | previous tick's commanded headway bin (feedback loop, as in sim) | static/feedback |
| leader_gap, leader_relative_speed | nearest in-corridor track: pinhole distance + dZ/dt slope | measured |
| left/right_lane_front_gap | nearest track with lateral offset ≈ ∓1 lane | measured |
| follower_*, *_rear_gap, rear_required_decel | spec "empty road" values (inf / 0) - no rear sensing | fallback_neutral |
| target_lane_* | = current-lane values (sim defaults target lane to current) | derived |
| active_vehicle_count_local | forward in-range track count ×2 (symmetric extrapolation, `symmetrize_counts`) | derived / derived_empty (zero in-range tracks) |
| local_density_bin | sim formula count/(2·range/1000), sim bin edges (12, 30) | derived / derived_empty (zero in-range tracks -- the disagreement rule's only firing condition under shipped constants) |
| local_mean_speed_bin | mean(ego + rel_speed) over valid tracks, sim edges (8, 18) | derived |
| local_queue_estimate | tracks with absolute speed < 5 m/s (sim queue_speed) | derived / derived_empty (no in-range tracks) / fallback_neutral (tracks present, none measurable) |
| uncongested_low_speed_flag | mirrors `safety/etiquette.py` (density < 12 ∧ v < vf − 8), against a locally sensed density where the sim reads a segment one | approximated |
| distance_to_next_merge | 0.0 - **sim parity**: the sim itself hardcodes 0.0 | sim_parity |
| distance_to_downstream_bottleneck | inf (no map matching; sim's off-bottleneck value) | sim_parity |
| time_since_last_lane_change, lane_changes_last_km | inf / 0 (no lane-change detection) | fallback_neutral |
| nearby_av_*, cooperation.* | V2V beacons when enabled; else spec neutral fallbacks (count 0, mean speed = free-flow, pressure/congestion 0) | measured / fallback_neutral |
| nearby_av_lane_distribution.{0,1,2} | share of heard peers reporting each lane | derived (a peer carried a lane id) / fallback_neutral (no peers, or none carried one) |

Encoding (scales, inf clamping, bool handling) is **bit-identical** to
`src/rl/encoders.py` - property-tested in `tests/test_sim_contract.py`.

## 6. Contract vendoring

The Jetson must not import the sim env stack (`src.rl.actions` →
`src.envs.*` → `highway_env`). `policy/sim_contract.py` therefore vendors,
from sim commit `d477dba`:
field lists + FIELD_SCALES + `encode_local_observation` (numpy twin),
action heads/values/forced defaults, `decode_speed_bin` / `decode_headway_bin`,
neutral fallbacks, and `_bin` semantics.

**When the sim contract changes:** update `sim_contract.py` (and
`SIM_COMMIT`), run `python3 -m pytest tests/test_sim_contract.py` on a
machine where the sim imports (encoder tests run everywhere torch exists;
action/wrapper tests need highway_env), then re-export the policy bundle
(`export_policy.py` refuses dim mismatches).

The actor architecture (`backbone.{0,2,4}` + `heads.<name>` state-dict
layout) is likewise mirrored in `export_policy.VendoredActor` and checked by
`test_actor_state_dict_layout_matches_sim`.

## 7. Deviations from plan_deployment.md (and why)

| plan | v0 implementation | rationale |
|---|---|---|
| monocular depth net (Depth Anything small) | closed-form pinhole geometry (ground-plane + width-prior fallback) | a depth net would roughly double the GPU budget; geometry costs ~0.1 ms. Interface (`distance.py`) is swappable for an A/B later. |
| ByteTrack / SORT | SORT-lite (greedy IoU + const-velocity) | sub-ms; adequate for ≤15 windshield vehicles. Upgrade if real footage shows ID churn. |
| `deployment/jetson/logging/` | `logio/` | a local `logging/` package shadows the stdlib for every script in the folder |
| OBD-II reader (`obd_reader.py`) | not implemented | no adapter on hand; GPS speed + hold-on-dropout suffices for v0. Stub path documented in roadmap. |
| TorchScript inference | TorchScript artifact + numpy execution mirror | TS interpreter cost ~4 ms p50 / 10 ms p95 for a 33 KB MLP; numpy is 0.5 ms with no jitter. TS remains the artifact of record and the mirror is verified at load. |

## 8. Extension roadmap

1. **Second (rear) camera** → fills `follower_*` and `*_rear_gap` fields:
   instantiate a second `CameraStream` + `TrtYoloDetector` (one more ~18 ms on
   the same GPU stream budget - measure; consider 448 engine for both),
   a mirrored `DistanceEstimator`, and pass rear vehicles to the builder.
   The observation builder already has the field slots.
2. **OBD-II speed** (`sensors/obd_reader.py`): python-obd over ELM327 BT/USB;
   prefer OBD speed over GPS when fresh; GPS-vs-OBD comparison feeds the
   plan's observation-quality metrics.
3. **Two-unit cooperative demo**: enable `v2v.enabled` on both cars on one
   WiFi hotspot; `nearby_av_*`/`cooperation.*` fields then come live. The
   schema's aggregate-only constraint is honored.
4. **Map matching**: GPS → road segment + distance-to-merge from a preloaded
   GeoJSON of the test route; replaces two sim_parity fields with measured.
5. **Lane detection** for `ego_lane` and better lateral assignment.
6. **INT8 detector** after collecting a calibration set from drive videos.

## 9. Evaluation hooks (plan §"Prototype evaluation metrics")

Everything needed for the paper's tables is in `metadata.jsonl`:
per-tick `stage_ms`/`jetson_ms`/`link_ms`/`e2e_ms`/`fps` (system metrics, where
`link_ms` is null for a local camera and null whenever the capture stamp was
proxied rather than converted), `vehicles` with
per-track distance/method (perception metrics), `obs` + `field_sources` +
`obs_diagnostics.missingness` (observation quality), `head_probs`/`confidence`
(policy), `type: system` records with power/utilization from jtop, and a
per-tick `thermal` block (Jetson temperature and both devices' throttle-event
status) beside `type: thermal_sample` (the Jetson's 1 Hz temperature and
cooling-state series, independent of the tick loop) and `type: thermal_event`
(one line per throttle transition, on either device). `summary.json`
aggregates p50/p95, including a `thermal` rollup; `bench_results.md` is the
static-compute table; replay agreement comes from `replay_summary.json`;
`eval_run.py` turns any run dir into `report.md`/`report.json` + timeline
plots with PASS/FAIL gates, and its `## Thermal` section is not gated -- no
peak-temperature threshold has been measured yet, and a run recorded before
this existed reports `thermal: null` rather than a failure.

The failure event log gives the failures this repository already detects --
GPS dropout, HERE quota exhaustion, dropped frames, transport stalls -- a time
axis, an episode and a reader, without instrumenting any of them a second
time: `type: failure_scan` is a 1 Hz record, written whether or not anything
failed, so "nothing failed" is distinguishable from "nothing was watching";
`type: failure_event` is two records per episode (`phase: "open"` and
`phase: "close"`) for the 17 registry rows with `event_records=True`; for the
other 13 the episode still opens, runs and closes, but only the summary row
in `summary["failures"]["sources"]` proves it, since it never writes a
`failure_event` line at all. The close record carries a `recovered` /
`open_at_end` / `unobservable` outcome -- `unobservable` for a source that
stopped being readable while the episode was open, so a lost instrument is
never reported as a recovery nobody observed. A per-tick `failures` block beside `thermal`
names what was open at decision time and how fresh that view is;
`summary["failures"]` rolls the whole run up by source, three-word status
(`fired`/`quiet`/`not_evaluable`) included. `log_health.json` is a separate
file, not a `summary.json` key, carrying the metadata logger's own final
state (`dropped_records`, `writer_failure`) -- written after `close()`, since
`write_summary` runs before the logger's last drops are known. `eval_run.py`'s
`## Failures` section, between `## Thermal` and `## GPS`, renders all of it,
plus the phone's own failures (a fourth `SessionLog` line shape, rate-capped,
read only when a `--phone-log` is supplied) under a `phone (offline)` heading.
A run recorded before this existed reports `failures: null` rather than a
failure, the same convention `## Thermal` set.

## 10. Simulated-drive test harness

Closed-loop hardware-free validation: real dashcam footage through the
real detector, with GPS synthesized from a scripted profile.

```
scenario.json ─┬─ video ────────► CameraStream (file:, paced to clip fps)
               ├─ camera block ─► config overrides (fx/horizon/height/hood line)
               └─ gps profile ──► SimulatedGps ──► GpsFix @ 5 Hz
                                   (dead-reckoned, noise, dropouts, cold start)
```

- `SimulatedGps` mirrors `GpsReader`'s consumer surface; the pipeline
  cannot tell them apart. Its core (`GpsSimulator.state_at/fix_at`) is
  pure and deterministic — the evaluator re-instantiates it from the
  profile logged in `metadata.jsonl` (`type: scenario` record) to compare
  observed ego speed against scripted truth (RMSE gate).
- Scripted dropouts just stop publishing fixes, so the builder's
  hold-on-stale path runs exactly as in a real antenna outage.
- Found by this harness and fixed in v0: FFmpeg's VP9 decoder state can be
  poisoned mid-file by one bad cluster (CameraStream now reopens + seeks
  past it, bounded); YOLO boxes the ego hood and its reflections as a
  phantom leader at minimum range (`camera.hood_line_y_px` filter, plus
  ground-plane → width-prior fallback when a bbox bottom is occluded by
  the hood / clipped by the frame edge).
- Honesty limits on borrowed footage: fx rests on an assumed HFOV, so
  absolute distances are order-of-magnitude only (lane assignment and
  closing-speed signs are fx-invariant); ego speed comes from the script,
  not the video, so it won't match the visual motion. Neither limit
  applies to the real calibrated camera + real GPS.
