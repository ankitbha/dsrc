# Architecture — Jetson Advisory Deployment

How the prototype works internally, why it is shaped this way, and how it
maps onto the simulation. Written to be sufficient for resuming work cold.

## 1. Design priorities

1. **Latency over accuracy** (project decision). Every choice below trades
   estimation quality for milliseconds; the budget table in §4 shows the result.
2. **Measure what is read.** The observation carries the seven fields its
   four readers actually read (§5), each tagged with how it was obtained. It
   used to mirror the simulator's 39-dim encoder contract, because a local
   actor consumed the whole vector; that actor is gone.
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
 |                       7 measured fields + provenance tags           |
 |                    -> dsrc_runtime.py (one speed action per         |
 |                       super-segment, every 60 s)                    |
 |                    -> segment_advisory.py (decode = speed limit     |
 |                       x action fraction)                            |
 |                    -> safety_gate.py (bounds the ego segment's      |
 |                       recommended speed)                            |
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
perception/observation_builder.py  sensors -> the 7 measured fields + provenance
policy/export_dsrc_policy.py  DSRC checkpoint -> TorchScript bundle (+ --random)
policy/dsrc_runtime.py     bundle loader; argmax over three speed fractions
policy/segment_advisory.py one speed per super-segment -> the driver's number
policy/safety_gate.py      bounds that number; two speed rules, with a census
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
| policy + advisory | 0.5 ms | measured when the local-sensing actor ran this block. It now holds the DSRC step (a decision every 60 s, held over in between) and the safety gate, and is not comparable tick for tick with the number above. |
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

## 5. What one tick measures

Seven fields, each tagged per tick in `field_sources` from the closed
vocabulary in `perception/provenance.py`. `OBS_FIELDS` in
`perception/observation_builder.py` is the one place the set is written down,
and `_covers_obs` checks the provenance map against it by NAME rather than by
count.

| field | prototype source | provenance | read by |
|---|---|---|---|
| ego_speed | GPS RMC speed-over-ground (5 Hz); held during dropouts | measured / measured_converted / measured_arrival_proxy / fallback_neutral (held value during a dropout) | safety gate, sensing scheduler, advisory, dashboard |
| ego_acceleration | least-squares slope of GPS speed (~1 s window), refused whenever this tick's GPS fix is not fresh | derived / fallback_neutral | sensing scheduler (the free-tier event rule) |
| leader_gap | nearest in-corridor track: pinhole distance | measured / fallback_neutral (no leader) | safety gate (`forward_ttc`), dashboard |
| leader_relative_speed | that track's dZ/dt slope, once the window spans 0.2 s | measured / fallback_neutral | safety gate (`forward_ttc`) |
| local_density_bin | count/(2·range/1000), bin edges (12, 30) | derived / derived_empty (zero in-range tracks) | safety gate (`low_speed_uncongested`), sensing scheduler, advisory |
| active_vehicle_count_local | forward in-range track count ×2 (symmetric extrapolation, `symmetrize_counts`) | derived / derived_empty | dashboard |
| segment_target_speed | mean of heard V2V peers' speeds, else the configured free-flow speed | derived / fallback_neutral | safety gate (free-flow bound) |

The lane split still runs -- it is what decides which track is the leader --
but only the ego lane's nearest reaches a field. The traffic feed owns no
field at all: it is published beside the observation on
`ObservationResult.feed`, where the sensing scheduler reads it and the record
keeps it.

**What used to be here.** Thirty-two more fields, mirroring the simulator's
39-slot encoder contract, plus the encoded vector itself. They were the
local-sensing actor's input. That actor was removed; nothing read the vector
afterwards, and most of the fields were the simulator's "empty road"
constants for sensors this rig does not have -- a constant carried into a
decision that never compares it is indistinguishable, in the record, from a
measurement that did not matter. `policy/sim_contract.py`,
`specs/sim_contract_golden_vectors.json` and its generator went with them.
`eval_run.py` still reads the 39-key shape (`LEGACY_ENCODER_SLOTS`), because
every drive in the recorded corpus was written under it.

## 6. Contract vendoring

### 6.1 The safety and etiquette contract (task 144)

`policy/safety_gate.py` vendors `src/safety/{constraints,etiquette,safety_layer}.py`
by copy rather than by import: the Jetson must not import the simulation
stack. This is the one vendored contract left on the device --
`policy/sim_contract.py`, which vendored the 39-slot encoder, went with the
actor that read it. `SafetyConstraints`
and `SafetyContext` are checked against a committed reference, `specs/
safety_contract_golden.json` (field names, defaults, and a hash over both),
by two tests that never import each other's side:
`deployment/jetson/tests/test_safety_contract.py` (the vendored copy,
unconditional, no `importorskip`) and `tests/test_safety_contract_matches_
golden.py` (`src/safety/` itself). This replaces the older idiom of comparing
the vendored copy against the original: task 143 found that check goes
vacuous the moment the original it compares against is deleted
(`src/rl/encoders.py` and its siblings were, and the test then reported `1
skipped` and said nothing). A golden file has no side that can disappear out
from under it.

**Where the gate runs, and what it bounds.** `pipeline.step` runs the gate
right after the DSRC step, on every tick that has an ego row to bound. The
gate is not run at all on a tick with no recommendation -- a rig with no
policy for its road, or a fix off the network -- and `tick.safety_gate` is
`None` there, which is a different fact from a gate that found nothing to
do. Where it does run it runs whole (`gate_ms`/`stages["gate"]` are
`measured`, never `absent` -- there is no early-return path here the way
`dsrc_infer`'s coverage-gate refusal has one).

Only the ego segment's row is bounded. The other eleven rows are the
controller's output for stretches of road this vehicle is not on, and the
gate holds no evidence about those stretches. The ego row's
`recommended_speed_mps` and `recommended_speed_display` are overwritten with
the bounded values, so that field keeps meaning "the number shown to the
driver" for every reader that already treats it that way (`eval_run.py`,
`replay_demo.py`, `transport/messages.py`). The target headway is a fixed
rig setting rather than a fed-back action: the controller emits one speed
fraction per super-segment and no headway at all.

**Decision 3's per-field input-class partition** is what keeps a
`fallback_neutral` observation from either silently never firing a rule (a
statement about the fallback, not about traffic) or firing on a substituted
constant (worse). Each of the seven `SafetyContext` fields is exactly one
of: (A) *configured road property* (`free_flow_speed_mps`) -- never blocks a
rule; (B) *evidence-required* (`ego_speed_mps`, `leader_gap_m`,
`leader_relative_speed_mps`, `local_density_veh_per_km`, the last with its
own carve-out: a `derived_empty` density counts as evidence only when
`obs_diagnostics.last_detection_age_s` bounds how recently the camera saw
anything at all) -- blocks the rules that read it whenever its
`perception.provenance` class is in `SUBSTITUTED`; (C) *structurally absent*
(the merge-conflict pair, which needs a map match to a joining node this rig
cannot make) -- blocks unconditionally, no sensor exists. Both rules are
evaluated as total, independent predicates for the per-tick `safety` record
(`Tick.to_record()`, beside `advisory`), never short-circuited by chain
position.

**A not_evaluable rule changes nothing in the decision, not only in the
record (validator round 1, F1).** `apply_safety_layer` runs against
`SafetyInputs.inert_context()`, not the raw observed context: every field
lacking evidence this tick is replaced by its `INERT_CONTEXT_VALUES` entry
(`policy/safety_gate.py`, beside `RULE_READS`) before it reaches the
decision, so a not_evaluable rule cannot move the recommended speed or
`emergency_override` by reading the observation's own substituted default --
`local_density_veh_per_km`'s substituted `0.0` was the reproduced case:
below the uncongested threshold, the opposite of inert, and it used to raise
the recommended speed 20.0 -> 22.0 with every rule not_evaluable. `evaluate_rules`'s own census
(above) keeps reading the unmodified context, so the record still says
what was actually observed.

**What is left of the twelve rules, and why.** Ten were removed, and the
test was effect rather than name: a rule earns its place by bounding the
recommended speed. `low_speed_uncongested` raises a low recommendation on an
uncongested road and `forward_ttc` drives the emergency override, so both
survive. The other ten only nulled the lane action, which a speed-only
advisory does not carry -- and seven of the ten were `not_evaluable` on every
tick this rig could ever produce anyway (no rear sensor, no lane-change
detector, no lane index but the assumed one). `config.yaml`'s
`safety.withhold_lane_when_not_evaluable` went with them: a flag over a
decision the controller no longer makes had nothing left to withhold.
`safety.enabled` (default `true`; validator round 1, Fix 3) is the rollback
for the whole gate: `false` still runs the full census and writes the
complete record, but `bounded_*` equals `proposed_*` exactly.
`deployment/jetson/score_safety.py` measures the resulting per-rule
evaluability census against recorded runs, refuses to print a firing rate
for a rule evaluable on zero ticks, and replays each run under its OWN
recorded `safety.config` rather than this tool's own defaults (validator
round 1, F3/F4) -- refusing, and naming the missing key, for a run
recorded before that config block existed.

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
   The observation builder already has the field slots. On the safety gate
   (§6.1) it would restore nothing on its own: the three rear rules
   (`target_lane_rear_gap`, `target_lane_rear_ttc`,
   `target_lane_rear_braking`) were removed along with the lane action they
   nulled, so a rear camera means reinstating them and the lane advisory they
   guard, which also needs a lane-change detector and a lane index for the
   four guards (`lane_change_dwell`, `lane_changes_per_km`,
   `target_lane_missing`, `target_lane_front_ttc`) that read those.
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
`obs_diagnostics.missingness` (observation quality), `advisory` and `dsrc`
(the ego segment's recommended speed, and the whole-network decision it came
from), `safety` (the gate's proposed/bounded pair and its per-rule census),
`type: system` records with power/utilization from jtop, and a
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

`report.md` opens with a `## Session summary` section, above `## Gates`:
seven instrument axes (latency, rates, API calls, triggers, failures,
thermal, provenance), each reporting two independently counted integers --
`answered` of `attempted` -- and the census of its own reason words when they
differ, never a percentage or a single health word. Ten cross-record
reconciliations follow, each `held`, `failed` (both numbers named, neither
resolved) or `unavailable` (an input, usually `summary.json`, was missing --
a population of zero records compared is also `unavailable`, never `held`).
The tenth needs only tick records and checks the reference shape rule below.
A
`## Sensing` section, between `## Advisory` and `## Phone join`, renders the
rate/trigger/HERE detail behind three of those axes. `here_calls` and
`here_errors` -- the phone's own count of HERE HTTP calls placed and non-2xx
responses, on the wire since task 27 -- now ride on every tick's per-tick
`sensing.reference` block alongside `achieved`/`dropped`, null together with
the rest of that block on a phone never heard from.

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
