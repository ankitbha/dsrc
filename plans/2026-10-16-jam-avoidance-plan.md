# Jam avoidance: a shared feed plan with a private camera correction

Written 2026-10-16 by the planning agent, on branch `jam-avoidance` at `d699ac2`. Every
design decision below was taken by recommendation without asking; the decisions table records
each one, and the open items list the decisions where the planner had no real preference or
where the code or data contradicted the brief. Nothing in this plan has been implemented. The
appendix records the read-only measurements and scratch pilots made while planning; all of them
ran from the session scratchpad (`scratchpad/dsrc_jam/plan_measure/`) and wrote nothing into the
repository.

## Short version

### Summary

The question is whether a vehicle that already follows a shared plan computed from a public
traffic feed avoids more jams when its own camera may lower its advisory speed by one step. The
feed reports speed, and it reports it late: on the drives a publication was first received 34 to
110 s after its stamp (median 63 s), and the plan decides once per 60 s publication. When a
jam's speed drop begins less than that delay before onset, no controller that reads only the
feed can change a vehicle's speed before the jam forms. A camera measures the density of the
vehicles ahead, and in the no-control burst runs of the previous program the density on the
first segment to jam rose a median of 640 s before onset while the speed fell a median of 320 s
before onset.

This plan specifies the two-layer controller, the camera error model measured on the
re-processed drive frames, the SUMO sensing model that carries that error model to hundreds of
vehicles, seven arms, seven scenarios, a lag sweep, a penetration sweep and a camera-bias sweep,
three short theory statements, the step table, the experiment matrix, the figures, the validator
attack points and the risks. The first experiment is a go/no-go gate: the oracle correction
against the plan alone on training seeds, because if the noise-free correction does not reduce
jams, no camera arm can.

Four measurements made while planning shape the design.

- **Bursts do not produce fast onset on this network.** On training seeds 1 and 2, no control
  first falls below the 19.0 km/h jam threshold 620 to 790 s after the start of every burst
  tried (5,000, 6,000, 7,500 and 9,000 veh/h), because the first segment to jam is always
  segment 10 or 11 at the exit, 2.2 to 13.3 km along the routes from the eight entries. One
  lane of the exit road blocked by a stopped vehicle gives onset 40 to 80 s after the block. The burst sweep
  therefore varies how long the feed's own quantity, segment speed, warns before onset, and the
  lane block supplies the scenario whose onset is shorter than the feed's delay plus its
  decision interval (open item 2).
- **A simplified oracle correction gives less jam time than the plan alone in a pilot.** On
  training seeds 1 to 3, under the placeholder feed lag, the plan alone has a jam spell of at least
  120 s on all 3 seeds of the 9,000 veh/h burst (19.2 to 28.3 jammed segment-minutes) and the
  oracle correction on none of them (0.0 segment-minutes); on the exit lane block the oracle's
  jam time is 9.8 to 10.5 segment-minutes against the plan's 12.0 to 17.2. On the background
  demand with no perturbation the same oracle is active on 10% of controlled vehicle-ticks and
  its served flow is 44 veh/h below the plan's. This pilot is not the gate; it is the reason the
  gate runs first, and it shows that false alarms cost served flow.
- **The sensing pass costs 1.4 s per 2,500 s episode.** One sorted pass over every lane of every
  segment edge every 5 s, counting leaders within 80 m for about 800 controlled vehicles, took
  3.0 ms per pass (445 passes), and the episode took 8.8 s against 7.5 s without it. The pass
  did not change the trajectory: served flow was 3,648.89 veh/h in both runs.
- **The drive frames can be re-processed in about 27 minutes.** The six indexed drives hold
  18,910 frames; YOLOv8n on this laptop's CPU takes 18.8 ms per frame at 640 px and 62.0 ms at
  1,280 px. The frames were captured every 0.20 s on four drives and every 0.67 s on two, not
  at the 30 fps the video files are tagged with.

The total compute is about 11 hours of single-process time, inside the 12-hour budget.

### Scope boundary

In scope: the local sensing model and camera model in SUMO; incidents in `MainzEnv`; the
lookahead rule, the take-the-slower rule, the rule correction, the learned correction and the
oracle; a jam episode engine; jam metrics and lead-time analysis; frame re-processing and the
camera error model; the drive case table; the predictor training script; a runner with resume,
pause and provenance; figure scripts; tests; three new demand files built with the existing
builder options.

Out of scope: retraining the plan's Q-network (`mainz_here_best.pt` is frozen); any change under
`paper/`; any edit to `scripts/poll_here_feed.py` or `scripts/analyze_feed_process.py`; any
drive; any HERE request; the reward, the network and the 0.5/0.75/1.0 action set; running the
experiments (the implementer runs smoke episodes only).

### Open items

1. **The learned correction is a supervised onset predictor, not reinforcement learning on the
   density-and-speed reward.** The author was told the correction would be trained on SRC's
   reward plus a per-step correction cost. Retraining SRC's Q-network under the cap actuation
   with the published protocol selected a constant action vector (0 changes in 37 decisions,
   Amendment 4), and a per-vehicle keep-or-step-down policy trained on a shared reward has no
   per-vehicle credit assignment for a rare jam event. The plan trains a classifier whose label
   comes from the simulator (a jam spell of at least 120 s begins on the vehicle's segment or the
   next downstream one within 120 s) and chooses its threshold on the selection seeds by the
   pre-stated criterion. SRC's per-episode return is recorded for every candidate so the author
   can compare the choice against the reward. This departs from what the author was told.
2. **Burst size does not move onset.** The brief asks for a burst-size sweep from slow to fast
   onset. Measured onset after burst start is 620 to 790 s at every size from 5,000 to
   9,000 veh/h (appendix A3), so the sweep cannot reach onset shorter than the feed delay plus
   the 60 s decision interval (137 s at the placeholder's lag, 146 to 156 s at the polling
   preview's). The plan keeps three burst sizes because they change the feed's speed-precursor
   lead, and takes fast onset from the exit lane block.
3. **Super-segments are 0.13 to 2.3 km long in this port, not 2 to 3 km.** The longest
   along-route run through each super-segment is 1,906, 1,015, 1,458, 2,324, 1,221, 2,198,
   2,015, 1,247, 517, 1,251, 131 and 428 m (segments 0 to 11). The jams start on segments 10 and
   11, where an 80 m camera view covers 61% and 19% of the road, against 3.4% on segment 3.
4. **"No range limit" for the oracle is read as perfect detection inside the same 80 m window.**
   The oracle counts every vehicle on the visible lanes within 80 m, with no occlusion, no missed
   detection and no distance error, so it isolates the camera's error at the same window. A wider
   variant (200 m) runs on two scenarios for the record. If the author meant a window to the end
   of the segment, the wide variant is the closer one.
5. **The plan's speed is refreshed for every vehicle every 5 s in every arm.** The deployed
   policy was trained with `step()` semantics, under which a vehicle that changes segment between
   decisions keeps its old command. Refreshing is what a deployed vehicle with map matching does,
   and it makes the plan-alone arm differ from the corrected arms only by the correction. The
   cost is that the plan-alone numbers here are not the block A and block D numbers; the
   equivalence test (attack point 11) reproduces those with refresh off.
6. **Onset is the start of the first jam spell lasting at least 120 s.** Block D used the first
   jammed 10 s sample. In block D's sustained controlled traces 34 of 66 jam spells last one
   sample, so first-sample onset is set by brief dips. First-sample onset is reported beside the
   primary; the paper's onset numbers will not equal block D's.
7. **Incidents are stopped vehicles and their sizes are the planner's choice.** One lane of the
   2-lane exit road at 100 m, and lanes 0 and 1 of the 3-lane edge 14 (segment 6) at 600 m, each
   for 300 s from 900 s. A permission-based lane closure is the alternative; SUMO's lane-change
   model would then see the closure in advance and merge early, which makes onset less sudden.
   On edge 14 the 120 s-spell onset under no control is the secondary jam at the exit after the
   block clears (1,010 to 2,000 s in the pilot), not the block itself.
8. **The selection criterion's exact form.** Criterion C (section 5.7) averages, over the five
   perturbation scenarios, the paired difference in jammed segment-minutes against the plan alone
   divided by no control's jam time. The alternatives (prevented share, KM median onset, SRC's
   return) are recorded for every candidate. Weak preference.
9. **The lookahead rule's depth and speeds** (one segment ahead gives 0.5, two give 0.75) are the
   planner's; only its jam-factor threshold is tuned.
10. **Measured polling sessions are shorter than a 7,200 s episode, so the replay wraps.** The
    sessions run 6,900 s (`--duration-s 6900` in the running poller's command line) and an
    episode needs 7,380 s of publications. Amendment 1 (A9) forbids wrapping a measured artefact.
    This plan allows the wrap with each wrapped stamp keeping its recorded second of the minute,
    as the placeholder path already does, and records the wrap count per episode.
11. **The replayed session is the Mainz-area session**, because the network is Mainz. The NJ
    session is the drives' area. Weak preference; one config value.
12. **HERE's own aggregation delay is not in the measured lag.** The measured lag runs from a
    version's stamp to its first receipt. In SUMO the snapshot at the stamp is the instantaneous
    segment state, while HERE's speed at its stamp is computed from probe data collected before
    it over a window HERE does not publish. The lag multiplier of 2 is the plan's coverage of
    this.
13. **The camera's absolute distance scale cannot be measured from the drives.** The phone's
    focal length was never recorded (no intrinsics in `run_config.yaml` or the phone sessions;
    `fx_px: 800` is the config's assumed webcam value). The horizon row and camera height can be
    fitted from the detections with the deployment's own `calibration/auto_horizon.py`; the scale
    is swept as a multiplicative bias of 0.8, 1.0 and 1.25 on reported distance.
14. **The existing CPU reconstruction path times frames at 30 fps.** `make_perception_figure.py`
    calls the tracker with `t = i / 30.0`; the recorded frames are 0.20 s or 0.67 s apart. The new
    frame processing takes each frame's time from its tick's `t_capture_mono_ns`. The figure
    script is left as it is (it renders one frame).
15. **Two index defects in the drive data.** `run_20260908_155910/video_index.jsonl` has 3,421
    lines for 3,334 distinct frame ids, and `run_20260908_183538` has one unparseable index line
    and 92 indexed frames with no tick. Duplicates are dropped (first position kept) and frames
    without a tick are skipped, with both counts recorded.

### Decisions table

All taken by recommendation. "Real" means the planner had a reasoned preference; "weak" means
the alternative is defensible and the item is listed above.

| # | Decision | Taken | Alternatives not taken | Preference |
|---|---|---|---|---|
| J1 | Stage 1, the shared plan | `mainz_here_best.pt` (sha256 `941ac210...`) frozen; version-keyed rule of the previous program, `P_q = 10 s`, `D = m L_bar + P_q`; replayed publication process with each version's lag multiplied by `m`; `m = 1` in every block except the lag sweep | Synchronous switch at availability; a fixed delay | Real |
| J2 | Feed-process artefact | Path and session from the config (`--feed-artefact`, `--feed-session`); placeholder only with `--allow-placeholder`; header records path, sha256, placeholder flag, session, `L_bar`, `D`, `m`, wraps; blocks T to K refuse to mix artefacts | Hard-coded path | Real |
| J3 | Plan refresh | Every controlled vehicle on a segment edge is written its applied version's value for its current segment every 5 s and at every version switch, in every arm; refresh off only in the equivalence test | `step()` semantics | Weak (item 5) |
| J4 | Correction cadence | 5 s, the same ticks for every vehicle | 10 s; per-vehicle phases | Real |
| J5 | Correction action | Keep the plan's fraction or the next lower fraction on 0.5/0.75/1.0; at 0.5 the correction is a no-op | A 0.25 step (15 km/h, below the 19 km/h jam threshold) | Real |
| J6 | Never above the plan | One function `shown_fraction(plan, correction) = min(plan, correction)`; every write counted against the plan, violations recorded and required to be 0 | Per-arm write paths | Real |
| J7 | Which leaders a camera sees | Own and adjacent lanes, leaders within `R_max` along the route (continuing onto the route successor edge, lane index clamped), adjacent-lane vehicles only inside the field of view, closed-form occlusion by closer vehicles (section 1.2) | No occlusion; SUMO context subscriptions | Real |
| J8 | Camera errors | Detection probability `p(d)` and distance noise `sigma(d)` from the drive tables, independent per 5 s tick; multiplicative distance bias `b` nominal 1.0, swept 0.8 and 1.25 | Gaussian noise of a fixed size | Real |
| J9 | Density estimate | `k_hat` = mean over visible lanes; own lane `L/(g_hat + L)` from the nearest detected leader's gap; adjacent lanes `n_hat L / R_eff`; occupancy units | Counts on every lane (saturates under occlusion) | Real |
| J10 | Trend | Least-squares slope of the vehicle's own `k_hat` over its last 30 s (7 samples) | Difference of two samples | Real |
| J11 | Leaders' speed | True leader speed plus noise `sigma_v(d)` measured from the tracked closing rate; ego GPS speed is never an input | Ego speed plus raw closing rate | Real |
| J12 | Detection confidence | Per-detection confidence resampled from the drive's distribution in the same distance bin; inputs are the mean confidence and the detection count | Omit | Weak |
| J13 | Network-wide feed lookahead (arm 2) | Standalone rule on the applied feed version: nearest downstream segment with jam factor `>= J_th` one segment ahead gives 0.5, two ahead 0.75, otherwise 1.0; `J_th` in {2, 4, 6} by criterion C | A local lookahead | Weak (item 9) |
| J14 | Take-the-slower (arm 4) | Q-network re-run with the own row's speed and jam factor replaced by the camera's mean leader speed; shown = min(plan, re-run); may fall two steps | Clip to one step | Real |
| J15 | Rule correction (arm 5) | F4 margin `M = (k_c - k_hat - tau_l [slope]_+)/k_c`, `k_c = 0.178`, `tau_l = 10 s` from the delay budget; on at `M <= m_on`, `m_on` in {0, 0.1, 0.2} by criterion C; off at `M >= m_on + 0.05` on 2 ticks after at least 30 s on | `tau_l` tuned | Real on `tau_l`, weak on the grid |
| J16 | Learned correction (arm 6) | Supervised onset predictor (section 5.5): logistic regression or a 2x32 MLP by validation log-loss on training seeds 9 and 10, one DAgger round, threshold in {0.3, 0.45, 0.6, 0.75} by criterion C, hysteresis as J15 | RL on SRC's reward plus a step cost | Weak (item 1) |
| J17 | Oracle (arm 7) | Arm 5's rule and selected `m_on` fed the true occupancy of the visible lanes within 80 m, no occlusion, detection loss or noise; `oracle_wide` at 200 m on B9 and I_exit | Window to the segment end | Weak (item 4) |
| J18 | Scenarios | B5 (existing `mainz_burst`), B7.5, B9 (7,500 and 9,000 veh/h on [900, 1,200) s over 3,572 veh/h), I_exit, I_mid (lane blocks on [900, 1,200) s), BG (3,572 veh/h, no perturbation), S45 (committed sustained 4,500 veh/h) | A 6,000 veh/h burst | Weak (items 2, 7) |
| J19 | Incident model | Stopped vehicle per blocked lane (`vehicle.add` plus `setStop`), excluded from segment counts, speeds and arrivals, visible to cameras | Lane permission closure | Weak (item 7) |
| J20 | Jam metrics | `v_jam = 19.0 km/h` read from `results/feed/drain_thresholds.json` (network pooled fit, not refitted); onset = first jam spell of at least 120 s after the perturbation; prevented = no such spell by the cap; jam time in segment-minutes; drain by `drain.py` with `t_0` at the end of the perturbation; held and pending vehicle-minutes; Kaplan-Meier for onset and drain | First-sample onset | Real; spell length weak (item 6) |
| J21 | Episode caps | 7,200 s for perturbation scenarios in blocks M, L, P, K; 3,600 s in G, T, S and for BG; 2,500 s for S45 | 12,000 s everywhere | Weak |
| J22 | Seeds | Train 1-10 (blocks G, T), select 11-15 (S), evaluate 16-30 (M, L, P, K), paired on seed; smoke on seed 1 | Fewer evaluation seeds | Real |
| J23 | Selection criterion C | Section 5.7; candidates for the best model: plan alone, take-the-slower, rule, learned | Prevented share; return | Weak (item 8) |
| J24 | Go/no-go gate | First experiment: oracle against plan alone, seeds 1-5, B9 and I_exit, 3,600 s, pre-stated pass rule; a 6-setting grid on failure; stop on no-go | Run the full matrix and see | Real |
| J25 | Lag levels | `m` in {0, 1, 2}; `m = 0` is the synchronous rule at zero lag on the same publication stamps | {1, 2, 3} | Real |
| J26 | Penetration | {1.0, 0.75, 0.5}, run-time floor rule of the previous program | Add 0.25 | Real |
| J27 | Bias sweep | `b` in {0.8, 1.25} for the best model, on B9 and I_exit | Every camera arm | Real |
| J28 | Frames | All 18,910 indexed frames, rotated clockwise, official `yolov8n.pt` fetched once into `outputs/models/` with its sha256 recorded, 640 px primary and 1,280 px sensitivity, confidence 0.30, frame times from ticks | A sample of frames | Real |
| J29 | Drive lead time | A descriptive case table on the two drive events; not a paper claim | A lead-time estimate | Real |
| J30 | Lead time in simulation | From block M's no-control episodes with passive sensing on every vehicle: feed lead `W_feed` and camera lead `W_cam` on the first segment to jam (section 4.3) | True density | Real |
| J31 | Runner | One process, sequential, resume by key, pause sentinel `results/jam/PAUSE`, provenance header and footer, seeds enforced per block | Parallel workers | Real |
| J32 | Storage | Summaries and small artefacts in `results/jam/`; traces, frame records, training data and weights in `outputs/jam/` and `outputs/models/` (gitignored) | Everything in `results/` | Real |
| J33 | Random streams | Camera draws `default_rng(seed + 5000)`, in sorted vehicle-id order per tick; predictor `torch.manual_seed(0)` | Unseeded | Real |
| J34 | Paper items | Three result figures and one table, three arms (section 9) | Four figures | Real |

## 1. The local sensing model in SUMO

### 1.1 What a controlled vehicle measures, and why each piece is there

The correction must decide on what a phone camera on the dashboard can measure, so the SUMO
model has to reproduce the camera's view, its losses and its errors, and nothing else. Five
quantities come out of it every 5 s for each controlled vehicle on a segment edge: a local
density estimate, its 30 s trend, the leaders' speed, a detection confidence, and the per-lane
counts and own-lane gap they are built from. The feed's age and the plan's values for the
vehicle's segment and the next one come from stage 1, not from the camera.

### 1.2 Which leaders are visible

At each 5 s tick, one pass reads every lane of every segment edge (`lane.getLastStepVehicleIDs`,
then `vehicle.getLanePosition` and `vehicle.getSpeed` per vehicle, through libsumo) and sorts
each lane by position. For an ego on lane `i` of edge `e` at front position `x`:

- **Candidates.** Vehicles ahead on lanes `i-1`, `i`, `i+1` of `e` (where they exist) with gap
  `d = x_leader - length_leader - x` in `(0, R_max]`, plus, when `x + R_max` passes the end of
  `e`, vehicles on the route successor of `e` (every edge has exactly one successor across the
  eight routes, checked while planning) on lanes `j-1`, `j`, `j+1` with `j = min(i,
  lanes(next) - 1)`. Internal junction lanes are not scanned; a vehicle crossing a junction is
  invisible for that tick. This is a stated approximation.
- **Field of view.** The upright frame is 720 px wide with `fx = 800 px`, so the horizontal
  half-angle is `atan(360/800) = 24.2 deg`. An adjacent-lane vehicle is visible only when its
  centre lies inside the field of view, `|y| / d < 0.449`, with `y` the lateral offset (lane
  index difference times the ego lane's width from the net, 3.75 m on 300 of 427 lanes).
- **Occlusion.** Vehicles are boxes 1.8 m wide and taller than the 1.25 m camera, centred on
  their lanes, on a flat road. A vehicle's azimuth interval is `[(y - 0.9)/d, (y + 0.9)/d]`
  (small-angle form). A candidate is occluded when closer candidates cover at least half of its
  interval. On equal-width lanes this reduces to three rules the implementer tests against the
  general interval computation: in the own lane only the nearest leader is visible; in an
  adjacent lane a vehicle at `d2` is occluded by the next closer vehicle in that lane at `d1`
  when `d2 <= 1.32 d1`; and an adjacent-lane vehicle at `d2` is occluded by the own-lane leader
  at `d0` when `d2 >= 4.11 d0` (with `y = 3.7 m`; the general computation uses the net's width).
- **Detection.** Each visible candidate is detected with probability `p(d)` from the drive
  table (section 2), drawn independently at every tick; `p = 0` beyond `R_max`, the largest
  distance bin with at least 50 drive observations and `p >= 0.2`.
- **Distance.** A detected leader is reported at `d_hat = b d + eps`, `eps ~ N(0, sigma(d)^2)`,
  `sigma(d)` interpolated from the drive table, `b = 1.0` nominal.
- **Speed.** A detected leader's speed is reported as `v_leader + eta`, `eta ~ N(0,
  sigma_v(d)^2)` from the drive table of closing-rate noise. This is "ego speed plus tracked
  closing rate" with the ego speed term cancelling exactly; ego speed itself is never a
  correction input, because the advice lowers it.
- **Confidence.** Each detection carries a confidence resampled from the drive's empirical
  confidences in the same 10 m bin.

### 1.3 The estimates

- Per-lane: `n_hat_l` = detected leaders in lane `l` with `d_hat <= R_eff`, `R_eff =
  min(80 m, R_max)` (80 m is the deployment's `observation.effective_range_m`).
- Own-lane gap `g_hat` = `d_hat` of the nearest detected own-lane leader, or none.
- Density, occupancy units (the same units as `DENSITY_CRITICAL = 0.178`):
  `k_hat = mean over visible lanes of k_l`, with `k_own = L/(g_hat + L)` when an own-lane leader
  is detected within `R_eff` and 0 otherwise, `k_adj = n_hat_l L / R_eff`, `L = 4.5 m`
  (`VEHICLE_LENGTH_M`). The own-lane term uses the gap because occlusion leaves one visible
  own-lane leader in a queue; a count would read one vehicle in 80 m.
- Trend `s_hat`: least-squares slope of the vehicle's last 7 values of `k_hat` (30 s); 0 with
  fewer than 3 values. It mixes time and the vehicle's own travel, which is what a device
  computes.
- Leaders' speed: own-lane leader's reported speed, and the mean reported speed over all
  detected leaders; none when nothing is detected.
- Confidence: mean confidence of the detections in range and their count.
- **True local occupancy** (the oracle's input, and a trace quantity in every arm): `k_true` =
  (all candidates on the visible lanes within `(0, R]`, without the field-of-view limit,
  occlusion or detection loss, at true distances) times `L / (R x visible lanes)`, `R = 80 m` (200 m for
  `oracle_wide`).

### 1.4 Efficiency, measured

The measured pass (appendix A2) read every lane of every segment edge once per 5 s tick and
counted leaders for every controlled vehicle with two `searchsorted` calls per visible lane:
3.0 ms per pass on average (p95 3.8 ms) for a mean of 809 controlled vehicles (maximum 1,034)
and 3,765 vehicle-leader pairs (maximum 7,120), 1.36 s over a 2,500 s episode, against 7.5 s for
the episode without it. Occlusion, detection draws, noise draws and the estimates are numpy
operations on per-lane arrays; the implementer measures the full camera pass on one 600 s smoke
episode and the runner records per-episode wall time. No per-vehicle `getLeader` calls and no
context subscriptions are used: a context subscription is evaluated at every 0.5 s step for every
subscribed vehicle, 10 times as often as the correction needs.

### 1.5 Penetration and no control

Only controlled vehicles sense and correct; the camera sees every vehicle. In the no-control arm
every vehicle senses passively (the sensing set is the controlled set at penetration 1.0) and
nothing is commanded, so block M's no-control episodes give the camera's view of uncontrolled
jams for the lead-time measurement at no extra episode cost.

## 2. The camera error model from the drive frames

### 2.1 What the drives can and cannot give

The device's own detector returned detections on 0 to 6 ticks per drive (rotated frames), so the
error model must come from re-processing the recorded frames. The frames carry no ground-truth
distance (no radar, lidar or second camera), so the model can measure what is visible in the
detections themselves: how far vehicles are detected, how often a tracked vehicle is missed, how
much one vehicle's distance jitters from frame to frame, and how the confidence falls with
distance. It cannot measure the distance bias, because the phone's focal length was never
recorded.

### 2.2 Processing (`scripts/process_drive_frames.py`)

- Frames: every indexed frame of the six indexed drives (`155910`, `161422`, `170849`,
  `173809`, `183538`, `190548`), 18,910 distinct frame ids. Duplicated frame ids keep their first
  position; frames without a tick are skipped; both counts are written. Each frame is rotated with
  `cv2.ROTATE_90_CLOCKWISE` to a 720 x 1,280 upright image. Its time is the matched tick's
  `t_capture_mono_ns`; the AVI's 30 fps tag is never used.
- Detector: official Ultralytics `yolov8n.pt`, fetched once into `outputs/models/yolov8n.pt`
  (gitignored through `outputs/`), sha256 written to `results/jam/weights.json` and checked at
  every run; CPU; classes 2, 3, 5, 7; confidence 0.30; NMS IoU 0.5; `imgsz` 640 (the device's
  engine size) as primary and 1,280 as sensitivity, each a full pass.
- Tracker and distance: the deployment's `IouTracker` and `DistanceEstimator`
  (`deployment/jetson/perception/`) unchanged, `fx = 800`, `cx = 360`, hood line 1,010 px; the
  horizon row and camera height per drive from a Theil-Sen fit of `v_bottom` on `w_px / W_class`
  over that drive's detections (the method of `calibration/auto_horizon.py`, applied to the
  upright frames). Every detection record keeps the raw ground-plane distance, the width-prior
  distance, the EMA distance, the closing rate and its validity, the track id, the box, the
  confidence and the frame time.
- Records go to `outputs/jam/frames/<run>/detections_<imgsz>.jsonl`; the drive data directory is
  only read.

### 2.3 What is measured (`scripts/fit_camera_model.py` writes `results/jam/camera_error_model.json`)

All tables use 10 m bins of EMA distance and are reported separately for the 0.20 s drives and
the 0.67 s drives; the SUMO model uses the 0.20 s table, whose frame interval is closer to a
running camera.

- **Detection range.** The distribution of first-detection distances of tracks whose distance
  decreases over their life; `R_max` as in section 1.2.
- **Detection rate by distance.** For tracks with at least 3 detections, every frame between the
  first and last detection is an opportunity; missed frames get a distance by linear
  interpolation in time; `p(bin) = detected / (detected + missed)`. A vehicle never detected is
  invisible to this estimate, and the tracker's 10-frame maximum age splits longer gaps, so
  `p(d)` is the detection rate of a vehicle that the tracker holds, not of every vehicle in view.
- **Distance noise.** Per track, residuals of the raw ground-plane distance and of the EMA
  distance about a local linear fit over 5 consecutive detections; `sigma(bin) = 1.4826 MAD`. The
  SUMO model uses the EMA residual, which is what the deployment reports.
- **Closing-rate noise.** Per track, the closing rate is fitted as the least-squares slope of the
  EMA distance over non-overlapping windows of 8 detections; `sigma_v(bin)` is 1.4826 times the
  median absolute difference between successive windows' slopes, divided by `sqrt(2)`. Real
  relative acceleration between windows also enters the difference, so `sigma_v` is an upper
  bound on the estimator's noise.
- **Confidence by distance**: the empirical confidences per bin, stored whole for resampling.
- **A bias check that does not depend on `fx`.** The ratio of ground-plane to width-prior
  distance per bin. Both scale with `fx`, so the ratio tests the horizon and camera-height fit
  against the 1.8 m car-width prior (which itself varies by about 10%), not the absolute scale.
- **Not measurable: the absolute scale.** A phone's main camera has a horizontal field of view
  between about 65 and 80 degrees on the 1,280 px axis, which puts `fx` between 763 and 1,005 px,
  so the assumed 800 px gives distances between 0.95 and 1.26 times the truth from `fx` alone;
  camera height adds about 10%. The sweep `b` in {0.8, 1.0, 1.25} covers that range.

### 2.4 Cost

Measured on 20 frames of `run_20260908_170849` with an architecture-only YOLOv8n built from
`yolov8n.yaml` (random weights, so the timing and not the detections are valid): decode and
rotation 1.7 ms, inference 18.8 ms at 640 px (p95 23.5 ms) and 62.0 ms at 1,280 px (p95
66.0 ms); 1 and 4 torch threads gave the same times. All 18,910 frames take about 6.5 minutes at
640 px and 20 minutes at 1,280 px. Tracking and distance cost under 0.1 ms per frame.

## 3. Camera against feed lead time on the drives

The measurement is feasible only as a case description, and the simulation lead time carries the
claim. Three facts decide this. First, only two indexed drives carry HERE responses with their
frames: `run_20260908_170849` (93 responses at 16.7 s intervals, frames every 0.67 s) and
`run_20260908_190548` (28 responses at 60 s intervals, frames every 0.20 s). Second, a scan for
a HERE flow item within 60 m of the vehicle whose jam factor rose by at least 2 to at least 4
between consecutive versions finds two events: Scudders Mill Rd on `170849` (jam factor 3.8 to
8.1 in version 21:18:36Z, first received 533 s after the run's first tick, while the vehicle
slowed from about 60 km/h to a stop) and CR-610/Deans Ln on `190548` (3.2 to 8.1 in version
23:16:36Z, received 719 s after the run's first tick, while the vehicle slowed from about
100 km/h to a stop). Third, both roads are signalised arterials, and the vehicle reached each
queue once; when the camera first sees a queue depends on when the vehicle arrived, and the feed may have reported a queue that existed
before the vehicle got there. The difference between the two first reports is therefore a
property of the vehicle's arrival time, not of the two sensors.

`scripts/drive_lead_cases.py` writes `results/jam/drive_lead_cases.json` with, per event: the
version's stamp and first receipt on the run's clock (through the `metadata.jsonl` wall-clock
anchor), the first frame at which the camera estimate `k_hat` from the re-processed detections
reaches `0.9 k_c` or the own-lane leader's estimated speed falls below half of its value 30 s
earlier, and the time the vehicle's GPS speed first fell below 5 m/s. The paper does not quote a
drive lead time; the case table is for the author.

## 4. The scenarios

### 4.1 Why these scenarios

The claim needs jams whose feed-visible warning is longer than the feed's delay plus its decision
interval (where the plan should cope), jams where it is shorter (where only the camera can act),
the cost of the correction when no jam forms, and the dependence on lag and penetration. The
network's background of 3,572 veh/h lies between the served flow of a jammed network (3,535 veh/h
under no control at 4,500 veh/h demand) and the flow at which it breaks down (about 3,900 veh/h),
so once it jams under no control it does not recover (block D: 0 of 15 drained by 12,000 s).
Preventing a jam is therefore worth more than draining it faster (section 6.3).

### 4.2 The set

| id | demand and event | built by | role | no-control onset after the event, training seeds 1 and 2 (first jammed sample) |
|---|---|---|---|---|
| B5 | 3,572 veh/h, 5,000 veh/h on [900, 1,500) s | existing `mainz_burst` | slow onset; continuity with block D | 760, 640 s |
| B7.5 | 3,572 veh/h, 7,500 veh/h on [900, 1,200) s | `--demand-burst 900 1200 0.416667` | burst sweep | 770, 760 s |
| B9 | 3,572 veh/h, 9,000 veh/h on [900, 1,200) s | `--demand-burst 900 1200 0.5` | burst sweep; feed lead near the delay | 620, 790 s |
| I_exit | 3,572 veh/h; one stopped vehicle on `exit_road` lane 0 at 100 m, [900, 1,200) s | `mainz_bg` plus incident spec | fast onset | 40, 80 s |
| I_mid | 3,572 veh/h; stopped vehicles on edge 14 lanes 0 and 1 at 600 m, [900, 1,200) s | `mainz_bg` plus incident spec | block mid-network, secondary jam at the exit | 200, 250 s |
| BG | 3,572 veh/h, no event | `mainz_bg` | false alarms | none |
| S45 | 4,500 veh/h sustained, 2,500 s | committed `mainz` | throughput, continuity with the record | (already jammed) |

Build commands (the implementer runs them; they write beside `mainz.net.xml` with `--demand-only`):

```
.venv/bin/python scripts/build_mainz_scenario.py --demand-only --out-name mainz_bg \
    --duration-s 7200 --demand-duration-s 7200 --demand-scale 0.198444
.venv/bin/python scripts/build_mainz_scenario.py --demand-only --out-name mainz_b7500 \
    --duration-s 7200 --demand-duration-s 7200 --demand-scale 0.198444 --demand-burst 900 1200 0.416667
.venv/bin/python scripts/build_mainz_scenario.py --demand-only --out-name mainz_b9000 \
    --duration-s 7200 --demand-duration-s 7200 --demand-scale 0.198444 --demand-burst 900 1200 0.5
```

B5 reuses the committed 12,000 s file and stops at the 7,200 s cap. The builder's docstring still
names a 3,650 veh/h background for `mainz_burst`; the committed file is 3,572 veh/h (Amendment 2,
B4), measured at 3,568 veh/h on [0, 900) s and 3,572 veh/h after the burst.

### 4.3 The quantities that place a scenario relative to the lag condition

For each no-control episode with passive sensing, on the first segment to start a 120 s jam spell:

- `W_feed` = onset minus the first time the segment's space-mean speed is below 80% of its mean
  over [600, 900) s for 30 s (jam factor at least 2 under the feed's formula). The plan can first
  respond at the first publication after that time plus `D`.
- `W_cam` = onset minus the first time the mean `k_hat` over sensing vehicles on that segment is
  above its [600, 900) s mean plus 3 standard deviations for 30 s. The orchestrator's check used
  true density; this uses the camera model.

A scenario is "fast" for the feed when `W_feed < D + 60 s` (about 137 s on the placeholder, 146
to 156 s at the polling preview's lags). In the planning pilots on training seeds 1 and 2,
measured against first-sample onset, `W_feed` on the first-jam segment ranged from 70 to 620 s
across the four bursts, and on the exit block it was 30 s on seed 2 while seed 1 had no speed
precursor before onset (appendix A3). Because it varies this much between seeds of one scenario,
`W_feed` is measured per episode rather than assigned per scenario.

### 4.4 Sweeps

- **Lag** (block L): `m` in {0, 2} on B9 and I_exit for the lookahead, the plan alone and the best
  model; `m = 1` comes from block M.
- **Penetration** (block P): {0.75, 0.5} on B9 and I_exit for the plan alone and the best model;
  1.0 from block M. The plan alone gains -5 +/- 54 veh/h over no control at 0.75 and loses at 0.5
  on the sustained scenario, so the sweep tests whether the correction changes that.
- **Camera bias** (block K): `b` in {0.8, 1.25} on B9 and I_exit for the best model, if the best
  model reads the camera.
- **False alarms**: BG and S45 in block M, every arm: served flow and the share of controlled
  vehicle-ticks with the correction active.

### 4.5 Incidents in `MainzEnv`

`MainzEnv.schedule_incident(edge, lanes, pos_m, start_s, duration_s)` adds, at the first step at
or after `start_s`, one vehicle per lane (`incident_<k>`, type `human`, a one-edge route) with
`departLane`, `departPos` and `departSpeed=0`, and `setStop(edge, pos, lane, duration)`; it
removes the vehicle at `start_s + duration_s`. Incident ids are registered in the departure index
as uncontrolled. They are excluded from `_per_edge` counts and speeds (count minus incident
vehicles on the edge, speed recomputed without their zero), from `segment_state`, from arrivals
(the step counts `getArrivedIDList` minus incident ids) and from the feed snapshot. They remain
visible to the camera and to the oracle as stopped leaders, because they are on the road.

## 5. The controllers

### 5.1 Common structure

Every controlled arm runs the entry gate of the previous program, which holds a controlled
vehicle at an entry link while that link is at or above critical density. Because all controlled
arms have the gate, a difference between two controlled arms is not a difference in the gate.
Arms 3 to 7 use stage 1:

- **Stage 1.** The feed process from the active artefact (J1, J2). Each version `k` is
  snapshotted (`observe()`) at the first tick at or after its stamp `T_k`, its action vector
  `a_k = greedy_actions(model, x_k)` computed once. Each controlled vehicle's applied version
  follows the version-keyed rule with `P_q = 10 s` and `D = m L_bar + P_q`, using the previous
  program's `VersionKeyed` and per-vehicle query phases. For `m = 0`, the synchronous rule with
  lag 0 on the same stamps. The plan's fraction for a vehicle is `a_k[e]` for its current segment
  `e` and applied version `k`.
- **Writes.** At every 5 s tick and at every tick a vehicle's applied version changes, every
  controlled vehicle on a segment edge is written `shown_fraction(...) x free-flow speed of its
  edge` through `_apply` (`setMaxSpeed`). `shown_fraction(plan, c) = min(plan, c)` for arms 4 to
  7; the write counter compares every written fraction with the plan's and records violations.
- **Hysteresis (arms 4 to 7).** Once active, a correction stays active for at least 30 s and
  releases only after its release condition holds on 2 consecutive ticks.

### 5.2 Arm 2: network-wide feed lookahead

On the vehicle's applied feed version, jam factor `J_j = 10 (1 - v_j / v_free_j)` clipped to
[0, 10] for every segment. For a vehicle on segment `e`, `j* = min{j > e : J_j >= J_th}`
(downstream is `j > e` on every route, checked while planning). Shown fraction: 0.5 if
`j* = e + 1`, 0.75 if `j* = e + 2`, 1.0 otherwise, including on a slowing segment itself.
`J_th` in {2, 4, 6} chosen by criterion C on the selection seeds. It reads the same versions as
the plan, with the same lag, and it is not combined with the plan.

### 5.3 Arm 4: take-the-slower

The state `x_k` with row `e`'s speed replaced by the vehicle's mean reported leader speed and its
jam factor by `10 (1 - v_loc / v_free_e)` clipped; `a' = greedy_actions(model, x')`; shown
`min(a_k[e], a'[e])` as fractions. No detection leaves `x_k` unchanged. The states of all
controlled vehicles at a tick go through the network as one batch. Release condition: `a'[e]` is
no lower than `a_k[e]`.

### 5.4 Arm 5: rule correction

`M = (k_c - k_hat - tau_l [s_hat]_+) / k_c`, `k_c = 0.178` (the measured critical occupancy),
`tau_l = 10 s`: the 5 s tick, about 1 s of perception, and about 4 s for a driver to respond and
for a vehicle to slow by 15 km/h at 1.5 m/s^2. F4 states that the horizon is the response time,
not a tuning parameter, so it is fixed; `tau_l = 30 s` appears only in the gate's fallback grid.
On at `M <= m_on`; release at `M >= m_on + 0.05`. `m_on` in {0, 0.1, 0.2} by criterion C.

### 5.5 Arm 6: learned correction

- **Inputs per vehicle and tick** (all available on a device): `k_hat`, `s_hat`, `n_hat` for
  left, own and right lanes and the number of visible lanes, `g_hat` (80 m when none), the
  own-lane leader's reported speed, the mean reported leader speed and its 30 s slope, mean
  confidence, detection count, feed age (`t - T_k` of the applied version), the plan's fractions
  for the own and the next downstream segment, the feed's speed and jam factor for both, the own
  segment's length and lane count, and the remaining along-route distance to the end of the own
  segment. Excluded: ego GPS speed and every true density.
- **Label.** `y = 1` when a jam spell of at least 120 s (at `v_jam`) starts on the own segment or
  the next downstream one within `(t, t + 120 s]`; samples whose own segment is jammed at `t` are
  dropped. 120 s is close to `D + 60 s`, the interval in which only the local layer can warn.
- **Data.** Block T: plan-alone episodes with passive camera features, training seeds 1-10, the
  six scenarios B5, B7.5, B9, I_exit, I_mid and BG, 3,600 s; at each tick 5% of controlled
  vehicles are sampled (`default_rng(seed + 6000)`). Labels from the 10 s trace after the
  episode.
- **Model.** Standardised features; logistic regression and an MLP (2 hidden layers of 32,
  ReLU), class-weighted binary cross-entropy, Adam, early stopping; trained on seeds 1-8,
  validated on 9-10; the lower validation log-loss wins. One DAgger round: 60 more block T
  episodes with the round-1 model active at threshold 0.5, labelled the same way, and a refit on
  the union. The model card (`results/jam/onset_predictor.json`, weights `.pt` beside it) lists
  features, scaler, seeds, rounds and sha256.
- **Decision.** Step down when `p >= theta`, release at `p < theta - 0.1`; `theta` in {0.3, 0.45,
  0.6, 0.75} by criterion C on the selection seeds.

### 5.6 Arm 7: oracle

Arm 5's rule with arm 5's selected `m_on` and `tau_l`, fed `k_true` and its slope instead of
`k_hat` and `s_hat`. `oracle_wide` uses `R = 200 m`. Not deployable; an upper reference for what
the rule can do with a perfect camera at the same window.

### 5.7 Selection: criterion C, stated before any selection run

For a candidate `a` and scenario `s` in {B5, B7.5, B9, I_exit, I_mid}, on selection seeds
11-15, with `J` = jammed segment-minutes on [900, 3,600] s:

`C(a) = mean over s of [ mean over seeds of (J_a - J_plan) / mean over seeds of J_nc ]`

Lower is better; `C(plan) = 0`. Within an arm's grid the lowest `C` sets its threshold (the
lookahead's `J_th`, the rule's `m_on`, the learned `theta`). The best model is the lowest `C`
among plan alone, take-the-slower, rule and learned at their selected thresholds; candidates
within 0.02 of each other are ordered by BG served flow. A candidate with any collision, any
collision teleport or any write above the plan is excluded. The selection code reads only
records with seeds 11-15, writes `results/jam/S_summary.json` with every candidate's `C`, each
scenario's terms, prevented counts, KM onset medians, BG flow and active share, and SRC's return,
and blocks M to K read their thresholds and best arm from it and from nowhere else.

### 5.8 What stays unchanged

The safety layer is the car-following model under the `setMaxSpeed` cap: EIDM brakes for its
leader whatever the cap says, which is what the device's `apply_safety_layer` does by bounding the
shown speed by gap and closing rate. Every controlled episode records colliding vehicles and
collision teleports, and every one must show zero.

## 6. Theory the paper can print in a few lines

### 6.1 The lag condition

Setting: snapshots of segment speed at stamps `T_k` with gaps of at most `g` (60 s here); every
vehicle applies the action computed from snapshot `k` from `T_k + D` on. A feed-only controller is
any rule whose shown speeds at `t` depend only on snapshots with `T_k + D <= t` and on static data.
Let `t_v` be the first time the segment speed differs from its no-event value, so that snapshots
stamped before `t_v` are the same with and without the event, and let `t_on` be onset; `W = t_on -
t_v`.

**Proposition 1.** (a) A feed-only controller's shown speeds are the same with and without the
event until `t_r = T_{k*} + D`, where `T_{k*}` is the first stamp at or after `t_v`; so `t_r` lies
in `[t_v + D, t_v + D + g)`. (b) If `W < D`, no feed-only controller changes any vehicle's speed
before onset. If `D <= W < D + g`, it can do so only when the first stamp after `t_v` falls in
`[t_v, t_on - D)`, which, with gaps equal to `g` and the stamp phase uniform, has probability
`(W - D)/g`. (c) A camera sampled every `Delta_c` with processing delay `l_c` responds by
`t_v^c + l_c + Delta_c`, where `t_v^c` is when local density first departs from its no-event
value.

Proof. The controller's output is a function of the applied snapshots; up to `t_r` these coincide
with the no-event snapshots, so the outputs coincide, which is (a). (b) follows from `t_r >= t_v +
D` and, for the probability, from `T_{k*} - t_v` being uniform on `[0, g)`. (c) is (a) with the
camera's own sampling. QED.

Assumptions to print: the event leaves earlier snapshots unchanged; "acting" means a change of
shown speed, which still needs F4's response time to change the flow, so the condition is
necessary and not sufficient. With `D` from 77 to 96 s (placeholder and polling preview), a
feed-only controller cannot respond before onset when `W` is below 77 to 96 s, and responds before
onset with probability `(W - D)/60 s` when `W` is below 137 to 156 s. In the exit block `W` was
30 s.

### 6.2 The margin a sampled, delayed rule needs (F2 and F4)

**Proposition 2.** Let local density obey `|k'| <= r` on `[0, T]` with `k(0) < k* - r (Delta +
l)`. A rule observes `k` at instants `j Delta`, triggers at the first sample `s Delta` at which
`k >= k* - r (Delta + l)`, and from `s Delta + l` on keeps `k' <= 0` (F2's inward-pointing
transfer). Then `k < k*` on `[0, T]`. Proof: let `s Delta` be the first triggering sample. Every
earlier sample is below the trigger level, so between earlier samples `k < k* - r l`. From the
sample `(s - 1) Delta`, where `k < k* - r (Delta + l)`, to the action at `s Delta + l`, at most
`Delta + l` elapses and `k` rises by at most `r (Delta + l)`; after the action it does not rise.
QED.

This is F4's margin with `tau = Delta + l` and F2's limit stated as a number: a rule applied only
at sample instants protects the boundary between samples only with a margin proportional to the
sampling interval plus the delay. For the feed layer `Delta + l = 60 s + D`, 137 to 156 s; for the
camera layer `5 s + about 1 s`, 6 s. The feed reports speed, which stays near free flow until density
reaches `k_b`, the density where the speed-density relation leaves its free-flow branch; a
speed-triggered rule cannot trigger below `k_b`, so Proposition 2's trigger level is available to
it only for ramps with `r <= (k* - k_b) / (60 s + D)`. For the camera rule the condition is
`r <= (k* - k_0) / (Delta_c + l_c)`, `k_0` being the density at which `k_hat` departs from its
noise. Block M measures `k_b` (binned median segment speed below 95% of free flow, from
no-control S45 traces) and `r` (the largest 30 s rise of segment occupancy before onset), so the
paper can print both bounds with numbers.
Assumptions: bounded drift, immediate reversal once acting (F4's actuation time can be folded into
`l`), and `k*` the productive density.

### 6.3 Point-queue drain time and why prevention matters here

**Proposition 3.** A point queue holding `Q` vehicles at `t_0`, with constant arrival rate
`lambda` and service rate `mu`, empties at `t_0 + Q / (mu - lambda)` if `mu > lambda`, and never if
`mu <= lambda`. With a capacity drop, service is `mu_f` before breakdown and `mu_q < mu_f` after;
if `mu_q <= lambda < mu_f`, no queue forms while the bottleneck has not broken down, and none
drains after it has. Proof: `Q' = lambda - mu` while `Q > 0`. QED.

On this network `lambda = 3,572 veh/h`, no control's served flow at sustained overload is
3,535 veh/h (`mu_q`), so no control never drains, which block D measured (0 of 15). The plan under
the cap serves 3,609 veh/h, 37 veh/h above `lambda`, so each queued vehicle adds 97 s of drain
time, and block D's KM median drain of 5,740 s for the plan corresponds to about 59 queued
vehicles. Assumptions: one bottleneck, constant rates, served flow equal to the service rate; the
gate lowers the effective arrival rate by the vehicles it holds, which is why held vehicles are
reported.

## 7. Step table for the implementer

Test-driven: unit tests first (concrete inputs and outputs, no SUMO), then code, then sanity tests
(short SUMO episodes). Each test file runs in its own process,
`.venv/bin/python -m pytest -q tests/<file> -p no:cacheprovider`, never with `-n`; check
`memory_pressure` before a SUMO test. Smoke runs use seed 1 and 600 s. The implementer runs no
experiment block. `scripts/poll_here_feed.py` and `scripts/analyze_feed_process.py` are not edited
in any step: a polling session runs the first at 22:00 tonight and another session analyses with
the second. Nothing under `paper/` is read as a spec, edited or staged. The existing test files
must pass unchanged.

| step | files | functions | unit tests (concrete I/O) | sanity tests (SUMO) |
|---|---|---|---|---|
| S0 files not to edit | `scripts/poll_here_feed.py`, `scripts/analyze_feed_process.py`, everything under `paper/` | none: not edited in any step (a polling session runs the first at 22:00 tonight; another session analyses with the second; the paper is rewritten later by someone else) | none | none |
| S1 scenario files | `data/mainz/mainz_{bg,b7500,b9000}*` (built), `scripts/train_mainz_src.py` (`SCENARIOS`) | none new | the three schedules have 7,144, 7,468 and 7,594 departures (measured with the builder in the scratchpad); rates on [0, 900) s are 3,572 +/- 10 veh/h; the burst windows carry 7,500 and 9,000 veh/h; `mainz.net.xml` sha256 unchanged | a 600 s no-control episode on each starts and inserts vehicles |
| S2 incidents | `src/sumo/mainz.py` | `schedule_incident`, incident-aware `_per_edge`, `segment_state`, arrival counting, `controlled` | an edge with 10 vehicles at 50 km/h and one incident vehicle reports count 10 and 50 km/h; arrivals exclude incident ids | 1,300 s no-control episode with the I_exit spec: the stopped vehicle exists exactly on [900, 1,200) s at 0 m/s; arrivals equal the run without the incident up to 900 s; segment 11 speed falls below 19 km/h within 300 s |
| S3 lane pass and visibility | `src/sumo/sensing.py` | `LaneSnapshot.build(env)`, `successors(routes)`, `visible_candidates(snapshot, ego, R)`, `occluded(d, y, width)` | three-lane synthetic scene: own-lane leaders at 10, 20, 30 m give one visible; adjacent leaders at 10, 12, 30 m give 10 and 30 visible (12 <= 1.32 x 10); adjacent at 50 m with own leader at 10 m is occluded (50 >= 41.1); a leader 20 m past the edge end appears through the successor; FOV removes an adjacent vehicle at 6 m; the closed-form rules equal the interval computation on 1,000 random scenes | 600 s episode: every controlled vehicle's true candidate count equals a brute-force per-vehicle computation on 3 ticks; one pass takes under 10 ms at the measured vehicle count |
| S4 camera model and estimates | `src/sumo/sensing.py` | `CameraModel.from_artefact`, `CameraModel.observe(candidates, rng)`, `OracleSensor`, `LocalEstimate`, `TrendTracker` | with `p = 1`, `sigma = 0`, `b = 1` the camera equals the oracle on unoccluded scenes; with `p(d) = 0.5` the detection share over 10,000 draws is 0.5 +/- 0.02; `b = 1.25` multiplies reported distances by 1.25 and lowers `n_hat` within `R_eff`; `k_own = 4.5 / (g + 4.5)`; slope of 7 points rising 0.01 per 5 s is 0.002 /s; fixed rng seed gives identical draws | 600 s episode under the plan: `k_hat` differs from `k_true` on at least one tick per vehicle; switching off occlusion raises mean `k_hat`; the camera arm's trace and the oracle's differ |
| S5 controllers | `src/sumo/correction.py` | `step_down`, `shown_fraction`, `Hysteresis`, `RuleCorrection`, `LearnedCorrection`, `TakeTheSlower`, `LookaheadRule` | `step_down(1.0) = 0.75`, `step_down(0.5) = 0.5`; `shown_fraction` never exceeds the plan over all 9 pairs; hysteresis holds 30 s and needs 2 release ticks; rule fires at `M = 0.1` with `m_on = 0.1` and releases at 0.15; lookahead with `J_th = 4` and jam factor 5 on segment 9 gives 0.5 on 8, 0.75 on 7, 1.0 on 9 and 10; take-the-slower returns min of the two fractions on a stub network | none |
| S6 jam episode engine | `src/sumo/jam_episode.py`; `src/sumo/feed.py` (add `FeedProcess.with_lag_multiplier(m)`) | `run_jam_episode`, `JamEngine.on_tick`, modes for arms 1-7 | `with_lag_multiplier(2)` doubles every lag and keeps stamps; a fake env receives writes only through `shown_fraction` and the violation counter stays 0; refresh writes every listed vehicle at 5 s ticks | equivalence: refresh off, `m = 0`, publication at the env's decision instants, no correction reproduces the committed deployed-cap `here` seed 16 row exactly (3,635.56 veh/h, 1,936 arrivals); passive sensing leaves flow and arrivals unchanged; a 600 s B9 smoke per arm with 0 collisions and 0 violations; an active corrected vehicle's `getMaxSpeed` equals the stepped-down fraction times its edge's free flow |
| S7 frame processing | `scripts/process_drive_frames.py`, `src/analysis/camera_error.py` | `frame_times(run)`, `iter_frames(run)`, `detect`, `track_and_measure` | duplicate ids are dropped keeping the first position; a frame without a tick is skipped and counted; frame times are tick times (a synthetic index at 0.2 s gives 0.2 s gaps, not 1/30 s); rotation maps a 1280x720 array to 720x1280 | real smoke: 50 frames of `run_20260908_190548` with the fetched weights write detection records with times from ticks; weights sha256 matches `results/jam/weights.json` |
| S8 camera error model | `scripts/fit_camera_model.py`, `src/analysis/camera_error.py` | `detection_rate_table`, `distance_noise_table`, `closing_rate_noise_table`, `confidence_table`, `horizon_fit`, `bias_ratio_table` | synthetic tracks with a known 1-in-4 miss rate give `p = 0.75 +/- 0.02`; synthetic linear tracks with N(0, 1.5 m) noise give `sigma = 1.5 +/- 0.1`; the Theil-Sen fit recovers a planted horizon of 825 px and height 1.25 m | real smoke: the 50-frame records produce a parseable artefact with every table present (values not interpreted) |
| S9 drive case table | `scripts/drive_lead_cases.py` | `find_events`, `camera_first_report`, `feed_first_report` | a synthetic response sequence with jam factor 3.8 then 8.1 within 60 m yields one event; one 70 m away yields none | real smoke: on the HERE bodies of both drives the two events of section 3 are found (frames not required) |
| S10 jam metrics | `src/analysis/jam_metrics.py` | `jam_spells`, `onset`, `prevented`, `jam_time`, `held_minutes`, `drain` (calls `drain.drain_time`), `km_onset`, `leads`, `speed_density_break`, `max_ramp`, `criterion_c` | spells of 6, 12 and 13 samples give onset at the 12-sample spell's start with `H = 120 s`; no spell gives `onset = None` and censoring at the cap; KM on (100, 200, cap-censored) gives 2/3, 1/3; criterion C on a hand table gives the hand value; the module source contains neither `DENSITY_CRITICAL` nor `densities(` | none |
| S11 onset predictor | `src/rl/onset_predictor.py`, `scripts/train_onset_predictor.py` | `feature_vector`, `label_samples`, `fit`, `save`, `load`, `predict` | labels on a hand trace (spell starting 60 s ahead on the next segment gives 1; 130 s ahead gives 0; own segment jammed drops the sample); feature vector has no ego-speed or true-density field (name check against a deny list); a separable synthetic set reaches validation log-loss below 0.1; save then load predicts identically | real smoke: 2 block T episodes of 600 s, fit runs, a model card is written |
| S12 runner | `scripts/run_jam_experiments.py` | `block_specs`, `run_block`, `header`, `summarize_*`, `read_selection` | seeds outside a block's set are refused (S refuses 16, T refuses 11); resume skips present keys; the pause file stops before the next episode and writes `results/jam/<block>_state.json`; headers carry checkpoint, feed artefact, camera model, predictor and weights sha256; M refuses an `S_summary.json` built on another feed artefact; arms share identical seed lists | `--smoke` runs one 600 s episode per arm of each block on seed 1 into `outputs/jam/smoke/` |
| S13 figures | `scripts/figures/jam_figures.py` | `fig_warning`, `fig_onset_prevention`, `fig_lag`, `table_main` | each returns the plotted series from loaded JSON and the test checks them against the JSON; a missing artefact raises | none |
| S14 smoke record | `plans/implementation_records.md` (append) | none | none | the S12 smoke once per block, the S7 frame smoke and the S11 predictor smoke, with wall times |

## 8. The experiment matrix

### 8.1 Order, with the gate first

1. **Block G, the go/no-go gate** (before frames, training or selection). Oracle (`tau_l = 10 s`,
   `m_on = 0.1`) against the plan alone, training seeds 1-5, B9 and I_exit, 3,600 s, whatever feed
   artefact exists (recorded). **Pass rule:** for at least one of the two scenarios, the paired mean
   jammed segment-minutes of the oracle is below the plan's by more than two standard errors of the
   paired differences (n = 5), or on B9 the oracle has no 120 s jam spell on at least 2 more seeds
   than the plan; and in neither scenario is the oracle's paired mean above the plan's by more than
   two standard errors. **On failure:** run the grid `tau_l` in {10, 30} s by `m_on` in {0, 0.1,
   0.2} on the same seeds; pass if any setting passes, and record which. **On no-go:** no camera
   arm runs; the orchestrator reports to the author that a noise-free local correction does not
   reduce jams below the plan in this simulator, with the block G table, and the program stops.
   The two fallbacks for the author are to keep the paper on feed agreement and drain time, or to
   amend this plan to restrict the correction to segments upstream of the first-jam segments. The
   simplified oracle of appendix A4 meets this pass rule on both scenarios on its 3 seeds (on B9 it
   has no 120 s spell on 3 more seeds than the plan; on I_exit its paired mean jam time is
   4.1 segment-minutes below the plan's against a two-standard-error bar of 2.9).
2. **Block F, frames**: section 2, about 27 minutes.
3. **Block T, training data and the predictor**: section 5.5.
4. **Block S, selection** on seeds 11-15 (section 5.7).
5. **Block M, main evaluation** on seeds 16-30.
6. **Block L, lag sweep**; **Block P, penetration**; **Block K, camera bias**.

Blocks T to K run on one feed-process artefact, recommended the measured one; if it does not
exist when block T starts, the orchestrator asks the author whether to wait or to run on the
placeholder, and every summary states which was used.

### 8.2 Matrix and compute

Per-episode wall times measured while planning (appendix A2, A4): 7.5 s for a 2,500 s plan-alone
episode under the lagged feed and 8.8 s with the sensing pass; 11.5 to 16.3 s for 3,600 s
controlled episodes with the simplified oracle; 7.4 to 10.4 s for 3,600 s no-control episodes. Budgets
below assume 15 s per 3,600 s controlled episode, 30 s per 7,200 s controlled episode, 20 s per
7,200 s no-control episode with passive sensing, and 10 s per 2,500 s episode.

| block | arms and variants | scenarios | seeds | episodes | budget |
|---|---|---|---|---|---|
| G | oracle, plan alone (grid of 6 oracle settings only on failure) | B9, I_exit, 3,600 s | 1-5 | 20 (+60) | 5 min (+15) |
| F | frame re-processing, 640 and 1,280 px | six drives | none | 18,910 frames x 2 | 27 min |
| T | plan alone with passive camera features; DAgger round with round-1 model | B5, B7.5, B9, I_exit, I_mid, BG, 3,600 s | 1-10 | 60 + 60 | 35 min with fitting |
| S | no control; plan; lookahead x 3; take-the-slower; rule x 3; learned x 4 | B5, B7.5, B9, I_exit, I_mid, BG, 3,600 s | 11-15 | 13 x 6 x 5 = 390 | 1.6 h |
| M | no control, lookahead*, plan, take-the-slower, rule*, learned*, oracle | B5, B7.5, B9, I_exit, I_mid at 7,200 s; BG at 3,600 s; S45 at 2,500 s | 16-30 | 525 + 105 + 105 | 5.0 h |
| M+ | `oracle_wide` | B9, I_exit, 7,200 s | 16-30 | 30 | 15 min |
| L | lookahead*, plan, best; `m` in {0, 2} | B9, I_exit, 7,200 s | 16-30 | 180 | 1.5 h |
| P | plan, best; `p` in {0.75, 0.5} | B9, I_exit, 7,200 s | 16-30 | 120 | 1.0 h |
| K | best; `b` in {0.8, 1.25} | B9, I_exit, 7,200 s | 16-30 | 60 | 0.5 h |
| total | | | | about 1,650 episodes | about 11 h |

`*` marks a threshold read from `S_summary.json`. If measured per-episode times exceed the budget
by more than 20% after the first 30 episodes of block M, the orchestrator cuts in this order:
block K, then penetration 0.75 in block P, then S45 in block M, then `m = 0` in block L; each cut
is recorded in the summary.

Training compute for the learned correction is block T's 35 minutes plus its share of block S
(4 thresholds x 6 scenarios x 5 seeds = 120 episodes, about 30 minutes).

### 8.3 Checkpoint, resume, pause and environment

- One JSONL per block under `results/jam/<block>.jsonl`, one line per episode, appended and flushed
  after every episode; traces under `outputs/jam/traces/<block>/`. Key: `(block, arm, variant,
  scenario, seed)`.
- `--resume` skips every key present. Between episodes the runner checks `results/jam/PAUSE`; when
  it exists it writes `results/jam/<block>_state.json` naming the next key and exits 0 with
  `paused before <key>`.
- Header: git SHA and dirty flag (`_provenance`), `sumo --version`, libsumo, Python, platform,
  command line, config, checkpoint path and sha256, feed artefact path, sha256, placeholder flag,
  session, `L_bar`, `D`, `m`, camera model artefact sha256, predictor sha256, weights sha256,
  `S_summary.json` sha256 (blocks M to K), scenario files' sha256, ultralytics and torch versions
  (block F). A footer repeats the provenance and records whether the tree changed during the run.
- Every episode record: flow, arrivals, held, pending, running, colliding vehicles, collision
  teleports, write count, writes above the plan, active share, onset (spell and first sample),
  prevented, jam time, held and pending vehicle-minutes, drain time and flag, wall time, the feed
  wraps and the trace path. Traces every 10 s: per segment speed, density, count, mean `k_hat`,
  mean `k_true`, mean plan fraction, mean shown fraction, share active; held, pending, arrivals,
  collision counters.

## 9. Artefacts and figures

Artefacts: `results/jam/{G,T,S,M,L,P,K}.jsonl` and `*_summary.json`; `camera_error_model.json`;
`weights.json`; `onset_predictor.json` and `.pt`; `drive_lead_cases.json`; `leads.json` (`W_feed`,
`W_cam` per no-control episode, `k_b`, `r`); figures as PDF under `results/jam/figures/`.

The paper shows three arms: no control, the network-wide feed lookahead, and the best model from
`S_summary.json`.

| item | content | result it carries |
|---|---|---|
| Fig. 1 | (a) camera error model from the drives: detection rate and distance noise against distance, 640 px, with the 1,280 px curve thin; (b) per scenario, `W_feed` and `W_cam` on the first segment to jam (no-control episodes, seeds 16-30), with the band `D + [0, 60] s` | the camera warns earlier than the feed, by how much, and which scenarios fall below the lag condition (Proposition 1) |
| Fig. 2 | per scenario, ordered by no-control median onset in block M: share of seeds with no 120 s jam spell, and jammed segment-minutes with paired two-standard-error bars, for the three arms | onset, prevention and jam time across slow and fast onset |
| Fig. 3 | jammed segment-minutes against lag multiplier 0, 1, 2 on B9 and I_exit for the three arms | the feed-only arm degrades with lag and the corrected arm degrades less, or the measured alternative |
| Table 1 | three arms by scenario: KM median onset, prevented count of 15, jam time, fraction drained by the cap and KM median drain time, served flow, held vehicle-minutes; BG rows: active share and served flow against the plan alone | the full result and the false-alarm cost |

The remaining arms, the penetration and bias sweeps, `oracle_wide` and the drive case table go to
the result files for the record and the author.

## 10. Validator attack points

1. **Never above the plan.** Every episode's `writes_above_plan` is 0; recompute on one B9
   episode of each of arms 4 to 7 from a per-write log (vehicle, tick, edge, applied version,
   written fraction) that the written fraction is at most `a_k[e]`. Replacing `min` by `max` in
   `shown_fraction` must fail a unit test and produce violations.
2. **The step down reaches SUMO.** For active vehicles, `vehicle.getMaxSpeed` equals the shown
   fraction times the edge's free-flow speed after the write; corrected arms' traces show mean
   shown fraction below mean plan fraction on B9.
3. **The camera model is applied.** Its parameters equal the tables the validator recomputes from
   `outputs/jam/frames/*/detections_640.jsonl`, not from the summary; in a camera arm `k_hat` and
   `k_true` differ; turning off occlusion, detection loss and noise one at a time changes `k_hat`;
   `b = 1.25` lowers mean `k_hat` in block K.
4. **Only the oracle reads true density.** The sensor type per arm is in every record; `k_true`
   enters a decision only in arms 7 and `oracle_wide`; a static check that `RuleCorrection` and
   `LearnedCorrection` receive `LocalEstimate` from `CameraModel` in arms 5 and 6.
5. **The lag is injected.** Recorded switch instants equal `T_k + D` with `D = m L_bar + P_q` for
   the plan and the lookahead; `m = 2` doubles each version's lag; the lookahead's applied versions
   equal the plan's; at `m = 0` snapshot age at the switch is 0.
6. **Zero collisions and zero collision teleports** in every controlled episode of every block.
7. **Selection never touches seeds 16-30.** `S_summary.json` lists only seeds 11-15; block T data
   only seeds 1-10; M to K headers carry the `S_summary.json` sha256 and read every threshold from
   it; a test feeds a seed-16 record to the selector and expects a refusal.
8. **Censored runs are kept.** KM denominators are 15; an episode without onset is censored at the
   cap; undrained runs keep `drained = False`.
9. **Held-at-entry vehicles are counted.** Drain requires held plus pending zero; held and pending
   vehicle-minutes are in every record and in Table 1.
10. **The jam threshold is the speed threshold.** `jam_metrics.py` has no `DENSITY_CRITICAL` and no
    `densities(`; `v_jam = 19.0` is read from `results/feed/drain_thresholds.json` with its sha256
    recorded; recomputing jam time at the fixed 30 km/h sensitivity changes at least one value.
11. **Equivalence with the record.** Refresh off, zero lag and the env's decision instants
    reproduce the committed deployed-cap `here` seed 16 row exactly; passive sensing leaves flow
    and arrivals unchanged.
12. **Incidents.** The incident vehicles stand exactly on [900, 1,200) s, are absent from segment
    counts, speeds, the feed snapshot and arrivals, and present as camera candidates.
13. **Frame times and orientation.** Frame times in the detection records come from ticks
    (median gap 0.20 s or 0.67 s per drive), never 1/30 s; images are rotated clockwise; duplicate
    and tick-less frames are counted.
14. **Predictor inputs.** The model card's feature list has no ego GPS speed and no true density;
    labels come from the simulator trace; training seeds 1-10 only.
15. **Pairing and clocks.** Identical seed lists across arms; the perturbation start (900 s) and
    cap are one number per scenario.
16. **One artefact per program run.** Blocks T to K share the feed artefact sha256, the camera
    model sha256 and the predictor sha256; the summaries say whether the feed artefact was the
    placeholder.
17. **Timing.** Per-episode walls within a factor of 2 of section 8.2's budgets.
18. **Untouched files.** `git diff` over the program's commits is empty for `paper/`,
    `scripts/poll_here_feed.py` and `scripts/analyze_feed_process.py`; `yolov8n.pt` is not tracked.

## 11. Risks

| risk | what it does to the paper's claim | what this plan does |
|---|---|---|
| The oracle does not reduce jams below the plan alone in the fast scenarios | The pitch fails: even a perfect local view does not help, so no camera arm can | Block G runs first on training seeds; its grid; stop and report on no-go |
| The best model on the selection seeds is the plan alone | The paper's best model has no correction; the three printed arms are no control, lookahead and plan | Reported as the result; the oracle and the camera arms stay in the record |
| The camera model is too generous in dense traffic (drives were light traffic on arterials) | Camera arms' gains are overstated | Occlusion is modelled geometrically; `p(d)` is conditional on a held track and stated as such; bias sweep; the oracle bounds the camera arms from above |
| The distance scale is wrong by more than the sweep | `k_hat` is biased by the inverse of the scale | `b` from 0.8 to 1.25 covers the phone field-of-view range; the `fx`-free ratio check is reported |
| False alarms cost throughput | A corrected controller trades flow for jam avoidance where no jam would form | BG and S45 in block M for every arm; active share and flow loss in Table 1; ties in criterion C go to BG flow |
| Prevention cannot be measured on I_exit because the block jams the exit whatever any controller does | Prevented share is 0 for every arm there | I_exit is reported on jam time, onset and recovery; prevention is claimed only where some arm prevents |
| The learned predictor has too few positive labels or does not generalise from plan-alone data | Arm 6 is no better than the rule | One DAgger round; class weighting; the rule competes in selection |
| The measured feed artefact is late | Blocks T to K wait or run on the placeholder | The orchestrator asks the author; every summary records the artefact |
| HERE's aggregation adds lag that is not in the measured lag | The plan's real delay is longer than simulated | `m = 2` in block L; stated in the paper |
| Compute overruns | Sweeps are cut | Stated cut order; resume and pause |
| Refresh changes the plan's behaviour relative to its training | Plan-alone numbers differ from blocks A and D | The equivalence test isolates the engine; `m = 0` gives the zero-lag reference |
| The gate holds vehicles, so a controlled arm avoids jams by holding demand outside | Prevention is partly the gate's | Held and pending vehicle-minutes in every record and in Table 1 |
| 15 seeds cannot resolve differences in prevented share | Prevented-share differences fall inside their uncertainty | Jam time is the continuous primary; prevented counts are reported as counts |

## Appendix: measurements made while planning (2026-10-16)

All read-only on the repository. Scripts and outputs in `scratchpad/dsrc_jam/plan_measure/`.
Environment: `.venv` with libsumo 1.27.1, torch 2.14.0, ultralytics 8.4.150, OpenCV 5.0.0.

**A1. Geometry.** Super-segment along-route lengths (open item 3); every edge has one route
successor; every route visits segments in increasing index (routes from entries at segments 0, 2,
5 and 8). Trunk lanes: edge 14 (segment 6) 3 lanes and 1,138 m, edge 218 (segment 11) 3 lanes,
`exit_road` 2 lanes. Lane widths 3.75 m on 300 of 427 lanes and 3.5 m on 86. The deployed
policy on the sustained scenario (block A `published`, 555 decisions) sets segment 5 to 0.5 on
67% of decisions, segment 7 on 55% and segment 8 on 58%, where the correction is a no-op; segment 9 is at 0.75 on
every decision and segments 10 and 11 at 1.0 on 80% and 100%.

**A2. Smoke episode and sensing cost.** Deployed policy, version-keyed rule, `P_q = 10 s`, `D =
77.08 s` (placeholder session `drive_run_20260908_170849`, `L_bar = 67.08 s`), seed 16, 2,500 s,
0.5 s step, one process: 7.5 s wall, flow 3,648.89 veh/h, 0 collisions. The same with the 5 s
sensing pass: 8.8 s wall, same flow and arrivals; 445 passes, 3.05 ms mean, 3.82 ms p95, 1.36 s
total; 809 controlled vehicles per pass on average (maximum 1,034), 3,765 vehicle-leader pairs
within 80 m (maximum 7,120); true local occupancy mean 0.102, maximum 0.506. Seed 16 is an
evaluation seed; this run times the engine and chooses nothing.

**A3. No-control onset.** Demands built in the scratchpad with the repository's builder
(`main(argv, data_dir=scratch)`, `--demand-only`), no control, SUMO's own insertion, first jammed
sample at 19.0 km/h, first segment to jam, and the leads of the speed precursor (below 80% of the
[600, 900) s mean for 30 s) and the true-density precursor (3 SD above that mean for 30 s):

| scenario | seed | onset after 900 s | first segment | speed lead | density lead | jam segment-min, [900, 3,600] s |
|---|---|---|---|---|---|---|
| B5 | 1, 2 | 760, 640 | 11, 10 | 80, 80 | 570, 100 | 25.2, 18.2 |
| 6,000 veh/h on [900, 1,500) | 1, 2 | 760, 680 | 11, 11 | 490, 70 | 570, 480 | 49.5, 55.8 |
| B7.5 | 1, 2 | 770, 760 | 10, 10 | 620, 320 | 610, 430 | 35.5, 51.7 |
| B9 | 1, 2 | 620, 790 | 10, 11 | 470, 150 | 460, 600 | 66.7, 60.3 |
| BG | 1, 2 | none | | | | 0, 0 |
| I_exit | 1, 2 | 40, 80 | 11, 11 | none, 30 | none, 40 | 50.5, 49.3 |
| I_mid | 1, 2 | 200, 250 | 6, 6 | 150, 200 | 120, 190 | 15.8, 19.7 |

Before these, the same runs on seed 16 gave B5 800 s (speed lead 250 s, density lead 640 s, which
equals the block D trace), 6,000 veh/h 780 s, B7.5 880 s, B9 700 s, I_exit 70 s, I_mid 210 s,
blocking 2 of 3 lanes of edge 218 70 s, and blocking 1 lane of edge 14 no onset. The scenario
parameters were set after the seed-16 runs; no threshold, model or arm was chosen on them. A run
that inserted no-control vehicles through the env's gate path at penetration 0 instead of SUMO's
route-file insertion gave B5 onset 1,200 s on seed 16 against 800 s, so no-control episodes must
use SUMO's insertion, as `run_feed_episode` does.

**A4. Simplified oracle pilot (training seeds 1-3).** Plan alone and plan plus a simplified oracle (true
occupancy within 80 m on own and adjacent lanes, no occlusion, 30 s trend, `tau = 30 s`, on at `M
<= 0.1`, off at `M >= 0.15`, no minimum dwell), both with 5 s refresh, placeholder lag, 3,600 s;
onset is the first 120 s jam spell after 900 s; jam time on [900, 3,600] s:

| scenario | arm | onset (s1, s2, s3) | jam segment-min | jammed at 3,600 s | served flow veh/h | active share |
|---|---|---|---|---|---|---|
| B9 | no control | 740, 790, 800 | 66.7, 60.3, 63.0 | segments 8 and 9 | 3,105 to 3,230 | |
| B9 | plan | 960, 1,320, 1,520 | 19.2, 21.7, 28.3 | segment 11 on s1 | 3,134 to 3,184 | |
| B9 | oracle | none, none, none | 0.0, 0.0, 0.0 | none | 3,370 to 3,406 | 0.28 |
| I_exit | no control | 60, 80, 70 | 50.5, 49.3, 44.3 | segment 9 on all three, 11 on s1 | 2,878 to 2,939 | |
| I_exit | plan | 100, 80, 90 | 12.0, 13.7, 17.2 | none | 2,930 to 3,088 | |
| I_exit | oracle | 130, 90, 120 | 9.8, 10.5, 10.2 | none | 3,039 to 3,075 | 0.14 |
| I_mid | no control | 1,980, 2,000, 1,010 | 15.8, 19.7, 21.7 | 11, 9, none | 2,961 to 3,012 | |
| I_mid | plan | none, none, none | 5.2, 7.7, 4.3 | none | 2,935 to 3,015 | |
| I_mid | oracle | 200, 200, none | 2.3, 2.0, 1.3 | none | 3,033 to 3,046 | 0.16 |
| BG | plan (s1, s2) | none | 0.0 | none | 3,110, 3,109 | |
| BG | oracle (s1, s2) | none | 0.0 | none | 3,067, 3,064 | 0.10 |

Served flow here is measured from 300 s, the window `reset()` opens, so it includes the network's
fill and is lower than the demand; it compares arms at the same window only. Every episode had 0
collisions, 0 collision teleports and 0 writes above the plan. On I_mid the oracle's onset at
200 s on two seeds is a 120 s spell on the blocked segment, which the plan's shorter spells did not
reach, while its total jam time is lower; this is why jam time and not onset is the selection
quantity.

**A5. Drives.** Indexed frames 3,334, 2,553, 2,426, 2,103, 478 and 8,016 distinct ids for
`155910`, `161422`, `170849`, `173809`, `183538`, `190548` (18,910); median frame gap 0.20 s, 0.67
s, 0.67 s, 0.20 s, 0.20 s, 0.20 s from `t_capture_mono_ns`; the AVI is tagged 30 fps; the device's
`n_detections` was positive on 4, 0, 2, 6, 0 and 4 ticks. HERE responses exist on `170849` (93),
`183538` (2) and `190548` (28); the request URLs in `here_index.jsonl` carry no key. Event scan as
in section 3.

**A6. Frame inference timing.** Section 2.4.

## Orchestrator decisions at commit (2026-10-16)

Recorded by the orchestrator before implementation, so that no stage stops to ask.

- **The feed-process artefact for blocks G to K** is the measured artefact built by
  `scripts/analyze_feed_process.py` from the afternoon polling session alone
  (`outputs/feed/polling/afternoon_20261016`), written to `results/jam/feed_process_afternoon.json`,
  with `--feed-session afternoon_20261016_mainz` (open item 11 taken as recommended). The night
  session is not part of the replay, so the experiments do not wait for it and `L_bar` and `D`
  cannot change between blocks; the night session supplies only the time-of-day comparison of the
  lag. Block G runs on the same artefact. Section 8.1's "ask the author" step therefore does not
  arise.
- **Open items 1 to 15 are accepted as the plan recommends**, including the supervised onset
  predictor in place of RL (open item 1) and wrapping the measured replay with each wrapped stamp's
  second of the minute kept (open item 10, which supersedes the previous program's Amendment 1, A9,
  for this program only). Open item 1 is reported to the author with the results.
- **Models:** the implementer and the validator both run on Opus, as the author asked for the
  previous program.

## Amendment 1 (2026-10-16, during implementation; orchestrator)

**Block L adds the lag multiplier `m = 4`.** Wang et al. (TRR 2018, arXiv 1711.02688) measured
the latency of three commercial GPS-probe speed feeds against Bluetooth re-identification on
freeways in three states: the time from a speed change on the road to its appearance in the vendor
data had a mean of 4.4 minutes and a standard deviation of 2.3 minutes. That latency is upstream of
the publication-to-receipt lag the polling measures, so the plan's `m = 2` (about 180 s of total
delay at the polling preview's `L_bar`) does not reach it. `m = 4` gives `D = 4 L_bar + P_q`, about
350 s, which is the content age a vehicle acts on if the feed's aggregation latency is near the
reported mean. Block L becomes `m` in {0, 2, 4} on B9 and I_exit for the lookahead, the plan alone
and the best model: 90 more episodes, about 45 minutes. Figure 3's lag axis is 0, 1, 2, 4. The
cut order of section 8.2 is unchanged, and `m = 4` is cut after `m = 0`.

## Pause state (2026-10-16, about 15:20 EDT; orchestrator, at the author's `pause`)

### The blocking unknown

**Whether block G passes:** does the oracle correction (true local occupancy, rule of section 5.4)
reduce jams below the plan alone on training seeds 1-5, on B9 and I_exit, under the pass rule of
section 8.1? Every later block, the paper's pitch and the rewrite depend on it. The planning pilot
(appendix A4, simplified oracle, seeds 1-3, placeholder lag) met the pass rule on both scenarios,
but block G itself has not run. It needs steps S4 to S6, S10 and S12 built, then about 5 minutes of
compute. **Open decision for the author:** run block G as soon as those steps exist, before the
validator rounds (a no-go then stops the program before the validation cost is spent), or keep
the plan's order (all validation first). The orchestrator recommends running it early, on training
seeds only, and re-running it after validation if any validator fix touches the code it uses.

### Where everything stopped

- Branch `jam-avoidance` in `/Users/ankit_nash/Desktop/ankit_summer_2026/dsrc` (no worktree),
  branched from `feed-agreement` at `d699ac2`. Nothing is pushed.
- Committed: the plan (`70b9237`), Amendment 1 (`394e3cd`), and the implementer's steps S1
  (`2abab94`, demand files), S2 (`3ae9f72`, incidents) and S3 (`68e425b`, lane pass and
  visibility). The implementer (Opus) stopped at the pause after S3 with nothing uncommitted;
  its stopping point is the next subsection.
- Uncommitted and not this program's: everything under `paper/` (the author's edits plus the
  earlier 6-page feed-agreement rewrite; the old version is in `paper/icra_version/`). The author
  commented out `\appendix` and `\input{appendix}` in `paper/main_extended.tex` at 08:28; left as is.
- Validator: never started. Findings file `plans/2026-10-16-jam-avoidance-validation.md` does not
  exist yet. Validation rounds used: 0 of 3.
- Experiments: none run. `results/jam/` does not exist yet.
- Agent briefs and the paper outline: `plans/2026-10-16-jam-avoidance-briefs.md` (copied out of the
  session scratchpad; the scratchpad is not needed to resume).

### The implementer's stopping point (its report, 2026-10-16 15:20)

- **Done and committed:** S1 (three demand files; departures 7,144, 7,468 and 7,594;
  `mainz.net.xml` sha256 unchanged), S2 (`MainzEnv.schedule_incident`, `incident_report`;
  incident vehicles excluded from edge counts and speeds, `segment_state`, the feed snapshot,
  `_segment_vehicles`, arrivals, `pending()` and `running()`, and kept on the lanes for cameras),
  S3 (`src/sumo/sensing.py`: `LaneSnapshot`, `successors`, `CandidateTable`, `occluded`,
  `closed_form_occluded`, `visible_candidates`).
- **Tests:** `tests/test_jam_scenarios.py` 18 passed, `tests/test_jam_incidents.py` 12 passed,
  `tests/test_jam_visibility.py` 17 passed; `test_feed_env` 24, `test_feed_episode` 22 and
  `test_mainz_port` 12 still pass. `tests/test_feed_builder.py::test_burst_rates_are_3650_then_5000_then_3650`
  fails, and also fails on an archive of `394e3cd`: it asserts the 3,650 veh/h background that the
  previous program's Amendment 2 (B4) replaced with 3,572 veh/h. Not this program's defect; it needs
  its expected rate updated.
- **Not started:** S4 to S14, and Amendment 1 (block L `m` in {0, 2, 4}, `fig_lag` axis 0, 1, 2,
  4), which lands in S12 and S13. Resume at S4: `CameraModel`, `OracleSensor`, `LocalEstimate`,
  `TrendTracker` in `src/sumo/sensing.py`, tests in a new `tests/test_jam_camera.py`.
- **Where the plan could not be built as written (for the validator to check):**
  1. SUMO inserts a stopped incident vehicle only when its spot is clear: insertion came 0 to
     15.5 s after the 900 s request on three seeds of both blocks, with zero collisions. The
     incident therefore stands on [insertion, 1,200) s, not exactly [900, 1,200) s;
     `incident_report()` records the insertion time and the sanity test asserts insertion within
     30 s. Attack point 12 and the S2 sanity test are to be read with this.
  2. The closed-form occlusion rules of section 1.2 are sufficient but not necessary (the own-lane
     leader and a closer same-lane vehicle can together cover half of an interval when neither does
     alone), so "the closed-form rules equal the interval computation on 1,000 random scenes" is
     false. Production uses the general interval rule; the tests check that the vectorised table
     equals the general rule on 1,000 scenes and that every disagreement with the closed form is the
     combined case.
  3. The field-of-view bound is 360/800 = 0.45 (the plan's 0.449 is tan(24.2 deg) rounded).
- **Choices later steps depend on:** the lane pass also scans each segment edge's route successor,
  because the I_exit incident stands on `exit_road`, which belongs to no segment (the planning
  pilot's oracle scanned segment edges only); "visible lanes" are the offsets -1, 0, +1 that exist
  on the ego's own edge; an incident is removed from an edge's mean speed by
  `(mean x count - incident speeds) / (count - incidents)`, because SUMO's edge mean speed is not
  the plain mean of vehicle speeds (they differed by up to 8.3 m/s).

### Polling (the feed's update process)

- The afternoon session (`outputs/feed/polling/afternoon_20261016`, 5 s per area, NJ 40.3525 N
  -74.6239 W and Mainz 49.9312 N 8.3091 E, 500 m circles) was left to reach its own 6,900 s end
  at about 15:20, because `scripts/poll_here_feed.py` has no signal handling and an interrupt
  would skip its footer line. Its footer records the requests sent (cap 2,762).
- A preview analysis on the first 1,187 lines (orchestrator, 14:40, not an artefact of record):
  NJ 78 versions, `L_bar` 84.0 s, `L_low` 33.9 s, `g_min` 28 s; Mainz 50 versions, `L_bar`
  89.5 s, `L_low` 39.4 s, `g_min` 60 s. The analysis script ran without error on this session's
  format.
- **The night session (22:00, 6,900 s) was cancelled by the pause.** The author approved at most
  5,760 requests in total on the HERE key that is shared with Nash production; the night session
  may use at most 5,760 minus the afternoon footer's `sent`. To run it:
  `.venv/bin/python scripts/poll_here_feed.py --session-id night_<date> --duration-s 6900 --max-requests <remaining> --budget-confirmed`
  from the repository root, with `caffeinate -i -w <pid>` alongside. The orchestrator decision at
  commit makes the experiments independent of the night session.
- Nothing is committed from `outputs/feed/polling/` (gitignored outputs).

### Still to do from the previous program (independent of this plan)

The polling analysis and the block A re-run on the measured process. Not started; the job that
would have run them was cancelled while it was still waiting for the poller. Commands, in order,
from the repository root:

1. `.venv/bin/python scripts/analyze_feed_process.py --polling outputs/feed/polling/afternoon_20261016 --out results/jam/feed_process_afternoon.json`
   (this file is also the artefact blocks G to K replay, per the orchestrator decision at commit).
2. Block A runs from a frozen copy of the feed-agreement code, so they cannot collide with this
   program's edits: `M=<any empty directory>; git archive d699ac2 | tar -x -C $M; ln -s $PWD/.venv $M/.venv;
   mkdir -p $M/results/feed_measured; cp results/jam/feed_process_afternoon.json $M/results/feed_measured/feed_process.json`, then in `$M`:
   `.venv/bin/python scripts/run_feed_experiments.py --block A --checkpoint results/checkpoints/mainz_here_best.pt --feed-artefact results/feed_measured/feed_process.json --results-dir results/feed_measured --traces-dir outputs/feed_measured/traces`
   (the default session, the one with the most versions, which is NJ), and the same with
   `--feed-session afternoon_20261016_mainz --results-dir results/feed_measured_mainz --traces-dir outputs/feed_measured_mainz/traces`
   as a recorded sensitivity. 150 episodes each, about 15 minutes each. The run of record
   (`results/feed/A.jsonl`) is the placeholder run; copy the new results into the repository
   under a new directory, never over it.

### Task order to resume

1. Confirm the afternoon footer and the request count; decide whether to run the night session.
2. Build `results/jam/feed_process_afternoon.json` (command 1 above).
3. Resume the implementer from its recorded stopping point with the implementer brief in
   `plans/2026-10-16-jam-avoidance-briefs.md`, steps S4 to S14 (plus whatever of the in-progress
   step it left uncommitted), Amendment 1 included.
4. The open decision above on block G's timing.
5. Validator rounds 1 to 3 (Opus, kept alive), with the validator brief in the briefs file.
   Confirm every finding before sending fixes.
6. Experiments in the order of section 8.1 on the artefact from step 2, session
   `afternoon_20261016_mainz`.
7. The paper rewrite from the outline in the briefs file: three arms (no control, the
   network-wide feed lookahead, the best model from `S_summary.json`).
8. The block A re-runs (commands above), whenever compute is free.

## Resumed (2026-10-16, 15:20 EDT; orchestrator)

- **The afternoon poller did not reach its own end.** The Claude process that ran it exited during
  the pause, and the poller went with it: the last request was sent at 15:17:33 EDT, 167 s before
  the 6,900 s end, and the session has no footer. All 2,696 requests returned HTTP 200 and no line
  is partial. With the dry session's 122, 2,818 of the author's 5,760 requests are spent, leaving
  2,942.
- **The measured artefact of record for blocks G to K** is `results/jam/feed_process_afternoon.json`
  (sha256 prefix `8f5430bfc68b`), built from the afternoon session alone: NJ 179 versions, `L_bar`
  84.0 s, `L_low` 28.9 s, `J` 55.1 s, `g_min` 28 s; Mainz 113 versions, `L_bar` 89.5 s, `L_low`
  34.4 s, `J` 55.1 s, `g_min` 60 s. Replayed session: `afternoon_20261016_mainz`.
- **Block G's timing (the open decision of the pause) is taken by the orchestrator's
  recommendation:** block G runs as soon as S4, S5, S6, S10 and S12 are committed, from an archive
  mirror of that commit, on training seeds 1-5 only, before the validator rounds. If a validator fix
  later touches code block G uses, block G is re-run on the fixed commit before block F.
- The night session is rescheduled for 22:00 with its cap of 2,762 requests, inside the 2,942 left.
- The block A re-runs (NJ default session, then Mainz) are running from the archive mirror of
  `d699ac2`.

## Block G result (2026-10-16, 15:47 EDT; orchestrator)

**Decision: go, on the base setting** (`tau_l` 10 s, `m_on` 0.1); the fallback grid was not needed.
Run from an archive of commit `1ede200` (the headers record `commit: null` because an archive has
no `.git`; later blocks run from a detached git worktree so the header carries the SHA), measured
artefact `results/jam/feed_process_afternoon.json` (sha256 `8f5430bf...`), session
`afternoon_20261016_mainz`, `L_bar` 89.49 s, `D` 99.49 s, training seeds 1-5, 3,600 s, 20 episodes,
298 s of wall time. Zero colliding vehicles, zero collision teleports, zero writes above the plan,
zero `setMaxSpeed` read-back mismatches.

| scenario | plan alone, jammed segment-min | oracle, jammed segment-min | paired difference (oracle - plan) | no 120 s jam spell | served flow, plan / oracle |
|---|---|---|---|---|---|
| B9 | 27.5, 29.5, 27.2, 19.2, 31.3 | 0.0, 0.0, 2.2, 0.2, 2.0 | -26.07 +/- 3.89 (2 SE) | plan 0 of 5, oracle 4 of 5 | 3,508-3,572 / 3,688-3,707 veh/h |
| I_exit | 13.5, 18.3, 20.3, 21.0, 21.2 | 9.2, 8.5, 8.3, 8.2, 7.3 | -10.57 +/- 3.38 (2 SE) | 0 of 5 for both | 3,141-3,321 / 3,351-3,377 veh/h |

On I_exit both arms start a jam spell 90 to 130 s after the request time (the incident vehicle was
inserted 0.0 to 18.0 s after 900 s), so the oracle shortens the jam rather than preventing it, as
the risk table anticipated. The oracle's correction was active on 36% to 38% of controlled
vehicle-ticks on B9 and about 20% on I_exit.

## Amendment 2 (2026-10-16, after validation round 1; orchestrator)

Round 1 (findings in `plans/2026-10-16-jam-avoidance-validation.md`) confirmed block G's "go"
independently: recomputed from the traces, four episodes re-run and reproduced exactly, the oracle's
input equal to a vehicle-by-vehicle SUMO count, no write above the plan in 596,763 writes. The
orchestrator reproduced R1-1 and R1-2 before accepting them.

- **Fixes sent to the implementer:** R1-1 (edge speed removes stopped incidents twice), R1-2 (each
  incident vehicle is removed at its own insertion time + 300 s, so both arms of a seed face blocks
  of equal length), R1-4 (tests for the 11 surviving mutants on the block G path, and "incomplete"
  unless all seeds are present), R1-7 (a counter of active ticks on which the plan is above 0.5),
  and O-1 (orchestrator: fit the horizon and camera height only on boxes whose centre lies within
  15% of the frame width of the centre column; on `run_20260908_155910` all boxes give 0.70 m and
  centre boxes 0.89 m, because an obliquely viewed vehicle's box includes part of its side).
- **Block G is re-run on the fixed commit before block T,** because R1-1 and R1-2 touch code block
  G uses. The recorded block G (`706f5b5`) stays in the record; the re-run is the gate of record.
- **R1-3, documented limit.** On I_exit the stopped vehicle stands at least 95.5 m beyond any vehicle
  on a segment edge, so no 80 m sensor (camera or oracle) ever sees it; the corrections respond only
  to the queue it causes. `oracle_wide` (200 m) sees it. On I_mid the stopped vehicles are seen.
  Section 4.5's "visible to cameras" holds only for sensors whose range exceeds 95 m on I_exit.
- **R1-5, documented limit.** Under the version-keyed rule, `L_bar` is at least every replayed lag,
  so the whole fleet switches together at `T_k + D` and the lag multiplier acts only through `D`:
  block L is a sweep of the plan's switch delay, `D` = 0, 99.5, 189.0 and 368.0 s for `m` = 0, 1, 2
  and 4. Figure 3's caption and the paper say so.
- **R1-6** (the block header records the feed at `m = 1`; each episode carries its own `m` and `D`)
  is not fixed.
