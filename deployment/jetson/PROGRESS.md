# Deployment Progress — resume here

Status against `../../plans/plan_deployment.md` tasks. Updated 2026-06-12.

## Snapshot

Code-complete, benchmarked, and now **validated end-to-end on simulated
drives**: real dashcam footage + scripted GPS through the full pipeline in
real time, scored by `eval_run.py` — **all gates PASS** on both shipped
scenarios. Camera and road testing still pending hardware. Policy bundle is
**random-init** until simulation training delivers a checkpoint.

Synthetic bench: **e2e p50 19.8 ms / p95 20.2 ms, 48.5 FPS** — `models/bench_results.md`.
Simulated drives (paced 30 fps, real TRT detector + MJPG file decode on CPU):
e2e p50 ~22–25 ms / p95 ~34–38 ms. Target < 200 ms.
56/57 tests pass on-device (`python3 -m pytest tests/`); 1 skip = sim-env
comparisons that need highway_env (run those on the dev machine).

Reference simulated runs (each has `report.md` + plots from `eval_run.py`):
- `~/dsrc_logs/run_20260612_163118` — i495_cruise, 8382 ticks / 280 s, PASS
  (GPS speed RMSE 0.121 m/s = injected noise floor; dropout handled)
- `~/dsrc_logs/run_20260612_162817` — highway_decel, 901 ticks / 30 s, PASS
  (leader acquired at ~42 m, closes to ~5 m as scripted braking ends)

## Plan tasks

| task | status | notes |
|---|---|---|
| P1 Jetson setup | DONE (software) | deps installed; engine built; `run_demo.py --selfcheck` implemented. Pending on hardware: **camera not yet attached** (no /dev/video0); **GPS blocked by dialout group** (fix below). |
| P2 detection + tracking | DONE (code + engine) | YOLOv8n FP16 TRT @640 = 3.9 ms GPU / 17.7 ms incl. pre/post; SORT-lite. Pending: demo video on real footage. |
| P3 distance estimation | DONE (geometric v0) | ground-plane + width-prior instead of depth net (latency call - see ARCHITECTURE §7). Pinhole math unit-tested; needs real-world sanity pass after calibration. |
| P4 observation builder | DONE | all 39 fields with provenance tags; encoder bit-identical to sim (tested); neutral fallbacks per spec. |
| P5 actor export + inference | DONE (untrained weights) | export_policy.py (checkpoint or --random); numpy runtime 0.5 ms. **Blocked on a trained checkpoint from the sim side.** |
| P6 dashboard | DONE (code) | HUD + advisory panel + UNTRAINED banner; not yet seen on a physical display. |
| P7 replay mode | DONE (code) | `replay_demo.py --log <run_dir>`; needs a recorded run (requires camera) to exercise for real. |
| P8 in-vehicle demo | NOT STARTED | needs camera, mount, calibration, drive plan. |
| (extra) simulated-drive harness | DONE + EXERCISED | scenarios = dashcam clip + camera geometry + scripted GPS (`sensors/gps_sim.py`); scored by `eval_run.py` gates. Both shipped scenarios PASS on-device. See README "Simulated drives", ARCHITECTURE §10. |

## Hardware blockers (one-time, need the operator)

1. **GPS permissions**: `sudo usermod -aG dialout $USER` then re-login.
   (`/dev/ttyACM0` is root:dialout; the u-blox 8 is enumerated and waiting.)
2. **Attach the USB camera**, then `python3 run_demo.py --selfcheck`.
3. Optional for stable benchmarks: `sudo nvpmodel -q` to record the power
   mode and `sudo jetson_clocks` before timing runs (couldn't be captured
   this session - no passwordless sudo; jtop logs the mode during runs).

## Next actions (in order)

1. Operator runs the dialout fix; re-run `--selfcheck`; confirm GPS sentences
   and the 5 Hz rate config (selfcheck prints observed rate).
2. Attach camera → desk run `python3 run_demo.py` pointed out a window;
   verify detections/distances look sane; record a clip and run
   `replay_demo.py` once over it (closes the P7 loop on real data).
3. Calibrate (README "Calibration") and set `camera:` values in config.yaml;
   if the mount sees the hood, also set `camera.hood_line_y_px` (phantom-
   leader filter — see ARCHITECTURE §10).
4. DONE 2026-07-02 — trained checkpoint (workstation Stage-0 v3 `clean_s7`)
   exported and validated on both scenarios (PASS; head switches 170→31/min
   on i495; advisory gap-conditioned with zero exceptions). Optional polish:
   advisory-layer hysteresis to damp leader-acquisition flicker; review and
   commit the sim fixes + training configs on the sim side.
5. Re-run `bench_latency.py` + a live logged run for final paper numbers
   (live e2e expected ~25–55 ms including camera frame interval; the
   simulated drives already measured p95 ~34–38 ms paced at 30 fps).
6. Windshield mount + P8 drive (passenger-operated, advisory-only protocol
   from the plan).

## Session log

- **2026-09-05** — Phone app installed on `ZY227VV4XC` (implement stage,
  tasks 40/42/43/47). The app was absent from the handset going into this
  session (`pm path com.dsrc.phone` empty); tasks 40, 42, 43 and 47 all
  need a live phone session, so this is a precondition for all four.
  Built `:app:assembleDebug` on this laptop (JDK 17, `phone/gradlew`) from
  commit `a61458d`, applicationId `com.dsrc.phone`, versionCode 1,
  versionName `0.1`. SHA-256 of `app-debug.apk`:
  `e179e95282ced6a99a71b03547350079254ceda3caabbbd2778dc7c15f327602`
  (identical across a clean rebuild, so the build is reproducible for this
  tree). Copied to `jetson-orin` by `scp` (checksum verified after
  transfer, unchanged) and installed from there —
  `adb -s ZY227VV4XC install -r`, since the Mac's `adb` cannot see the
  phone while it is cabled to the Jetson. `pm path com.dsrc.phone` on the
  device now resolves; `dumpsys package com.dsrc.phone` reports
  versionCode 1 / versionName 0.1, matching the build. No other package or
  device setting was touched. Any run record from here should cite this
  SHA-256 as the build that produced it.

- **2026-07-02 (final)** — TRAINED POLICY DEPLOYED. v3 training (5 runs,
  3000 updates, ring/heterogeneous/2 AVs + 8 humans, crash_penalty 2.0,
  entropy 0.005 — config `shared_ppo_stage0`) produced genuinely different
  strategies per seed; winner `clean_s7` (nominal with leader, fast when
  clear) exported to `models/actor_policy` (trained=True, UNTRAINED banner
  gone; smoketest bundle deleted). Sim-side eval: all learned policies ≈
  cooperative_smoothing (11.8–12.6 vs 11.1 m/s; no controller survives the
  full 120 steps — aggressive humans eventually rear-end any slower AV);
  no_av collapses (2.6–6.4 m/s, jam 0.43–0.78). Scenario re-runs with the
  trained bundle — both PASS all gates:
  - highway_decel `run_20260702_214612` (902 ticks, e2e p50 21.2/p95 22.2 ms):
    advisory gap-conditioned — leader <60 m → nominal 100%, open road →
    fast 100%, 97% nominal while braking; head switches 52.5/min
    (random bundle was ~170/min). Edge case: stationary pileup cars can
    drop out of tracking at very close range → brief fast flickers (3% of
    low-speed ticks).
  - i495_cruise `run_20260702_214755` (8383 ticks/280 s, e2e p50 24.2/
    p95 25.0 ms, 29.9 Hz): 93.3% nominal / 6.7% fast, rec p50 23.8 m/s
    (= free_flow−3 ✓), head switches 30.6/min, leader→nominal with zero
    exceptions over all ticks.
  Remaining advisory roughness = leader-track acquisition flicker at the
  fast/nominal boundary; a small hysteresis in the advisory layer would
  cut the switch rate further (not implemented).
  NOTE for the sim side: training uncovered 3 sim bugs (fixed, uncommitted,
  `ws:~/dsrc/outputs/sim_fixes.patch`) + the crash-penalty/entropy training
  changes (`shared_ppo_stage0.yaml`); tests 133→138. Review + commit.

- **2026-07-02 (afternoon update)** — first training launch was INVALID and
  surfaced three sim bugs (fixed, uncommitted — patch at
  ws:~/dsrc/outputs/sim_fixes.patch, review before committing):
  (1) ring spawn packed all 14 vehicles onto one 65 m arc (per-arc modulo +
  single spawn-lane entry) → 12 pre-crashed at t=1, every episode terminated
  at step 1, so PPO saw a constant world (scores frozen at 18.4286 = the
  crash-clamped speed mean); (2) `road.step(dt=1.0)` did one 1 s Euler step
  per decision → IDM phantom collisions/negative speeds; now 10 physics
  substeps per 1 Hz decision (highway_env's intended design); (3)
  run_baseline gave initial ring humans only to no_av → baselines and
  evaluate_policy ran near-empty rings (my earlier "no_av collapses vs
  learned 17.2 m/s" comparison was apples-to-oranges — retracted).
  Also two design findings: training default human-model "normal" is
  homogeneous (no seed-to-seed variation), and the contract's speed-bin
  decode (slow = free_flow−10, floor 12 m/s) has no authority when flow
  equilibrium < 15 m/s — density must keep flow in the actionable band.
  v2 training env (validated by probes): ring, heterogeneous humans,
  2 AVs + 8 humans — humans-only collapses into stop-and-go with crashes;
  2 slow-commanding AVs fully stabilize it; slow-vs-fast trajectories
  diverge (authority ✓). Deployment contract untouched. Sim tests 133→137.
  v2 fleet: 4 clean seeds + 1 deploysense, 3000 updates × 256 steps,
  ~1.8 s/update (~1.5–2 h). eval staged (evaluate_all.sh, matched env).

- **2026-07-02** — training kickoff (Claude session, in progress). Stage-0
  policy training is RUNNING on the user's workstation (`ssh dsrc-ws` =
  cims-phd-de8-1, i9-14900K/RTX 4000 Ada — evaluated sufficient; env
  stepping is CPU-bound, ~19 s/update). Repo rsynced there at 46fa023 +
  uncommitted sensing passthrough (TrainingConfig.sensing → env_config;
  new `configs/training/shared_ppo_deploysense.yaml` with Jetson-measured
  noise: range 100 m, pos σ 1.5 m, speed σ 0.15 m/s; patch copy in
  ws:~/dsrc/outputs/sensing_passthrough.patch); 134/134 sim tests pass
  there (py3.12, torch 2.12.1, highway-env 1.11). Runs: 4 clean seeds
  (7/17/27/37) + 1 deploysense (seed 7), shared_ppo speed_only, ring/
  medium/normal, 1500 updates × 256 steps each, ETA ~9 h from 12:45.
  State: `ssh dsrc-ws 'bash ~/dsrc/outputs/stage0/status.sh'`; eval staged
  at ws:~/dsrc/outputs/stage0/evaluate_all.sh (argmax vs no_av +
  cooperative_smoothing, seeds 101–103). Export path dry-run VERIFIED
  end-to-end with the 5-update smoke checkpoint: ws actor.pt →
  `export_policy.py --checkpoint` → trained bundle → highway_decel
  scenario runs with a stable advisory (rec 44.7 mph constant, e2e p50
  20.2 ms; delete `models/actor_policy_smoketest.*` after the real
  export). Dry-run reference (ring/medium seed 101): no_av collapses
  (mean speed 0.26 m/s, jam 0.99, 12 collisions) vs smoke policy 17.2 m/s,
  0 collisions — AV presence alone is a huge effect; tonight's eval
  separates learning from presence.

- **2026-06-12** — simulated-drive harness (Claude session). New:
  `sensors/gps_sim.py` (profile-scripted GPS twin, dropouts/noise/loop),
  `--scenario`/`--sim-gps` in run_demo, `eval_run.py` (gated reports +
  plots), `calibration/auto_horizon.py` (Theil–Sen horizon/height fit from
  detections), `transcode_clip.py`, 2 scenarios + 2 CC dashcam clips
  (`data/`, licenses in SOURCES.md). Both scenarios PASS all gates.
  Fixed along the way: FFmpeg VP9 state poisoning on one bad cluster →
  CameraStream reopen+seek recovery (the I-495 Commons transcode has a
  glitch at frame ~4153 and a dead tail past ~8390, hence the 280 s AVI);
  ego hood detected as phantom leader at minimum range →
  `camera.hood_line_y_px` detector filter + ground-plane→width-prior
  fallback for hood/frame-clipped bboxes. Tests 37→56. Known cosmetics:
  matplotlib Axes3D warning in eval_run (same ultralytics/matplotlib
  duplication, harmless).
- **2026-06-11** — initial build (Claude session). Probed device (JetPack 6 /
  L4T R36.4.7, TRT 10.3, CPU-only torch → TRT-direct detection + CPU actor
  design). Vendored sim contract from commit `d477dba` with equality tests.
  Implemented full pipeline, entry points, tests, docs. Built YOLOv8n FP16
  engine (3.9 ms GPU). Optimized: numpy preprocess → blobFromImage (−5.5 ms),
  TorchScript actor → verified numpy mirror (−3.5 ms, jitter gone);
  e2e 29.2 → 19.8 ms p50. Fixed rel_speed_window default (6→8: at 30 fps the
  window must span the 0.2 s minimum or relative speed never validates).
  Known cosmetics: ultralytics prints a matplotlib Axes3D warning (harmless);
  ONNX exported at opset 18 (requested 12, converter fell back - TRT 10.3
  handles 18 fine).
