# DSRC Project Task List

This task list is status-aware. The simulator foundation is already substantially implemented and the deployment prototype is code-complete and validated on simulated drives, so the remaining work is closing the prototype's hardware loop, converting placeholder observation fields into measured ones, and building experiment-launch infrastructure. Final training, evaluation sweeps, and paper plots happen only after the implementation and launch scripts are in place.

## A. Simulator Foundation Status

The following foundation is substantially implemented and should be maintained with regression tests rather than rebuilt from scratch:

1. **Simulator integration**
   Installed `highway_env` integration, reset/step behavior, local source fallback, and environment smoke checks.

2. **Project interfaces**
   Public environment API, v2 action schema, observation schema, controller contract, config loading, active vehicle lifecycle, aggregate-only cooperation, and common execution-time safety contract.

3. **Topology ladder**
   Ring, straight single-lane, straight multi-lane, merge, inverted tree, and inverted tree bottleneck topologies.

4. **Demand and vehicle lifecycle**
   Reproducible spawning, inflow/outflow handling, AV penetration, branch splits, demand profiles, and exited-vehicle accounting.

5. **Vehicle roles and human behavior profiles**
   AV/human role accounting plus cautious, normal, aggressive, and heterogeneous human-driver profiles.

6. **Metrics and logging**
   Canonical JSON/CSV artifacts, step metrics, segment metrics, fairness metrics, safety diagnostics, obstruction metrics, and validation reports.

7. **Local sensing**
   AV-local observations, aggregate local cooperation fields, neutral fallback, range/noise/latency support, density and speed bins.

8. **Common safety, etiquette, and physical-control layer**
   Shared execution-time action filtering for all controllers, including speed/headway decoding, bounded acceleration, lane-change dwell, follower-disruption checks, low-speed-uncongested blocks, emergency overrides, and rolling lane-change limits.

9. **Baseline ladder**
   `no_av`, `random_av`, `selfish_av`, `density_lookup`, `dynamic_speed_limit`, `av_mediated_speed_harmonization`, `backpressure`, and `cooperative_smoothing`.

10. **Model-free RL entrypoints**
    Shared PPO, IPPO, MAPPO, learned-policy evaluation, checkpointing/resume, and smoke validation.

## A2. Deployment Prototype Status

The prototype is code-complete, benchmarked, and validated end-to-end on simulated drives with a trained policy. Maintain rather than rebuild:

1. **Pipeline and entrypoints**
   Capture-to-advisory loop, live demo, selfcheck, headless and scenario modes, replay, latency bench, and gated run scoring.

2. **Perception and observation**
   TensorRT detector, lightweight tracker, geometric distance estimation, and the full observation builder with per-field provenance tags.

3. **Policy and advisory path**
   Vendored sim contract with equality tests, checkpoint export, fast inference runtime, and the driver-facing advisory decode.

4. **Simulated-drive harness**
   Scripted-GPS twin, scenario definitions, test footage, and PASS/FAIL gates on latency, tick rate, GPS freshness, ego-speed error, and perception coverage.

## B. Finish Implementation Before Final Experiments

Do these tasks before running final paper sweeps.

1. **Resolve GPS device permissions**
   Add the operator account to the serial device group, re-login, and confirm the observed fix rate through selfcheck.

2. **Attach and exercise the camera**
   Attach the USB camera, pass selfcheck, run a desk session, and confirm detections and distances are sane.

3. **Record and replay a live run**
   Capture a real logged run and replay it to confirm tick-for-tick decision agreement on real data.

4. **Calibrate the mounted camera**
   Run the calibration helpers, commit the resulting camera values to config, and set the hood line if the mount sees the hood.

5. **Add advisory hysteresis**
   Damp the leader-acquisition flicker at the fast/nominal bin boundary and re-measure the advisory switch rate.

6. **Fix the driver-facing readout**
   Show the safety filter's gap-aware cap rather than the raw decode.

7. **Run the in-vehicle advisory-only drive**
   Windshield mount plus the passenger-operated, non-actuating drive protocol from the deployment plan.

8. **Add rear sensing**
   Second camera to convert the follower and rear-gap fields from neutral fallback into measured values.

9. **Add OBD-II speed**
   Prefer a fresh OBD speed over GPS, and use the GPS-vs-OBD comparison as an observation-quality measurement.

10. **Run the two-unit cooperative demo**
    Enable beacons on two units so the nearby-AV and cooperation fields come live while preserving the aggregate-only constraint.

11. **Add map matching**
    Derive segment and distance-to-merge from a preloaded route so the two sim-parity fields become measured. Note that the simulator itself currently hardcodes `distance_to_next_merge = 0.0`, so this task has a simulator-side counterpart.

12. **Add the local-plus-aggregate observation regime**
    Support local-only, aggregate-assisted, and oracle/global observation conditions behind one config switch so they can be compared under a single safety layer.

13. **Add partial-deployment sweep support**
    Config and runner support for message loss, delay, staleness, compliance, and penetration sweeps, including bottleneck-seeded versus uniform placement at equal fleet size.

14. **Add experiment matrix launcher**
    Add `scripts/run_experiment_matrix.py` with dry-run and launch modes for all paper experiments.

15. **Add plot/table generation scripts**
    Add scripts for speed heatmaps, queues, throughput, branch fairness, merge delay, spillback, safety diagnostics, observation-regime comparisons, and deployment metrics.

16. **Package configs**
    Add experiment configs for the observation-regime comparison, partial-deployment sweeps, and final evaluation.

## C. Validation Before Final Sweeps

1. **Validate the current simulator stack**
   Keep running project interface, topology, baseline, metrics, safety, and model-free training smoke tests.

2. **Validate observation-contract parity**
   Whenever the simulator's observation contract changes, update the vendored prototype contract, re-run the contract-equality tests, and re-export the policy bundle.

3. **Validate the prototype gates**
   Re-run both shipped scenarios and require all gates to pass after any contract, calibration, or advisory change.

4. **Validate the observation regimes**
   Confirm that local-only, aggregate-assisted, and oracle conditions differ only in observation content, and that the aggregate path still degrades to local-only when no peers are heard.

5. **Validate launch scripts**
   Dry-run the complete experiment matrix and verify every expected output path before starting long training jobs.

## D. Final Experimentation

Run final experiments only after Sections B and C are complete.

1. **Model-free reference experiments**
   Run baselines, Shared PPO, IPPO, and MAPPO across the selected topology/demand/human-model matrix.

2. **Observation-regime experiments**
   Compare local-only, aggregate-assisted, and oracle controllers under one safety layer to quantify what the aggregate signal buys.

3. **Partial-deployment experiments**
   Sweep penetration, compliance, message loss, delay, and sensing noise, and compare bottleneck seeding against uniform adoption at equal fleet size.

4. **Robustness sweeps**
   Sweep demand, human-driver model, sensing range/noise/latency, and topology.

5. **Safety and obstruction analysis**
   Report collisions, hard braking, follower disruption, lane-change rates, low-speed-uncongested behavior, all-lane low-speed occupancy, rolling-roadblock score, and branch starvation.

6. **Deployment prototype metrics**
   Report advisory-only latency, FPS, policy inference time, ego-speed accuracy under dropout, observation quality and missingness, and sim-to-prototype observation alignment.

7. **Paper figures and tables**
   Generate the deployment feasibility table, highway-env final performance tables, speed heatmaps, queue plots, fairness plots, observation-regime and partial-deployment tables, and the observation provenance breakdown.

## E. Reproducibility Package

1. **One-command dry runs**
   Provide commands to dry-run all experiment matrices without launching long jobs.

2. **One-command smoke validation**
   Keep smoke validation small enough for routine regression testing.

3. **HPC/container instructions**
   Document SIF/overlay use, output roots, seeds, and expected artifacts.

4. **Artifact manifest**
   Standardize where checkpoints, metrics, plots, prototype run logs, and validation summaries live.
