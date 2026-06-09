# DSRC Project Task List

This task list is status-aware. The simulator foundation is already substantially implemented, so the remaining work should first complete the full world-model code path and experiment-launch infrastructure. Final training, evaluation sweeps, and paper plots happen only after the implementation and launch scripts are in place.

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

## B. Finish Implementation Before Final Experiments

Do these tasks before running final paper sweeps.

1. **Add world-model package skeleton**
   Create `src/world_model/` with graph, field, data, model, policy, loss, reward, and evaluation modules.

2. **Build GraphSpec adapter**
   Convert existing `TopologySpec` objects into `GraphSpec` using canonical segment IDs, segment lengths, lane counts, lane-segment maps, merge nodes, and bottleneck segments.

3. **Implement traffic field extractor**
   Convert highway-env vehicle state into edge-bin traffic fields `Z_t` with density, speed, flow, AV/human channels, speed variance, occupancy, and jam indicators.

4. **Implement executed-action field aggregator**
   Convert actually executed safe AV interventions into edge-bin action fields `U_t`, including AV count, target speed/headway summaries, lane preference fractions, merge-mode fractions, and action intensity.

5. **Implement world-model transition dataset**
   Add `WorldModelTransition`, dataset writer/loader, storage format, metadata, and same-topology batching.

6. **Add data collection script**
   Add `scripts/collect_world_model_data.py` to collect simulator transitions across topologies, demand profiles, seeds, and controllers.

7. **Implement functional graph world model**
   Add input encoding, edge-wise Conv1D dynamics, relation-typed graph message passing, residual prediction, and bounded output handling.

8. **Implement world-model losses and validation metrics**
   Add one-step prediction loss, multi-step rollout loss, flow consistency, bounds loss, optional conservation loss, reward auxiliary loss, and prediction reports.

9. **Add world-model baselines and ablations**
   Add persistence, scalar-GNN, and no-graph Conv1D models so the functional edge-field contribution can be isolated.

10. **Add world-model train/evaluate scripts**
    Add `scripts/train_world_model.py`, `scripts/evaluate_world_model.py`, and `scripts/validate_world_model_pipeline.py`.

11. **Add model-based policy hook**
    Add a frozen-world-model imagined rollout objective that can be combined with model-free PPO/MAPPO loss. Use differentiable soft safety penalties only as training regularizers, while final execution still goes through the common environment safety layer.

12. **Add model-based policy script**
    Add `scripts/train_model_based_policy.py` with an `alpha=0` setting that reproduces model-free behavior and positive-alpha settings for imagined rollout training.

13. **Add experiment matrix launcher**
    Add `scripts/run_experiment_matrix.py` with dry-run and launch modes for all paper experiments.

14. **Add plot/table generation scripts**
    Add scripts for world-model prediction plots, speed heatmaps, queues, throughput, branch fairness, merge delay, spillback, safety diagnostics, and deployment metrics.

15. **Package configs**
    Add `configs/world_model/`, `configs/training/model_based_policy.yaml`, and experiment configs for world-model data collection, world-model training, model-based policy training, and final evaluation.

## C. Validation Before Final Sweeps

1. **Validate current simulator stack**
   Keep running project interface, topology, baseline, metrics, safety, and model-free training smoke tests.

2. **Validate world-model data path**
   Confirm graph construction, field extraction, action aggregation, transition writing, transition loading, and shape consistency for ring, merge, and inverted tree.

3. **Validate world-model prediction**
   Require the functional graph model to beat persistence on one-step and multi-step prediction before using it for control.

4. **Validate model-based policy hook**
   Confirm imagined rollout gradients update the policy, not the frozen world model, and `alpha=0` matches model-free behavior.

5. **Validate launch scripts**
   Dry-run the complete experiment matrix and verify every expected output path before starting long training jobs.

## D. Final Experimentation

Run final experiments only after Sections B and C are complete.

1. **Model-free reference experiments**
   Run baselines, Shared PPO, IPPO, and MAPPO across the selected topology/demand/human-model matrix.

2. **World-model prediction experiments**
   Compare persistence, scalar-GNN, no-graph Conv1D, functional graph, and functional graph plus physics losses.

3. **Model-based policy experiments**
   Compare model-free policies against model-based policies using functional graph world models and ablations.

4. **Robustness sweeps**
   Sweep AV penetration, demand, human-driver model, sensing range/noise/latency, and topology.

5. **Safety and obstruction analysis**
   Report collisions, hard braking, follower disruption, lane-change rates, low-speed-uncongested behavior, all-lane low-speed occupancy, rolling-roadblock score, and branch starvation.

6. **Deployment prototype metrics**
   Report Jetson advisory-only latency, FPS, policy inference time, observation quality, and sim-to-prototype observation alignment.

7. **Paper figures and tables**
   Generate prediction-error tables, sample-efficiency curves, highway-env final performance tables, speed heatmaps, queue plots, fairness plots, ablation tables, and deployment feasibility figures.

## E. Reproducibility Package

1. **One-command dry runs**
   Provide commands to dry-run all experiment matrices without launching long jobs.

2. **One-command smoke validation**
   Keep smoke validation small enough for routine regression testing.

3. **HPC/container instructions**
   Document SIF/overlay use, output roots, seeds, and expected artifacts.

4. **Artifact manifest**
   Standardize where checkpoints, datasets, metrics, plots, and validation summaries live.
