# Functional Graph World Model Plan

This document is the implementation plan for the paper's main technical contribution: a differentiable functional graph world model for mixed-autonomy traffic control.

The current repository already contains the simulator-facing foundation:

- `src/envs/`: highway-env wrapper and public CTDE-style API.
- `src/road/`: topology ladder and `TopologySpec` metadata.
- `src/demand/`: demand profiles, branch routes, and spawning.
- `src/sensing/`: local AV observations.
- `src/safety/`: common execution-time safety, etiquette, and physical-control layer.
- `src/baselines/`: infrastructure-free decentralized baselines.
- `src/rl/`: model-free PPO/IPPO/MAPPO actors, critics, rollout buffers, and trainers.
- `src/metrics/`: global, segment, fairness, safety, and output logging.
- `scripts/`: baseline, training, evaluation, and validation entrypoints.

The world model should extend this structure rather than create a parallel project.

## 1. Main Objective

Learn a differentiable surrogate of traffic dynamics from highway-env rollouts:

```text
Z_{t+1} ~= f_psi(Z_t, U_t, G)
```

where:

- `G` is the directed road graph derived from the current `TopologySpec`.
- `Z_t` is a discretized traffic field on graph edges.
- `U_t` is an executed AV intervention field on graph edges.
- `f_psi` is a PyTorch functional graph model.

The learned model is not the final environment. It is a training aid used for prediction, ablation, and model-based policy improvement. Final policy performance claims must always come from highway-env evaluation through the current `HighwayTopologyEnv`.

Scientific claim:

> We learn a differentiable functional graph simulator of mixed-autonomy traffic, where each road segment is represented as a spatial traffic field. This model enables model-based policy improvement for sparse AV traffic regulation while preserving decentralized execution and a common runtime safety layer.

## 2. Repo-Native Architecture

Add a new `src/world_model/` package:

```text
src/world_model/
  graph/
    graph_spec.py
    graph_builder.py

  fields/
    traffic_field.py
    action_field.py
    channels.py

  data/
    transitions.py
    dataset.py
    storage.py

  models/
    functional_graph_world_model.py
    edge_conv.py
    graph_message.py
    scalar_gnn_world_model.py
    no_graph_conv_model.py
    persistence_model.py

  policies/
    field_policy.py
    model_based_policy.py

  losses.py
  rewards.py
  evaluation.py
```

Add scripts matching existing conventions:

```text
scripts/collect_world_model_data.py
scripts/train_world_model.py
scripts/evaluate_world_model.py
scripts/train_model_based_policy.py
scripts/validate_world_model_pipeline.py
scripts/run_experiment_matrix.py
```

Add configs under existing config families:

```text
configs/world_model/functional_graph.yaml
configs/world_model/scalar_gnn.yaml
configs/world_model/persistence.yaml
configs/training/model_based_policy.yaml
configs/experiments/exp_world_model_ring.yaml
configs/experiments/exp_world_model_merge.yaml
configs/experiments/exp_world_model_tree.yaml
```

Generated datasets and model outputs should use canonical output roots:

```text
outputs/world_model/data/
outputs/world_model/checkpoints/
outputs/world_model/evaluation/
outputs/metrics/
outputs/validation/
```

## 3. Graph Representation

Create a `GraphSpec` that adapts the current `TopologySpec` from `src/road/segment_graph.py`.

Use canonical segment IDs as graph edge IDs. Do not invent a separate naming system.

Initial fields:

```python
@dataclass(frozen=True)
class GraphSpec:
    graph_id: str
    num_edges: int
    edge_ids: tuple[str, ...]
    segment_lengths_m: torch.Tensor      # [E]
    lane_counts: torch.Tensor            # [E]
    speed_limits_mps: torch.Tensor       # [E]
    edge_attr: torch.Tensor              # [E, A]
    num_bins: int
    bin_centers_m: torch.Tensor          # [E, K]
    bin_centers_norm: torch.Tensor       # [E, K]
    bin_valid: torch.Tensor              # [E, K], bool
    edge_index: torch.Tensor             # [2, M]
    edge_type: torch.Tensor              # [M]
```

Build it from:

- `TopologySpec.segment_ids`
- `TopologySpec.segment_lengths`
- `TopologySpec.lane_counts`
- `TopologySpec.lane_segments`
- `TopologySpec.merge_nodes`
- `TopologySpec.bottleneck_segments`
- road-network connectivity already used by the topology builders

Initial relation types:

```text
0 upstream_to_downstream
1 downstream_to_upstream
2 lateral_same_segment
3 merge_relation
4 bottleneck_relation
```

Start with fixed `K` spatial bins per segment and same-topology batches. Multi-topology padding can be added after the first working model.

## 4. Traffic Field Representation

Convert vehicle-level simulator state into:

```text
Z_t: [E, K, C]
```

Initial channels:

```python
STATE_CHANNELS = [
    "density_total",
    "speed_mean_total",
    "flow_total",
    "density_av",
    "speed_mean_av",
    "density_human",
    "speed_mean_human",
    "speed_std",
    "occupancy",
    "jam_indicator",
]
```

Vehicle-to-field conversion:

1. Use the current environment's active vehicle records or vehicle snapshots.
2. Map each vehicle to a canonical segment ID.
3. Compute longitudinal position within that segment.
4. Convert position to bin index.
5. Accumulate density, speed, flow, role-specific density/speed, occupancy, and jam indicators.

Start with hard binning. Add triangular kernel smoothing only after the first dataset/model loop works.

Density should be normalized by bin length and lane count. Speed channels should use local mean speed with a neutral value for empty bins, plus occupancy/count channels so the model can distinguish empty bins from stopped traffic.

## 5. Executed Action Field

The world model operates on aggregate intervention fields:

```text
U_t: [E, K, C_u]
```

Initial channels:

```python
ACTION_CHANNELS = [
    "av_count",
    "target_speed_mean",
    "target_speed_min",
    "target_speed_max",
    "desired_headway_mean",
    "lane_pref_left_fraction",
    "lane_pref_keep_fraction",
    "lane_pref_right_fraction",
    "merge_create_gap_fraction",
    "merge_hold_lane_fraction",
    "action_intensity",
]
```

Important rule:

> Aggregate the action that was actually executed by the common DSRC safety/physical-control layer, not merely the raw controller proposal.

The first implementation may approximate executed action fields from post-step AV target speed, target headway, merge mode diagnostics, and lane-change diagnostics. Later versions can expose a compact per-agent executed-action record from `HighwayTopologyEnv.step()` if needed.

If no AV is present in a bin, set `av_count = 0` and neutralize other action channels. This avoids confusing no-control bins with active zero-valued commands.

## 6. Safety Semantics

All controllers use the same public v2 action schema:

```python
{
    "desired_speed_bin": str,
    "desired_headway_bin": str,
    "lane_preference": str,
    "merge_mode": str,
}
```

All proposed actions are executed through the common DSRC safety layer in `HighwayTopologyEnv.step()`:

```text
controller action -> apply_safety_layer(...) -> executable physical behavior
```

This applies to:

- non-learning baselines,
- SharedPPO,
- IPPO,
- MAPPO,
- future model-based policies.

For imagined world-model rollouts, use differentiable soft safety penalties as regularizers in the model-based objective. Do not introduce a second runtime safety mechanism. Final safety and performance evaluation remains in highway-env.

## 7. Dataset Format

Each transition should contain:

```python
@dataclass(frozen=True)
class WorldModelTransition:
    z_t: torch.Tensor
    u_t: torch.Tensor
    z_tp1: torch.Tensor
    reward: float
    reward_components: dict[str, float]
    topology_id: str
    graph_id: str
    step_index: int
    episode_id: str
```

Batch shape for v1:

```python
z_t:      [B, E, K, C]
u_t:      [B, E, K, C_u]
z_tp1:    [B, E, K, C]
bin_valid:[B, E, K]
graph:    GraphSpec
```

Start with same-topology batches. Store metadata so later loaders can group by topology and support padding.

Data collection should cover:

- `no_av`
- `random_av`
- `density_lookup`
- `av_mediated_speed_harmonization`
- `backpressure`
- `cooperative_smoothing`
- early model-free RL checkpoints when available

The world-model dataset must include diverse policies so the model does not only learn one controller's narrow state distribution.

## 8. Functional Graph World Model

Model:

```text
z_next = z + Delta_psi(z, u, graph)
```

Recommended architecture:

1. **Input encoding**
   - Concatenate `z`, `u`, bin position features, and edge attributes.
   - Encode each edge-bin with an MLP.

2. **Intra-edge dynamics**
   - Use residual 1D convolution blocks across the spatial bins of each segment.
   - This captures waves, density buildup, and speed changes along a link.

3. **Graph message passing**
   - Pool each edge into an edge summary.
   - Pass relation-typed messages across `edge_index`.
   - Inject graph context back into edge-bin representations.

4. **Residual decoder**
   - Predict `Delta Z`.
   - Clamp or bound normalized channels in v1.
   - Add bounded channel heads later if needed.

Add ablation models:

- `PersistenceModel`: predicts `Z_{t+1} = Z_t`.
- `ScalarGNNWorldModel`: uses one vector per segment.
- `NoGraphConvModel`: uses edge field convolution without graph messages.

The functional graph model should beat persistence and no-graph baselines on multi-step prediction before it is used for policy improvement.

## 9. Losses And Evaluation

Training losses:

- one-step Huber or MSE prediction loss;
- multi-step rollout loss with horizon `H=5` initially;
- flow consistency loss for `q ~= rho * v`;
- bounds loss for impossible density/speed values;
- optional global mass/conservation consistency;
- auxiliary prediction for reward components such as mean speed, jam fraction, hard braking, fairness, and rolling-roadblock score.

Evaluation metrics:

- one-step channel MSE/MAE;
- multi-step prediction error by horizon;
- density/speed/flow error separately;
- rollout stability;
- error by topology and demand;
- comparison against persistence, scalar graph, and no-graph baselines.

Never use learned-model rollout performance as final controller performance. Use it only to evaluate the surrogate.

## 10. Model-Based Policy Improvement

The model-based policy objective should sample real `Z_t` states from the replay dataset, roll them forward through a frozen world model, compute differentiable traffic reward, and update the policy.

Training loop:

```text
real highway-env rollout
  -> collect simulator transition data
  -> update or refresh world-model dataset
  -> train/evaluate world model
  -> freeze world model
  -> apply imagined-rollout policy loss
  -> evaluate resulting policy in highway-env
```

The imagined rollout loss should be an auxiliary term combined with the existing model-free PPO/MAPPO loss:

```text
L_policy = L_model_free + alpha * L_imagined + beta * L_soft_safety
```

Use `alpha=0` as the no-world-model baseline. Increase `alpha` only after the model beats persistence on multi-step validation.

The first implementation can train a field policy that maps traffic fields to action fields. A later bridge can map field-level decisions back to AV-local v2 actions for highway-env evaluation.

## 11. Experiment Matrix

World-model method comparisons:

```text
no_av
hand-designed baselines
model-free SharedPPO/IPPO/MAPPO
persistence model baseline
scalar-GNN world-model method
no-graph Conv1D world-model method
functional graph world-model method
functional graph + physics losses
```

Sweeps:

- topology: ring, straight single-lane, straight multi-lane, merge, inverted tree, inverted tree bottleneck;
- demand: low, medium, high, burst;
- AV penetration: 5%, 10%, 20%;
- human model: normal, heterogeneous, aggressive;
- sensing: deterministic first, then noise/range/latency;
- model-based loss weight: `alpha = 0, 0.1, 0.3, 1.0`;
- world-model horizon: `H = 1, 5, 10`.

Paper questions:

- Does the functional edge-field representation improve prediction?
- Does graph message passing matter?
- Do physics-inspired losses improve multi-step rollout stability?
- Does the learned world model improve sample efficiency or transfer?
- Does final highway-env evaluation improve over model-free baselines without increasing obstruction metrics?

## 12. Development Milestones

### Stage W1: Graph and field extraction

Deliver:

- `GraphSpec` adapter for all current topologies.
- `TrafficFieldExtractor`.
- `ActionFieldAggregator`.
- Unit tests for tensor shapes and conservation sanity.

### Stage W2: Dataset and collection script

Deliver:

- `WorldModelTransition` schema.
- dataset writer/loader;
- `scripts/collect_world_model_data.py`;
- small smoke dataset from ring and merge.

### Stage W3: One-step world model

Deliver:

- functional graph model forward pass;
- persistence baseline;
- one-step loss;
- `scripts/train_world_model.py`;
- `scripts/evaluate_world_model.py`.

Acceptance:

- beats persistence on one-step prediction for ring and straight.

### Stage W4: Multi-step world model

Deliver:

- multi-step rollout loss;
- flow/bounds losses;
- scalar-GNN and no-graph ablations.

Acceptance:

- beats persistence on multi-step prediction.

### Stage W5: Model-based policy hook

Deliver:

- imagined rollout objective;
- `scripts/train_model_based_policy.py`;
- alpha sweep with `alpha=0` reproducing model-free baseline behavior.

Acceptance:

- policy still evaluates in highway-env;
- no final claim is based only on learned-model rollouts.

### Stage W6: Experiment launcher

Deliver:

- `scripts/run_experiment_matrix.py`;
- configs for final sweeps;
- validation script for all required artifacts.

Acceptance:

- one command can launch or dry-run all paper experiments.

## 13. Pitfalls And Guardrails

- A good one-step model can still fail multi-step rollout. Always evaluate multi-step prediction.
- A policy can exploit model errors. Use model uncertainty, conservative imagined rewards, and final highway-env evaluation.
- Do not train only on one controller. Collect diverse data.
- Do not treat the world model as the environment for paper claims.
- Do not add a second runtime safety path for model-based methods.
- Keep the public v2 action schema unchanged.

## 14. Paper Positioning

Suggested abstract-level sentence:

> We introduce a differentiable functional graph world model for mixed-autonomy traffic control. Each road segment is represented as a spatial traffic field rather than a scalar graph feature. The model learns topology-aware traffic dynamics under local AV interventions and enables model-based policy improvement through differentiable imagined rollouts. All controllers use the same public action interface and common execution-time safety layer, and final policies are evaluated in the original non-differentiable simulator.

The first real milestone is not model-based RL. The first milestone is a clean field extraction and world-model training pipeline that beats persistence on multi-step prediction. Everything else depends on that.
