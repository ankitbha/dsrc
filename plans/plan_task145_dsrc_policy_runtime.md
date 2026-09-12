# Task 145 — a device-side runtime for the DSRC controller

## Short version

Written under `plan_dsrc_rec`: every decision was taken by recommendation, without asking.
**None carries user sign-off.** The open items below are the ones that would have been
questions.

**The defect.** The Jetson runs a 39 -> 128 -> 128 -> 4x3 MLP over the local-sensing
contract. The paper's controller is `SrcQNetwork`, a 60 -> 120 -> 36 MLP over the whole
Mainz network's state. The rig has never executed the second.
`paper_deploying_self_regulating_cars.md:642` says of the 39-field contract "It is not the
policy's input", which is true of the simulator and false of the device.

**What this task builds.** A second runtime on the device — `DsrcRuntime` — that loads a
DSRC checkpoint, assembles 12 super-segments x 5 features, runs the argmax, and emits one
speed advisory per super-segment. Demonstrated by requiring it to reproduce
`src.rl.src_q.greedy_actions` exactly on the same input, against a frozen golden file, and
by measuring its latency into the existing per-tick `stages` block.

**What this task does not build, and cannot.** A policy for a road anyone can drive to.
`SrcQNetwork`'s input is the entire Mainz network and its weights are Mainz's. The drive
corpus is New Jersey — 123 HERE bodies, 915 road segments, 68 named roads such as
`US-1/Brunswick Pike` (task 71). There is no Mainz observation in the corpus and no
Westfield policy. That gap closes only by defining a drivable network, querying HERE over
it, and training a policy on it, which is a larger piece of work and is **out of scope
here**. Section 9 names its three pieces so the paper can state the boundary.

**The honest outcome.** After this task the paper can say the rig executes the paper's
controller and reproduces the simulator's actions on the simulator's own states. It cannot
say the rig has driven under that controller. That makes the deployment claim narrower and
more precise than the current wording, not stronger. This is stated again in section 10.

### Scope boundary

| In | Out |
|---|---|
| `DsrcRuntime` on the device | Training a policy for a drivable network |
| Vendored `SrcQNetwork` + network identity | Collecting new drives (the corpus is closed) |
| Super-segment feature assembly from `here_feed` | A HERE query shaped to a network |
| Action-equality demonstration vs the simulator | Rewriting the paper's wording |
| Latency, into the existing `stages` block | Task 143's golden encoder vectors |
| A per-super-segment advisory | Task 144's safety filter (called by reference) |

### Decisions, all taken by recommendation and none signed off

| # | Question | Taken |
|---|---|---|
| 1 | Where the runtime lives, and does `SrcQNetwork` get vendored | `policy/dsrc_runtime.py` + `policy/dsrc_contract.py`; **vendor it**, for deploy-integrity reasons, not import reasons |
| 2 | Export format and what refuses a wrong checkpoint | Reuse `<out>.ts` + `<out>.json`; manifest carries a `network_fingerprint` over ordered segment ids, feature names and per-segment speed limits |
| 3 | Partial network coverage | Every segment must be feed-measured or the runtime emits **no action**, with a named outcome; per-segment basis recorded as `measured` / `substituted` / `absent` |
| 4 | The demonstration | Action equality against `greedy_actions`, 0 mismatches required on ~180 recorded simulator states and 20,000 random states; Q-values within 1e-5 |
| 5 | Latency | Two new entries in the per-tick `stages` dict (`segment_assemble`, `dsrc_infer`) plus `STAGE_ORDER`; no new artifact |
| 6 | The advisory | A new `SegmentAdvisory` beside `Advisory`; `AdvisoryDecoder` is untouched; the base is the segment's **static** speed limit, not HERE's `freeFlow` |
| 7 | The closing statement | Section 9, three named pieces |

### Open items flagged for the user

1. **The 60 s decision interval against a 5-30 Hz tick loop.** `DECISION_INTERVAL_S` is
   60.0. The device tick loop runs at the camera rate. Step 6 puts the DSRC decision on its
   own 60 s cadence and holds the last advisory between decisions. An alternative — running
   the DSRC argmax every tick — would be cheap (measured 0.012 ms) but would not be the
   controller that was trained.
2. **Feature 3 is a different quantity on each side.** The simulator computes
   `jam_factor = clip(10*(1 - speed/free_flow), 0, 10)` (`src/sumo/mainz.py:83`). HERE
   returns its own `jamFactor`, which folds in incident data. Step 4 recommends recomputing
   the simulator's formula from HERE's `speed` and `freeFlow` and discarding HERE's own
   `jamFactor`, so the policy sees the quantity it was trained on. This is a choice between
   two defensible readings and is the item most likely to be reversed.
3. **Whether the golden file and the network definition belong in the repository.**
   Both are generated. `specs/transport_golden_frames.json` is the precedent for committing
   a small frozen generated file alongside its generator; sections 4 and 8 follow it.
4. **Whether this task should also produce the Westfield network definition** as a
   no-policy placeholder, so section 9's first piece is half-done. Recommended **no**: a
   network definition with no policy is a file nothing can load, and task 143's rule — a
   guard that has never been seen to fail is not a guard — applies to it.

---

## 1. Facts, each re-verified for this plan

Every number below was measured in this session against the working tree on branch
`mainz-src-port`.

### 1.1 The deployed path

`deployment/jetson/pipeline.py:264-272` is the whole chain, in order:

```
obs_result = self.builder.build(vehicles, gps, time.monotonic(), peers, feed)   # 39 slots
policy_out = self.actor.act(obs_result.encoded)                                 # ActorRuntime
advisory   = self.advisory_decoder.decode(policy_out, obs_result.obs)           # AdvisoryDecoder
```

`sim_contract.local_obs_dim()` returns 39 (33 local fields + 3 cooperation + 3 lane
distribution). `contract_fingerprint()` returns `918ec57cf2f2e1db`. The four action heads
are `desired_speed_bin`, `desired_headway_bin`, `lane_preference`, `merge_mode`.
`run_demo.py:165-169` builds `ActorRuntime` from `policy.bundle`, default
`models/actor_policy`.

### 1.2 The paper's controller

`src/rl/src_q.py` defines `SrcQNetwork(num_segments, num_features, num_actions)` as
`Linear(S*F, 2*S*F) -> ReLU -> Linear(2*S*F, S*A)`, with `forward` flattening the whole
`(segments, features)` state. `greedy_actions` applies `F.softmax` then `argmax` per
segment.

`src/sumo/mainz.py` gives `HERE_FEATURES = ("speed", "free_flow", "jam_factor", "lanes",
"length_km")`, `SPEED_ACTION_FRACTIONS = (0.5, 0.75, 1.0)` and `DECISION_INTERVAL_S = 60.0`.
`data/mainz/mainz_segments.json` holds 12 super-segments over 188 edges.

Measured from the two committed checkpoints:

| file | `stack.0.weight` | `stack.2.weight` | implied |
|---|---|---|---|
| `results/checkpoints/mainz_here_best.pt` | `[120, 60]` | `[36, 120]` | 12 segments x 5 features, 3 actions |
| `results/checkpoints/mainz_src_best.pt` | `[144, 72]` | `[36, 144]` | 12 segments x 6 features, 3 actions |

**Both files are bare `OrderedDict` state dicts.** They carry no metadata: no segment
count, no feature list, no network name, no speed limits. `scripts/train_mainz_src.py:204`
writes `torch.save(model.state_dict(), out / "best.pt")` and nothing else.

`SrcQNetwork` imports `torch`, `torch.nn` and `torch.nn.functional` and nothing else.
Verified: importing `src.rl.src_q` loads no `sumolib`. `src/rl/__init__.py` is a docstring.

### 1.3 Per-segment static values, from the map

Read with `sumolib` from `data/mainz/mainz.net.xml` against `mainz_segments.json`:

| segment | speed limit (km/h) | length (km) | mean lanes |
|---|---|---|---|
| 0 | 60.0 | 7.238 | 1.96 |
| 1 | 60.0 | 1.945 | 2.60 |
| 2 | 60.0 | 7.876 | 1.64 |
| 3 | 60.0 | 4.848 | 2.00 |
| 4 | 60.0 | 2.680 | 1.82 |
| 5 | 60.0 | 9.122 | 2.10 |
| 6 | 60.0 | 3.822 | 3.33 |
| 7 | 60.0 | 4.072 | 2.47 |
| 8 | 60.0 | 2.600 | 2.83 |
| 9 | 60.0 | 2.370 | 5.00 |
| 10 | 60.0 | 3.735 | 1.73 |
| 11 | 60.0 | 0.790 | 3.50 |

Every Mainz edge is limited to 60 km/h, so on this network the fraction set `{0.5, 0.75,
1.0}` is `{30, 45, 60}` km/h on every segment. The per-segment limit is still carried in
the manifest, because `_command` (`src/sumo/mainz.py:316-323`) multiplies the fraction by
**each edge's own** `free_flow_kmh`, and `inverted_tree` is 108 km/h
(`src/sumo/mainz.py:38`).

The 12 segments span 11,444 m east-west and 6,198 m north-south. A circle covering all of
them has a radius of about 6.5 km.

### 1.4 What `here_feed.py` produces, and what stands between it and the policy's input

`parse_flow` returns `list[FlowLink]`. `FlowLink` carries `points`, `speed_mps`,
`free_flow_mps`, `jam_factor`, `confidence`, `traversability`, `length_m`. `HereFeed.offer`
stores every link of one response in a `_Snapshot`; `HereFeed.at(gps, t_mono)` returns a
`FlowReading` holding **one** link, chosen by cross-track then distance.

Seven gaps stand between that and a 12 x 5 state:

1. **One link, not twelve aggregates.** `at()` selects the single link ahead of the
   vehicle. The whole snapshot is in `HereFeed._current`, which is private and has no
   public accessor. A new accessor is needed; `at()` must not change.
2. **No link identity.** `parse_flow` keeps only `location.shape` and `location.length`.
   `location.description` — the source of the names task 71 counted — is discarded. A link
   can therefore only be assigned to a super-segment by geometry against a stored polyline.
3. **No lane count.** HERE v7 flow returns no lane count. `lanes` is the fourth of the
   policy's five features and comes from the map in the simulator
   (`_sumo.edge.getLaneNumber`). On the device it must come from the network definition.
4. **Units differ.** `FlowLink` is metres per second and metres; the policy's row is
   kilometres per hour and kilometres.
5. **`jam_factor` is a different quantity.** See open item 2.
6. **Aggregation rule.** The simulator pools a segment's edges with `_nanmean` for
   `speed`, `free_flow` and `lanes`, and sums `length_m` / 1000 for `length_km`
   (`src/sumo/mainz.py:388-405`). An edge with no vehicles contributes `nan`, not zero.
7. **The gating is per-query, not per-network.** `at()` applies an association radius, a
   heading cone and a downstream horizon, all of which ask "what is ahead of this vehicle".
   A whole-network state asks a different question and must not inherit them. Response
   staleness (`MAX_RESPONSE_AGE_S = 30.0`) does still apply.

### 1.5 The device can run torch

`deployment/jetson/PROGRESS.md:177` records the probe: JetPack 6 / L4T R36.4.7, TRT 10.3,
CPU-only torch, so detection runs through TensorRT and the actor runs on the CPU.
`requirements_jetson.txt` lists torch as already present from JetPack and instructs not to
pip-install it. `ActorRuntime` already imports torch unconditionally and calls
`torch.jit.load`. **The torch-availability risk for this task is therefore already
retired by the existing actor path**; what remains is the cost, measured in section 5.

### 1.6 What is rsynced to the device, and what the integrity hash covers

`scripts/record_deployed_commit.py` documents the deploy step as
`rsync -a --exclude .git ./ jetson:~/dsrc-task40/` — the **whole repository**, so
`src/rl/src_q.py` does land on the device. But `source_tree_sha256` hashes only
`deployment/jetson/`, excluding `models`, `__pycache__`, `.pytest_cache` and `.git`. A
module imported from `src/` would therefore be code the deploy-integrity check does not
cover. This is the argument for decision 1.

### 1.7 Measurements taken for this plan

Run on this machine (arm64, CPython 3.12, `.venv`), with `mainz_here_best.pt` loaded into
`SrcQNetwork(12, 5, 3)`:

| measurement | result |
|---|---|
| parameters | 11,676 |
| `greedy_actions` latency, torch eager, 5,000 calls | mean 0.0123 ms, p50 0.0123 ms, p95 0.0129 ms |
| a numpy mirror (`x@W0+b0`, ReLU, `x@W2+b2`, argmax), 5,000 calls | mean 0.0029 ms, p50 0.0029 ms, p95 0.0030 ms |
| numpy mirror action vs `greedy_actions` action, 20,000 random states | 0 mismatches |
| `argmax(softmax(logits))` vs `argmax(logits)`, 20,000 random states | 0 differences |
| substituting one segment's row with the simulator's zero-fill, 4,000 draws | changed at least one **other** segment's action in 675 draws (16.9%); mean 0.27 of the other 11 changed, maximum 7 |

The last row is the evidence for decision 3. The input is flattened through a dense layer,
so one segment's substituted values reach every output. A per-segment coverage rule cannot
confine the damage to the segment that was missing.

---

## 2. Decision 1 — where the runtime lives, and whether `SrcQNetwork` is vendored

**Taken by recommendation, not user sign-off.**

**Recommendation.** Two new modules under `deployment/jetson/policy/`:

- `dsrc_contract.py` — a vendored twin of `SrcQNetwork`, plus `SPEED_ACTION_FRACTIONS`,
  `DECISION_INTERVAL_S`, `HERE_FEATURES`, and a `network_fingerprint()` function.
- `dsrc_runtime.py` — `DsrcRuntime`, parallel to `ActorRuntime`: load a bundle, refuse a
  mismatch, run the argmax, report latency.

**Vendor it, and the reason is not the one that applied to `sim_contract.py`.** The
existing vendoring exists because `src.rl.actions` pulls in the simulation environment
stack, which must not be on the Jetson. That argument does not apply here: section 1.2
verified that `src.rl.src_q` imports only torch, so the device could import it directly
and nothing would break.

The reason to vendor anyway is section 1.6. The deploy step rsyncs the whole repository but
`source_tree_sha256` hashes only `deployment/jetson/`. A `DsrcRuntime` that imports
`src.rl.src_q` would execute code outside the hashed tree, so `run_demo.py`'s
`_build_provenance` would report a tree hash that says nothing about the network
definition the policy was run against. Vendoring keeps every file the device executes
inside the one tree the integrity check covers.

**The vendored copy is 24 lines** — the class and its `forward`. `greedy_actions`,
`epsilon_actions` and `td_loss` stay in `src/rl/` and are not vendored: only `forward` is
needed at inference, and `greedy_actions` is the reference the demonstration compares
against, which is the one thing that must not be duplicated.

**The equality check is task 143's idiom, not a `pytest.importorskip`.** A new test
`deployment/jetson/tests/test_dsrc_contract.py` asserts that the vendored class produces
the same `state_dict` key set and the same layer shapes as `src.rl.src_q.SrcQNetwork` for
`(12, 5, 3)`, guarded by `pytest.importorskip("src.rl.src_q")` for the device case, **and**
asserts the frozen golden actions from section 4, which run with no sim import at all.
Task 143 establishes why the second is the one that must exist.

---

## 3. Decision 2 — the export format and the mismatch refusal

**Taken by recommendation, not user sign-off.**

**Recommendation.** Reuse `export_policy.py`'s shape exactly: a new
`policy/export_dsrc_policy.py` writing `<out>.ts` (TorchScript) and `<out>.json`
(manifest). The config gains `policy.dsrc_bundle`, default `models/dsrc_policy`.

**Why TorchScript and not the raw state dict.** The state dict alone is what caused the
problem: section 1.2 measured that both committed checkpoints carry no metadata whatever.
A bundle is a module plus a manifest, and the manifest is where the identity goes. Keeping
the same two-file shape also means `resolve_model_path`, the `models/` gitignore and the
deploy sidecar all already work.

**The manifest must carry the network's identity, and a wrong checkpoint must be refused.**
The failure mode `contract_fingerprint` exists to catch — "same dimension, different
fields or scales; silent and total" (`actor_runtime.py:88-102`) — is sharper here, because
the network definition is data, not code:

- `mainz_segments.json` is an ordered list of lists. **Reordering two entries changes
  nothing about the shapes** and sends every action to the wrong road.
- A different 12-segment network with 5 features loads with identical shapes.
- Swapping two entries of `HERE_FEATURES` leaves the input at 60.

So the manifest carries, and `DsrcRuntime.__init__` checks:

| field | refuses |
|---|---|
| `num_segments`, `num_features`, `num_actions` | a shape mismatch, which torch would also catch |
| `feature_names` (ordered) | a reordered or renamed feature set |
| `network_id` (e.g. `mainz`) | a checkpoint for a different named network |
| `network_fingerprint` | everything below, in one hash |
| `segment_ids` (ordered list of ordered edge-id lists) | a reordered or edited segment definition |
| `segment_speed_limits_kmh` (ordered) | a network whose limits moved under the same ids |
| `action_fractions` | a changed action set |
| `trained` | an untrained bundle, shown as a banner rather than refused |
| `source`, `created_utc` | provenance, not a check |

`network_fingerprint()` is a sha256 over a canonical JSON of `network_id`,
`feature_names`, `segment_ids` and `segment_speed_limits_kmh`, truncated to 16 hex
characters, matching `contract_fingerprint`'s form. `DsrcRuntime` computes it from the
loaded network definition and refuses on inequality, with the bundle's value and the
device's value both in the message.

**Unlike `contract_fingerprint`, this check has no grandfather clause.** `actor_runtime.py`
accepts a bundle exported before the fingerprint existed, because refusing every older
bundle would have been a more disruptive rule than the one intended. No DSRC bundle exists
yet, so there is nothing to grandfather, and a missing `network_fingerprint` is a refusal.

**Prove the guard fires before anything rests on it.** Per
`feedback_run_your_guard_against_a_control`, step 8 requires a test that mutates each
manifest field in turn — a swapped pair of `segment_ids`, a changed speed limit, a
reordered `feature_names` — and asserts `DsrcRuntime` raises. A guard that has only ever
passed is not a guard.

---

## 4. Decision 3 — assembling the super-segment state, and partial coverage

**Taken by recommendation, not user sign-off.**

### 4.1 The network definition file

`specs/dsrc_network_mainz.json`, generated by `scripts/export_dsrc_network.py` from
`data/mainz/mainz.net.xml` and `data/mainz/mainz_segments.json`, and committed beside its
generator, the way `specs/transport_golden_frames.json` is. It holds, per super-segment in
order: the segment id, the ordered edge ids, the WGS84 polyline of each edge, the lane
count, the length in metres, and the speed limit in km/h. The polylines are what a HERE
link is matched against; the rest are the static half of the feature row.

The file is a few hundred kilobytes and its inputs are already committed, so this does not
breach "only code lives in the repo": the generator is the artifact of record and the
frozen output exists so the device does not carry a SUMO dependency to re-derive it.

### 4.2 Assembly

New module `deployment/jetson/perception/segment_state.py`, class `SegmentStateBuilder`,
returning a `SegmentStateResult` Pydantic-style dataclass (per
`feedback_use_data_models_not_anonymous_structures`) carrying:

```
state:          np.ndarray            # (12, 5) float32, or None when not complete
segment_basis:  tuple[str, ...]       # one of the three classes below, per segment
outcome:        str                   # a named outcome, never a silent fill
matched_links:  tuple[int, ...]       # how many links matched each segment
response_age_s: float | None
```

Per segment, the three dynamic fields are built from the HERE links whose polylines lie
within a match tolerance of that segment's polyline:

- `speed` = `nanmean` of matched links' `speed_mps * 3.6`
- `free_flow` = `nanmean` of matched links' `free_flow_mps * 3.6`
- `jam_factor` = the simulator's own formula, recomputed from the two above (open item 2)

and the two static fields come from the network definition:

- `lanes` = the definition's lane count, averaged over the segment's edges unweighted,
  matching `_nanmean` in `src/sumo/mainz.py:404`
- `length_km` = the definition's summed edge length / 1000, matching line 405

**The link-to-segment match is by polyline proximity**, reusing `FlowLink.distance_m`,
which already measures to a link's shape rather than to its vertices and whose docstring
records why. A match tolerance is a new constant and a new failure mode; step 3 measures it
against the New Jersey corpus before fixing a value, because that is the only real HERE
geometry this project has.

### 4.3 Partial coverage: the runtime emits no action

**The rule: `DsrcRuntime` produces an action only when every segment's basis is
`measured`. Otherwise it produces a named outcome and no advisory.**

The three classes, mirroring `time_sync.STAGE_BASIS_*` in form and
`perception/provenance.py` in discipline:

| class | meaning |
|---|---|
| `measured` | at least one HERE link matched this segment in this decision's snapshot |
| `substituted` | only the network definition's static fields are known; no link matched |
| `absent` | the network definition has no entry for this segment — a load-time refusal, unreachable at run time |

`substituted` belongs in `provenance.SUBSTITUTED`: it is not evidence about this decision.

**Why not a coverage threshold.** Section 1.7 measured it. Substituting one of 12 segments
with the simulator's own zero-fill changed at least one **other** segment's action in 675
of 4,000 draws (16.9%), and changed up to 7 of the other 11 at once. The input is flattened
through a dense layer, so a substituted row is not confined to its own output. A rule of the
form "act on the segments we observed" would be acting on segments whose action was decided
partly by a fabricated row. `here_feed.py`'s module docstring states the governing
principle: nothing returns a congestion number for a question it could not answer, and a
zero reads as "clear road", not "unknown".

**The simulator fills zeros and the device must not — and the demonstration needs both.**
`MainzEnv.observe` ends in `np.nan_to_num(...)`, so an unobserved edge reaches the policy
as 0.0 during training. That is a property of the training environment, and the replay path
in section 5 must reproduce it exactly or the equality test would compare two different
inputs. So:

- **the replay path reproduces `nan_to_num`**, because it is replaying the simulator;
- **the live path refuses**, because a zero the simulator generated from its own ground
  truth is not the same object as a zero the device generated from an absence.

This split is recorded in the module docstring, because a reader who finds the two paths
disagreeing will otherwise read it as a defect.

### 4.4 Feed access

`HereFeed` gains one method and `at()` is not touched:

```python
def snapshot_links(self, t_mono: float) -> tuple[tuple[FlowLink, ...], FlowReading]
```

returning every usable link of the current snapshot and a `FlowReading` carrying the
outcome and the age provenance, with `link=None`. It applies the response-age and
snapshot-presence checks and **not** the association radius, heading cone or fix checks:
those answer "what is ahead of this vehicle", which a whole-network state does not ask.
Section 1.4 item 7 is the reason, and it goes in the method's docstring.

---

## 5. Decision 4 — the demonstration and its acceptance criterion

**Taken by recommendation, not user sign-off.**

**The claim to be demonstrated: for the same input state, the device's action equals the
simulator's action.** Not "the runtime agrees with itself".

### 5.1 The golden file

`scripts/export_dsrc_golden.py` runs `MainzEnv` under the committed
`mainz_here_best.pt` over `TEST_SEEDS = (16, 17, 18, 19, 20)` and writes
`specs/dsrc_golden_actions.json`:

- `network_fingerprint` and `checkpoint_sha256`, so the file names what produced it;
- every decision's `(12, 5)` float32 state, serialised as exact hex float32 so no decimal
  rounding enters;
- that state's `greedy_actions` output, as 12 integers;
- that state's raw Q-values, as `(12, 3)` hex float32;
- plus 20,000 states drawn from a fixed seed over the plausible feature box, with their
  actions — the same population section 1.7 measured on.

An episode is `duration_s = 2500`, `warmup_s = 300`, `DECISION_INTERVAL_S = 60`, so about
36 decisions per seed and about 180 recorded simulator states across the five.

### 5.2 Acceptance criterion, stated numerically

`deployment/jetson/tests/test_dsrc_runtime.py` loads the bundle through `DsrcRuntime`,
feeds each golden state, and requires:

| quantity | criterion |
|---|---|
| action, on all ~180 recorded simulator states | **exactly equal**, 0 mismatches of ~180 |
| action, on all 20,000 random states | **exactly equal**, 0 mismatches of 20,000 |
| Q-values, TorchScript path, same machine that exported | **exactly equal** (bit-for-bit) |
| Q-values, any path, any machine | **maximum absolute difference below 1e-5** |

Actions are integers, so "exactly equal" is bit-for-bit by construction. Q-values are
float32 through two matrix multiplications, and the BLAS on the Jetson is not the BLAS on
the export machine, so cross-machine bit equality is not available. 1e-5 is
`ActorRuntime._verify_numpy_mirror`'s existing tolerance, and this plan does not introduce
a second convention on the same axis.

**Where "bit for bit" does hold and must be asserted:** the action is the deliverable, and
it is an integer. The criterion above therefore is the bit-for-bit claim, on the quantity
that reaches the driver.

### 5.3 The test must be seen to fail

Per `feedback_run_your_guard_against_a_control`, step 9 requires the test to be run against
three deliberately broken runtimes before it is trusted:

1. a transposed weight load — must fail on Q-values;
2. two segments swapped in the network definition — must fail on the network fingerprint
   at load, before any action is computed;
3. `jam_factor` computed from HERE's own field instead of the simulator's formula — records
   how many of the ~180 actions change, which is the measurement open item 2 needs.

A pass on all three without a prior failure means the test is not measuring what it names.

---

## 6. Decision 5 — latency, recorded beside the existing stages

**Taken by recommendation, not user sign-off.**

**Recommendation.** Two new entries in the per-tick `stages` dict that
`PerceptionPolicyPipeline._stages` already builds, using the existing `StageTiming`:

| stage | what it measures | basis when the decision did not run this tick |
|---|---|---|
| `segment_assemble` | `SegmentStateBuilder` over the snapshot | `absent`, reason `no dsrc decision this tick` |
| `dsrc_infer` | `DsrcRuntime.act` on the assembled state | `absent`, same reason |

Both go into `eval_run.STAGE_ORDER` after `decode`, so the existing ten-stage table becomes
a twelve-stage table and `stage_timings` aggregates them with no change: it already
computes statistics only over values that exist, reports `n`, and counts the bases of the
rest.

**No new artifact.** `eval_run.py`'s `latency` section and `report.md` table pick the new
stages up from `STAGE_ORDER` alone.

**The `absent` basis carries most of the entries, and that is correct, not a gap.** The
DSRC decision runs once per `DECISION_INTERVAL_S = 60.0` while the tick loop runs at the
camera rate — 1 Hz idle, 5 Hz active (`policy/sensing_controller.py:41-42`). At 5 Hz that
is one decision in 300 ticks. `StageTiming.absent(reason=...)` is exactly the construct for
this, and the alternative — a zero — is the defect `time_sync.py:115-127` was written to
prevent.

**Expected magnitude, so the measurement has a prior to be checked against.** On this
machine `greedy_actions` took 0.0123 ms p50 in torch eager (section 1.7). `ActorRuntime`'s
docstring records that TorchScript interpreter dispatch costs about 4 ms with multi-ms
jitter on the Orin's CPU for a network about 3.3 times larger. A TorchScript `DsrcRuntime`
on the Orin should therefore land in the low single-digit milliseconds, against a p95
budget of 200 ms for the whole Jetson path (`eval_run.py:8`). Step 10 measures it rather
than assuming it. **A numpy mirror is not in scope for this task**: section 1.7 measured 0
action mismatches over 20,000 states, so the mirror is available if the Orin measurement
warrants it, and adding it before the measurement would be optimising against a guess.

---

## 7. Decision 6 — what the advisory is

**Taken by recommendation, not user sign-off.**

The two controllers produce different objects and cannot share a decoder:

| | 39-field actor | DSRC |
|---|---|---|
| output | one bin per head, for the ego vehicle | one action per super-segment, 12 of them |
| speed semantics | an offset in m/s against a base speed | a fraction of that segment's own speed limit |
| base | `cooperation.segment_target_speed`, defaulting to 30.0 m/s | the segment's static speed limit from the network definition |
| floor | `max(12.0, base + offset)` | none in the simulator; `setSpeed` is bounded by the car-following model |

**Recommendation.** A new `SegmentAdvisory` dataclass and a `SegmentAdvisoryDecoder` in
`policy/advisory.py`, beside the existing ones. `AdvisoryDecoder` is not modified and
`Advisory` is not extended. The two coexist because they are two controllers, and a single
decoder that branched on which one produced the action would hide exactly the difference
this task exists to surface.

`SegmentAdvisory` carries, per super-segment: the segment id, the action index, the
fraction, the recommended speed in m/s, and the display value in the configured units. It
also carries `ego_segment`, the index of the segment the vehicle's own fix falls on, or
`None` — because the driver is shown one number and that is the one.

**The base is the segment's static speed limit, not HERE's `freeFlow`.** `_command` in
`src/sumo/mainz.py:316-323` multiplies the fraction by `self._static[edge]
["free_flow_kmh"]`, which `_load_static` reads from the map with
`_sumo.lane.getMaxSpeed`. That is the posted limit, not an observation. HERE's `freeFlow`
is a reported quantity that moves. Using it would make the advisory depend on the feed
twice — once through the observation and once through the decode — where the simulator
depends on it once. On Mainz the two coincide at 60 km/h, so this choice is invisible on
the demonstration and would only appear on a different network, which is the reason to fix
it now rather than later.

**Task 144 bounds this output and this plan does not duplicate it.** Section L item 144
records that `apply_safety_layer` has exactly one caller in the repository,
`tests/test_safety_layer.py`, and that it imports `src.envs.base_ctde_env` so it cannot run
on the Jetson as written. Task 144's job is to vendor that filter and call it on the
advisory path. This task's obligation is to leave the call site available: `DsrcRuntime`
returns the unfiltered `SegmentAdvisory`, and the pipeline applies the filter where task
144 puts it. **This plan does not add a floor of its own**, because two floors applied in
sequence is the defect `feedback_a_fix_replaces_what_it_should_add` describes.

---

## 8. The steps

| # | Step | Produces | Verified by |
|---|---|---|---|
| 1 | `scripts/export_dsrc_network.py` + `specs/dsrc_network_mainz.json` | the network definition: 12 ordered segments, polylines, lanes, lengths, speed limits | round-trips to the same lane counts and lengths section 1.3 measured from `sumolib` |
| 2 | `policy/dsrc_contract.py` | vendored `SrcQNetwork`, the constants, `network_fingerprint()` | `test_dsrc_contract.py` asserts shape and key equality against `src.rl.src_q` when importable |
| 3 | Match tolerance, measured | one constant, with the measurement in its comment | run the matcher over the 123 New Jersey HERE bodies; report how many of the 915 segments match at each tolerance, and pick from the curve |
| 4 | `perception/segment_state.py` | `SegmentStateBuilder`, `SegmentStateResult`, the three basis classes | unit tests over synthetic snapshots: full coverage, one segment missing, all missing, a stale response |
| 5 | `HereFeed.snapshot_links` | whole-snapshot access with age provenance | `test_here_feed.py` additions; existing `at()` tests must be unchanged and still pass |
| 6 | `policy/dsrc_runtime.py` | `DsrcRuntime`, the manifest checks, the 60 s cadence | loads the bundle, refuses each mutated manifest |
| 7 | `policy/export_dsrc_policy.py` + the bundle | `models/dsrc_policy.ts` and `.json` from `mainz_here_best.pt` | manifest carries every field in section 3's table |
| 8 | Mismatch guards run against controls | three failing cases, each seen to fail | swapped `segment_ids`, changed speed limit, reordered `feature_names` |
| 9 | `scripts/export_dsrc_golden.py` + `specs/dsrc_golden_actions.json` + `test_dsrc_runtime.py` | the demonstration | section 5.2's table; then section 5.3's three broken runtimes, each seen to fail first |
| 10 | `advisory.py`: `SegmentAdvisory`, `SegmentAdvisoryDecoder` | the per-segment advisory | decode tests over known action vectors and known limits |
| 11 | Pipeline wiring + the two `stages` entries + `STAGE_ORDER` | latency in the existing table | `test_pipeline_smoke.py` addition; `eval_run` report shows twelve stages |
| 12 | Measure on the Orin | the p50 and p95 for `segment_assemble` and `dsrc_infer` | a replay run over one recorded drive, reported against section 6's prior |
| 13 | `plans/implementation_records.md` item 145 | the record, with the numbers | the actual measurements, not this plan's estimates |

Steps 1, 2 and 5 are independent of each other. Step 3 gates step 4. Step 9 gates step 13.

**Step 12 needs the device and nothing in steps 1-11 does.** Every other step runs on this
machine. This is stated because it decides what can proceed while the rig is not to hand.

---

## 9. What still stands between this and a vehicle running the controller on a road

**This is the section the paper quotes.** After every step above is complete, the following
three pieces are untouched, and none of them is small.

1. **A network definition for a drivable road.** `specs/dsrc_network_mainz.json` describes
   Mainz. A vehicle in New Jersey needs the same file for roads it can reach: a partition of
   those roads into super-segments, each with an ordered edge list, a polyline, a lane count,
   a length and a speed limit. Task 71 measured what the corpus contains — 915 road segments
   across 68 named roads, with names such as `US-1/Brunswick Pike` and `RT-27/Nassau St` —
   which is raw material for such a partition and is not itself one. The partition is a
   modelling decision about which stretches of road share a control action, and SRC's own
   Mainz partition came from the Vissim model, not from a traffic feed.

2. **A HERE query shaped to that network rather than to the vehicle.** The existing query
   is a circle centred on the vehicle with a radius of `speed / here_hz * 2`, clamped to
   500 m and 10,000 m (`policy/sensing_controller.py:799`). The 12 Mainz segments span
   11,444 m by 6,198 m, so a circle covering them needs a radius of about 6.5 km — within
   the existing clamp, but only while the vehicle sits near the network's centre. A
   whole-network observation needs a query bounded by the network's extent, issued whether
   or not the vehicle is inside it, and its quota cost is the product of the network's area
   and the decision rate rather than of the vehicle's speed.

3. **A training run on that network.** `SrcQNetwork`'s weights are a function of one
   network's segment count, segment ordering and dynamics. There is no transfer path: the
   first layer is `Linear(S*F, 2*S*F)`, so a different `S` is a different network and the
   same `S` with a different segment ordering is a different function. Training needs that
   network in SUMO — geometry, demand and a calibrated fleet — which is the work
   `data/mainz/` represents for Mainz.

**Only piece 3 spends compute; pieces 1 and 2 are the ones with no default.** Neither has a
defensible recommendation available from inside this repository, which is why they are named
here rather than decided.

---

## 10. Risks

1. **A checkpoint for the wrong network loads silently.** The two committed checkpoints
   differ in input width (60 against 72), so torch refuses that pair. Every other confusion
   is shape-compatible: a reordered `mainz_segments.json`, a different 12-segment network,
   a swapped feature pair. Section 3's `network_fingerprint` is the mitigation and section
   3's last paragraph requires it to be seen failing. **Residual risk:** the fingerprint
   covers the network definition the device loaded, not the network the checkpoint was
   trained on — nothing in `best.pt` records that. Until `train_mainz_src.py` writes the
   fingerprint into the checkpoint, the manifest asserts the pairing rather than proving it,
   and the export step is where a human can still pair the wrong two files. Step 7's manifest
   records `checkpoint_sha256` so the pairing is at least auditable after the fact.

2. **Torch on the Jetson.** Retired as a blocker: section 1.5 verified CPU-only torch is
   present from JetPack and that `ActorRuntime` already depends on it. The live risk is cost,
   not availability. `ActorRuntime`'s docstring records about 4 ms of TorchScript dispatch
   with multi-ms jitter on the Orin for a network 3.3 times larger than this one. Step 12
   measures it. If it exceeds the budget, the numpy mirror is available and section 1.7
   already measured 0 action mismatches over 20,000 states.

3. **A demonstration that only proves the runtime agrees with itself.** The specific way
   this fails: the golden file is generated by the same code path the test then replays, so
   a defect present in both passes. Two mitigations, both required. The golden states come
   from `MainzEnv` through `src.rl.src_q.greedy_actions` — the simulator's own function,
   which is deliberately not vendored — and the runtime under test uses the vendored class,
   so the two paths share only the weights file. And section 5.3 requires three broken
   runtimes to be seen failing before any pass is credited.

4. **The match tolerance has no reference.** Step 3 measures link-to-segment matching
   against New Jersey HERE bodies and Mainz polylines, which are different road networks.
   There is no ground truth for which HERE link belongs to which Mainz super-segment,
   because no HERE response was ever collected over Mainz. The tolerance is therefore
   calibrated on geometry alone. This is a real limitation and belongs in the record, not
   in a caveat that implies it was checked.

5. **`jam_factor` is a modelling choice made without measurement.** Open item 2.
   Section 5.3's third broken runtime measures how many of the ~180 golden actions change
   when HERE's own field is substituted for the simulator's formula. That number is the
   evidence, and it does not exist yet.

6. **This task makes the deployment claim narrower, not stronger.** Stated as an intended
   outcome. The paper plan's contribution section says "A system was built and driven in a
   car, and that is the paper". What the rig drove was the 39-field local-sensing actor,
   which the same plan describes as belonging to the superseded formulation
   (`plans/detours.md`). After this task the rig can execute the paper's controller and
   reproduce the simulator's actions on the simulator's states, and it still will not have
   driven under it, for the reasons in section 9. The claim that becomes available is
   narrower than the current wording and is true; the current wording is wider and is not.
   **Rewriting that wording is out of scope here** and is named in the scope table.

---

## 11. Sign-off

- [ ] Section 9 reviewed: is the three-piece boundary the one the paper should state?
- [ ] Open item 1: the 60 s decision cadence against the tick loop.
- [ ] Open item 2: the simulator's `jam_factor` formula against HERE's own field, with
      step 9's measurement in hand.
- [ ] Open item 3: `specs/dsrc_network_mainz.json` and `specs/dsrc_golden_actions.json`
      committed, or generated on demand.
- [ ] Open item 4: whether a Westfield network definition with no policy belongs in this
      task. Recommended no.
- [ ] Decision 1 confirmed: vendoring `SrcQNetwork` for deploy-integrity coverage rather
      than for import isolation.
- [ ] Decision 3 confirmed: the runtime emits no action on partial coverage.
- [ ] Risk 6 confirmed: a narrower deployment claim is the accepted outcome.
