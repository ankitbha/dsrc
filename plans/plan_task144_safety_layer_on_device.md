# Task 144 — run the safety and etiquette layer on the device

Written 2026-09-12 under `plan_dsrc_rec`. **Every decision below was taken by recommendation,
not by user sign-off.** The open items in the last section of the short version are the ones
that would have been questions under `plan_dsrc`.

## Short version

`apply_safety_layer` has one caller in the repository, `tests/test_safety_layer.py`, and the
paper plan says the advisory is bounded before it reaches the driver. This task makes the
filter run on the device advisory path and records what it does, per rule, per tick.

**The measurement the paper's third open item asks for cannot be obtained from what is in the
repository.** Measured on the four recorded runs in `outputs/task42_usb/` (3,913 tick records,
2026-09-02 and 2026-09-05): **zero of the layer's twelve rules had evidence for their inputs on
any of the 3,913 ticks.** Eleven rules read a field whose provenance is in
`provenance.SUBSTITUTED` on every tick. The twelfth, `target_lane_front_gap`, appears evaluable
on 1,229 ticks only because the 2026-09-02 run tagged that slot `derived` unconditionally; the
value on all 1,229 is `inf`. `leader_gap` is finite on 0 of 3,913 ticks, which is task 63's
rotated camera. So the deliverable is a per-rule **evaluability census**, not a clamp rate, and
the tool must refuse to print a rate for a rule that was never evaluable.

**What a firing rate would have said, if one had been reported.** Running `apply_safety_layer`
over the same 3,913 ticks with the recorded actions clamps on 0 of 3,913. Forcing
`desired_speed_bin` to `slow` and changing nothing else clamps on 3,913 of 3,913, every one for
`low_speed_uncongested`, raising the recommended speed from 20.0 to 22.0 m/s. The reason is
algebraic: the device's `free_flow_speed_mps` is the configured 30.0, so
`decode_speed_bin` gives {slow: 20.0, nominal: 27.0, fast: 30.0} and the etiquette threshold is
`30.0 − 8.0 = 22.0`. The predicate `target < free_flow − 8` reduces to `desired_speed_bin ==
"slow"`. A "clamp rate" from this rig would be the policy's speed-bin distribution under a
safety label.

**Scope.** Making the existing filter run, and measuring it.

**Out of scope, named:** writing new safety or etiquette rules; changing any value in
`SafetyConstraints`; anything that needs a new drive; wiring the gate into `src/sumo/mainz.py`
(decision 7 below says why it is a different task); changing `ObservationBuilder` to feed the
target lane's own gap (decision 3, finding F3 — it changes the policy's input).

### Decisions, all taken by recommendation

| # | Question | Taken |
|---|---|---|
| 1 | Vendor or import `src/safety`? | **Vendor** as `deployment/jetson/policy/safety_gate.py`, guarded by a golden file in the task-143 idiom. Import cost is no longer the reason; the reason is that importing would put the whole `src/` package on the device's `sys.path`, and that the device would then carry two copies of `decode_speed_bin`. |
| 2 | Where in the tick? | Inside `pipeline.step`, after `advisory_decoder.decode` and **before** `set_target_headway`, which keeps taking the **raw** decode so the next observation matches what the policy was trained against. |
| 3 | Substituted inputs? | Three input classes — configured road property (3 fields), evidence-required (6), structurally absent (14). A rule with any evidence-required or structurally-absent input substituted is `RULE_NOT_EVALUABLE` and changes nothing — **except** the lane and merge action, which is withheld when its guards cannot be evaluated. |
| 4 | What is recorded? | A `safety` block in `Tick.to_record()`, beside `advisory`. Estimated **1,518 bytes** on a device-typical tick against a measured 10,125-byte mean record, so **+15.0%**. The estimate is to be checked against a measurement in step 11. |
| 5 | Etiquette on the device? | Yes, evaluated and recorded. All three record `not_evaluable` on this rig and change nothing. Deleting them would delete the record of why the paper's etiquette claim is unsupported on the device. |
| 6 | How is the firing rate measured? | `deployment/jetson/score_safety.py` over `outputs/task42_usb/*/metadata.jsonl`, in `score_shadow.py`'s refuse-before-misleading idiom. It reports the evaluability census first and refuses to print a rate for any rule with zero evaluable ticks. |
| 7 | Gate the simulation too? | **No — separate task.** `mainz.py`'s action is a speed fraction per super-segment written with `setSpeed`; there is no `AVAction`, so `apply_safety_layer` does not typecheck against it, and `setSpeedMode` is never called, so there is no ungated arm to build without changing the speed mode. |

### Open items flagged for the user

1. **The lane advisory becomes permanently withheld on this rig.** Decision 3 fails closed on
   the lane action, and the three rear guards can never be evaluated, so `lane_text` reads
   "Keep lane" on every tick until a rear camera exists. This changes what a driver is shown.
   A flag (`safety.withhold_lane_when_not_evaluable`, default `true`) makes the previous
   behaviour reachable; the default is the decision the user may want reversed.
2. **The etiquette clamp raises the recommended speed**, 20.0 → 22.0 m/s in the counterfactual
   above. The layer as written treats a low advisory on an empty road as the fault to correct.
   Changing the constant is out of scope, but a filter called "safety" that raises a speed
   recommendation is worth knowing about before it runs on a rig.
3. **What the driver sees on `emergency_override` is undefined.** Step 6 recommends withholding
   the speed number. It cannot fire on this rig (no finite `leader_gap`), so the choice is
   untestable against the corpus.
4. **`target_lane_front_gap` is fed the current lane's leader gap** while `left_lane_front_gap`
   and `right_lane_front_gap` are measured and unused (finding F3). Fixing it changes the
   observation the policy reads, so it is out of this task.
5. **The HERE feed's jam factor could supply the density the etiquette rule needs**, but task 28
   deliberately gives the feed no observation field. Using it in the gate would put the gate on
   a quantity the policy never saw. Not done here.
6. **+15.0% on every tick record** for a block that will read `not_evaluable` twelve times over
   on this rig.
7. **The corpus predates the task-63 rotation fix.** No drive in the repository has a leader
   track. This task cannot close the paper's third open item; it can only state what the gate
   would do and why the rig cannot answer.

---

## Evidence

Every number below was measured on 2026-09-12 with `.venv/bin/python` against the repository at
`98032dc`. Nothing was run on the Jetson.

### E1 — the stated facts, re-verified

- `apply_safety_layer` callers: one, `tests/test_safety_layer.py` (14 call sites, 18 tests,
  `18 passed in 0.01s`). The only other matches are three prose references in `plans/`.
- Under `deployment/`: two matches for `safety`, both comments —
  `perception/observation_builder.py:402` and `:505`. No import.
- `AdvisoryDecoder.decode` (`policy/advisory.py:83-108`) bounds nothing but
  `sim_contract.decode_speed_bin(bin, base_speed, min_contextual_speed_mps)`, and `base_speed`
  is `obs["cooperation"]["segment_target_speed"]`.

### E2 — the import blocker is no longer a dependency blocker

`src/safety/safety_layer.py` imports `src.envs.base_ctde_env` and `src.envs.wrappers`. Both are
now pure standard-library typing and literal tables: `base_ctde_env` is `TypedDict`/`TypeAlias`
declarations plus an ABC, `wrappers` is bin tuples and three decode functions. `import
src.safety` completes in 0.01 s and pulls in neither `torch` nor `highway_env`. The claim "it
cannot be imported on the device as written" is true for a different reason, given in decision 1.

### E3 — the corpus

`outputs/` is gitignored (`.gitignore:5`) and present on disk, which is this project's standing
pattern for harness data. Four runs:

| file | tick records | ticks with finite `leader_gap` |
|---|---|---|
| `outputs/task42_usb/baseline_run_20260902_183446/metadata.jsonl` | 1,229 | 0 |
| `outputs/task42_usb/run_20260905_113757/metadata.jsonl` | 899 | 0 |
| `outputs/task42_usb/run_20260905_114546/metadata.jsonl` | 900 | 0 |
| `outputs/task42_usb/run_20260905_115156/metadata.jsonl` | 885 | 0 |
| **total** | **3,913** | **0** |

`ego_speed`: n=3,913, min 0.00, median 0.00, max 7.33 m/s — these are bench and low-speed runs,
not road drives. The action is the same tuple on all 3,913 ticks (`fast`, `normal`,
`prefer_left_if_safe`, `create_gap`) and `advisory.recommended_speed_mps` is 30.0 on all 3,913,
which is an untrained bundle. `obs_diagnostics.density_veh_per_km` is 0.0 and
`obs_diagnostics.last_detection_age_s` is `null` on all 3,913 ticks: the builder has never seen
an in-range detection in any recorded run.

### E4 — the evaluability census (the deliverable's headline)

A rule is evaluable on a tick when every input it requires as evidence carries a provenance
class outside `provenance.SUBSTITUTED`.

| rule | evaluable ticks of 3,913 | the input that blocked it |
|---|---|---|
| `low_speed_uncongested` | 0 | `cooperation.segment_target_speed` = `fallback_neutral` |
| `all_lane_low_speed_occupancy` | 0 | `nearby_av_count`, `nearby_av_mean_speed`, `downstream_congestion_estimate`, all `fallback_neutral` |
| `passing_lane_slow_hold` | 0 | `ego_lane` = `static_config`, `local_mean_speed_bin` = `fallback_neutral` |
| `lane_change_dwell` | 0 | `time_since_last_lane_change` = `fallback_neutral` |
| `lane_changes_per_km` | 0 | `lane_changes_last_km` = `fallback_neutral` |
| `target_lane_missing` | 0 | `ego_lane` = `static_config` |
| `target_lane_front_gap` | 1,229 — see below | `target_lane_front_gap` = `fallback_neutral` on the other 2,684 |
| `target_lane_rear_gap` | 0 | `target_lane_rear_gap` = `fallback_neutral` |
| `target_lane_front_ttc` | 0 | no observation field carries any lane's relative speed but the leader's |
| `target_lane_rear_ttc` | 0 | as above, and `target_lane_rear_gap` = `fallback_neutral` |
| `target_lane_rear_braking` | 0 | `target_lane_rear_required_decel` = `fallback_neutral` |
| `forward_ttc` | 0 | `leader_gap` and `leader_relative_speed`, both `fallback_neutral` |

The 1,229 are the whole 2026-09-02 run, where `target_lane_front_gap` was tagged
`SOURCE_DERIVED` unconditionally. The builder now sets it to `src["leader_gap"]`
(`observation_builder.py:484`). **On all 1,229 of those ticks the recorded value is `inf`.** So
the honest count is zero evaluable ticks for all twelve rules, and one stale provenance tag,
recorded as finding F4 below.

### E5 — what a firing rate would have reported

`apply_safety_layer` run over the 3,913 ticks with the mapping in decision 3:

| arm | ticks the speed was clamped | mean change | rules that produced an event |
|---|---|---|---|
| recorded actions (`fast` on every tick) | 0 of 3,913 (0.0%) | — | none |
| `desired_speed_bin` forced to `nominal` | 0 of 3,913 (0.0%) | — | none |
| `desired_speed_bin` forced to `slow` | 3,913 of 3,913 (100.0%) | **+2.00 m/s (raised)** | `etiquette_blocked_action: low_speed_uncongested`, 3,913 times |

`emergency_override` fires on 0 ticks in all three arms, and the lane action is never masked in
any arm: the seven lane guards run on all 3,913 ticks (`lane_preference` is
`prefer_left_if_safe` and `merge_mode` is `create_gap`, so `lane_action` is not `None`) and pass
every time, because `last_lane_change_time_s` is `None` (dwell `inf`), `target_lane_exists` is
the substituted `True`, all three gaps are `inf` and `target_lane_rear_required_decel` is 0.0.
Seven guards passed 3,913 times on substituted constants.

### E6 — record size

`MetadataLogger.write` uses `json.dumps(record, default=_json_default)` with default separators.
Measured over `run_20260905_113757`: 899 tick records, mean 10,125 B, p50 10,125 B, p95 10,133 B.
Existing sub-blocks for comparison: `sensing` 2,699 B, `field_sources` 1,722 B,
`obs_diagnostics` 1,415 B, `obs` 1,240 B, `advisory` 209 B.

A drafted `safety` block carrying all twelve rules with a device-typical verdict distribution
(eleven `not_evaluable` with their `missing` lists, one `quiet` with evidence) measures
**1,518 B**, which is **+15.0%** on a 10,125 B record. A variant carrying evidence only for
evaluable rules and a flat name→missing map for the rest measures 1,103 B (+10.9%).
**Recommendation: the full 1,518 B form**, because the flat variant drops each substituted
field's provenance class, which is the one thing a reader of this block needs. Step 11 measures
the real figure and the plan is wrong if it lands outside 1,300–1,800 B.

### E7 — findings recorded while reading, none of them the task

- **F1.** `SafetyContext` has 23 fields. Three are read by nothing in `safety_layer.py`:
  `follower_gap_m`, `follower_relative_speed_mps`, `near_merge`. So the absent rear sensor
  blocks three fields that a rule actually reads (`target_lane_rear_gap_m`,
  `target_lane_rear_relative_speed_mps`, `target_lane_rear_required_decel_mps2`), not five.
- **F2.** `_lane_preference_safe` (`safety_layer.py:201-218`) has no callers. It duplicates the
  `elif` chain inside `apply_safety_layer`, and the two can drift.
- **F3.** `ObservationBuilder` sets `target_lane_front_gap` to the **current** lane's
  `leader_gap` (`observation_builder.py:438`), while `left_lane_front_gap` and
  `right_lane_front_gap` are measured from lateral offsets and reach no target-lane slot. For a
  `prefer_left_if_safe` advisory the measured quantity exists and the wrong one is used.
- **F4.** `target_lane_front_gap` carries `derived` on 1,229 archived ticks whose value is `inf`.
  A provenance tag on an archived log can be wrong for the contract that log was written under.
- **F5.** The observation vector has six rear slots (`follower_gap`, `follower_relative_speed`,
  `left_lane_rear_gap`, `right_lane_rear_gap`, `target_lane_rear_gap`,
  `target_lane_rear_required_decel`); `SafetyContext` has five rear fields. They are not the same
  five: the observation has no `target_lane_rear_relative_speed` and `SafetyContext` has no
  left/right rear gap. The paper plan's "all six of the absent ones rear-facing" counts
  observation slots and is correct as written.

---

## The decisions in full

### Decision 1 — vendor, do not import

`policy/sim_contract.py` is the precedent and its docstring states the rule: the Jetson must not
import the simulation stack. E2 shows the dependency half of that reason has expired —
`src.envs.wrappers` is now three functions and four tuples. Two reasons remain, and both are
about drift rather than weight.

1. `run_demo.py:38-39` puts only `JETSON_DIR` on `sys.path`. The repository root is added only by
   `deployment/jetson/tests/conftest.py`, for tests. Importing `src.safety` from the runtime
   means adding the repository root, which puts every module under `src/` within reach of every
   device module, and **no test forbids a `src.` import under `deployment/`** — checked. The
   next `from src.sumo import ...` would work and nothing would say so.
2. `src/envs/wrappers.decode_speed_bin` and `sim_contract.decode_speed_bin` are already two
   copies of the same function with the same constants. Importing `src.safety` puts the second
   copy on the device, reachable from the gate, while `AdvisoryDecoder` uses the first. Two
   decoders on one device is the failure mode, not a hypothetical.

**Taken:** vendor into `deployment/jetson/policy/safety_gate.py`. The vendored module calls
`sim_contract.decode_speed_bin` and `sim_contract.decode_headway_bin` rather than carrying its
own, so the device keeps exactly one decoder. The vendored `SafetyConstraints` is a frozen
dataclass with the same 16 fields and the same values.

**How this one avoids the drift task 143 is fixing.** Task 143's finding is that a check written
as "compare the vendored copy against the original" becomes vacuous when the original is
deleted: `test_sim_contract.py` reports `1 skipped` and says nothing. The guard here is a file,
not a comparison between two copies:

- `specs/safety_contract_golden.json` holds the `SafetyConstraints` field names and values, the
  `SafetyContext` field names in order with their defaults, and a short hash over both.
- `deployment/jetson/tests/test_safety_contract.py` asserts the **vendored** copy matches the
  file, unconditionally, with no `importorskip`. It runs everywhere and cannot go vacuous.
- `tests/test_safety_contract_matches_golden.py` asserts `src/safety/` matches the same file.
  If `src/safety/` is ever deleted, this test goes away and the one that matters does not.
- A change on either side must edit the golden file, which appears in a diff.
- **Before anything relies on it:** mutate one constant in the vendored copy, run the golden
  test, see it fail, revert. A guard that has never been seen to fail is not a guard. This is
  step 4 and it is a gate on the rest of the work.

### Decision 2 — inside `pipeline.step`, feeding back the raw headway

`pipeline.py`'s docstring says `run_demo.py`, `replay_demo.py` and `bench_latency.py` all drive
the same object so live, replay and bench numbers are directly comparable. Putting the gate in
`run_demo.py` would leave it out of replay and out of the bench, which means the bench's latency
would describe a system without the gate and `replay_demo.py` could not exercise it. The gate's
output also belongs in `Tick.to_record()`, which `pipeline.py` owns, and `pipeline.step` already
splits `infer_ms` from `decode_ms`, so a `gate_ms` fits the same accounting.

**Taken:** in `pipeline.step`, between line 271 (`advisory_decoder.decode`) and line 273
(`set_target_headway`).

**`set_target_headway` keeps taking the raw `advisory.headway_target_s`.** The observation's
`target_headway_s` is defined as the previous action's commanded headway, and task 87 records
that the simulator stores the raw `desired_headway_bin`. The gate can add
`merge_gap_headway_bonus_s` to the headway; feeding the bounded value back would make the
device's observation differ from the training distribution in a way nothing measures. The gate
bounds what the driver is shown, not what the policy reads.

**What the record must carry.** `advisory.recommended_speed_mps` continues to mean *the number
shown to the driver*, because `eval_run.py:966,1171,2185`, `replay_demo.py:145` and
`transport/messages.py:1159` all read it as that. The `safety` block carries both values
explicitly:

```
"safety": {
  "proposed": {"speed_mps": …, "headway_s": …, "lane_action": …},
  "bounded":  {"speed_mps": …, "headway_s": …, "lane_action": …},
  "delta_speed_mps": …,
  "emergency_override": false,
  "lane_withheld": null | "not_evaluable",
  "evaluable": 1, "not_evaluable": 11,
  "rules": { "<rule>": {"status": …, "missing": […], …evidence} , … × 12 }
}
```

An offline reader that wants the ungated series reads `safety.proposed.speed_mps`; one that
wants what the driver saw reads `advisory.recommended_speed_mps`, which equals
`safety.bounded.speed_mps` by construction. Step 9 asserts that identity.

### Decision 3 — the per-field policy for substituted inputs

This is the crux. A filter fed `fallback_neutral` inputs either never fires, which makes the
measured rate a statement about the fallbacks, or fires on a constant, which is worse. E5 shows
both outcomes on the same corpus, one per arm.

**The rule: a rule's verdict is decided by its inputs' provenance before its predicate runs.**
Three input classes, and every `SafetyContext` field is in exactly one.

**(A) Configured road property — 3 fields.** Accepted as given; never makes a rule
`not_evaluable`. These are operator statements about the road, not measurements this tick owes.

`time_s` (the device's monotonic clock), `free_flow_speed_mps` (`observation.free_flow_speed_mps`
via the cooperation block), `min_contextual_speed_mps` (`policy.min_contextual_speed_mps`).

**(B) Evidence-required — 6 fields.** A rule reading one of these is `RULE_NOT_EVALUABLE` when
that field's provenance is in `provenance.SUBSTITUTED`.

`ego_speed_mps`, `leader_gap_m`, `leader_relative_speed_mps`, `local_density_veh_per_km`,
`local_mean_speed_mps`, `target_lane_front_gap_m`.

*One extra test, for `local_density_veh_per_km` only.* `SOURCE_DERIVED_EMPTY` is deliberately
outside `SUBSTITUTED` — `provenance.py:66-70` gives the reason, and removing it would delete the
disagreement rule. But a density of 0.0 tagged `derived_empty` from a camera that has never
produced a track is a blind camera, not an empty road, and the project already has the
instrument that separates them: `obs_diagnostics.last_detection_age_s`, whose docstring calls it
"the only bound available on whether an empty `local_density_bin` is an empty road or a blind
camera". So: `derived_empty` counts as evidence only when `last_detection_age_s` is not `None`
and is within the builder's `gps_stale_after_s` × 2 (`BuilderConfig.gps_stale_after_s`, merged
from `config["gps"]["stale_after_s"]`). On the recorded corpus it is `None` on all
3,913 ticks, so the density is not evidence on any of them.

**(C) Structurally absent — 14 fields.** No sensor exists on the rig. `not_evaluable`
unconditionally, and the record names the field, not a substituted number.

Rear, no sensor at all (5): `follower_gap_m`, `follower_relative_speed_mps`,
`target_lane_rear_gap_m`, `target_lane_rear_relative_speed_mps`,
`target_lane_rear_required_decel_mps2`.
No lane detection (3): `target_lane_exists`, `in_passing_lane`, and
`target_lane_front_relative_speed_mps` (no observation field carries a non-leader relative speed).
No cooperating peers (2): `all_lanes_av_occupied`, `av_mean_speed_mps`.
No feed-to-observation path (1): `downstream_congested` — `downstream_congestion_estimate` is a
literal 0.0 in both branches of the builder; `observation_builder.py:487-491` says so.
No map matching (3): `near_merge`, `merge_conflict_gap_m`, `merge_conflict_relative_speed_mps`.

Plus the two `SafetyState` counters the device cannot advance, treated as class (C) for the two
rules that read them: `last_lane_change_time_s` and `lane_changes_last_km` / 
`lane_change_distances_m`. There is no lane-change detection on the device;
`time_since_last_lane_change` is `fallback_neutral` and `lane_changes_last_km` is 0 on every
tick.

3 + 6 + 14 = 23, the whole of `SafetyContext`.

**What a `not_evaluable` rule does.** Nothing. It is recorded and the advisory is unchanged.
**With one exception, and it is deliberate:** the lane and merge action. There is no rear sensor,
so `target_lane_rear_gap`, `target_lane_rear_ttc` and `target_lane_rear_braking` are
`not_evaluable` on every tick this rig will ever produce. Showing a driver "Prepare left (if
safe)" with no evidence about the lane being entered is the one case where leaving the advisory
alone is not the conservative option, and it is the case the paper's sentence about bounding the
advisory is most directly about.

So: **when any rule guarding the lane action is `not_evaluable`, the lane action is withheld**,
`lane_text` falls back to "Keep lane", and the record carries `"lane_withheld":
"not_evaluable"` with the rules that could not be evaluated named in `rules`.

The consequence is stated plainly, and is open item 1: on this rig the lane advisory is withheld
on every tick, permanently, until a rear camera exists. `config.yaml` gains
`safety.withhold_lane_when_not_evaluable`, default `true`, gating the entry to the changed
behaviour — the project's stated rollback mechanism. Setting it `false` restores exactly the
current display and still records every verdict, which is also what makes a gated/ungated
comparison possible on the device side.

**Vocabulary.** `RULE_FIRED` / `RULE_QUIET` / `RULE_NOT_EVALUABLE` and the `missing` tuple come
from `policy/sensing_controller.py:164-166` and `RuleCheck`; the substitution partition comes
from `perception/provenance.py`. No fourth word is introduced. One point needs care:
`apply_safety_layer`'s lane checks are an `elif` chain, so only the first failing guard produces
an event. The record must not confuse a short-circuited guard with an unevaluable one, so **each
of the twelve rules is evaluated independently as a total predicate for the record**, while the
decision itself keeps the `elif` chain. The three-state word is therefore decided by input
provenance and by the predicate, never by chain position. Step 9 asserts the two agree: the
first `fired` guard in chain order must be the one whose reason the decision carries.

### Decision 4 — what is recorded, where, and what it costs

The `safety` block goes in `Tick.to_record()` beside `advisory`, written by the pipeline. It is
not attached in `run_demo.py` the way `sensing`, `thermal` and `failures` are, because those are
produced outside the pipeline and this is produced inside it.

The per-rule shape copies `sensing.attribution.rules` exactly: `{"status": …}`, plus `missing`
only on `not_evaluable`, plus the compared value and threshold on `fired` and `quiet`, plus each
input's provenance class. `RuleCheck.to_record`'s reasoning applies unchanged — a threshold
compared only against a value that produced it can drift with every test still green, so the
record states both.

Size: **1,518 B estimated, +15.0% on a measured 10,125 B mean tick record** (E6). Step 11
measures the real figure on a replay and the plan is wrong if it lands outside 1,300–1,800 B.
`eval_run.py` gains a `safety` rollup beside its `advisory` one: per-rule evaluable counts,
per-rule fired counts, the clamp-delta distribution, and the count of ticks where the lane
action was withheld. It is reported and **not gated**, for the reason `eval_run.py:15` already
gives about an untrained bundle.

### Decision 5 — the etiquette rules run, and record `not_evaluable`

`is_low_speed_uncongested` reads a density the device derives from a camera that has never seen
a vehicle. Under decision 3 that density is not evidence, so the rule is `not_evaluable` and
changes nothing. `is_all_lane_low_speed_occupancy` needs peers (V2V is `enabled: false` and the
rig is one vehicle) and `is_passing_lane_slow_hold` needs a lane index the device sets from
`observation.assumed_lane`. All three are `not_evaluable` on every tick of every recorded run.

They are kept rather than deleted because the block that says "this rule could not be evaluated,
and here is the input that was missing" is the evidence for the paper's own statement that the
etiquette layer is unexercised. Deleting them replaces that evidence with silence.

Note for the record: `is_passing_lane_slow_hold` appends a diagnostic and changes no output even
when it fires. It is a counter, not a filter.

### Decision 6 — how the firing rate is measured, and what it cannot support

`~/dsrc_logs` is on the Jetson and unreachable. The measurement runs against
`outputs/task42_usb/*/metadata.jsonl`, 3,913 tick records on this laptop. Every `SafetyContext`
input is reconstructable from a tick record's `obs`, `obs_diagnostics`, `field_sources` and
`action`, and the gate is a pure function of them, so no video and no device are needed. This is
the same reconstruction E4 and E5 were measured with.

`deployment/jetson/score_safety.py`, following `score_shadow.py`:

- **Census first, rate second.** It prints the per-rule evaluable count before any rate, and for
  any rule whose evaluable count is zero it prints `not evaluable on any tick` and **no
  percentage**. A rule evaluable on 0 ticks must never appear as "0% firing rate".
- **It refuses before it misleads.** On a log that already carries a `safety` block it must
  reproduce that block exactly from the recorded inputs before reporting anything, which is
  `score_shadow.py`'s incumbent-replay gate. On a log with no `safety` block it labels every
  number **counterfactual** in the output itself, not only in prose.
- **It reports the counterfactual arms of E5** — recorded action, and each speed bin forced —
  because the difference between 0.0% and 100.0% on the same inputs is the finding.

**What this measurement can support:** that zero of twelve rules had evidence on any of 3,913
recorded ticks; which input blocked each rule; that `low_speed_uncongested` reduces to
`desired_speed_bin == "slow"` under a configured free-flow speed; the magnitude and sign of the
clamp when it does fire (+2.00 m/s, raising); the gate's own latency, measured on this laptop
and again on the device when one is next available.

**What it cannot support:** any statement about how often local safety clamps an advisory on a
road. That needs a drive with the gate running, a leader track, and the task-63 rotation fix,
and the repository has none of the three. The paper's third open item stays open, and this task
should say so in its own record rather than reporting a rate that answers a different question.

### Decision 7 — the simulation side is a separate task

`src/sumo/mainz.py:317-330` writes the advisory as `setSpeed(vehicle, fraction ×
free_flow_kmh/3.6)`, one speed per super-segment per decision interval. There is no `AVAction`:
no speed bin, no headway bin, no lane preference, no merge mode. `apply_safety_layer`'s first
parameter is an `AVAction` and its lane branch is the majority of its logic, so wiring it into
Mainz is not plumbing — it needs an action adapter that does not exist and would have nothing to
feed eight of the twelve rules.

`setSpeedMode` is never called anywhere in `src/` or `scripts/`, so SUMO's car-following bound is
always on. The paper's gated/ungated/no-control comparison on Mainz is therefore a question
about `setSpeedMode` and `release()`, not about `apply_safety_layer` — which is the distinction
task 87 and the paper plan already draw when they say the gate is real on the device and
implicit in the simulation, and that the paper cannot claim the two are the same rule.

**Taken: out of this task.** It should be its own task, and what it needs is: a third arm with
`setSpeedMode` relaxed so "ungated" means something, a control arm with `release()`, and a
statement of what SUMO's bound does that the explicit layer does not. Decision 3's flag gives
the device side its own gated/ungated pair in the meantime.

---

## Steps

| # | Step | Produces | Done when |
|---|---|---|---|
| 1 | Freeze `specs/safety_contract_golden.json`: `SafetyConstraints` names and values, `SafetyContext` names in order with defaults, and a hash over both. | the golden file | The file exists and both hashes are computed from `src/safety/` as it stands. |
| 2 | Write `deployment/jetson/tests/test_safety_contract.py`, no `importorskip`, asserting the golden file against the (not yet written) vendored module; and `tests/test_safety_contract_matches_golden.py` asserting it against `src/safety/`. | two tests | The `src/safety/` one passes; the vendored one fails for want of a module. |
| 3 | Vendor `deployment/jetson/policy/safety_gate.py`: `SafetyConstraints`, `SafetyContext`, `SafetyDecision`, the twelve rules as **total independent predicates**, and the decision path. It calls `sim_contract.decode_speed_bin`/`decode_headway_bin`; it defines no decoder of its own. | the gate module | Step 2's vendored test passes. |
| 4 | **Gate on the rest of the work.** Mutate one constant in the vendored copy, run both golden tests, see each fail, revert. Record which line was mutated and what the failure said. | a line in the record | Both tests were seen to fail and then to pass again. |
| 5 | Implement the three input classes of decision 3 as a `SafetyInputs` model built from `ObservationResult` (`obs`, `diagnostics`, `field_sources`) plus config, returning each field's value and its provenance class. Use a data model, not a dict of dicts. | `safety_gate.SafetyInputs` | Unit tests cover one field from each of (A), (B), (C), and the `derived_empty` + `last_detection_age_s` test. |
| 6 | Wire the gate into `pipeline.step` between lines 271 and 273. `set_target_headway` keeps the raw headway. Add `gate_ms` to `stage_ms` and `PipelineStats`. Decide the `emergency_override` display: withhold the speed number (open item 3). | the gate runs | `test_pipeline_smoke.py` passes and shows a `safety` block. |
| 7 | Add `safety.withhold_lane_when_not_evaluable: true` to `config.yaml`, thread it into the gate, and make `run_demo.build_components` pass it. | the flag | A test runs both settings and asserts the lane action differs and the recorded rules do not. |
| 8 | Add the `safety` block to `Tick.to_record()` in the shape from decision 2. | the record | A replay tick carries all twelve rules. |
| 9 | Assert the two identities: `advisory.recommended_speed_mps == safety.bounded.speed_mps`, and the first `fired` guard in chain order is the one whose reason the decision carries. | two invariant tests | Both pass, and each was seen to fail against a deliberately broken gate. |
| 10 | Write `deployment/jetson/score_safety.py` per decision 6: census first, refuse a rate for a zero-evaluable rule, label counterfactual output as counterfactual, reproduce an incumbent `safety` block exactly when one exists. | the tool | Run over the four `outputs/task42_usb` runs; it reproduces the E4 and E5 tables. |
| 11 | Measure the real `safety` block size on a replay and compare against the 1,518 B estimate. Measure `gate_ms` p50/p95 on this laptop. | two numbers | Recorded. If the size is outside 1,300–1,800 B, the estimate was wrong and the record says by how much. |
| 12 | Add the `safety` rollup to `eval_run.py` beside `advisory`, reported and not gated. | rollup | `eval_run.py` on an `outputs/task42_usb` run prints it. |
| 13 | Update `ARCHITECTURE.md`: a new subsection under section 6 for the safety contract's vendoring and its golden file, and a row in section 8's roadmap saying which rules a rear camera makes evaluable (`target_lane_rear_gap`, `target_lane_rear_ttc`, `target_lane_rear_braking`). | docs | Written, with no file:line citations in anything paper-facing. |
| 14 | Write the task 144 record in `plans/implementation_records.md`: the census, the two arms of E5, the algebraic reduction, findings F1–F5, and the plain statement that the paper's third open item is not closed. | the record | Written. |
| 15 | Run the scoped test set: `deployment/jetson/tests/`, `tests/test_safety_layer.py`, and the two new contract tests. Not the wider suite. | green | All pass. |

## Sign-off

- [ ] The seven decisions above were taken by recommendation. Confirm decision 3's fail-closed
      lane behaviour and its default, which is the one that changes what a driver sees.
- [ ] Confirm open items 1, 2 and 3 — the withheld lane advisory, the etiquette clamp raising the
      recommended speed, and the `emergency_override` display.
- [ ] Confirm that a census reporting "zero of twelve rules evaluable on 3,913 ticks" is the
      deliverable, and that no clamp rate will be published from this corpus.
- [ ] Confirm decision 7, that the Mainz gated/ungated/no-control comparison is a separate task,
      and that it is about `setSpeedMode` rather than about `apply_safety_layer`.
- [ ] Confirm findings F2 (`_lane_preference_safe` is dead) and F3 (`target_lane_front_gap` is
      fed the current lane's gap) are recorded and left for separate tasks, not fixed here.
