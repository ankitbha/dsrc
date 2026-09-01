# Task 36 — Per-tick field provenance and missingness

> Written by `plan_dsrc_rec`: every decision below was taken by recommendation,
> without putting the question to the user. The decisions table says what was
> chosen and why; **none of it is user-approved**, and the "Open items flagged
> for the user" section holds every point where the recommendation was weak or
> where the code contradicted the first reading. This plan is the fixed target
> a validator audits the implementation against.

## The short version

Task 36 (plans/task_list.md:1244) asks for per-tick field provenance and
missingness. **Both halves already exist and both are wrong in a specific,
measurable way.** `field_sources` has recorded a provenance class per
observation field since before task 28 (observation_builder.py:5-12, :53), it
is written to every tick record (pipeline.py:138), and `missingness` is
computed from it every tick (observation_builder.py:447). What this task adds
is three things, each an instance of the project's recurring defect class — a
record that cannot distinguish failure from success:

1. **Provenance never reaches the controller.** `observation_builder.py:216`
   writes `src["ego_acceleration"] = "derived" if accel_derived else
   "fallback_neutral"`, and `inputs_from` (sensing_loop.py:127) copies only
   the float. So the controller's event rule reports `quiet` at 0.0 where the
   truth is "no value was computed". Task 34 built the three-state vocabulary
   and pinned `not_evaluable` precisely because this task would make it the
   live path (plan_task34.md:384-391); that branch is unreachable today
   (sensing_controller.py:464-466 is reached only by a hand-built `Inputs`).
2. **Two fields are mislabelled at the source, and one of the two is not the
   field task 34 named.** `local_density_bin` is `derived` unconditionally
   (observation_builder.py:413), so a bin of 0 from an empty detection set is
   indistinguishable from a measured light road — and under the shipped config
   that is the disagreement rule's *only* firing condition (proved below).
   Separately, and not previously recorded anywhere: **`ego_acceleration` is
   reported as `derived` for the whole of a GPS dropout of any length**,
   because `_speed_slope` (observation_builder.py:463-488) never asks how old
   its own window is. Verified by running: after 60 s with no fresh fix, the
   builder still reports `ego_acceleration = 0.000, source = derived`.
3. **The map covers 33 of the 39 encoded slots, and `missingness` divides by
   33.** Verified by running: `set(field_sources) == set(LOCAL_OBS_FIELDS)`,
   `len == 33`, `encoded.shape == (39,)`. The six uncovered slots are
   `cooperation.{segment_target_speed, merge_pressure,
   downstream_congestion_estimate}` and `nearby_av_lane_distribution.{0,1,2}`
   — all six neutral-filled on a lone instrumented car, so the metric
   under-reports exactly where this vehicle lives.

**A correction to the brief's framing, grounded in the code.** "A dead
accelerometer" is not the failure mode. `Inputs.ego_acceleration` is a
least-squares slope of *GPS speed*; the phone's 50 Hz IMU stream is drained,
counted, and read by nothing — `phone_link.py:173-184` says so outright
("`Inputs.ego_acceleration` still comes from a finite difference of 1 Hz GPS
speed, so the `EVENT_ACCEL_MPS2` trigger has never seen an IMU sample";
verified by grep — no consumer of `PhoneLink.imu` exists outside that class).
The condition that substitutes a neutral is therefore **a stale or absent GPS
fix, or a speed window shorter than 0.3 s**, which is a tunnel, an urban
canyon, a cold start, and the opening ticks of every drive.

The plan, in one paragraph: (1) give the provenance vocabulary one home
(`perception/provenance.py`), close it, and extend `field_sources` to all 39
encoder slots so `missingness` divides by the whole vector; (2) fix the two
mislabels at source — a slope fitted over an expired window is not `derived`,
and a bin derived from an absence of detections gets its own class
`derived_empty`; (3) carry provenance into `Inputs` through `inputs_from`,
which nulls a *substituted* value so task 34's pinned `not_evaluable` branch
becomes the live path with no change to `decide`'s structure; (4) publish the
result on the three surfaces that exist — `SensingLoop.to_record`,
`score_shadow`, and `eval_run`'s `report.md` — because task 33's own
experiment found the measurement present and the surface absent
(task_list.md:1110-1112). The wire is untouched and the phone is untouched.

**Scope boundary.** In: new `deployment/jetson/perception/provenance.py`;
`perception/observation_builder.py`; one helper in `policy/sim_contract.py`
(`encoded_slot_names()`, additive — `contract_fingerprint` hashes only
`LOCAL_OBS_FIELDS` and `FIELD_SCALES`, sim_contract.py:253-258, so it does not
move); `policy/sensing_controller.py` (four `Inputs` fields, evidence keys);
`policy/sensing_loop.py` (`inputs_from`, one counter); `score_shadow.py` (one
refusal, one generalisation, one rollup); `eval_run.py` (the `observation`
block and its report lines); `perception/feed_fusion.py` (re-export
`SOURCE_FEED` from its new home); tests; pinned mutations in
`scripts/remutate.py`; the provenance table in `ARCHITECTURE.md:108-129`. Out:
`pipeline.py` and `run_demo.py` (both already carry `field_sources` through
unchanged — verified); the wire, `specs/`, and every codec; the phone (zero
Kotlin); wiring the IMU (named, not done); rendering beyond `eval_run`
(task 39); simulator-side provenance parity (task 47); and any HERE call.

**Open decisions flagged for the user** (details at the bottom): the
disagreement over-report is *named*, not removed, and the reason is that
removing it deletes the rule; the stale-window fix is the one change here that
can remove a rate raise and was found by reading rather than commissioned; the
`missingness` denominator moves 33 → 39, changing a number already quoted in
the paper; and every log recorded before this task becomes unscoreable by
`score_shadow`.

## What `field_sources` already records, and who reads it

Verified by reading, plus one probe run against the real builder.

- **The vocabulary is prose, not code.** Five classes are named in the module
  docstring (observation_builder.py:5-11: `measured`, `derived`,
  `fallback_neutral`, `static_config`, `sim_parity`). Four more are written
  by code and appear in no list: `measured_converted` and
  `measured_arrival_proxy` (`_speed_provenance`, :115-127), `approximated`
  (:412, for `uncongested_low_speed_flag`), and `feed_derived`
  (`feed_fusion.SOURCE_FEED`, :63-66 — reserved, emitted by nothing today).
  There is no constant, no membership test, and no closure test.
- **The map is built in two passes.** Nine keys are written explicitly
  (:208, :212, :216, :232-237, :244, :257, :286, :293) and twenty-four by
  `defaults` + `setdefault` (:384-426). `defaults` contains a **dead duplicate
  key**: `"local_queue_estimate"` at :414 is overwritten by the same key at
  :420 in the same dict literal.
- **Coverage, measured.** `len(field_sources) == 33`; `set(field_sources) ==
  set(sim_contract.LOCAL_OBS_FIELDS)`; the obs keys with no entry are exactly
  `cooperation`, `current_segment`, `nearby_av_lane_distribution`, `sensor`;
  `encoded.shape == (39,)`. So provenance covers 33 of the 39 slots the actor
  reads (`local_obs_dim()`, sim_contract.py:110-111 = 33 + 3 + 3).
  *(Verified by running a probe against `ObservationBuilder(BuilderConfig())`.)*
- **`missingness` and `fallback_fields`** are derived inside the builder
  (:429, :447): `missingness = |{v == "fallback_neutral"}| / len(src)`. On a
  no-peer, no-vehicle, fresh-GPS tick this measured **0.636 = 21/33**
  *(verified by running)*.
- **Readers.** `pipeline.py:138` writes it into every tick record.
  `run_demo.py:180` strips it from the **UI telemetry** record only
  (`telemetry_record`, :177-182) — the metadata log keeps it.
  `eval_run.py:463` reads exactly one key
  (`field_sources["leader_relative_speed"] == "measured"`) and
  `eval_run.py:490-497` reads the builder's derived `missingness` /
  `fallback_fields`, rendered at `eval_run.py:822-824` as
  "encoder-field missingness". `scripts/run_loopback_pipeline.py:434` reads
  `ego_speed`. Tests read it in test_phone_source.py, test_observation_builder.py,
  test_feed_fusion.py, test_eval_run.py. **Nothing under `policy/` reads it**
  (verified by grep) — that is the gap this task closes.
- **Provenance-shaped machinery already exists elsewhere, and this task must
  not duplicate it.** Task 33's `StageTiming` (sensors/time_sync.py:115-181)
  has measured / converted-with-a-bound / absent-with-a-named-reason. Task 34's
  `RULE_FIRED` / `RULE_QUIET` / `RULE_NOT_EVALUABLE` with a `missing` tuple
  (sensing_controller.py:144-151, :169-189). `feed_fusion.Decline` (:69-77) is
  a six-value closed set of named absence reasons, already carried into
  `Inputs.feed_declined` (sensing_controller.py:234-237, whose own comment says
  those two absences "are not told apart at this layer, which is task 36's
  subject"). This plan adds **one** vocabulary and reuses all three.

## The three defects, exactly

### 1. A substituted acceleration reads as a calm road

`_speed_slope` returns `(0.0, False)` on both fallback paths
(observation_builder.py:481-486); the caller writes `fallback_neutral`
(:216); `obs["ego_acceleration"] = 0.0` (:341); `inputs_from` copies the float
(sensing_loop.py:127); `abs(0.0) >= 1.5` is False, so the event rule records
`quiet` with `{"value": 0.0, "threshold": 1.5}` (sensing_controller.py:468-473).

**A useful consequence for auditing: the substituted value is *always exactly*
0.0, and 0.0 never fires.** So making this input `not_evaluable` changes the
record and cannot change a rate. This is asserted by test, not argued.

### 2. The same field is mislabelled `derived` for the whole of a GPS dropout

`speed_samples` is appended only under `gps_fresh` (observation_builder.py:201).
`_speed_slope` takes the last ten samples and asks only whether they span
0.3 s (:480, :485) — never how old the newest one is. During a dropout no
sample is appended, the window freezes, and the slope is reported as `derived`
forever.

Verified by running: 8 samples of constant −3.0 m/s², then the same fix
replayed with no new fresh fixes —

```
t+  1.0s  fresh=True   accel=-1.273  src=derived  speed_src=measured
t+  4.0s  fresh=False  accel= 0.000  src=derived  speed_src=fallback_neutral
t+ 20.0s  fresh=False  accel= 0.000  src=derived  speed_src=fallback_neutral
t+ 59.8s  fresh=False  accel= 0.000  src=derived  speed_src=fallback_neutral
```

`ego_speed` correctly falls back at the same instant; `ego_acceleration`
claims a measurement for a minute. This is the **dominant live path** for the
defect task 34 named, and the fix in §1 alone does not catch it, because the
label says `derived`. When the frozen window's slope is non-zero and above
`EVENT_ACCEL_MPS2`, the event rule fires on every tick of the dropout,
satisfies the 0.5 s dwell, and latches `ACTIVE_RATES` for the tunnel.

### 3. A blind camera fires the disagreement rule, and bin 0 means exactly "nothing detected"

`local_density_bin` is `derived` unconditionally (:413) from
`density = n_local / (2·range/1000)` (:243) with `n_local = 2·n_forward`
(:241). Under the shipped config (`effective_range_m: 80.0`,
`symmetrize_counts: true`, `density_bin_edges_veh_per_km: [12.0, 30.0]` —
config.yaml:59-68) density is `12.5 · n_forward`, so `bin_index` (0 edges ≤
0.0) is **0 if and only if `n_forward == 0`**, and 1 at a single in-range
track. `disagreement` fires on `feed_congestion >= 0.5 and
camera_density_bin <= 0` (sensing_controller.py:395-407). So the rule's only
firing condition, on shipped constants, is *zero in-range detections* — which
is what an empty road, a lens-blocked camera, and a broken detector all
produce.

**And the Jetson has no evidence that distinguishes them.** A tick only runs
on a decoded fresh frame (`camera.wait_for_fresh`, run_demo.py:487-492), so
"no frames" is not the failure shape; what remains — a dark frame, a covered
lens, a detector returning nothing — is invisible to every signal on this
device. That is why D4 below records rather than removes.

## Decisions taken (by recommendation — not signed off by the user)

| # | Question | Options | Taken | Why |
|---|----------|---------|-------|-----|
| D1 | How provenance reaches the controller | (a) a second scheme in `policy/`; (b) carry `field_sources` through `inputs_from` | (b) | The brief's instruction, and the map is already on `tick.obs_result` at the one seam that builds `Inputs` (sensing_loop.py:116-147). A second scheme is drift by construction. |
| D2 | How the controller *sees* a substitution | (a) named `*_source` fields the rules interpret; (b) `inputs_from` nulls the value and carries the source string for the record | (b) | (b) makes task 34's already-pinned branch (sensing_controller.py:464-466) the live path with **no structural change to `decide`**, and keeps the perception vocabulary out of `policy/sensing_controller.py`. `decision_inputs` then records `null`, which is *true* of what the controller saw; the substituted float and its class remain in the same tick record under `obs` and `field_sources`. |
| D3 | Which classes are substitutions | (a) `fallback_neutral` only; (b) `{fallback_neutral, static_config, sim_parity, unattributed}` | (b) | All four are values no measurement or derivation of *this tick* produced. No controller input is `static_config` or `sim_parity` today, so (b) costs nothing and is right if one ever is. `derived_empty` and `approximated` are deliberately **not** members. |
| D4 | The disagreement over-report | (a) treat a `derived_empty` bin as not evaluable; (b) keep firing, and record the basis plus a liveness bound | (b) | Under shipped constants bin 0 ⟺ zero in-range tracks (verified), so (a) deletes the rule's only firing path and trades a false positive for a guaranteed false negative — and there is no camera-liveness signal on this device to key (a) on. The rule's own purpose is "spend samples to resolve" an unreconcilable pair (:398-402), which is the right action whether the camera is blind or the road is empty. What was wrong was the *claim*, so the claim is fixed. Flagged as open item 1. |
| D5 | Naming a zero derived from an absence | (a) reuse `fallback_neutral`; (b) a new class `derived_empty` | (b) | The count really was zero — nothing was substituted — so (a) would move `missingness` for the wrong reason. `derived_empty` says the derivation is correct and its only input was an absence. Applied to `local_density_bin`, `active_vehicle_count_local`, and (for consistency) the empty-population arm of `local_queue_estimate`, whose *other* fallback arm — tracks present, none measurable — stays `fallback_neutral`, because that one is a measurement failure (:415-420). |
| D6 | The stale slope window | (a) out of scope (the brief named two fields); (b) refuse a window whose newest sample is older than `gps_stale_after_s` | (b) | It is the same defect in the same field, it is the *dominant* live path, and the current label is a positive false claim. Reusing `gps_stale_after_s` rather than a new constant makes acceleration exactly as fresh as the speed it is fitted from. **This is the one change in the task that can remove a rate raise**; flagged as open item 2. |
| D7 | Provenance coverage | (a) leave 33 obs-keyed entries; (b) all 39 encoder slots, dotted names for the six | (b) | The report already calls it "encoder-field missingness" (eval_run.py:822) while dividing by 33, and the six omitted slots are neutral on every tick of a lone-vehicle drive — the metric under-reports exactly where this vehicle is. Consequence disclosed as open item 3. `current_segment` and `sensor` stay uncovered: the encoder ignores them, so they would be noise in a metric about the actor's input. |
| D8 | Where the vocabulary lives | (a) constants in `observation_builder.py`; (b) a new `perception/provenance.py` | (b) | Three modules need it after this task (builder, `sensing_loop`, `eval_run`) and a fourth already half-owns it (`feed_fusion.SOURCE_FEED`, :63-66, which becomes a re-export). One home is what stops the second scheme the brief warns about. |
| D9 | A field with no provenance entry | (a) treat as measured; (b) a named class `unattributed`, in `SUBSTITUTED` | (b) | A value the builder did not tag is not evidence. It fails safe (the rule refuses rather than deciding), and it is named rather than null so the record can tell "the builder wrote nothing" from "the builder wrote a class". Only `inputs_from` can produce it; the coverage test makes it unreachable in production. |
| D10 | Pre-task-36 logs in `score_shadow` | (a) default the new keys; (b) treat as a different incumbent (ship a versioned controller); (c) a named refusal derived from the `Inputs` schema | (c) | (a) is exactly what task 35 D2 forbids: "a silently defaulted input is a silently different replay". (b) means maintaining a second controller whose only consumer is old logs, and the identity gate's value is that it certifies *the code that ships*. Without (c) the tool does not degrade — `Inputs.from_record` raises `ValueError` (sensing_controller.py:293-297) inside `_replay_incumbent` (score_shadow.py:147) and the tool **crashes**, which contradicts its own doctrine ("refuses before it misleads", score_shadow.py:20-25). Made schema-derived rather than a hard-coded key list, so the next `Inputs` change gets a refusal instead of a traceback for free. |
| D11 | `_rules_never_exercised`'s reason map | (a) keep the special-cased `feed_declined` (score_shadow.py:352-357); (b) a general `why: {evidence_key: {value: count}}` | (b) | After this task a not-evaluable entry can carry `ego_acceleration_source` as well, and the "why" is this task's whole contribution. A task-35 output-shape change, disclosed as open item 5; no existing test asserts the `feed_declined` key (verified — the three tests at test_score_shadow.py:657, :682, :705 assert `rule`, `ticks`, `missing` only). Also printed by `render_table`, which prints `missing` and drops the reason today. |
| D12 | `Inputs.feed_declined` (task 34 open item 1) | (a) remove as redundant; (b) keep | (b) | It is not redundant: `feed_congestion` is not an observation field and has no `field_sources` entry at all (task 28's conclusion, feed_fusion.py:20-27). It is the same *shape* — a named reason for an absence — and the new fields follow its idiom. Task 34's device run already found it non-null on 899 of 899 ticks. Resolves task 34 open item 1 toward keep. |
| D13 | Drive-level surfaces | (a) per-tick record only; (b) `SensingLoop.inputs_by_source`, `score_shadow.input_provenance`, an `eval_run` block **and report lines** | (b) | Task 33's experiment: all 900 ticks carried the `stages` block and `report.md` printed none of it (task_list.md:1110-1112). A measurement with no surface is this project's most repeated mistake. `SensingLoop` is the right live home for the same reason task 34 D6 put counters there — `summary["sensing"]` *is* `SensingLoop.to_record()`. |
| D14 | `camera_last_detection_age_s` | (a) diagnostics only; (b) an `Inputs` field carried into the rule's evidence | (b) | The misleading record is `{"status": "fired", "camera_density_bin": 0}`; the bound belongs at the point of the claim, not in a block a reader has to join. It is the only evidence available about whether the perception chain was alive, and it is free (one float of builder state). |

## The record, exactly

### `perception/provenance.py` (new)

The closed vocabulary and the one partition anything keys on:

```python
SOURCE_MEASURED               = "measured"
SOURCE_MEASURED_CONVERTED     = "measured_converted"
SOURCE_MEASURED_ARRIVAL_PROXY = "measured_arrival_proxy"
SOURCE_DERIVED                = "derived"
SOURCE_DERIVED_EMPTY          = "derived_empty"      # NEW
SOURCE_APPROXIMATED           = "approximated"
SOURCE_FEED                   = "feed_derived"       # moved from feed_fusion.py:66
SOURCE_STATIC_CONFIG          = "static_config"
SOURCE_SIM_PARITY             = "sim_parity"
SOURCE_FALLBACK_NEUTRAL       = "fallback_neutral"
SOURCE_UNATTRIBUTED           = "unattributed"       # NEW; only inputs_from emits it

SOURCES = frozenset({...all eleven...})

#: Classes whose value is NOT evidence about this tick: a number the builder
#: put there that no measurement or derivation of this tick's inputs produced.
#: `derived_empty` is deliberately absent -- a zero count is a real, if weak,
#: observation, and excluding it would delete the disagreement rule (D4).
SUBSTITUTED = frozenset({SOURCE_FALLBACK_NEUTRAL, SOURCE_STATIC_CONFIG,
                         SOURCE_SIM_PARITY, SOURCE_UNATTRIBUTED})

def is_substituted(source: str | None) -> bool: ...
def summarise(field_sources: Mapping[str, str]) -> dict[str, Any]: ...
```

`summarise` returns `{"fields": n, "by_source": {class: count},
"missingness": round(fallback/n, 3), "fallback_fields": [...]}` — the existing
two diagnostics keys move here unchanged in meaning, so the builder stops
computing them inline and no reader changes.

`feed_fusion.py:66` becomes `SOURCE_FEED = provenance.SOURCE_FEED` (re-export,
so nothing that imports it moves).

### `sim_contract.encoded_slot_names()` (new, additive)

```python
def encoded_slot_names() -> tuple[str, ...]:
    """The 39 slot names `encode_local_observation` fills, in encoder order."""
    return (*LOCAL_OBS_FIELDS,
            *(f"cooperation.{f}" for f in COOPERATION_FIELDS),
            *(f"nearby_av_lane_distribution.{lane}" for lane in LANE_DISTRIBUTION_LANES))
```

Not part of the mirrored contract (it introduces no new constant), and
`contract_fingerprint` is unaffected. `len(encoded_slot_names()) ==
local_obs_dim() == 39` is a test.

### `field_sources`, after

Exactly 39 keys, `set(field_sources) == set(encoded_slot_names())` on every
tick. Every pre-existing key keeps its spelling and its meaning except the
three named below. The six new entries:

- `cooperation.segment_target_speed`, `cooperation.merge_pressure`,
  `cooperation.downstream_congestion_estimate` — the class of the flat field
  of the same name, because they are the same value (observation_builder.py:362-364,
  :381 read one dict).
- `nearby_av_lane_distribution.{0,1,2}` — `derived` when
  `_peer_lane_distribution` returned a non-empty map (the shares are computed
  from measured peers), `fallback_neutral` when it returned `{}` (no peers, or
  peers with no `lane_id`, :490-499). Today these three encoded slots are 0.0
  with no provenance at all.

Three classes change:

| field | today | after |
|---|---|---|
| `ego_acceleration` | `derived` whenever a slope was fitted (:216) | `fallback_neutral` also when the newest speed sample is older than `cfg.gps_stale_after_s` (D6) |
| `local_density_bin` | `derived` always (:413) | `derived_empty` when `n_forward == 0`, else `derived` (D5) |
| `active_vehicle_count_local` | `derived`/`measured` by `symmetrize_counts` (:244) | `derived_empty` when `n_forward == 0`, else as today |
| `local_queue_estimate` | `derived` if `abs_speeds` else `fallback_neutral` (:420) | `derived` if `abs_speeds`; `derived_empty` if `in_range` is empty; `fallback_neutral` if `in_range` is non-empty and none measurable |

The dead duplicate at :414 is deleted.

### `obs_diagnostics`, after

`missingness` and `fallback_fields` keep their names, positions and meanings;
their **denominator becomes 39** because `src` now has 39 entries. Two
additions:

```
"provenance": {"fields": 39,
               "by_source": {"fallback_neutral": 26, "derived_empty": 3, ...},
               "covers_encoder": true},
"last_detection_age_s": 12.4    # or null before the first in-range detection
```

`last_detection_age_s` is `t_mono - <the last t_mono at which in_range was
non-empty>`, `None` until the first such tick. A measured 0.0 on a tick that
has one; never a substituted zero.

**Predicted, and pinned by test:** on a no-peer, no-vehicle, fresh-GPS tick,
`fields == 39`, `missingness == round(26/39, 3) == 0.667`, and
`by_source["derived_empty"] == 3`. (Today the same tick measures 21/33 =
0.636, verified by running. The arithmetic: 21 − 1 for `local_queue_estimate`
leaving the fallback set, + 3 cooperation slots + 3 lane slots.)

### `Inputs`, after — four new fields, seventeen in total

```python
#: The `field_sources` class of the value above, as `inputs_from` read it.
#: `ego_acceleration` is None exactly when this class is a substitution
#: (`provenance.SUBSTITUTED`): the controller is told the value was not
#: evidence rather than handed the neutral that stood in for it.
ego_acceleration_source: str | None = None
#: The class behind `ego_speed`. No rule keys on it -- the query radius is
#: still sized from a held speed (`_here_query`, :689-692) -- but the record
#: of what `here_radius_m` rested on has to exist somewhere.
ego_speed_source: str | None = None
#: The class behind `camera_density_bin` (the obs field `local_density_bin`).
#: `derived_empty` means the bin rests on an absence of detections, which is
#: what a blind camera and an empty road both produce.
camera_density_bin_source: str | None = None
#: How long since the perception chain last produced an in-range track. The
#: only bound available on whether "the camera saw nothing" is a statement
#: about the road. None means it never has on this drive.
camera_last_detection_age_s: float | None = None
```

`to_record` emits all four unrounded (`ego_speed_source` etc. are strings;
the age is a float carried at full precision, like every other `Inputs`
field — sensing_controller.py:258-265). `from_record` stays strict in both
directions, unchanged.

### `inputs_from`, after

Reads `tick.obs_result.field_sources` by attribute — no `getattr` default,
because `ObservationResult.field_sources` is a required field
(observation_builder.py:53) and a result without one is a caller bug, not a
missing measurement. Per field, `src.get(name)` missing →
`SOURCE_UNATTRIBUTED`. Then:

- `ego_acceleration` = the obs value, or `None` when
  `is_substituted(ego_acceleration_source)`.
- `ego_speed`, `camera_density_bin` = the obs values, **unchanged** (D4, and
  the `ego_speed` note above).
- `camera_last_detection_age_s` from `diagnostics.get("last_detection_age_s")`.

### The attribution record, after

`event_from_free_tier` carries `ego_acceleration_source` on all three
statuses; `source_disagreement` carries `camera_density_bin_source` and
`camera_last_detection_age_s` on all three. The `feed_declined` idiom
(sensing_controller.py:424-428) is copied exactly. Two illustrative entries:

```
"event_from_free_tier": {"status": "not_evaluable",
                         "missing": ["ego_acceleration"],
                         "ego_acceleration_source": "fallback_neutral"},
"source_disagreement":  {"status": "fired",
                         "feed_congestion": 0.9, "jammed_congestion": 0.5,
                         "camera_density_bin": 0, "empty_density_bin": 0,
                         "camera_density_bin_source": "derived_empty",
                         "camera_last_detection_age_s": 241.7}
```

The second is the brief's "worse half", after: the rule still fires and still
raises the camera rate, and the record now states that the camera view it
fired on was an absence, last corroborated four minutes ago.

Reading rules the validator audits against:

- `rules_fired == [r for r in RULES if rules[r].status == "fired"]` — task 34's
  identity, unchanged, still pinned.
- Every `status` is one of task 34's three; every provenance string in
  `field_sources` and in the four `Inputs` source fields is a member of
  `provenance.SOURCES`; `derived_empty ∉ SUBSTITUTED`.
- On any two ticks differing only in `ego_acceleration_source`
  (`derived` at 0.0 versus `fallback_neutral` at 0.0), **`rates`, `trigger`,
  `rules_fired`, `reasons`, `thermal_scale`, `clamped` and `here_radius_m` are
  equal**, and `attribution.rules.event_from_free_tier` differs. This is the
  no-rate-change contract for D2, asserted rather than argued.

### `SensingLoop.to_record()`, after

One new key beside `rules_by_status` (sensing_loop.py:209, :290):

```
"inputs_by_source": {"ego_acceleration":   {"derived": 812, "fallback_neutral": 87},
                     "ego_speed":          {"measured": 812, "fallback_neutral": 87},
                     "camera_density_bin": {"derived_empty": 899}}
```

Each field's counts sum to `ticks`.

### `score_shadow.py`, after

1. `REFUSAL_INPUTS_SCHEMA = "decision_inputs_schema"`, checked in
   `_log_refusal` (score_shadow.py:119-129) on **every** sensing tick by
   comparing `set(decision_inputs)` against `{f.name for f in fields(Inputs)}`.
   The result carries `{"refused": ..., "schema": {"missing": [...],
   "unknown": [...], "first_tick_id": n}}`, exit 2, no candidates, no
   traceback. `REFUSAL_PRE_TASK_35` keeps its meaning (the key absent
   entirely); this is the narrower "present and a different shape".
2. `_rules_never_exercised` (:334-359): the special-cased `feed_declined`
   block becomes a general `why: {evidence_key: {value: count}}` over every
   key of a `not_evaluable` check other than `status` and `missing`.
   `render_table` prints it (it prints `missing` and drops the reason today).
3. `result["input_provenance"] = {field: {source: count}}` over the drive's
   `decision_inputs`, computed beside `rules_never_exercised` so it is present
   with zero candidates, and printed in the table.

### `eval_run.py`, after

`observation` gains four keys and keeps its two:

```
"missingness": pctl(...),                  # unchanged code; denominator is now the log's own
"top_fallback_fields": {...},              # unchanged
"provenance_fields": 39 | 33 | None,       # the map's size on THIS run
"covers_encoder": true | false | None,     # provenance_fields == local_obs_dim()
"by_source": {class: fraction of field-ticks},        # sums to 1.0
"fields_by_source": {class: {field: fraction of ticks}}   # capped at 8 per class
```

All four are computed from each tick's own `field_sources`, which every log
back to the beginning carries — so an old 33-field log reports
`provenance_fields: 33, covers_encoder: false` rather than crashing, the
`jetson_ms_source` precedent (eval_run.py:447-451). **No new gate**: a run
recorded before this task is not a failed drive (open item 8).

`report.md` gains, under "Observation quality":

```
- encoder-field missingness: mean 66.7% of 39 provenance-tagged fields
- provenance covers 39 of 39 encoder slots
- by source: fallback_neutral 66.7%, derived_empty 7.7%, derived 12.8%, ...
- derived from an absence (a blind sensor and an empty road are the same
  number here): local_density_bin 100% of ticks, active_vehicle_count_local
  100%, local_queue_estimate 100%
```

## The work

1. **`perception/provenance.py`** — the vocabulary, `SUBSTITUTED`,
   `is_substituted`, `summarise`. `feed_fusion.SOURCE_FEED` becomes a
   re-export.
2. **`sim_contract.encoded_slot_names()`** — additive helper; no constant
   moves, `contract_fingerprint` unchanged.
3. **`observation_builder.py`** — import the constants and use them instead of
   bare strings; the six new entries; the three class changes (D5, D6); delete
   the dead `defaults` key at :414; `_speed_slope(t_mono)` gains the window
   -staleness guard against `cfg.gps_stale_after_s`; `_EgoState` gains
   `last_in_range_at: float | None`; diagnostics gain `provenance` and
   `last_detection_age_s` via `provenance.summarise`. **`missingness` and
   `fallback_fields` keep their names and their formula.**
4. **`sensing_controller.py`** — four `Inputs` fields with `to_record` /
   `from_record` symmetry; evidence keys on the two rules. `decide`'s control
   flow is untouched: the `ego_acceleration is None` branch at :464-466 is
   already there.
5. **`sensing_loop.py`** — `inputs_from` reads `field_sources`, nulls a
   substituted acceleration, fills the four fields; `inputs_by_source` counter
   and its `to_record` key.
6. **`score_shadow.py`** — the schema refusal, the general `why` map,
   `input_provenance`, and the two `render_table` lines.
7. **`eval_run.py`** — the four `observation` keys and the report lines.
8. **Test fixtures** — `FakeObs` in test_sensing_loop.py:38-42 and
   test_score_shadow.py:57-61 gains a `field_sources` default (a full,
   grounded 39-key map from a helper), so every existing test keeps its
   current behaviour and a fake that omits provenance is a visible choice.
9. **`ARCHITECTURE.md:108-129`** — the provenance table's `ego_acceleration`,
   `local_density_bin`, `local_queue_estimate` and `nearby_av_*` rows, plus a
   line saying the map covers all 39 encoder slots.
10. **Tests**, then **pinned mutations** in `scripts/remutate.py` under a
    "Task 36" block, scored on the pytest summary line matching the baseline
    and on the failing test name, per the harness's own rule.

Nothing touches `pipeline.py`, `run_demo.py`, `transport/`, `specs/`, or the
phone. Nothing calls or configures HERE; no key appears in any URL, log or
test in this task's surface.

## Tests

**Vocabulary and coverage**
- Every value the builder emits across a state sweep (no fix / stale fix /
  fresh fix; no tracks / tracks with and without measurable speeds; no peers /
  peers with and without `lane_id`; feed owned and declined) is a member of
  `provenance.SOURCES`.
- `set(field_sources) == set(sim_contract.encoded_slot_names())` on every tick
  of that sweep, and `len == local_obs_dim() == 39`.
- `encoded_slot_names()[:33] == LOCAL_OBS_FIELDS` in order, and
  `len(encoded_slot_names()) == local_obs_dim()`.
- `SUBSTITUTED ⊂ SOURCES`, and a test named for the decision asserting
  `SOURCE_DERIVED_EMPTY not in SUBSTITUTED` with D4's reason in its docstring.

**The accelerometer path — the caller waiting on this task**
- Fresh GPS, constant speed: `ego_acceleration == 0.0`, source `derived`,
  `inputs_from` passes 0.0, event `quiet`. *("the road was calm")*
- GPS unfresh for longer than `gps_stale_after_s`: source `fallback_neutral`,
  `inputs.ego_acceleration is None`, event `not_evaluable` with
  `missing == ["ego_acceleration"]` and
  `ego_acceleration_source == "fallback_neutral"`. *("the sensor was dead")*
- **Those two ticks' `rates`, `trigger`, `rules_fired`, `reasons`,
  `thermal_scale`, `clamped`, `here_radius_m` are equal.** The no-rate-change
  contract, asserted.
- Cold start (fewer than 3 samples, and a window shorter than 0.3 s): the same
  `not_evaluable`, distinct from the stale case by nothing but the class —
  which is correct, both are substitutions.
- **The threshold is the GPS window, not a new literal**: brake at −3 m/s²,
  then stop supplying fresh fixes; at 1.9 s of dropout the source is still
  `derived`; at 2.1 s it is `fallback_neutral` and the value is 0.0.
- **The one behaviour change, asserted directly**: a dropout entered with a
  frozen window whose slope exceeds `EVENT_ACCEL_MPS2`. Before the guard the
  event rule fires on every dropout tick and the camera latches at
  `ACTIVE_RATES`; after it, the rule is `not_evaluable` and the rate returns to
  `IDLE_RATES` once `HOLD_S` elapses.

**The density path**
- Zero in-range tracks: `local_density_bin` `derived_empty` at value 0;
  `active_vehicle_count_local` `derived_empty`; `local_queue_estimate`
  `derived_empty`.
- One in-range track: bin 1, `derived` — pinning that under shipped constants
  bin 0 ⟺ zero in-range tracks, which is what makes D4 a real choice.
- Six in-range tracks, none with a measurable relative speed:
  `local_queue_estimate` stays `fallback_neutral` (the distinction :415-420
  exists for) while `local_density_bin` is `derived`.
- `disagreement(0.9, 0)` on a `derived_empty` bin **still fires**, and the rule
  entry carries `camera_density_bin_source == "derived_empty"` and a
  `camera_last_detection_age_s`. The over-report is named, not removed.
- `camera_last_detection_age_s` is `None` before the first in-range detection,
  0.0 on a tick that has one, and grows monotonically after the last one.

**Missingness**
- The no-peer, no-vehicle, fresh-GPS tick: `provenance.fields == 39`,
  `missingness == 0.667`, `by_source["derived_empty"] == 3`, and the six new
  slots present with the classes stated above. (The pre-task value 0.636 = 21/33
  is recorded in the test's docstring so the move is legible.)
- Peers with `lane_id`: the three lane slots are `derived`; peers without:
  `fallback_neutral`.
- `sum(by_source.values()) == fields` on every tick of the sweep.

**Schema and replay**
- `Inputs.to_record` → `json` → `from_record` round-trips all seventeen fields,
  including a 17-significant-digit `camera_last_detection_age_s`;
  `from_record` refuses a 13-key record naming the four missing fields.
- A `SensingLoop` drive's `decision_inputs` carries seventeen keys on every
  tick.
- **`score_shadow` on a pre-task-36 log** (13-key `decision_inputs`) refuses
  with `decision_inputs_schema`, names the four missing keys and the first
  offending tick, exits 2, emits no `candidates` — asserted as *not* a
  `ValueError`, because a traceback is what happens without this change.
- `score_shadow` replay identity is 0-mismatched on a post-task-36 log, and
  the existing task-35 tests still pass unmodified.
- `_rules_never_exercised`'s `why` map carries
  `{"ego_acceleration_source": {"fallback_neutral": N}}` on a drive whose
  accel is substituted throughout, and `{"feed_declined": {...}}` on the
  disagreement rule — two reasons in one general map.

**Surfaces (D13, task 33's lesson applied at birth)**
- `SensingLoop.to_record()["inputs_by_source"]`: each field's counts sum to
  `ticks`; the key is present in `summary["sensing"]` with no `run_demo` change.
- `eval_run`'s `observation` block carries the four new keys; against the
  existing 33-field-free fixture (test_eval_run.py:39-43) it reports
  `covers_encoder: false` and does not crash.
- **`render_report` output is asserted, not just the JSON**: the rendered
  `report.md` contains a line per class present and names the `derived_empty`
  fields; `score_shadow.render_table` contains the `why` map and the
  `input_provenance` line.

**Compatibility**
- Full suite green from the 1615 baseline (verified by running before this
  plan: `1615 passed in 59.53s`).
- `eval_run.py:463`'s `leader_relative_speed == "measured"` reader is
  unchanged, and the four `field_sources["ego_speed"]` spellings pinned in
  test_phone_source.py (:259, :292, :302-314, :462) are unchanged.

## Pinned mutations (`scripts/remutate.py`, "Task 36" block)

1. `inputs_from` passes a substituted acceleration through instead of nulling
   it — a dead GPS reads as a calm road again. Caught by the two-tick
   record-differs/rates-agree test.
2. `_speed_slope` drops the window-staleness guard — an expired window
   reported as `derived`. Caught by the 1.9 s / 2.1 s boundary test.
3. `local_density_bin` reverts to unconditional `derived` — an empty detection
   set indistinguishable from a measured light road. Caught by the
   `derived_empty` test.
4. `is_substituted` returns False for `unattributed` — a field the builder
   forgot to tag is decided on as if measured. Caught by the unattributed test.
5. `SOURCE_DERIVED_EMPTY` added to `SUBSTITUTED` — the disagreement rule
   silently loses its only firing path. Caught by the "still fires on
   `(0.9, 0)`" test. **This pins the deliberate non-change**, which is the
   decision here most likely to be "fixed" by a later reader.
6. `score_shadow._log_refusal` skips the schema check — the tool crashes with
   a `ValueError` on a pre-task-36 log instead of refusing. Caught by the
   refusal test.

## What this change does to logs recorded before it

The brief's question, answered in three parts.

**`eval_run` still reads them, and says so.** Nothing in the tick record's
shape is removed. `provenance_fields` reports 33, `covers_encoder` reports
false, `by_source` shows the old vocabulary with no `derived_empty`. But
`missingness` on an old log is over 33 fields and on a new log over 39, and
the two are **not comparable** — which is why the denominator is now published
beside the ratio rather than left implicit.

This reaches a published number. `hotnets_submission/eval.tex:49` states "on
average almost half of the observation fields are placeholders (mean
missingness 0.477)". That figure is 0.477 of 33 slots. Recomputed over 39, the
same drive reads between **0.531 and 0.557** — `(0.477·33 + 6 − q)/39`, where
`q ∈ [0, 1]` is the tick-weighted fraction on which `local_queue_estimate`
leaves the fallback set (it leaves only on ticks with no in-range tracks, and
that drive has a leader present much of the time). The direction is worse,
which is the honest direction: the omitted six were neutral on every tick.
**Exact recomputation needs the archived run directory, which is not on this
machine** (`outputs/` holds only the two simulation studies) — open item 3.

**`score_shadow` refuses them by name (D10).** `Inputs.from_record` is strict
in both directions by design (sensing_controller.py:282-298), so a 13-key
`decision_inputs` cannot be replayed; and even with the keys defaulted, the
new controller would produce a different `attribution.rules.event_from_free_tier`
on every substituted-accel tick, so the identity gate would fail anyway — for
the right reason, with a misleading `first_mismatch`. Every drive recorded
before this task therefore becomes unscoreable: task 34's 899-tick run, task
35's 599 and 353, and `device-test-2026-08-25`. The cost is bounded by what
the tool actually does — it compares candidates against an incumbent *within
one log*, and no cross-log comparison exists in it — so refusing costs no
comparison anyone can make today.

**"A different incumbent" was considered and not taken.** It means shipping and
maintaining a pre-task-36 `SensingController` whose only consumer is
historical logs, and it inverts the identity gate's purpose: the gate exists
to certify that *the code that ships* reproduces the log, not that some
retired code does. A refusal costs one function; a versioned controller costs
a permanent fork of the decision function this project's whole shadow-scoring
argument rests on.

## What this establishes, and what it does not

It establishes: one closed provenance vocabulary with a single home; a
provenance entry for every one of the 39 slots the actor reads, so
`missingness` is a statement about the whole vector rather than about the 33
that happened to be tagged; two mislabels fixed at source, one of which
(the expired slope window) was reporting a positive false claim on every tick
of every GPS dropout; and provenance reaching the controller, so
**`quiet` at 0.0 and `not_evaluable` are now two different records** and task
34's pinned branch is the live path it was pinned for. It does this while
changing no rate on the accelerometer path at all — proved by test, not
argued — and it puts the result on three surfaces rather than only in the log.

It does not stop the disagreement over-report; it names it (D4). Under shipped
constants that rule fires if and only if the camera detected nothing, and this
device has no evidence that separates "detected nothing" from "could not
detect", so the record now carries the class and a liveness bound and the rate
still moves. It does not reach the phone: no wire change, no Kotlin, and a
phone-side sensor fault is still visible only as `achieved`/`dropped` in task
35's reference block. It does not wire the IMU, which is commanded at 50 Hz
and read by nothing (`phone_link.py:173-184`) — it only makes the GPS-derived
field say when it is not a measurement. It does not give the simulator's
`src/sensing/local.py` a provenance twin, so sim-versus-real provenance parity
(task 47) is untouched. It does not close the *conversion* half of the
recurring defect class — task 33's 5,700×-bound offline join was a converted
value reported without its bound, a different mechanism from the substitution
this task closes. And it renders nothing beyond `eval_run`'s report (task 39).

## To be measured on a device run (the shape tasks 34 and 35's plans took)

Filled in after implementation; the questions are fixed now so the run cannot
be scored on what it happened to show.

1. Every tick carries 39 provenance entries and a `provenance` block — count
   over the drive, and `covers_encoder` true on all of them.
2. The drive's `by_source` distribution, and specifically **how many ticks had
   `ego_acceleration` `fallback_neutral`** — the number that says how often
   the event rule was being fed a substitution before this task. Report it
   whatever it is.
3. `summary["sensing"]["rules_by_status"]["event_from_free_tier"]["not_evaluable"]`
   against `inputs_by_source["ego_acceleration"]["fallback_neutral"]`. The two
   are computed from one fact by different code and **must be equal**; a
   mismatch is a defect, not a rounding.
4. The `derived_empty` fraction for `local_density_bin` — the fraction of
   ticks on which the disagreement rule's "camera sees empty road" would have
   rested on an absence. On a stationary handset expect close to 100%.
5. `camera_last_detection_age_s`: p50 / p95 / max over the drive, and how many
   ticks had it null.
6. `missingness` with its denominator stated, beside the pre-task figure for
   the same footage if a pre-task run directory is recoverable.
7. Record growth. Estimate: `field_sources` +6 entries ≈ 200 B, `provenance`
   block ≈ 250 B, `decision_inputs` +4 keys ≈ 120 B, attribution evidence
   ≈ 60 B — about **+630 B** against a measured mean record of 8,511 B
   (task 35), so about **7%**. Measure, do not trust: task 34's estimate was
   five times off on the base size, task 35's held.
8. `score_shadow` against a pre-task-36 run directory: the named refusal with
   the four key names, exit 2 — not a traceback, not a score.
9. `report.md` and the `score_shadow` table actually contain the provenance
   lines. Task 33's experiment found 900 ticks of a measurement rendered
   nowhere; this is the check that would have caught it.

## Open items flagged for the user

1. **The disagreement over-report is named, not removed (D4).** This is the
   brief's "worse half" and the recommendation does not stop the rate moving.
   The reason is structural: under shipped constants the rule fires iff
   `n_forward == 0`, so gating on `derived_empty` deletes the rule. If a gate
   is wanted anyway, the concrete form is: `camera_density_bin` not evaluable
   when `camera_last_detection_age_s` is `None` or above a threshold — and
   **no data exists to choose that threshold**, which is why it is not
   proposed as a number. Worth deciding before a HERE key makes the rule
   reachable at all.
2. **The stale-window fix (D6) was not commissioned and it changes rates.** It
   was found by reading `_speed_slope` and confirmed by running. It can only
   ever *remove* a raise, and only on ticks where the GPS has been unfresh for
   more than `gps_stale_after_s`. It is the largest single behaviour change in
   this task and it is in a task otherwise designed to change none.
3. **`missingness`'s denominator moves 33 → 39**, changing a figure already in
   the paper (`hotnets_submission/eval.tex:49`, "mean missingness 0.477" →
   0.531–0.557). The exact recomputation needs the archived run directory,
   which is not on this machine. Someone has to decide whether the paper is
   restated or the old denominator is kept alongside.
4. **Every pre-task-36 log becomes unscoreable by `score_shadow` (D10)** —
   task 34's 899-tick drive, task 35's two drives, and
   `device-test-2026-08-25`. If any of those must stay scoreable, D10 has to
   be revisited before this lands, not after.
5. **`_rules_never_exercised`'s `feed_declined` becomes `why.feed_declined`
   (D11)** — a task-35 output-shape change. No existing test asserts the key
   (verified), but `shadow_score.json` files already written have the old
   shape.
6. **`local_queue_estimate` gains a third class (D5)**, which also moves
   `missingness`. It is the consistent application of the same rule, but it is
   a change to a field the brief did not name and it partly offsets item 3's
   increase.
7. **The IMU is commanded at 50 Hz and read by nothing** (`phone_link.py:173-184`,
   verified by grep). The controller pays for a modality it never looks at, and
   the "free always-on tier" that is supposed to say when to spend the
   expensive ones is, for acceleration, 1 Hz GPS. Out of scope; named because
   this task is the one that inventories what the controller can actually see.
8. **`ego_speed_source` reaches `Inputs` and no rule keys on it.** A dropout
   still sizes the HERE query radius from a held speed with no gate
   (`_here_query`, sensing_controller.py:689-692). Deliberate — gating it
   would change `here_radius_m` and therefore the command's content on every
   dropout, which is a rate-command change with no requester — but a reader
   may expect the symmetry, so it is stated rather than left to be noticed.
9. **`unattributed` fails safe rather than loudly (D9).** A field the builder
   stops tagging makes its rule refuse to evaluate. The alternative — raising
   at the first such tick — would catch a builder regression immediately
   instead of degrading a rule quietly. The coverage test makes it unreachable
   in production, which is the argument for the softer choice, and is also the
   argument that the harder one costs nothing.
10. **No `eval_run` gate on provenance coverage.** A run whose map covers 33
    slots is a run recorded before this task, not a failed drive, so
    `covers_encoder` is reported and not gated. If it should become a gate
    once every historical run is retired, that is a one-line change and a
    deliberate decision.

## Open items carried in

- Task 27's HERE parse has never met a real response body. (No HERE work in
  this task; no key appears in any URL or log.)
- Task 28's observation vector cannot be feed-informed without a
  simulator-side change; `SOURCE_FEED` therefore remains a class nothing
  emits, now named in the closed vocabulary rather than only in `feed_fusion`.
- `MAX_QUERY_RADIUS_M` (10 km) unchecked against HERE v7's accepted range.
- `achieved["camera_hz"]` overstates the sustained rate when the channel
  evicts; needs `Session.send` to surface the displacement.
- Task 33's residue: the CameraX clock assumption on
  `capture_to_encode_start`, `fuse`'s unreachable absent branch, one
  `Thread.sleep(200)`, ~1.5% unmatched joins on pre-task-33 phone logs.
- Task 34's residue: open item 2's first half (the substituted acceleration)
  is what this task closes, and its second half (the density over-report) is
  named and counted rather than removed — see open item 1. Crossing two quiet
  entries in the emitted record still survives; `RuleCheck.to_record` still
  accepts non-finite floats.
- Task 35's residue: the staleness reconciliation agrees only at zero on
  hardware; `first_differ_tick_id` is exercised but not pinned;
  `source_disagreement` was `not_evaluable` on every tick of both drives; bare
  `NaN` from NaN GPS appears in the metadata log.
