# Task 34 — Trigger attribution in the controller: which rule fired, for which sensor, and why

> Written by `plan_dsrc_rec`: every decision below was taken by recommendation,
> without putting the question to the user. The decisions table says what was
> chosen and why; **none of it is user-approved**, and the "Open items flagged
> for the user" section holds every point where the recommendation was weak or
> where the code contradicted the first reading. This plan is the fixed target
> a validator audits the implementation against.

## The short version

Task 34 (plans/task_list.md:1135) asks the controller to attribute every
firing: which rule, for which sensor, and why. Half of this landed with
task 29 and is verified below: every per-tick record already carries
`trigger` (one word from a closed six-value set, sensing_controller.py:121-131),
`rules_fired` (every rule that fired, :176-179), free-text `reasons`,
`thermal_scale` and `clamped` (:188-197), spread into the tick log at
run_demo.py:526-532. What does **not** exist is exactly the project's
recurring defect class: the record cannot distinguish *"this rule fired"*
from *"this rule was evaluated and stayed quiet"* from *"this rule could not
be evaluated because its input was absent"* — a rule missing from
`rules_fired` is all three at once. Nothing maps rules to the sensors they
move (the decision is one global level bit plus per-key modifiers,
:338-359), nothing states the dwell/hold/bridge gate that let or blocked a
firing, and the run summary counts sends by reason but no decision by
trigger at all (sensing_loop.py:204-211).

The plan, in one paragraph: (1) give `Decision` a required `attribution`
record built where the rules are checked, with one entry per rule from a
closed three-state vocabulary — `fired` / `quiet` / `not_evaluable` with the
missing inputs named — following task 33's `StageTiming` discipline
(basis stated in the record, absence named, never a silent default;
sensors/time_sync.py:115-181); (2) a `gates` block stating the level
machinery per tick (wants_more, dwell elapsed/required/satisfied, hold
remaining, bridged, gapped, level), because "why did the rate move" is as
often a gate as a rule; (3) a `per_sensor` block giving each of the four
rate keys its full composition chain — profile base, level sensitivity,
thermal scale applied or exempt, clamp, previous value, changed — so "for
which sensor" is answered by the record instead of by a reader who has
memorized `IDLE_RATES`/`ACTIVE_RATES`; (4) drive-level counters
(`decisions_by_trigger`, `rules_by_status`) in `SensingLoop.to_record`, the
block `summary["sensing"]` already publishes. Nothing about what is decided
changes: rates, `trigger`, `rules_fired`, `reasons` stay byte-identical, and
the wire is untouched.

**Scope boundary.** In: `sensing_controller.py` (the attribution record and
the plumbing to build it), `sensing_loop.py` (one new `Inputs` field,
`feed_declined`, and the two counters), tests for all of it, and pinned
mutations in scripts/remutate.py. Out: any change to the decided rates or
to the existing record keys; the wire and `RateCommand` (the spec pins
`rates`, `trigger`, `shadow` at specs/transport_protocol.md:342 and the
phone consumes nothing more — no golden-frame or `PROTOCOL_VERSION` work);
run_demo.py (the `**decision.to_record()` spread at :527 carries the new
key with zero changes); eval_run.py (reads nothing under `sensing` today,
verified by grep — presentation is task 39's session summary generator);
the shadow decision log (task 35); field provenance reaching the controller
(task 36 — see open item 2); and the phone.

**Open decisions flagged for the user** (details at the bottom): carrying
the feed's `declined` reason into `Inputs` so disagreement's absence is
named rather than bare (borders task 36); a dead accelerometer showing as
`quiet` at 0.0 because the builder substitutes `fallback_neutral` before
the controller looks (the one place the three-state vocabulary can still
mislabel — deliberately left to task 36); the status word `quiet`; about
0.8-1.0 KB of tick-record growth; whether counters should split by
shadow/live segment; and the `raises[0]` priority that decides which word
wins when several raise rules fire at once.

## What already exists (verified by reading the code; suite run: 1506 passed in 60s)

- **The closed vocabulary and the per-decision record.** `Trigger` is closed
  "because task 34 attributes on it" (sensing_controller.py:121-131, module
  docstring :18-19; plan_task29:62,:108 lists it under "Needs sign-off").
  `Decision.to_record()` emits `rates`, `trigger`, `rules_fired`, `reasons`,
  `thermal_scale`, `clamped`, `here_radius_m` (:188-197). The tick loop
  spreads it into `record["sensing"]` beside `shadow`, `advisory_sent`,
  `command_sent`, `send_reason` (run_demo.py:526-532). Tests pin: the word
  describes the rate level, not the last rule checked
  (test_sensing_controller.py:323-355); every word produced is a member
  (:96-105); a dwelling tick reports `idle` while `rules_fired` keeps the
  event (:325-334); a bridged tick names itself (`TestABridgedTickNamesItself`, :613).
- **How the rates are actually decided — global level, per-key modifiers.**
  One boolean `active = dwelled or holding or bridged` selects the whole
  profile (:338-339); thermal multiplies exactly `camera_hz` and `here_hz`
  (:355-356, the free tier deliberately exempt :352-354); the floor clamps
  per key (:358-359). `gps_hz` and `imu_hz` are identical in both profiles
  (:41-42), so no rule ever moves them — with shipped constants the clamp is
  reachable only under monkeypatch (0.05 x 0.15 = 0.0075 > MIN_RATE_HZ
  0.001; the existing floor test at :568 patches constants to reach it).
  So "for which sensor" is a composition chain per key, not a per-sensor
  rule table; a record shaped as rule-x-sensor would be fiction.
- **What the summary attributes today, corrected against the brief.**
  `summary["sensing"] = SensingLoop.to_record()` carries `ticks`,
  `sends_by_reason` (`first`/`changed`/`query_moved`/`heartbeat`,
  sensing_loop.py:176-191), `rate_commands_sent`, `heartbeat_s`, and `mode`
  (run_demo.py:614-618). The `mode` block (shadow_mode.py:151-170) holds
  `mode`, `flips`, `flip_count`, `shadow_predicts`, `structurally_absent`,
  `feed_possible_from_mono`, `reference_rates_hold` — **no `trigger` key**,
  contrary to the brief's starting observation. No trigger count exists
  anywhere in the summary; triggers survive only per-tick and on the wire.
- **Reachability audit of the six `Trigger` values** (the brief asks which
  values the scheme can never emit):
  - `IDLE`, `EVENT`, `NARROW_MARGIN`, `THERMAL`, `HOLD`: reachable live;
    each has a test reaching it (:109, :122, :175, :352, and HOLD via :142).
  - `DISAGREEMENT`: fires only when `feed_congestion` and
    `camera_density_bin` are both present (:248-252); a missing view is
    deliberately no disagreement — task 29's rule, pinned as a mutation
    (scripts/remutate.py:824-828) and tested (:203-207). On a pure-shadow
    drive the feed is structurally absent, so the rule **cannot fire at
    all** (shadow_mode.py:17-24; `ABSENT_IN_PURE_SHADOW` names
    `feed_congestion` and `source_disagreement`, :79-82). As the single
    `trigger` word it is further shadowed by evaluation order: `raises[0]`
    (:381-387) means it names the tick only when neither `EVENT` nor
    `NARROW_MARGIN` fired (:278-286 sets the order).
  - Vocabulary split the current record leaves implicit: only `EVENT`,
    `NARROW_MARGIN`, `DISAGREEMENT`, `THERMAL` can ever appear in
    `rules_fired` (:278-286, :350-351); `IDLE` and `HOLD` are rate-level
    words, never rules. The attribution record makes this split explicit:
    the rules block has exactly those four entries, and level words belong
    to the gates block.
- **Which absences are reachable through the live path.** `policy_margin`
  is None before the first inference or with no head probabilities
  (`_margin`, sensing_loop.py:74-87) — reachable live. `feed_congestion` is
  None whenever the feed owns nothing; the *named* reason already exists as
  `FeedOwnership.declined` (feed_fusion.py:81-97, reasons assigned at
  :138-152) but `inputs_from` discards it (sensing_loop.py:107).
  `ego_acceleration` is **never** None through the live path: the builder
  always writes a float, substituting 0.0 with provenance
  `fallback_neutral` (observation_builder.py:216, :341), so a dead
  accelerometer reaches the controller as a genuine-looking 0.0 — see open
  item 2. `thermal_status`/`skin_temp_c` both-None is reachable (never
  heard), and the controller already treats silence as its own case
  (:431-433).
- **The thermal "why" is computed and then thrown away.** `_thermal_scale`
  knows whether the scale came from silence, staleness, the status word, or
  a skin threshold overriding the status via `min` (:429-465), but returns
  only the float; the cause survives as free text in `reasons`. The
  precedent for structured basis-plus-reason is task 33's `StageTiming`
  (time_sync.py:115-181, basis constants :108-111) and its "absent with a
  named reason" rule.
- **Nothing else consumes the decision record.** `command_for` reads
  `rates`/`trigger`/`here_query` off the object (shadow_mode.py:185-190);
  the dashboard reads no trigger; eval_run.py never touches `sensing`
  (verified by grep over the file); no test constructs `Decision(...)`
  directly (verified by grep), so a new required keyword-only field is safe.

## Decisions taken (by recommendation — not signed off by the user)

| # | Question | Options | Taken | Why |
|---|----------|---------|-------|-----|
| D1 | Where attribution lives | (a) Jetson tick record; (b) also on the wire in `RateCommand` | (a) | The spec and golden frames pin the `rate_cmd` extensions (transport_protocol.md:342); the phone consumes nothing beyond `rates`/`trigger`/`shadow`; task 33's amendment shows a frozen-vector change forces a `PROTOCOL_VERSION` discussion nothing here needs. |
| D2 | Per-rule record shape | (a) keep fired-list only; (b) three-state status per rule with missing inputs named; (c) boolean matrix | (b) | (a) is the defect class verbatim; (c) cannot say *why* a rule was quiet. Statuses: `fired` / `quiet` / `not_evaluable`, mirroring StageTiming's measured/absent-with-reason discipline. Decision behavior is untouched: a missing feed still does not fire (the pin at remutate.py:824-828 must keep passing unchanged). |
| D3 | "For which sensor" | (a) invent a rule-x-sensor matrix; (b) publish the real chain: global gates + per-key composition | (b) | The code decides one level bit and applies per-key modifiers (:338-359); a matrix would attribute agency the code does not have. Per key: base, level-sensitivity, scale applied/exempt, clamp, previous, changed. |
| D4 | Thermal's entry | (a) it can be `not_evaluable` when telemetry is absent; (b) always evaluable, with a closed `cause` | (b) | Silence maps to the `unknown` tier by doctrine ("no news is cool news" refused, :55-57, :431-433); calling that not-evaluable would contradict the code. `cause` ∈ {`status`, `skin_warm`, `skin_hot`, `no_telemetry`, `stale_telemetry`}, null when the scale is 1.0; on a `min` tie the earlier cause stands (only a strict lowering claims it). |
| D5 | How `Decision` carries it | (a) optional field defaulting None/{}; (b) required keyword-only field (`field(kw_only=True)`) | (b) | `_record` is the only construction site (verified); an optional default is a silent-absence path, which is the defect class. Existing keys of `to_record` stay byte-identical; the new key is additive. |
| D6 | Drive-level aggregation | (a) none, leave it to task 39; (b) counts in `SensingLoop` (`decisions_by_trigger`, `rules_by_status`); (c) full tables/percentiles | (b) | (a) leaves `summary["sensing"]` unable to say one word about triggers, which is this task's title; (c) is task 39's session summary generator ("trigger counts" is on its list, task_list.md:1143-1144). Counters live in the loop because `summary["sensing"]` *is* `SensingLoop.to_record()` (run_demo.py:618) and the controller has no record hook. |
| D7 | Naming the feed's absence | (a) `missing: ["feed_congestion"]` bare; (b) also carry `FeedOwnership.declined` into `Inputs` as `feed_declined` | (b) | The named reason exists one call upstream (feed_fusion.py:138-152) and `inputs_from` throws it away (sensing_loop.py:107); writing "missing, reason unknown" when the system knows the reason is a record that names less than it knows. Borders task 36 — flagged as open item 1. |
| D8 | Evidence in rule entries | (a) reason strings only; (b) structured value+threshold echoed per rule | (b) | "Why" as numbers survives aggregation; free text is the text-mining problem task 29 named. The two disagreement literals (0.5, <=0 at :250-252) become named constants echoed in the entry, consistent with `TestTheConstantsAreValuesNotSelfReferences` (:792). `reasons` texts stay exactly as they are. |

## The record, exactly

New module constants in sensing_controller.py: `RULE_FIRED = "fired"`,
`RULE_QUIET = "quiet"`, `RULE_NOT_EVALUABLE = "not_evaluable"`;
`RULES = (Trigger.EVENT, Trigger.NARROW_MARGIN, Trigger.DISAGREEMENT,
Trigger.THERMAL)` in evaluation order; `THERMAL_CAUSES`; and
`JAMMED_CONGESTION = 0.5`, `EMPTY_DENSITY_BIN = 0` used by
`disagreement()` and echoed in its evidence. New frozen dataclasses
`RuleCheck` (status, missing tuple, evidence dict, `to_record` rounding
floats to 4 places) and `Attribution` (rules, gates, per_sensor,
first_decision, `to_record`). `Decision` gains
`attribution: Attribution = field(kw_only=True)` and `to_record()` emits it
under `"attribution"`. The emitted shape, per decision:

```
"attribution": {
  "first_decision": false,
  "rules": {
    "event_from_free_tier":   {"status": "fired", "value": -2.1, "threshold": 1.5},
    "advisory_margin_narrow": {"status": "not_evaluable", "missing": ["policy_margin"]},
    "source_disagreement":    {"status": "not_evaluable", "missing": ["feed_congestion"],
                               "feed_declined": "no_reading"},
    "thermal_backoff":        {"status": "fired", "scale": 0.6, "cause": "skin_warm",
                               "thermal_status": "nominal", "skin_temp_c": 41.2,
                               "telemetry": "fresh", "telemetry_age_s": 0.8}
  },
  "gates": {
    "wants_more": true, "gapped": false,
    "dwell": {"elapsed_s": 0.0, "required_s": 0.5, "satisfied": false},
    "hold":  {"active": false, "remaining_s": null},
    "bridged": false,
    "level": "idle"
  },
  "per_sensor": {
    "camera_hz": {"hz": 0.6, "base_hz": 1.0, "level_sensitive": true,
                  "thermal_exempt": false, "scale": 0.6, "clamped": false,
                  "previous_hz": 1.0, "changed": true},
    "gps_hz":    {"hz": 1.0, "base_hz": 1.0, "level_sensitive": false,
                  "thermal_exempt": true, "scale": 1.0, "clamped": false,
                  "previous_hz": 1.0, "changed": false},
    "imu_hz":    {... same shape ...},
    "here_hz":   {... same shape ...}
  }
}
```

Reading rules, stated here because the validator audits against them:

- `rules` always has exactly the four `RULES` keys. `status` is from the
  closed three-value set. `missing` appears only on `not_evaluable` and
  lists `Inputs` field names (`policy_margin`, `feed_congestion`,
  `camera_density_bin`, `ego_acceleration`) — the same names
  `ABSENT_IN_PURE_SHADOW` uses for the feed (shadow_mode.py:79-82).
  Evidence values appear on `fired` and `quiet` (the signed input and the
  threshold; `event`'s comparison is on `abs()`, the record carries the
  signed value). `feed_declined` rides disagreement's entry when
  `Inputs.feed_declined` is set; absent otherwise, never invented.
- `rules_fired == [name for name in RULES if rules[name].status == "fired"]`,
  order included. This is an identity, not a resemblance; it gets a test
  and a pinned mutation.
- `gates.dwell.elapsed_s` is `now - _raised_since` while a dwell is armed
  (0.0 on the arming tick is a measured zero, the instant discipline of
  StageTiming.instant), and null when no dwell is running — null and 0.0
  mean different things and must not be conflated. `hold.remaining_s` is
  positive while `holding`, null otherwise (never a negative from the
  initial `_holding_until = 0.0`). `level` ∈ {"idle", "active"} and equals
  the profile actually copied at :339.
- `per_sensor[k]`: `hz` equals `rates[k]` (pinned by test — duplication is
  deliberate self-containment, divergence is a bug a test must catch);
  `base_hz` is the chosen profile's value; reconstruction identity
  `_clamp(base_hz * scale) == hz` holds on every decision, with
  `scale == thermal scale` for `camera_hz`/`here_hz` when applied and 1.0
  when exempt or nominal; `clamped` agrees with membership in
  `Decision.clamped`; `previous_hz` is the previous decision's emitted rate
  (null on the first decision, `first_decision: true`, `changed: false` —
  "first" and "unchanged" are distinguishable, mirroring `send_reason`'s
  separate `first`).
- The trigger word remains derived exactly as today (:381-393); a
  consistency test asserts the word against the gates block per branch
  (raise word ⇔ active ∧ (dwelled ∨ bridged) ∧ raises; `holding_after_event`
  ⇔ active ∧ holding without that; `thermal_backoff` ⇔ scale < 1 at idle;
  `idle` otherwise).

## The work

1. **Constants and dataclasses** in sensing_controller.py: statuses,
   `RULES`, `THERMAL_CAUSES`, the two disagreement constants (used inside
   `disagreement()`, behavior identical), `RuleCheck`, `Attribution`.
2. **`decide` builds the checks where it checks.** Each rule site
   (:278-286) produces a `RuleCheck` — fired with evidence, quiet with
   evidence, or not_evaluable with `missing` — and `fired` is derived from
   the checks so the identity with `rules_fired` holds by construction.
   `_thermal_scale` returns `(scale, cause, evidence)` (internal signature,
   no external caller — verified); its `reasons` appends are unchanged.
   `gapped` is passed through to `_record`.
3. **`_record` assembles gates and per_sensor** from what it already
   receives plus `self._last` for `previous_hz`/`first_decision`, and
   constructs the `Attribution` for the required kw-only field on
   `Decision`. Existing keys, texts and the trigger derivation are
   byte-identical.
4. **`Inputs.feed_declined`** (default None) and `inputs_from` carrying
   `feed.declined` when the ownership record is present
   (sensing_loop.py:100-117). The controller reads it only into
   disagreement's not_evaluable entry — it must not influence any decision.
5. **Loop counters**: `SensingLoop.on_tick` increments
   `decisions_by_trigger[decision.trigger]` and
   `rules_by_status[rule][status]` every tick; `to_record` publishes both
   beside `sends_by_reason` (:204-211). `summary["sensing"]` picks them up
   with no run_demo change.
6. **Tests** (below), then **pinned mutations** in scripts/remutate.py under
   a "Task 34" block: (i) the not_evaluable branch reports `quiet` — the
   record stops distinguishing absence from quietness; (ii) the fired set
   is decoupled from the checks so `rules_fired` and the attribution can
   drift; (iii) thermal's `cause` names the status while a skin threshold
   set the scale. Each pin scored on the pytest summary line and failing
   test name per the harness's rule.

No step touches run_demo.py, transport/, specs/, eval_run.py,
shadow_mode.py, or anything on the phone. Nothing calls or configures HERE.

## Tests

- **Three states are three states.** `calm(ego_acceleration=9.0)` → event
  `fired` with value/threshold; `calm()` → event `quiet` at 0.0;
  `calm(policy_margin=None)` → margin `not_evaluable` missing
  `policy_margin`; `calm(feed_congestion=None)` → disagreement
  `not_evaluable` missing `feed_congestion`, and `rules_fired` still lacks
  it (the remutate.py:824 pin's behavior, restated from the record side).
  With `feed_declined="feed_stale"` passed, the entry carries it.
- **Identity with `rules_fired`**, order included, on a decision where all
  four fire (the existing :336-351 scenario).
- **Gates tell dwell from idle.** One tick of evidence: event `fired`,
  `dwell.satisfied` False, `elapsed_s` 0.0 (not null), camera at idle —
  the record now distinguishes "fired, blocked by dwell" from "idle, quiet",
  which `trigger == "idle"` alone could not. A holding tick: all raises
  quiet, `hold.active` True with positive `remaining_s`, word
  `holding_after_event`. A bridged tick: `bridged` True (reusing :613's
  scenario). A redial-length gap: `gapped` True on the resuming tick.
- **Thermal cause.** Status `moderate` → `status`; skin 46 °C under status
  `nominal` → `skin_hot`; stale telemetry → `stale_telemetry`; total
  silence → `no_telemetry`; status `severe` plus skin hot (a `min` tie) →
  cause stays `status`; scale 1.0 → status `quiet`, cause null.
- **Per-sensor chain.** Across a mixed sweep (idle, dwell, raise, hold,
  thermal at every `THERMAL_SCALE` level): reconstruction identity holds on
  every decision and every key; `hz == rates[k]`; `clamped` agrees with
  `Decision.clamped` (reached via the same monkeypatch as :568); `gps_hz`
  and `imu_hz` are `level_sensitive` False, `thermal_exempt` True, and
  never `changed`; first decision has `first_decision` True, null
  `previous_hz`, `changed` False everywhere; a raise tick shows
  `camera_hz.changed` True from 1.0.
- **Word-versus-gates consistency** across the :633-style long mixed drive:
  every tick's `trigger` matches the branch its gates block implies.
- **Vocabulary closure**: every status, cause, and level ever produced is a
  member of its closed set (the :96-105 pattern).
- **Loop counters**: `sum(decisions_by_trigger.values()) == ticks`;
  statuses count correctly across a scripted sequence; both keys present in
  `to_record` and therefore in `summary["sensing"]`.
- **Compatibility**: the full suite (1506 tests at baseline, verified by
  running) passes unmodified except where a test constructs decisions via
  the controller — no existing assertion changes meaning; `to_record`'s
  pre-existing keys are unchanged.

## What this establishes, and what it does not

It establishes, per decision, the three answers the task names — which rule
(four per-rule entries with a closed status), for which sensor (the per-key
composition chain that reconstructs each emitted rate), and why (structured
evidence, gate state, thermal cause, and named missing inputs) — and makes
"never evaluated", "evaluated and quiet", and "fired" three different
records, which is this task's instance of the recurring defect class. It
also puts the first trigger-attribution numbers into the run summary, so a
drive can be asked "what raised the camera and how often" without re-reading
every tick.

It does not change a single decided rate, word, or reason string; does not
put attribution on the wire; does not attribute *sends* beyond what
`sends_by_reason` already does; does not see through the builder's
`fallback_neutral` substitution (a dead accelerometer still reads as a quiet
0.0 — task 36's provenance is the honest fix, open item 2); does not split
counters by shadow/live segment (task 35 owns segment scoring); and does not
render any of it (task 39).

## Measured on a device run, 2026-09-01

899 ticks over 180 s, phone dialling the Jetson over the tailnet on a direct path,
no USB in the data path. Every tick carried an attribution block. These answer
five questions the audit could not, because every check before this constructed
`Inputs` directly and nothing had run `inputs_from` against real telemetry.

- **All three states appear on real data.** `advisory_margin_narrow` fired on 899
  ticks, `event_from_free_tier` and `thermal_backoff` were quiet on 899, and
  `source_disagreement` was `not_evaluable` on 899 with `missing:
  ["feed_congestion"]`. That last row is the point of the task: HERE is not
  configured, so the rule genuinely cannot be evaluated, and the record says so
  rather than reporting the quiet 0.0 that would otherwise be indistinguishable
  from a calm road.
- **`feed_declined` is non-null on a real drive**, carrying `"feed_outcome"` on
  every tick. Open item 1 asked whether the field delivers anything; it does, and
  the answer is to keep it.
- **The thermal cause was null on all 899 ticks**, so the scale never left 1.0 and
  no tick reported `no_telemetry` or `stale_telemetry`. The telemetry thread did
  not die during this run.
- **`policy_margin` was never `not_evaluable`**, and `gapped` was false on all 899
  ticks, so the dwell was never re-armed by a gap.
- **Record growth, corrected.** Open item 4 estimated 0.8-1.0 KB of attribution on
  a roughly 1.5 KB record, about +60%. Measured: attribution is 1354 B on a mean
  tick record of 7847 B, so the absolute size is about a third larger than
  estimated while the proportional cost is 17%, not 60% — the base record is some
  five times the size the estimate assumed.

One consequence for the reachability audit: with HERE unconfigured,
`source_disagreement` cannot be evaluated at all on a real drive, so
`Trigger.DISAGREEMENT` is unreachable in practice as well as structurally
unreachable on shadow drives. The observed triggers were `advisory_margin_narrow`
on 896 ticks and `idle` on 3.

## Open items flagged for the user

1. **`Inputs.feed_declined` borders task 36.** Taken (D7) because the named
   reason already exists and discarding it writes a record that names less
   than the system knows — but provenance-into-the-controller is task 36's
   subject, and if task 36 lands a general mechanism this field may become
   redundant. Say whether to keep it, or to hold attribution to bare
   `missing` names until task 36.
2. **A dead accelerometer is recorded as `quiet` at 0.0.** The builder
   always writes a float (`fallback_neutral`, observation_builder.py:216)
   and its provenance never reaches `Inputs`, so event's `not_evaluable`
   branch is unreachable through the live path. My first instinct was to
   mark it `not_evaluable` off `field_sources`; the code contradicted the
   cheap version (the controller cannot see provenance today), so this is
   flagged rather than silently chosen. The branch still exists and is
   tested via direct `Inputs`.

   **The same substitution hides a second field, in the opposite direction.**
   `local_density_bin` is derived from the detection count and
   `field_sources` marks it `derived` unconditionally, so a camera that sees
   nothing yields `n_local = 0`, a bin index of 0, and `disagreement(0.9, 0)`
   returns True. A blind camera beside a congested feed therefore **fires**
   the rule and raises the camera rate, recorded as `{"status": "fired",
   "camera_density_bin": 0}` — faithful to what the controller saw, and
   misleading about the road. The accelerometer case under-reports and
   changes nothing; this one over-reports and moves a rate. Whoever acts on
   this item should carry both fields, not the accelerometer alone.
3. **The word `quiet`.** No strong preference against `did_not_fire` or
   `evaluated_false`; the vocabulary is load-bearing for every later reader,
   so the name deserves sign-off.
4. **Tick-record growth.** ~0.8-1.0 KB per tick on records of ~1.5 KB
   (load_records' own estimate, eval_run.py:74-75) — roughly +60% on the
   metadata log, ~30 MB over a two-hour 5 Hz drive. Judged acceptable and
   recorded because it is per-tick forever; a compact encoding was not
   chosen because dropped keys would reintroduce silent absence.
5. **Counters do not split by mode segment.** A drive that flipped
   shadow→live mixes both in one count; the flip instants are in
   `mode.flips` so task 39 can segment offline. Deferred, weak preference.
6. **`raises[0]` stays the word's tie-break** (:381-387): when several raise
   rules fire, the single word still under-reports by evaluation order. The
   attribution record supersedes the word for analysis, so re-ordering or
   pluralizing the wire word was left alone — but it remains the one field
   the phone sees.

## Open items carried in

- Task 27's HERE parse has never met a real response body. (No HERE work in
  this task; no key appears in any URL or log.)
- Task 28's observation vector cannot be feed-informed without a
  simulator-side change.
- `MAX_QUERY_RADIUS_M` (10 km) unchecked against HERE v7's accepted range.
- `achieved["camera_hz"]` overstates the sustained rate when the channel
  evicts; needs `Session.send` to surface the displacement.
- Task 33's own residue (task_list.md:1128-1134): the CameraX clock
  assumption on `capture_to_encode_start`, `fuse`'s unreachable absent
  branch, one `Thread.sleep(200)`, and ~1.5% unmatched joins on pre-task-33
  phone logs.
