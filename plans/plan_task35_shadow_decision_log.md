# Task 35 — Shadow-mode decision log emitted alongside the full-rate reference

> Written by `plan_dsrc_rec`: every decision below was taken by recommendation,
> without putting the question to the user. The decisions table says what was
> chosen and why; **none of it is user-approved**, and the "Open items flagged
> for the user" section holds every point where the recommendation was weak or
> where the code contradicted the first reading. This plan is the fixed target
> a validator audits the implementation against.

## The short version

Task 35 (plans/task_list.md:1190) asks for a "shadow-mode decision log emitted
alongside the full-rate reference, so every candidate policy can be scored
against identical traffic from one drive." Most of the *log* half landed with
tasks 30, 31 and 34 and is inventoried below: every tick already records the
full decision with its three-state attribution, the command's `shadow` flag,
and what was sent (run_demo.py:525-532), and the run summary already records
the mode, every flip, and the shadow log's structural limits
(shadow_mode.py:151-170 via sensing_loop.py:226). What does **not** exist is
the *scorable* half, and each gap is an instance of the project's recurring
defect class — a record that cannot distinguish failure from success:

1. **The exact inputs are not logged as a unit.** The controller's `Inputs`
   (sensing_controller.py:221-256, thirteen fields) can today be only partly
   scavenged from the tick record: `obs` carries three fields unrounded
   (pipeline.py:136), the attribution evidence carries others **rounded to
   four places** (`RuleCheck.to_record`, sensing_controller.py:188), and the
   thermal trio at the decide instant plus `feed_congestion`/`feed_declined`
   exist nowhere else at all. A candidate replayed from rounded copies can
   flip a threshold comparison and the log could not say whether the candidate
   differed or the reconstruction did.
2. **The decide instant is not logged.** Dwell, hold, bridge and gap all
   compare differences of the controller's own `now`
   (sensing_controller.py:456-497, :460-461, :495-496), so a stateful replay
   without the recorded instants is a different drive.
3. **The full-rate reference is asserted, never witnessed.** The phone
   reports the rates it actually achieved on every telemetry frame
   (`PhoneTelemetry.achieved`, transport/messages.py:598, decoded at :653),
   and the Jetson throws that field away: the per-tick record carries no
   telemetry and the summary's telemetry block (phone_link.py:908-918) has
   thermal fields only. Nothing in the artifact proves the phone was at
   full rate while the shadow decisions were being recorded.
4. **Nothing scores.** `eval_run.py` never reads a `sensing` key (verified
   by grep over the file), and nothing anywhere replays a controller against
   a logged drive. Task 30 explicitly deferred "the scoring itself" to this
   task (plan_task30_shadow_live.md, "Scope boundary": "Out: sending (task
   31); the scoring itself (35)").

The plan, in one paragraph: (1) log the exact `Inputs` once per tick as
`sensing.decision_inputs` (thirteen keys, unrounded), plus `decided_at_mono`
— the controller's own clock read — on the decision record, so `decide` is
replayable as a pure function of the log; (2) log a per-tick `reference`
block carrying the phone's latest `achieved`/`dropped` self-report and its
age, absent-with-a-named-reason when the phone never reported, so the
"full-rate reference" is a measurement instead of an assumption; (3) a new
offline tool `score_shadow.py` that first proves the log can reproduce the
incumbent byte-for-byte (the replay-identity gate), then scores candidate
controllers against the same recorded inputs, reusing task 34's closed
three-state rule vocabulary, with `rules_never_exercised` as a mandatory,
named output; (4) say in the score itself, quoting the mode record, what
"identical traffic" does and does not mean for this drive. Nothing the
controller decides changes, and the wire is untouched.

**Scope boundary.** In: `sensing_controller.py` (`Inputs.to_record`/
`from_record`, `Decision.decided_at_mono`), `sensing_loop.py` (`TickOutcome`
gains `inputs` and `reference` and a `to_record()`), `run_demo.py` (the
`record["sensing"]` construction moves onto `TickOutcome.to_record()`,
existing keys byte-identical), `phone_link.py` (the summary telemetry block
gains `achieved`/`dropped`), the new `deployment/jetson/score_shadow.py`,
tests for all of it, and pinned mutations in scripts/remutate.py. Out: the
wire and both codecs (`RateCommand`, `PhoneTelemetry` already carry
everything needed; no golden-frame or `PROTOCOL_VERSION` work); the phone
(zero Kotlin changes); `eval_run.py` (the scorer imports its `load_records`
and changes nothing in it); field provenance into `Inputs` (task 36); a
dedicated telemetry record type (task 37 — see open item 2); summary
rendering (task 39); the colocated live-vs-shadow check (task 43 — this task
builds the offline half of that property, not the bench half); and any
runtime mode-flip surface (nothing in run_demo flips today — verified, no
`flip_to` call site in run_demo.py — and the log format already handles
flips because the per-tick `shadow` flag is logged).

**Open decisions flagged for the user** (details at the bottom): where the
reference segment ends (mode-keyed, matching `reference_rates_hold`, versus
delivery-keyed); whether the per-tick `reference` block should instead be
task 37's 1 Hz telemetry record type; the candidate contract requiring
attribution rather than degrading to rates-only scoring; and the fact that
"candidate policy" is read as *candidate sensing controller*, not candidate
RL advisory bundle.

## What "identical traffic" can and cannot mean here

This is the spine of the task, and `shadow_mode.py`'s module docstring
already states it as three limits (shadow_mode.py:14-52). Quoted, because the
scorer's output must carry them rather than a paraphrase a reader can argue
with:

1. *"The traffic feed is structurally absent from a pure shadow drive."* —
   "The phone makes no HERE call until the Jetson tells it what to ask, and
   `ConfigApplier` returns on the shadow branch **before** it reaches
   `setHereQuery`. So a drive that has only ever been in shadow has no query,
   makes no call, and sends no `here` frame; `HereFeed` stays empty,
   `feed_congestion` is None on every tick, and `Trigger.DISAGREEMENT` -- one
   of the controller's three raise rules -- cannot fire at all. A shadow log
   therefore cannot credit or debit any candidate policy for that rule. This
   is not a degraded input; it is a missing one." (shadow_mode.py:17-24).
   Verified against the code on both ends: the shadow branch returns at
   ConfigApplier.kt:61-64, before every target call including `setHereQuery`
   (:73-82), and a null query on the phone is counted `unconfigured` with no
   HTTP call made (HerePipeline.kt:77-82). Verified in the field by task
   34's device run: `source_disagreement` was `not_evaluable` on 899 of 899
   ticks with `missing: ["feed_congestion"]` (plan_task34.md:349-356), and
   that plan's conclusion stands: "`Trigger.DISAGREEMENT` is unreachable in
   practice as well as structurally unreachable on shadow drives"
   (plan_task34.md:370-374).
2. *"Reference rates are only reference until the first live segment."* —
   "the discriminator is whether this drive has **ever** been live, not
   whether it has left live" (shadow_mode.py:29-32), which is what
   `reference_rates_hold` keys on (:169).
3. *"And the trajectory diverges."* — "a shadow log predicts the **decision
   function** exactly, tick for tick against the same inputs, and does
   **not** predict the **trajectory** -- a live drive feeds its own reduced
   observations back in and a shadow drive never does." (shadow_mode.py:43-48).

So, plainly:

- **"Identical traffic" CAN mean:** every candidate is evaluated against the
  same recorded per-tick inputs — the full-rate observation stream the
  reference rates produced — so candidate-versus-incumbent and
  candidate-versus-candidate comparisons are of decision functions on
  identical arguments, tick for tick, from one drive. The phone's startup
  defaults (SensingConfig.kt:17-20: camera 5.0 Hz, gps 1.0, imu 50.0, here
  0.2) equal the controller's `ACTIVE_RATES` (sensing_controller.py:42), so
  a pure shadow drive really is a *full-rate* reference — and after this
  task, that is witnessed per tick by the phone's own `achieved` report
  rather than assumed from a Kotlin default.
- **"Identical traffic" CANNOT mean the HERE traffic feed.** On a pure
  shadow drive there is no feed to be identical: no candidate can be
  credited or debited on the disagreement rule, ever, and a score that
  reports 100% agreement without saying so has silently never exercised one
  of the three raise rules. `rules_never_exercised` is therefore a mandatory
  output, not a footnote (D7).
- **"Identical traffic" CANNOT mean the traffic a candidate would itself
  have produced.** The score is of the decision function; no closed-loop
  claim survives the first tick on which a candidate would have changed a
  rate. The scorer prints limit 3 verbatim on every report.
- **And it stops meaning "the full-rate reference" at the first live
  moment.** Ticks from then on are still identical inputs across candidates,
  but they are inputs shaped by the incumbent's applied commands — the
  comparison remains valid as a decision-function comparison and stops being
  "against the reference". The scorer splits its counts into `reference` and
  `contaminated` segments ("Entering live is what contaminates a log" is the
  code's own word, shadow_mode.py:116-117) rather than pooling them.

**This task's instance of the recurring defect class**, designed against by
name: a shadow log that cannot distinguish *"this candidate policy would have
decided differently"* from *"this candidate policy was never given the inputs
to decide on"*. Both must be first-class, named states: the first is a
`differ` row with both decisions' evidence, the second is `not_evaluable`
with the missing inputs named (task 34's word, reused), excluded from every
agreement denominator with the exclusion count printed beside the ratio, and
rolled up into `rules_never_exercised` when it covers the whole drive.

## What already exists (verified by reading the code; suite run: 1547 passed in 61 s)

- **The per-tick decision log.** `record["sensing"]` spreads the full
  `Decision.to_record()` — `rates`, `trigger`, `rules_fired`, `reasons`,
  `thermal_scale`, `clamped`, `here_radius_m`, `attribution` with task 34's
  three-state rules/gates/per_sensor — beside `shadow` (the command's flag),
  `advisory_sent`, `command_sent`, `send_reason` (run_demo.py:525-532;
  Decision.to_record at sensing_controller.py:284-294). The command is built
  every tick whatever its fate, and `TickOutcome.command`'s own comment
  names this task: "Present even when nothing was sent, because the decision
  log task 35 scores is about what was DECIDED" (sensing_loop.py:64-67).
- **The mode record.** `summary["sensing"]["mode"]` carries `mode`, every
  flip with its instant, `flip_count`, `shadow_predicts` ("the decision
  function, not the trajectory"), `structurally_absent`
  (`["feed_congestion", "source_disagreement"]` for any drive not live from
  its first tick, shadow_mode.py:79-82, :166-167), `feed_possible_from_mono`,
  and `reference_rates_hold` keyed on ever-live (:151-170). Existing tests
  pin all of it (test_shadow_mode.py:178-347, e.g.
  `test_the_record_names_the_inputs_a_shadow_drive_cannot_have` :235,
  `test_the_disagreement_rule_really_is_unreachable_without_a_feed` :323).
- **The mode never reaches the decision.** `decide` takes no mode
  (pinned: test_shadow_mode.py:76), `command_for` is the one reader of the
  mode and the two modes differ by exactly one boolean
  (shadow_mode.py:173-191; pinned :44), and a drive replayed in both modes
  makes identical decisions (:85). This is the property the scorer's replay
  rests on and task 43 later checks against live gating.
- **Decisions are a deterministic function of `Inputs`, controller state,
  and the clock.** `decide(inputs)` reads `self._now()` once (:409), the
  gate machinery compares differences of that instant against stored
  instants (:456-497), and state updates happen only in `_record`
  (:615-617). The controller takes an injectable clock (:398-401), and
  `SensingLoop` shares its own clock with the controller it builds
  (sensing_loop.py:135-136). `Decision(` is constructed in exactly one
  place, sensing_controller.py:609 (verified by grep over deployment/jetson
  including tests), so new required kw-only fields are safe, as task 34's D5
  already established for `attribution`.
- **What the record can and cannot reconstruct today.** From the tick
  record: `ego_acceleration`, `ego_speed`, `local_density_bin` unrounded in
  `obs` (pipeline.py:136, values built at observation_builder.py:340-341,
  :366), `head_probs` unrounded (:141) from which `_margin` is recomputable
  (sensing_loop.py:74-87), gps lat/lon/valid (:152-163), `gps_age_s` in
  `obs_diagnostics`. From nowhere: the thermal trio as the controller saw it
  (`thermal_status`/`skin_temp_c`/`telemetry_age_s` survive only inside
  attribution evidence, floats rounded at sensing_controller.py:188),
  `feed_congestion` (in evidence only when both views were present, rounded)
  and `feed_declined` (in evidence only on the `not_evaluable` branch,
  :373-377), and the decide instant (not logged at all).
- **The reference is reported and discarded.** `PhoneTelemetry` carries
  `achieved` per `RATE_KEYS` and `dropped` per `DROP_KEYS`
  (transport/messages.py:598-599, :618, :625, decoded :653-654). On the
  phone, `achieved` is a windowed average of deliveries and `dropped` is
  cumulative (TelemetryReporter.kt:131, :184, doctrine at :12-19: "applies
  what arrives and reports what it achieved"). On the Jetson the latest
  report is kept as one atomic tuple (phone_link.py:624) and `inputs_from`
  reads only its thermal fields (sensing_loop.py:114-116); `achieved` and
  `dropped` appear in no record anywhere (summary telemetry block,
  phone_link.py:908-918, has thermal fields only; verified by grep for
  "achieved" over deployment/jetson excluding tests — only messages.py hits).
- **The offline surfaces.** `eval_run.load_records` collects `type=="tick"`
  records and returns unparseable-line counts rather than swallowing them
  (eval_run.py:70-99); it reads no `sensing` key (verified by grep). The
  metadata log is the flushed artifact; `summary.json` is written only at
  close (metadata_logger.py:72-74), so a truncated run may have ticks and no
  summary — the scorer must not require summary.json (D6 note).
- **Mode is fixed per run today.** `ModeHolder(LIVE if args.live_rates else
  SHADOW)` at construction (run_demo.py:449-450), no runtime flip call
  anywhere in run_demo (verified by grep). Segmentation logic must still
  exist in the scorer because the per-tick flag is the log's truth, not the
  process's argument.

## Decisions taken (by recommendation — not signed off by the user)

| # | Question | Options | Taken | Why |
|---|----------|---------|-------|-----|
| D1 | Where the shadow decision log lives | (a) a new `{"type": "shadow_decision"}` line per tick; (b) new keys inside the tick record's existing `sensing` block | (b) | The decision is *about* the tick; a second line needs a join key and can drift from the tick's own sensing block — two records that can disagree about one decision. `load_records` already collects ticks (eval_run.py:91-97) and the join is free. |
| D2 | Replay substrate | (a) scavenge `obs`/`head_probs`/attribution evidence offline; (b) log the exact `Inputs` once per tick with a symmetric `from_record` | (b) | Evidence floats are rounded to 4 places (sensing_controller.py:188) — a replay from rounded values can flip a threshold comparison, which is the offline-join defect's shape (task 33's 5,700x-bound instance, task_list.md:1084-1087). Three inputs exist nowhere else. (a) also re-implements `inputs_from` offline, a second copy that drifts. |
| D3 | The replay clock | (a) log the loop's `now`; (b) log the controller's own clock read as `Decision.decided_at_mono` | (b) | The gates compare differences of the controller's `now` (:456-497); the loop's read (sensing_loop.py:154) is microseconds earlier, and a dwell boundary tick can flip on that. The controller has `now` in hand where the Decision is built (:536-538), so (b) makes replay exact by construction rather than almost-always. |
| D4 | The full-rate reference witness | (a) assume SensingConfig defaults; (b) per-tick `reference` block from the latest `PhoneTelemetry`; (c) a separate 1 Hz telemetry record type | (b) | (a) writes a record that names less than the system knows — the phone measures `achieved` and sends it every second, and today the Jetson discards it. (c) is task 37's shape (thermal/throttle log) and needs a time join; flagged as open item 2. The per-tick block is ~180 B against a 7.8 KB measured mean record. |
| D5 | What a "candidate policy" is | (a) a candidate sensing controller: anything with `.decide(Inputs) -> Decision`; (b) also candidate RL advisory bundles | (a) | `Inputs` is the controller's whole world (sensing_controller.py:221-256). A different RL bundle changes `head_probs` and therefore `policy_margin`, which requires replaying the observation stream through the new bundle — that is upstream of the decision log and trajectory-shaped. The fixed upstream is exactly what makes the inputs identical; stated as an accepted limit, and flagged (open item 4) because it is an interpretation of the task's word "policy". |
| D6 | Scoring vocabulary | (a) invent comparison states; (b) import task 34's `RULE_FIRED`/`RULE_QUIET`/`RULE_NOT_EVALUABLE` and add only the comparison verdicts `agree`/`differ` | (b) | The brief's instruction, and the code already owns the words (sensing_controller.py:144-146). A second vocabulary for the same three facts is drift by construction. |
| D7 | Not-scoreable ticks | (a) fold into agreement (both sides did nothing); (b) exclude from every ratio, print the exclusion beside the ratio, and roll drive-wide absence into a mandatory `rules_never_exercised` | (b) | (a) is the defect class verbatim: a candidate that would differ on the disagreement rule scores 100% agreement on a HERE-less drive and the record cannot say the rule was never given its inputs. Two rules both `not_evaluable` is "not exercised", never "agree". |
| D8 | When replay identity fails | (a) warn and score anyway; (b) refuse to emit candidate scores, exit 2, name the first mismatching tick and keys | (b) | A log that cannot reproduce its own policy's decisions cannot referee anyone else's; scores from it would be numbers with no chain of custody. The identity is also this task's standing proof that the log is sufficient — the offline half of task 43's property. |
| D9 | Where the reference segment ends | (a) at the first tick whose command was live (`sensing.shadow == false`), regardless of delivery; (b) at the first live command actually sent (`command_sent == true`) | (a) | Matches `reference_rates_hold`'s own keying — "a drive that is still live has not held the reference rates either" (shadow_mode.py:29-32) — and is conservative: it can only under-claim reference ticks. (b) is physically truer (an unsent live command changed nothing on the phone) but invents a delivery semantics the mode record does not have. No strong preference — flagged as open item 1. |
| D10 | Pre-task-35 logs | (a) approximate replay from rounded attribution evidence; (b) named refusal: `not_scoreable: decision_inputs absent (pre-task-35 log)` | (b) | (a) is D2's rejected option wearing a compatibility hat. The refusal is per-run and loud, the precedent is eval_run's "an old run without `stages` still reports rather than crashing" — report, name, do not fabricate. |
| D11 | Who builds `record["sensing"]` | (a) keep run_demo's inline dict and append keys there; (b) move construction onto `TickOutcome.to_record()` and have run_demo call it | (b) | Task 34's round-1 blocker was that the emitted record was the one thing no test read (task_list.md:1154-1156). (b) puts the emitted shape where test_sensing_loop.py can assert it without a fake run_demo loop; run_demo's change is one expression. Existing keys stay byte-identical, pinned by test. |

## The record, exactly

Three additions to `record["sensing"]`; every pre-existing key is
byte-identical. Emitted per tick:

```
"sensing": {
  ... Decision.to_record() exactly as today (rates, trigger, rules_fired,
      reasons, thermal_scale, clamped, here_radius_m, attribution) ...,
  "decided_at_mono": 12345.678901234,          # NEW: the controller's own clock read
  "decision_inputs": {                          # NEW: the exact Inputs, one block
    "ego_acceleration": 0.03125,               # unrounded; null means the controller saw None
    "ego_speed": 13.4,
    "policy_margin": 0.03199999999999998,      # full precision, not the evidence's 4 places
    "feed_congestion": null,
    "camera_density_bin": 1,
    "feed_declined": "no_reading",
    "thermal_status": "nominal",
    "skin_temp_c": 33.8,
    "telemetry_age_s": 0.41,
    "lat": 37.42, "lon": -122.08,
    "position_valid": true,
    "position_age_s": 0.2
  },
  "reference": {                                # NEW: what the phone says it is running
    "achieved": {"camera_hz": 4.97, "gps_hz": 1.0, "imu_hz": 49.8, "here_hz": 0.0},
    "dropped": {"camera": 61, "gps": 0, "imu": 0, "here": 0},
    "age_s": 0.41,
    "absent": null                              # or: achieved/dropped/age_s all null, "absent": "no_telemetry"
  },
  "shadow": true,
  "advisory_sent": true, "command_sent": false, "send_reason": null
}
```

Reading rules, stated here because the validator audits against them:

- `decision_inputs` has exactly the thirteen `Inputs` field names
  (sensing_controller.py:222-256), values copied without rounding. JSON
  round-trips finite doubles exactly (Python's `json` serialises via the
  shortest-repr contract); a test pins a 17-significant-digit value through
  `to_record` -> `json.dumps` -> `json.loads` -> `from_record`. A null is a
  faithful copy of a None the controller actually saw — the absence
  semantics are already named by attribution's `not_evaluable`/`missing`,
  so null here is data, not missing logging.
- `Inputs.from_record` is strict both ways: a missing key and an unknown key
  are both refusals, not defaults — a schema drift between writer and reader
  must be loud, because a silently defaulted input is a silently different
  replay.
- `decided_at_mono` is the instant `decide` read at sensing_controller.py:409,
  carried into `_record` and onto the Decision as a required kw-only field
  (the same construction discipline as `attribution`, :282). Replay injects
  exactly this sequence as the candidate's clock. `reference.age_s` and
  `decision_inputs.telemetry_age_s` are computed against the loop's `now`
  (sensing_loop.py:154) and ride as data, so the microsecond skew between
  the two clock reads cannot reach any replayed comparison.
- `reference.achieved` is the phone's windowed average (TelemetryReporter.kt:131,
  :184); `dropped` is cumulative; `age_s` is now minus the report's arrival
  (the same `telemetry_at_mono` inputs_from uses, phone_link.py:254-256).
  When the phone has never reported: all three null and
  `"absent": "no_telemetry"` — never zeros, because a phone reporting zero
  achieved and a phone never heard from are different drives (the defect
  class, and the same rule the summary block already follows at
  phone_link.py:910-912).
- The identity the whole task rests on, pinned by test and mutation: a
  `SensingController` constructed with a clock replaying the logged
  `decided_at_mono` sequence and fed `Inputs.from_record(decision_inputs)`
  tick by tick emits `to_record()` dicts equal to the logged decision keys
  on every tick — including `reasons` free text, `attribution`, and
  `here_radius_m`. Compared at the record level, not the dataclass level
  (task 34's lesson: the emitted record is what must be pinned).

`summary["phone"]["telemetry"]` additionally gains `"achieved"` and
`"dropped"` from the latest report (null when never heard), beside the
thermal fields it already carries (phone_link.py:908-918) — the drive-level
form of the same witness.

### The scorer: `deployment/jetson/score_shadow.py`

```
python3 deployment/jetson/score_shadow.py <run_dir> \
    [--candidate label=module:factory ...] [--no-json]
```

- Loads ticks via `eval_run.load_records` (import, no change to eval_run).
  Refusals, each named in the output and exiting 2: no tick carries a
  `sensing` block (a phoneless run — sensing is built only with a phone,
  run_demo.py:444-445); `decision_inputs` absent (pre-task-35 log, D10).
- **Replay-identity gate first, always, candidates or none.** Replays the
  incumbent `SensingController` as above and reports
  `replay_identity: {status, ticks, mismatched, first_mismatch: {tick_id,
  keys}}`. On any mismatch: no candidate scores are emitted, exit 2 (D8).
  Running the tool with zero candidates is exactly this log-validity check.
- Candidates: `factory(clock)` returns an object whose
  `.decide(Inputs)` returns a Decision-like with `.to_record()`. The clock
  handed in returns the current tick's `decided_at_mono` for every call
  within that tick (held, not popped — a candidate may read its clock more
  than once per decide). A candidate whose records lack `attribution.rules`
  with the closed statuses is refused by name
  (`candidate_without_attribution`), not silently scored on rates alone
  (open item 3).
- Segments: `reference` = ticks strictly before the first tick with
  `sensing.shadow == false`; everything from that tick on is `contaminated`
  (D9). Every count below is reported per segment and totalled; the mode
  block is echoed from summary.json when present, else derived from the
  per-tick flags and marked `"mode_derived_from_ticks": true` (summary.json
  is written only at close, metadata_logger.py:72-74, and must not be a
  hard dependency).
- Output, `shadow_score.json` in the run dir plus a stdout table:

```
{
  "run": "...", "ticks": N, "unparseable_lines": u,
  "replay_identity": {"status": "ok", "ticks": N, "mismatched": 0, "first_mismatch": null},
  "segments": {"reference_ticks": R, "contaminated_ticks": C, "first_live_tick_id": null},
  "limits": {
    "shadow_predicts": "the decision function, not the trajectory",
    "structurally_absent": ["feed_congestion", "source_disagreement"],
    "reference_rates_hold": true
  },
  "reference_witness": {"ticks_with_achieved": n, "ticks_no_telemetry": m,
                        "achieved_mean": {...}, "dropped_final": {...}},
  "candidates": {
    "<label>": {
      "rules": {"<rule>": {"fired": n, "quiet": n, "not_evaluable": n}},
      "rules_never_exercised": [
        {"rule": "source_disagreement", "ticks": N,
         "missing": {"feed_congestion": N}, "feed_declined": {"no_reading": N}}
      ],
      "vs_incumbent": {
        "rates": {"same": n, "differ": n, "first_differ_tick_id": id,
                  "mean_commanded": {...}, "incumbent_mean_commanded": {...}},
        "trigger": {"same": n, "differ": n},
        "per_rule": {"<rule>": {"agree": n, "differ": n, "not_evaluable": n}}
      },
      "activity": {"ticks_active": n, "raises": n}
    }
  }
}
```

Reading rules for the score: `per_rule` verdicts come from the closed set
`{agree, differ, not_evaluable}` where `not_evaluable` means the inputs were
absent (shared across candidates by construction — evaluability is a
property of the inputs); every `agree`/`differ` ratio's denominator excludes
`not_evaluable` and the excluded count is printed beside it;
`rules_never_exercised` lists any rule `not_evaluable` on 100% of a
segment's ticks and is printed in the stdout table even when — especially
when — agreement is otherwise total. The three `limits` strings are emitted
on every report, verbatim from the mode record. `mean_commanded` is a
decision-function statistic (what the candidate would have asked for), never
labelled as consumption or outcome.

**Adjudication (2026-09-01): `reference_witness` is segment-scoped; `per_rule`
stays flat.** The JSON template above shows one flat `reference_witness`
block, which contradicts this section's own "every count reported per
segment" — a drive with zero reference ticks reported a full-rate witness
for a segment its own report calls zero ticks long. Resolved in favour of
the prose for `reference_witness` alone: it is computed over the `reference`
segment's ticks only, and the `contaminated` segment gets its own witness
under `reference_witness_contaminated` (`null` when the drive has no
contaminated ticks). `per_rule` and `rules` keep the flat, whole-drive
template as written: those counts are about candidate-versus-incumbent
agreement, not about what the phone was witnessed running, so nothing about
`reference_rates_hold` bears on them.

## The work

1. **`Inputs.to_record()` / `Inputs.from_record()`** in sensing_controller.py:
   exact copies, strict symmetric refusal (D2). No `decide` behaviour change.
2. **`Decision.decided_at_mono: float = field(kw_only=True)`**, set in
   `_record` from the `now` already in hand (:536-538), emitted by
   `to_record` as a new key; construction site :609 is the only one
   (verified by grep).
3. **`TickOutcome` gains `inputs: Inputs` and `reference: dict`** (both
   required — an optional default is a silent-absence path, task 34 D5's
   rule), and **`to_record()`** producing today's five run_demo keys
   byte-identically plus the three new blocks. `on_tick` builds the
   reference block from `phone.telemetry` / `phone.telemetry_at_mono`
   against the same `now` it already uses (sensing_loop.py:154). No-phone
   ticks get the named-absent reference.
4. **run_demo.py:525-532** becomes `record["sensing"] = outcome.to_record()`
   (D11). Nothing else in run_demo changes.
5. **phone_link.py:908-918**: `achieved` and `dropped` added to the summary
   telemetry block, null when never heard.
6. **`score_shadow.py`** as specified, importing `load_records` from
   eval_run and the vocabulary constants from `policy.sensing_controller`
   (never re-declaring them).
7. **Tests** (below), then **pinned mutations** in scripts/remutate.py under
   a "Task 35" block, scored on the pytest summary line and the failing test
   name per the harness's rule (remutate.py header).

No step touches transport/, specs/, phone/, ui/, or eval_run.py's own code.
Nothing calls or configures HERE; no key exists anywhere in this task's
surface.

## Tests

- **The inputs block is exact.** Round-trip `Inputs -> to_record -> json ->
  from_record -> Inputs` equality for an all-None and an all-set instance;
  a 17-significant-digit float survives unchanged (pins D2 against anyone
  "simplifying" to the rounded evidence); `from_record` refuses a missing
  key and an unknown key by name.
- **The emitted record is the tested record** (task 34's lesson applied at
  birth): `TickOutcome.to_record()` asserted key-for-key in
  test_sensing_loop.py — the five pre-existing keys byte-identical to the
  shape run_demo used to build inline, plus the three new blocks; and a
  run_demo-level test confirming the tick log line carries them
  (test_run_demo_loop.py's fake-logger pattern, :147-155).
- **Replay identity, stateful and exact.** Drive a controller through a
  scripted mixed sequence — idle, one-tick event (dwell armed, not
  satisfied), dwell satisfied, hold, bridge, a gap wider than
  `MAX_EVIDENCE_GAP_S`, thermal at every scale tier, a clamp via the MIN_RATE_HZ
  monkeypatch (test_sensing_controller.py:625-631) — logging each tick's record; rebuild from the records alone;
  require `to_record()` equality on every tick. A variant replays with the
  loop's `now` instead of `decided_at_mono` shifted by 300 µs across a
  dwell boundary and asserts the *shifted* replay diverges — demonstrating
  D3 is load-bearing, not decorative.
- **The reference block cannot lie about absence.** No telemetry ->
  `absent: "no_telemetry"` with three nulls, never zeros; with telemetry ->
  the achieved map echoed with its age; the summary block ditto.
- **The scorer's three-state discipline.** A synthetic pure-shadow run
  (feed absent every tick): a candidate differing *only* in the
  disagreement rule (e.g. `JAMMED_CONGESTION` halved) scores zero `differ`
  everywhere AND `rules_never_exercised` names `source_disagreement` on
  every tick with `missing` and `feed_declined` counts — the two facts the
  defect class conflates, asserted as two distinct fields in one test. A
  second run with the feed present on some ticks: the same candidate now
  shows `differ` rows exactly on the evaluable ticks, and the agreement
  denominator equals ticks minus not-evaluable ticks.
- **Replay-identity gate.** A log with one corrupted `decision_inputs`
  value: `replay_identity.status == "failed"`, first mismatch named, no
  `candidates` key in the output, exit 2 — a wrong log refuses to referee.
- **Segments.** A log whose ticks flip `shadow` false at tick k: reference
  = ticks 0..k-1, contaminated from k, both reported; `limits` echoes
  `reference_rates_hold: false` for that drive.
- **Refusals.** A pre-task-35 tick log (no `decision_inputs`) and a
  phoneless log (no `sensing`) each refuse with their named reason, exit 2,
  no fabricated score.
- **Vocabulary closure.** Every rule status in a score is one of task 34's
  three; every comparison verdict is one of `agree`/`differ`/`not_evaluable`;
  every segment tag is `reference`/`contaminated` (the membership
  pattern of test_sensing_controller.py:117 and :407).
- **Compatibility.** Full suite green from the 1547 baseline (verified by
  running before this plan); no pre-existing `record["sensing"]` key changes
  meaning or value.

## Pinned mutations (scripts/remutate.py, "Task 35" block)

1. `Inputs.to_record` rounds floats to 4 places (the evidence path's
   rounding copied where exactness is the contract) — caught by the
   17-digit round-trip test.
2. `decided_at_mono` emitted from the loop's clock read instead of the
   controller's — caught by the shifted-replay divergence test.
3. The scorer counts a `not_evaluable` tick into `agree` — caught by the
   denominator test; this is the defect class's most likely regression.
4. The reference block writes `0.0` achieved rates when telemetry is None —
   caught by the absence test.
5. `rules_never_exercised` computed from `quiet` instead of `not_evaluable`
   — caught by the pure-shadow candidate test (a calm road would then be
   reported as an unexercisable rule, the exact confusion task 34 closed).

## What this establishes, and what it does not

It establishes, from one drive: a per-tick record from which the incumbent's
every decision is reproducible byte-for-byte (proved by the tool on every
run, not asserted); a witnessed — not assumed — full-rate reference beside
those decisions; and a scoring surface on which any candidate sensing
controller is evaluated against identical recorded inputs, with "was never
given the inputs" a named, counted state distinguished from "agreed" and
from "differed", and with the one rule a pure shadow drive can never
exercise reported by name on every score.

It does not make the HERE feed exist in shadow (a protocol decision this
task may not take, shadow_mode.py:50-52); does not score trajectories or
outcomes, and says so on every report; does not score candidate RL advisory
bundles (D5); does not see through the observation builder's
`fallback_neutral` substitutions (task 34 open item 2 stands: a dead
accelerometer replays as the quiet 0.0 it was recorded as — the log is
faithful to what the controller saw, task 36 owns making the controller see
more); does not add a runtime mode-flip surface; and does not render any of
this into report.md (task 39).

## To be measured on a device run (the shape task 34's measured section took)

Filled in after implementation; the questions are fixed now so the run
cannot be scored on what it happened to show:

1. Every tick carries `decision_inputs`, `decided_at_mono` and `reference`
   (count over the drive).
2. `score_shadow.py` on the fresh drive: `replay_identity` status and
   mismatch count — the commitment is 0 mismatches over the full drive.
3. The reference witness in the field: achieved means against the
   SensingConfig defaults (camera 5.0 / gps 1.0 / imu 50.0 / here 0.2), and
   whether `achieved.here_hz` is 0.0 with HERE unconfigured — the
   structural-absence claim measured, not argued.
4. One non-trivial candidate (e.g. `EVENT_ACCEL_MPS2` 1.5 -> 1.0) scored:
   agreement counts, differ ticks, and `rules_never_exercised` naming
   `source_disagreement` on 100% of ticks (task 34 measured 899/899
   `not_evaluable`; this run should reproduce that through the scorer).
5. Record growth: estimate is ~560 B/tick (13-key inputs ~350 B, reference
   ~180 B, one float key) on a measured 7,847 B mean record (~7%);
   task 34's estimate was 5x off on the base size, so measure, don't trust.
6. `score_shadow.py` against a pre-task-35 run directory
   (device-test-2026-08-25): the named refusal, not a crash, not a score.

## Open items flagged for the user

1. **Where the reference segment ends (D9).** Mode-keyed was taken to match
   `reference_rates_hold`; delivery-keyed (`command_sent`) is physically
   truer when a live command was built but never sent. Weak preference,
   both defensible — say which, and the scorer's segmentation follows.
2. **The per-tick `reference` block borders task 37.** If task 37 lands a
   1 Hz `{"type": "phone_telemetry"}` record line, the per-tick block
   becomes ~5x-duplicated data and could shrink to the report's
   `t_capture_mono_ns` plus age. Taken per-tick here because task 35's
   claim ("alongside") wants the witness on the same record as the
   decision; revisit at task 37.
3. **The candidate contract requires attribution.** A candidate without
   `attribution.rules` is refused rather than scored on rates alone. The
   alternative — degrade with an explicit `rules: "not_reported"` state —
   was not taken because a rates-only score silently exempts a candidate
   from exactly the three-state discipline this task exists for. Flagged
   because it constrains what future candidates must implement.
4. **"Candidate policy" read as candidate sensing controller (D5).** The
   task's word is "policy"; in this repo the rate decider is the thing
   shadow mode records and task 43 checks. My first instinct included RL
   bundles; the log's contents contradicted it (a bundle's inputs are the
   observation stream, not `Inputs`). Explicitly not silently chosen —
   if candidate *advisory* policies must be scorable from one drive, that
   is a different task with an observation-replay substrate.
5. **The telemetry double-read straddle** (pre-existing, one line):
   `inputs_from` and the new reference block read `phone.telemetry` and
   `phone.telemetry_at_mono` as two property calls (phone_link.py:247-256),
   so a report landing between them can pair one report's fields with the
   next's arrival time. Benign today (both are real reports, age off by
   one interval), out of scope here, recorded so it is not rediscovered.

## Open items carried in

- Task 27's HERE parse has never met a real response body. (No HERE work in
  this task; no key appears in any URL or log.)
- Task 28's observation vector cannot be feed-informed without a
  simulator-side change.
- `MAX_QUERY_RADIUS_M` (10 km) unchecked against HERE v7's accepted range.
- `achieved["camera_hz"]` overstates the sustained rate when the channel
  evicts (task 32 residue) — now load-bearing for this task's reference
  witness: the block reports what the phone reports, and this caveat rides
  with it until `Session.send` surfaces the displacement.
- Task 33's residue: the CameraX clock assumption on
  `capture_to_encode_start`, `fuse`'s unreachable absent branch, one
  `Thread.sleep(200)`, ~1.5% unmatched joins on pre-task-33 phone logs.
- Task 34's residue: `local_density_bin`'s substitution blind spot (a blind
  camera fires disagreement), crossing two quiet entries survives, and
  `RuleCheck.to_record` accepts non-finite floats.
