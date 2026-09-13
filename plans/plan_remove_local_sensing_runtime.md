# Remove the local-sensing 39-field runtime

**Decision, 2026-09-13:** the 39-field local-sensing policy is deleted. It is the
superseded formulation. The paper's controller is the DSRC policy, which reads twelve
super-segments of five features and emits one speed fraction per segment, and it shares
nothing with the runtime being removed.

**Why this was not obvious.** The four action heads (`desired_speed_bin`,
`desired_headway_bin`, `lane_preference`, `merge_mode`) belong to the local-sensing
policy, not to the DSRC one. `SPEED_ACTION_FRACTIONS` is the DSRC action set and it is a
speed alone. No retraining is needed to make the advisory speed-only, because the policy
the paper reports already is.

## Already done

`a6ee919` removed the non-speed rules from `src/safety/`, the simulator-side layer. Kept:
`low_speed_uncongested`, which raises the target speed, and `forward_ttc`, which drives
the emergency override. Removed: `all_lane_low_speed_occupancy` and
`passing_lane_slow_hold` and the eight lane guards, all of which only null `lane_action`
or record a diagnostic. `SafetyDecision` lost `lane_action`.
`specs/safety_contract_golden.json` is unchanged and still matches, because the field
shapes of `SafetyConstraints` and `SafetyContext` were deliberately left alone. Root
suite green at 132.

## The three-way split

**Goes.** `policy/actor_runtime.py`, `policy/export_policy.py`, `policy/sim_contract.py`,
the `Advisory` / `AdvisoryDecoder` half of `policy/advisory.py`,
`specs/sim_contract_golden_vectors.json`, `specs/action_schema.md`,
`specs/observation_schema.md`, and every test that pins them.

**Stays.** `policy/dsrc_runtime.py`, `policy/dsrc_contract.py`,
`perception/segment_state.py`, the `SegmentAdvisory` / `SegmentAdvisoryDecoder` half of
`policy/advisory.py`, `specs/dsrc_network_mainz.json`, `specs/dsrc_golden_actions.json`.
`dsrc_runtime` imports only `perception.segment_state` and `policy.dsrc_contract`, so the
surviving path is independent of everything being deleted.

**Stays, reduced.** `perception/observation_builder.py` and the perception chain behind
it. This is the part that is easy to delete by mistake. The two surviving safety rules
read `leader_gap`, `leader_relative_speed`, `ego_speed` and `local_density_veh_per_km`,
and the observation builder is the only producer of them. Delete it and the gate has no
evidence for either rule by construction, and the camera feeds nothing. Reduce it to
those fields rather than removing it.

## Order of work

1. Split `policy/advisory.py` in two, so the DSRC decoder stops sharing a module with the
   one being deleted. 40 files import this module; do this first and separately.
2. Delete `actor_runtime`, `export_policy`, `sim_contract` and the `Advisory` half.
3. Unwire them from `pipeline.py` and `run_demo.py`.
4. Reduce `observation_builder` to the four fields the surviving rules read.
5. Mirror the `src/safety` rule removal into `policy/safety_gate.py`: `RULE_NAMES` down to
   two, `LANE_GUARD_RULES` and the lane-withholding logic gone, `RULE_READS` pruned.
6. Drop `lane_text` and `merge_text` from the Jetson dashboard and the phone display.
7. Delete the specs above and every test that pins them.
8. Whole-repo suite, then the Kotlin tests.

## Consequences to expect, not to be surprised by

`results/safety/gate_census_corpus.json` and `results/safety/drive_replay_gate_census.json`
both describe a twelve-rule filter that will no longer exist. `target_lane_front_gap`, one
of the three rules the drive census found evaluable, is among those removed, so the
surviving gate has two rules of which one needs a tracked leader. The paper's Section IV
describes twelve rules and is explicitly out of scope for this work.
