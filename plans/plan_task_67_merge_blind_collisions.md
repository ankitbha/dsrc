# Task 67 — AVs collide because the sensing model cannot see the vehicles that hit them

**Command:** `plan_dsrc_rec`. Every decision below was taken by recommendation
rather than put to the user. They are marked so nobody reads them as sign-off.

## Short version

Task 67 was filed as "fix the 40% episode-truncation rate". The rate is real and
worse on the one topology that matters, but truncation is a symptom. **Episodes end
because AVs crash, and AVs crash because `lane_gap_context` cannot report the
vehicle they are about to hit.** The safety layer brakes correctly for what it is
told; it is told the road ahead is clear.

Measured on `inverted_tree`, 108 runs mirroring the task 8 grid for one topology:

| controller | runs | completed | rate |
|---|---|---|---|
| `no_av` | 36 | 36 | **100%** |
| `cooperative_smoothing` | 36 | 7 | 19% |
| `backpressure` | 36 | 5 | 14% |
| **all** | **108** | **48** | **44%** |

`no_av` never fails because it has no AVs to crash. Completion falls to **0 of 6 at
`av_penetration` 0.20 on every demand level**: the failure rate rises with the number
of AVs, which is the signature of AVs being unable to see each other and the traffic
they merge into.

**51 terminating collisions, classified by the lane relation between the two
vehicles:**

| class | n | share |
|---|---|---|
| sibling merge arc — both arcs feed the same node | 27 | **53%** |
| same lane | 18 | 35% |
| adjacent lane, same arc | 6 | 12% |
| successor arc | 0 | 0% |

**Why the merge class is invisible.** `_lane_gaps` skips any neighbour whose
`lane_index` differs from the ego's (`src/sensing/local.py:362`). On `inverted_tree`
the graph is `a1_entry → b1`, `a2_entry → b1`, `a3_entry → b1`: several entry arcs
converge on one node. Two vehicles converging on that node are on different arcs, so
neither appears in the other's leader search, right up to the collision. The field
that should carry this, `distance_to_next_merge`, is **hardcoded to `0.0`** and was
recorded by task 5 as one of only two structurally uninformative fields in the whole
observation vector. The AV has no merge awareness on a merge topology.

**A measured run-up, `cooperative_smoothing`/medium/pen0.10/seed17:**

```
step  speed  sensed leader gap  true gap  commanded accel
  44   29.9                inf     145.0            +0.22
  47   27.2                inf      98.1            -0.27
  49   27.0              146.5      40.2            -0.04
  51   27.0              130.7      13.6            -0.01
```

The sensed gap is `inf` while the true gap closes from 176 m to 84 m, then reports
130.7 m when the truth is 13.6 m — it has locked onto a further vehicle on the ego's
own arc. Commanded deceleration decays to −0.01 m/s² as the gap collapses, because
the layer is braking for a leader 130 m away. The vehicle finally hit was on
`a4_entry → b2` while the ego was on `a5_entry → b2`: a sibling merge arc.

## Scope boundary

- **In scope:** the leader/follower/merge gap computation in `src/sensing/local.py`,
  and the `SafetyContext` fields fed from it in `src/envs/topology_env.py`.
- **Out of scope:** the safety layer's own braking law. It is not implicated: given a
  correct gap it brakes (−1.71 m/s² at 128 m in the trace above). If it turns out to
  brake too weakly once it can see, that is a separate task.
- **Out of scope:** highway-env's `IDMVehicle`. Human-on-AV rear-ends exist but are a
  third-party car-following model; see the open item below.
- **Out of scope:** every topology except `inverted_tree` for *measurement*. The fix
  is topology-agnostic because it lives in the sensing model, but only `inverted_tree`
  is re-measured.
- **Out of scope:** changing the termination rule. Ending an episode on an AV crash is
  the standard formulation and is what gives training its negative signal.

## Decisions, all taken by recommendation

| # | decision | taken | rejected, and why |
|---|---|---|---|
| D1 | Fix perception, not termination | Make the gap search see the vehicles that actually collide | Suppressing termination on collision hides the defect, and the safety filter is a headline claim of the paper |
| D2 | Merge conflicts get their own gap, not a fake leader | Add a merge-conflict gap and populate `distance_to_next_merge` | Reporting a merging vehicle as a "leader" would corrupt headway control, which is a different quantity |
| D3 | Route-aware search via the road graph | Walk `network.graph` successors/predecessors, accumulating lane lengths | Euclidean distance with a heading window is wrong on curves and links physically close but unconnected arcs |
| D4 | Adjacent-lane conflicts are in scope | Use `side_lanes` for the 12% class | Deferring leaves a known crash mode unaddressed for a small extra cost |
| D5 | Keep `range_m` as the visibility limit | Merge and successor searches stop at `range_m` | An unbounded search would make the AV clairvoyant and would not match the deployed system |
| D6 | Fix the sensing model only, then re-measure | Measure the residual before touching anything else | Changing two systems at once makes the delta unattributable |

## Open items for the user

1. **Human-on-AV rear-ends were raised as unrealistic, and they are 27% of the
   subset examined by collision partner.** They are not addressed by this plan.
   `IDMVehicle` has a bounded comfort deceleration, so when an AV brakes hard the
   follower can be unable to stop. The faithful comparison target (Flow, Vinitsky et
   al.) runs on SUMO, whose car-following is collision-free by construction, so a
   replication arguably should not have humans rear-ending anyone. **Recommendation
   if asked: after this fix, re-measure; if human-on-AV rear-ends remain material,
   make the human follower collision-free rather than changing the termination rule.**
   Flagged rather than decided because it changes the traffic model, not a bug.
2. **The 35% same-lane class may not be fully explained by this fix.** One sample had
   a correctly sensed gap of 7.8 m at 19.9 m/s onto a stopped vehicle — already
   unavoidable. Whether the leader became visible too late (a `range_m` or latency
   question, and therefore a task 9 question) or the layer braked too weakly is not
   yet established. Step 5 measures it.

## Steps

| # | step | done when |
|---|---|---|
| 1 | Characterise the same-lane class: log sensed gap, true gap and commanded accel for every same-lane collision | The 18 same-lane cases are split into "seen too late" and "seen, braked too weakly" with counts |
| 2 | `_route_gaps`: walk graph successors from the ego lane, accumulating `lane.length`, to find the true leader across arc boundaries; predecessors for the follower | Unit tests cover a two-arc chain, a three-arc chain, and the `range_m` cutoff |
| 3 | `_merge_conflict_gap`: find vehicles on sibling arcs sharing the ego's next node; return distance-to-merge-point for both, and the conflict gap | Unit tests cover two converging arcs, a non-merging pair, and a vehicle past the merge point |
| 4 | Populate `distance_to_next_merge` from step 3 instead of the hardcoded `0.0`; add the merge-conflict fields to `SafetyContext` | `observation_audit` no longer reports the field as structurally constant |
| 5 | Sanity tests: on `inverted_tree`, an AV converging with a sibling-arc vehicle reports a finite merge gap that decreases monotonically as they approach | Test fails against the current code |
| 6 | Re-run the 108-run grid and report completion before and after, by class | Completion rate and the class table are both reported; any class that did not improve is named |

## Risks

- **The observation vector changes.** `distance_to_next_merge` goes from constant
  `0.0` to a real value, and `leader_gap_m` changes on merge topologies. Nothing is
  trained yet, so no checkpoint is invalidated — but task 9's calibration and task 47's
  parity ledger both read these fields, and task 47 compares the sim sensing model
  against the live one. **The live model must be checked for the same defect**; if the
  Jetson's `observation_builder` has no merge concept either, the parity classification
  changes and task 47's ledger needs updating.
- **Cost per step.** The gap search is already O(n) per AV; a graph walk bounded by
  `range_m` adds a small constant. If the 108-run grid slows by more than ~25%, revisit.
- **A fix is new code.** The validator rounds exist for exactly this.

## Sign-off

- [ ] Steps 1–6 complete, tests passing
- [ ] Completion rate on `inverted_tree` reported before and after, by collision class
- [ ] Residual human-on-AV rear-end share reported, so open item 1 can be decided
- [ ] `validate_dsrc_3` clean
