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

---

# PAUSED 2026-09-09

## The blocking unknown

**Nobody has measured whether making the safety layer yield at a merge reduces
collisions, and the whole of task 67 now rests on it.**

The sensing half is committed and **bought nothing**: the 108-run grid returns
48/108 completed both before and after, identical run for run. That is not a
disappointing result, it is an expected one — `LaneGapContext` gained
`merge_conflict_gap_m` and `distance_to_next_merge_m`, but **`SafetyContext` has no
such fields, so the safety layer never receives them**. The route-aware leader
search that *is* wired up addresses the successor-arc class, and the classification
measured **0 of 51** collisions in that class. So the committed change is correct and
almost inert on this topology.

**How to settle it.** Finish the wiring, then re-run the grid:

1. Add `distance_to_next_merge_m`, `merge_conflict_gap_m` and
   `merge_conflict_relative_speed_mps` to `SafetyContext` in
   `src/safety/safety_layer.py` (they exist already on `LaneGapContext`).
2. Make `_forward_ttc` and `_speed_and_acceleration` consider a merge hazard:
   priority by arrival, so when `merge_conflict_gap_m < distance_to_next_merge_m`
   the other vehicle gets the joining point and the ego must be able to stop short
   of it. Treat the joining point as a stationary obstacle in that case.
3. Pass the three fields through `_safety_context_for_vehicle` in
   `src/envs/topology_env.py` (around line 666, where `SafetyContext` is built from
   `gap_context`).
4. Re-run the grid and compare against 48/108.

The probe that answers it is already written:
`scratchpad/repro67.py` runs the 108-run grid in ~220 s and writes
`scratchpad/repro67.json`. **It lives in the session scratchpad and will be gone** —
it is 40 lines and re-derivable from the "Steps" table above; the grid is
`inverted_tree` × demands (low, medium, high, burst) × pen (0.05, 0.10, 0.20) ×
controllers (`no_av`, `cooperative_smoothing`, `backpressure`) × seeds (7, 17, 27),
120 steps, via `src.analysis.simulator_health.run_condition`.

## Exactly where I stopped

- Repo `dsrc`, branch `main`, **no worktree**. Not pushed: `main` is **ahead 1** of
  `origin/main` (`origin/main` = `61b65f0`).
- **Committed:** `6efc3bf` "Task 67: the gap search could not see across an arc
  boundary" — `src/sensing/local.py` (route-aware gaps, merge context) and 7 tests in
  `tests/test_local_sensing.py`. Full simulator suite green at that commit: **263
  passed**.
- **Uncommitted:** `tests/test_safety_layer.py` — 4 new tests in `TestMergeConflict`,
  **currently failing with `TypeError`** because `SafetyContext` does not accept the
  keyword arguments yet. This is intended TDD red, not a broken tree.
- **Not started:** steps 1–4 above; nothing in `topology_env.py` has been edited.
- `validate_dsrc_3` has **not** been run. Per `implement_dsrc` there is no push until
  validation passes.

## What was learned that would otherwise be re-derived

- **`longitudinal_m` restarts at every arc.** That is why the original code required
  an exact `lane_index` match: a raw subtraction across a boundary is meaningless.
  Any fix must accumulate lane lengths along `road_network.graph`.
- **`inverted_tree` geometry:** six 500 m entry arcs, `a1..a3 → b1` and `a4..a6 → b2`;
  `b1 → c` and `b2 → c` are 600 m and two-lane; `c → exit`. 12 lanes total.
- **Two fields are dead on this topology.** `distance_to_next_merge` is hardcoded to
  `0.0` in `src/sensing/local.py` (task 5 recorded this). `near_merge` in
  `SafetyContext` is set from `"merge" in segment_id`, and `inverted_tree` segments
  are named `tree_leaf_a1`, so it is always `False` there.
- **Collision classification, 51 terminating collisions:** sibling merge arc 27
  (53%), same lane 18 (35%), adjacent lane 6 (12%), successor arc 0.
- **Same-lane sub-split:** 10 of 18 were seen too late to stop, 8 had room. When a
  same-lane leader first became visible the mean gap was 74 m at 27.2 m/s, against a
  75 m stopping distance at 5 m/s² — the leader appears at almost exactly the
  distance where it is already too late.
- **The safety layer is not the defect.** Given a correct gap it brakes: −1.71 m/s² at
  128 m in the traced run-up. It decayed to −0.01 m/s² at a true 13.6 m gap because it
  was told the leader was 130.7 m away.
- **Baselines to reproduce before quoting a delta:** simulator suite 263 passed;
  Jetson suite 2308 passed / 24 skipped; grid completion 48/108.
- Use `.venv/bin/python`; system `python3` is 3.14 with no pytest.

## Task order to resume

1. Steps 1–4 under "The blocking unknown" — finish the wiring and measure.
2. If the merge yield works, decide the adjacent-lane class (6 of 51, 12%) — plan
   decision D4 says it is in scope via `side_lanes`, and it is not yet started.
3. `validate_dsrc_3`: one independent `opus` validator, 3 rounds, kept alive, every
   finding verified before acting. Then push.
4. `experiment_dsrc`: the full before/after grid and the class table for the plan's
   sign-off list.

## Open decisions not yet made

- **Open item 1 in the plan is still open and is the user's**: human-on-AV rear-ends
  were raised as unrealistic. They are not addressed. The recommendation on record is
  to re-measure after the merge fix and, if they remain material, make the human
  follower collision-free rather than change the termination rule.
- Whether the 8 same-lane collisions that "had room to stop" indicate the braking law
  is too weak. That would be outside this task's scope boundary and needs its own
  task.

---

# FINDING 2026-09-09: the road produces collisions for plain human traffic

**Three fixes were implemented and measured. None moved the completion rate.**

| change | grid completion |
|---|---|
| baseline | 48/108 (44%) |
| route-aware leader/follower across arc boundaries | 48/108 (44%) |
| + merge yield in the safety layer, priority by arrival | 48/108 (44%) |
| + entry lanes separated to a full lane width | 49/108 (45%) |

**Why none of them could have worked.** `no_av` completes 36 of 36, which the plan
read as evidence that the road is safe and the AVs are at fault. It is not evidence
of anything: `terminated = any(vehicle.crashed for vehicle in self._av_vehicles)`,
and a `no_av` run has no AVs, so it **cannot terminate early whatever happens on the
road**.

Counting collisions instead of terminations:

| controller | collisions per 120-step run | runs with at least one |
|---|---|---|
| `no_av` | median **17**, max 57 | 30 of 36 |
| `cooperative_smoothing` | median 6, max 52 | |
| `backpressure` | median 5, max 52 | |

**Plain IDM human traffic crashes more than the AV runs do.** The AVs are not driving
badly; they are driving in traffic where collisions are endemic, and only their own
collisions end an episode.

**The mechanism is that every node in this topology reduces lane count and nothing
sequences vehicles through it:**

| node | lanes in | lanes out |
|---|---|---|
| `b1` | 3 | 2 |
| `b2` | 3 | 2 |
| `c` | 4 | 2 |

`highway_env`'s `IDMVehicle` follows the vehicle ahead **in its own lane**. Two
vehicles arriving at `b2` on different incoming lanes have no mutual awareness in the
human model at all, so they interpenetrate. The same blindness I fixed for the AV's
sensing exists in the human car-following model, and that is where the collisions
come from.

**This also means task 8's `episodes_complete` criterion compared every AV controller
against a reference that passes by construction.** 44 of 72 cells failing that
criterion is not 44 cells of controller failure.

**What is now known to be NOT the cause:** the safety layer's braking law (it brakes
correctly when given a correct gap), the AV's leader perception across arcs (0 of 51
collisions were successor-arc), and the entry-lane geometry alone (separating them to
a full lane width bought 1 percentage point).

## The decision this needs, and it is the user's

Three routes, and they produce materially different work:

1. **Give the human model merge awareness at funnel nodes.** The faithful option: the
   replication target (Flow, Vinitsky et al.) runs on SUMO, whose car-following is
   collision-free by construction, so a faithful replication should not have humans
   driving through each other. Largest change, and it touches a third-party vehicle
   class. **Recommended if asked.**
2. **Change the topology so no node reduces lane count.** Cheapest, but the funnels
   are what create the congestion this topology exists to produce — task 8 measured
   `inverted_tree` congesting in 12 of 12 cells — so this likely removes the
   phenomenon under study.
3. **Accept the collisions and change the success criterion.** Stop treating
   `episodes_complete` as a gate, replace the `no_av` reference with one that can
   actually fail, and let MAPPO learn to sequence. Cheapest path to a trained policy,
   but the paper would be reporting throughput on a road where traffic collides.

Option 1 and option 3 are compatible: 3 unblocks training now, 1 makes the result
defensible later.

## Decision 2026-09-09: option 1, taken by the user

Give the human model merge awareness at funnel nodes. Rationale on record: the
replication target runs on SUMO, whose car-following is collision-free by
construction, so a faithful replication should not have human traffic driving
through itself.

**The mechanism, and a defect it exposes in this task's own earlier work.** The
standard way to make a merge safe is to project every vehicle approaching a shared
node onto that node's axis: a vehicle whose distance to the node is smaller arrives
first, so it is a leader at a gap of `d_ego - d_other`, and ordinary car-following
handles the rest.

`_merge_context` as committed picks the conflicting vehicle with the **smallest
distance to the node**, which is the vehicle furthest ahead, not the nearest one.
The crash data shows what that costs: at one impact it reported a conflict 13.4 m
from the node while the ego was 99.1 m from it — a projected gap of 85.7 m — and the
vehicle actually struck was 4.1 m away, alongside. The layer was handed the wrong
vehicle, which is why a correct yield rule fired on 59% of samples and changed
nothing.

Both the human model and the safety layer use the same projection.
