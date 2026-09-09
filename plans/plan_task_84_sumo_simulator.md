# Task 84 — Replace highway_env with SUMO

**Command:** `plan_dsrc_rec`. Decisions are taken by recommendation, not put to the
user, and marked as such.

## Short version

**Move the physics, road network and demand layers onto SUMO. Keep everything
above the sensing seam unchanged.**

Three attempts to make `highway_env` collision-free produced diminishing returns:
the safe-velocity bound took human-only collisions from 98 to 6 at 120 steps, the
node-geometry fix took the worst lateral discontinuity from 10 m to 0 m, and
closest-point-of-approach took the residual from 23 to 14 over ten 600-step runs
while making four seeds worse. Each helped; none finished.

**SUMO does not need constraining.** Measured before writing this plan, on the same
3-into-1 merge shape that broke `highway_env`, at 6000 veh/h into a two-lane exit —
three times past capacity — for 1800 steps with 134 vehicles on the network:
**zero collisions.** Its car-following model cannot produce one, because the safe
speed is the model rather than a cap applied over it.

It is also what the replication target uses. Vinitsky et al. and the Flow benchmarks
run on SUMO, so this removes a difference between our setup and theirs that was
never deliberate.

## The seam, and why the change is smaller than it looks

`VehicleSnapshot` is the boundary between the simulator and everything this project
built. Above it sit the sensing model, the safety layer, the etiquette filters, the
observation encoders, the policy and the trainers — none of which know what produces
the snapshots. Below it sits vehicle physics, the road network and demand.

All eleven `VehicleSnapshot` fields are available from TraCI:

| field | TraCI source |
|---|---|
| `vehicle_id` | the SUMO id |
| `role` | our own bookkeeping, unchanged |
| `segment_id` | edge id, mapped through the topology spec |
| `lane_index` | edge id plus lane index, translated |
| `lane_id` | `getLaneIndex` |
| `position` | `getPosition` |
| `longitudinal_m` | `getLanePosition` |
| `speed_mps` | `getSpeed` |
| `acceleration_mps2` | `getAcceleration` |
| `free_flow_speed_mps` | `getAllowedSpeed` |
| `crashed` | always `False`; SUMO does not produce collisions |

The environment's whole public surface is six methods — `reset`, `step`,
`get_segment_metrics`, `get_global_state`, `get_episode_summary`,
`get_local_observations` — so a SUMO-backed class implementing those is a drop-in.

## Scope boundary

- **In:** a new `src/envs/sumo_env.py`; SUMO network generation for `inverted_tree`
  and `inverted_tree_bottleneck`; demand as SUMO flows; AV actuation through TraCI;
  metric collection from TraCI.
- **Preserved unchanged:** `src/sensing/`, `src/safety/`, `src/metrics/` (its inputs
  are re-populated, not rewritten), `src/rl/`, `src/baselines/`, the observation
  encoders and `sim_contract`.
- **Becomes dead code:** `CollisionFreeMixin`, `MergeAwareIDMVehicle`,
  `SafeControlledVehicle`, and the merge-projection work in `src/sensing/local.py`
  that exists only because `highway_env` vehicles could not see across an arc.
- **Out:** the other four topologies. Only `inverted_tree` is needed, per the
  section C scope decision. `merge` and `straight_multilane` congested in 0 of 12
  cells and are not used.
- **Out:** the Jetson tree. Nothing under `deployment/` changes.

## Decisions, taken by recommendation

| # | decision | taken | rejected, and why |
|---|---|---|---|
| D1 | Where to cut | At `VehicleSnapshot` | Cutting higher would mean rewriting the sensing model, which is the project's own contribution and is already validated |
| D2 | TraCI or libsumo | **libsumo** where available, TraCI as fallback | libsumo runs in-process and is roughly an order of magnitude faster; TraCI's socket per step would dominate a 3600-step episode |
| D3 | Network definition | Generate `.nod.xml`/`.edg.xml` from the existing `TopologySpec` and run `netconvert` | Hand-written `.net.xml` files would duplicate the topology definition and drift from it |
| D4 | Demand | SUMO `<flow>` elements, rate taken from the existing demand configs | Keeping our own spawner would reintroduce the spawn-side defects SUMO handles natively |
| D5 | AV actuation | `traci.vehicle.setSpeed` on the commanded speed, with the safety layer still producing the command | `setAcceleration` exists but SUMO applies its own safety bound afterwards either way; speed is the more direct expression of what the controller decides |
| D6 | Keep `highway_env` installed | Yes, and leave the old env in place | The parity ledger, the observation audit and 341 passing tests reference it; removing it in the same change would make a regression unattributable |
| D7 | Collision reporting | Record `getCollidingVehiclesNumber` every step and assert it is zero | A silent assumption that SUMO cannot collide is exactly the kind of unmeasured premise this project has been caught by three times |

## Open items for the user

1. **Whether the old `highway_env` path is eventually deleted.** Keeping both is the
   safe choice now, but two simulators is a maintenance cost and a source of
   divergent results. Recommendation if asked: delete it once the SUMO path has
   reproduced a full evaluation, not before.
2. **Whether the sensing model's route-aware machinery stays.** In SUMO a vehicle's
   route is explicit, so the graph walk added under task 67 becomes unnecessary for
   the simulator — but the parity argument for it was about the *deployed* system,
   which is unchanged. Recommendation: keep it, and feed it SUMO's route.

## Steps

| # | step | done when |
|---|---|---|
| 1 | `SumoNetwork`: generate `.nod.xml`/`.edg.xml` from `TopologySpec`, run `netconvert`, expose the edge↔segment mapping | `inverted_tree` builds and `netconvert` reports no errors; a test asserts the node and edge counts against the spec |
| 2 | `SumoTopologyEnv.reset/step` with libsumo, producing `VehicleSnapshot`s | A 600-step run completes and the snapshots pass the same field checks the old env's do |
| 3 | Demand as flows, rate from the demand config | Measured throughput at 1800 veh/h is within 10% of the configured rate |
| 4 | AV actuation and the safety layer through TraCI | An AV commanded to a lower speed measurably slows, verified per step |
| 5 | Metrics from TraCI into the existing `global_metrics` shape | `mean_speed`, `throughput_recent`, `jam_fraction` and `collision_count` all populated; collision count asserted zero |
| 6 | Re-measure the capacity sweep on SUMO | The demand that congests without gridlocking is identified, as task 79 did for the old simulator |
| 7 | Retrain MAPPO on the SUMO env | A checkpoint exists, trained on SUMO, with its config recorded |
| 8 | Evaluate against `no_av` over an hour | Throughput compared with every run completing, since collisions can no longer truncate |

## Risks

- **Speed.** libsumo is fast but a 3600-step episode with 130 vehicles has real cost.
  Step 2 measures it before step 7 depends on it.
- **The observation may change meaning.** SUMO lane positions and our
  `longitudinal_m` are both distance along a lane, but SUMO's lanes are per-edge
  while `highway_env`'s were per-arc. Step 1's mapping is where that is settled, and
  a wrong mapping would silently move every gap the sensing model computes.
- **Any trained checkpoint is invalidated.** The actor's input distribution changes,
  which is the same dependency that put task 9 ahead of task 68. The existing
  checkpoint is a `highway_env` policy and cannot be evaluated on SUMO.
- **`SUMO_HOME` is unset**, so XML validation is disabled. Harmless for generated
  files, but it means a malformed network is caught by `netconvert` rather than by
  a schema.

## Sign-off

- [ ] Steps 1–5 complete, tests passing, collision count asserted zero across a full run
- [ ] Capacity re-measured on SUMO
- [ ] MAPPO trained and evaluated over an hour, with every run completing
- [ ] `validate_dsrc_3` clean


---

# Steps 1-6 measured 2026-09-09

**Steps 1 to 5 are implemented and tested.** 22 new tests across network
generation, the environment, the road view and the control wiring; the simulator
suite is at 369 passed.

## What SUMO changed, measured

| | highway_env | SUMO |
|---|---|---|
| collisions, 600 steps at every demand | 14 to 23 over ten runs | **0** |
| wall time, 600 steps | ~40 s with CPA | **0.2 s** |
| runs completing | 6 or 7 of 10 | **all** |
| congestion reachable | only via crash queueing | **yes, by demand** |

An hour-long episode costs about 1.2 s, so the evaluation that was unmeasurable is
now nearly free.

## Step 6: capacity is junction-limited and much lower than it appeared

1800-step runs, no AVs, insertion keeping up throughout:

| veh/h | mean speed | arrived | departed | ratio |
|---|---|---|---|---|
| 450 | 26.26 | 222 | 228 | 1.01 |
| 600 | 18.11 | 289 | 300 | 1.00 |
| 750 | 20.25 | 361 | 384 | 1.02 |
| **900** | 7.35 | **378** | 456 | 1.01 |
| 1050 | 2.11 | 349 | 528 | 1.01 |
| 1800 | 1.18 | 425 | — | — |
| 3000 | 1.39 | 538 | — | — |

**Capacity is about 900 veh/h**: arrivals are maximal there and fall at 1050 while
insertion still keeps up. `configs/demand/sumo_saturating.yaml` is 1050, past the
knee, with 29 of 378 arrivals lost to congestion as the headroom a controller has.

**It is junction-limited, not lane-limited.** The three-into-two merges at b1 and b2
are unsignalised, so vehicles yield and throughput is throttled well below what the
two-lane trunk could carry. Realistic for an uncontrolled merge, and the reason
SUMO's capacity here is far below the ~1800 the previous simulator appeared to
manage — that figure was measured on a road where vehicles could drive through each
other.

**One instrument caveat.** `jam_fraction` as computed here is the share of vehicles
below `queue_speed_mps` of 5 m/s, and it reads 0.167 even at 450 veh/h where mean
speed is 26 m/s, because vehicles just inserted and vehicles crossing junctions are
momentarily slow. Mean speed and arrivals are the reliable congestion indicators at
this demand range; `jam_fraction` is not, and should not be used as the operating
point criterion.
