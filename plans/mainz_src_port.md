# The SRC Mainz network in SUMO, with a deployable observation

What this is, why each conversion decision was made, and what the runs have shown so
far. Written as the port progresses rather than after it, so the decisions are
recorded where they were taken.

## Why the observation had to change

SRC trains on six per-super-segment features: `[density, num_lanes, avg_speed, avg_gap,
input_rate, exit_rate]`. Four of the six come from vehicle-level ground truth. Its
`RL.py` computes the gap as the mean of each vehicle's `FollowDistGr`, and the entry
and exit rates as set differences between the vehicle ids present now and a minute ago.
The HERE Traffic API returns `speed`, `freeFlow`, `jamFactor`, `confidence` and
`traversability`, and nothing else.

So a vehicle cannot evaluate SRC's published policy: it cannot obtain two thirds of the
inputs. That is the reason for retraining, and it is not a modelling preference.

The reward does not change. Training is centralized in the simulator, so it may use
privileged information, and the reward keeps SRC's exact form on ground-truth density.
Only the state is restricted to what a vehicle can read.

## The observation

| field | source |
|---|---|
| `speed` | HERE `currentFlow.speed` |
| `free_flow` | HERE `currentFlow.freeFlow` |
| `jam_factor` | HERE `currentFlow.jamFactor` |
| `lanes` | the map |
| `length_km` | the map |

`jam_factor` is emulated as `clip(10 * (1 - speed/freeFlow), 0, 10)`. HERE does not
publish its formula and folds in incident data we do not have, so this is provenance
tier 3 under the project's convention: a stated theoretical model, not our data and not
prior work that measured it.

## Why decentralized execution needs no experiment

The policy's input is the whole network's state and the policy is a deterministic
argmax. Every vehicle holding the same weights and reading the same state computes the
same action, so distributed execution is identical to centralized by construction. The
whole-network query is 12 segments x 5 features = 60 numbers per decision per vehicle
per 60 s, and HERE's matrix latency is flat in the number of cells -- 42 ms at 4, 83 ms
at 8,100 -- so querying the network costs about what querying one segment costs.

This is why the paper does not need a decentralization experiment. What it needs is the
degradation from staleness, which is the one thing that does differ on the road.

## Conversion decisions

**netconvert reads the .inpx directly.** `--vissim-file` accepts the Vissim 2023 XML;
the legacy `.inp` limitation does not apply. 410 Vissim links become 410 SUMO edges,
191 named and 219 internal, over 51.5 km.

**`--flatten`.** The imported z-offsets produce edge grades up to 923%, which SUMO
applies to acceleration. Flattening removes every grade warning.

**`--speed.factor 1.2`.** Every edge imports at 13.9 m/s (50 km/h) and SRC's action set
tops out at 60 km/h. In Vissim a desired speed may exceed the link's nominal speed; in
SUMO `setSpeed` is capped by `getAllowedSpeed`, so the top action would clamp and become
indistinguishable from releasing the vehicle -- an action space inert at its own
maximum. Lifting to 16.67 m/s makes all three actions expressible and makes the released
state equal the fastest action, exactly as `def_speed = speeds[-1]` does in SRC.

**Link 206 splits.** A connector attaches part-way along it, so netconvert emits
`206[0]` and `206[1]`. It lies on every one of the eight routes, so a name-for-name
mapping would break all of them. Link ids are expanded to their SUMO parts everywhere.

**Links 24 and 68 are refused** by netconvert and dropped from super-segment 8. Neither
appears on any demand route.

**Demand and routing.** 8 entries carry 18,000 veh/h in total for the first 1,200 s and
nothing afterwards, so 6,000 vehicles are offered and the rest of the episode drains.
Every routing decision in the network has exactly one route and all of them end at link
218, so eight paths are the complete assignment and no route choice has to be modelled.
All eight resolve with zero breaks.

## A super-segment is the road that has traffic on it

`rl_links_mainz.txt` controls 188 edges. This demand traverses 60 of them; the other
128 are opposite-direction and unserved roads that stay empty for the whole episode.

Averaging their zeros into a super-segment held the mean density at **0.23** against
SRC's threshold of **0.3**, so the congestion penalty never fired and the reward reduced
to `0.2 * speed` with its anticipatory half dead. Excluding empty edges, the threshold
fires on **4 of 12** segments, against 21 of 59 occupied edges, with a segment maximum
of 0.464.

SRC's own `train.py` sets an empty link's density to 0 rather than nan, so this is
Ankit's stated intent for the aggregation rather than a line-for-line reproduction.

## Stage 1 acceptance: the network congests

Mainz has no signals -- `signalController` count is zero -- so its junctions are the
control, and that behaviour lives in 60 Vissim conflict areas which do not import. The
check is whether SUMO's inferred priorities still produce a jam.

Over a 2,500 s episode with no control, under the corrected demand and the paper's W99
calibration: the queue outside the network peaks near 3,500 as the surge arrives, the
running count climbs to about 2,300, and mean speed falls from 48 to 20 km/h. They do.

## Validating the port against the paper's own numbers

The cheapest check on a port is its NO-CONTROL row: it needs no training, and if the
uncontrolled network does not behave like the published uncontrolled network then
nothing measured on it is comparable. The paper's Table gives, over 5 seeds at 2,500 s:

| | No Control | SRC |
|---|---|---|
| VEHARR | 2,227 +/- 19 | 2,337.8 +/- 16.42 |
| SPEEDAVG km/h | 13.16 +/- 0.42 | |

That comparison found two errors that no amount of training would have surfaced.

**Demand is not constant.** Every entry runs at its stated volume for the first 1,200 s
and at zero afterwards, so the episode is a twenty-minute surge followed by a drain.
Reading only the first interval and holding it for 2,500 s offered 12,500 vehicles where
the scenario offers 6,000.

**The car-following model was SUMO's default.** Krauss computes a collision-free safe
speed exactly and recovers from a disturbance immediately, so it gave this network
roughly twice its proper capacity. `w99_calibrated` is the paper's own Vissim
calibration from its Appendix A, already transcribed in this repository, so using it is
part of reproducing the paper rather than a tuning choice.

| configuration | arrivals | speed km/h |
|---|---|---|
| Krauss, demand held constant | 3,455 | 24.01 |
| Krauss, demand intervals honoured | 3,398 | 29.75 |
| W99, demand correct | 1,610 | 20.27 |
| **paper, no control** | **2,227** | **13.16** |

A gap remains: 1,610 against 2,227, with speed high rather than low. Fewer arrivals at
higher speed means a tighter bottleneck than the paper's -- the network admits vehicles
and cannot discharge them, and at 2,500 s it is still filling. The likeliest cause is
the 60 Vissim `conflictArea` definitions, which do not import and leave SUMO inferring
right-of-way at the unsignalised junctions. Absolute comparability was judged
unnecessary, since the paper's claim is a relative +5%, so this offset is recorded
rather than closed.

## Penetration

**100%.** The published `RL.py` writes the advisory to `veh.No % 2 == 0` with
`mix = mixed_traffic[1]`, which is 50% -- but that script produces the paper's ABLATION
row, not its main table. The main table is at full penetration. Reading the code and
assuming it corresponded to the headline result gave the wrong answer, which is the
second time in this port that an assumption from the published script was wrong; the
first was taking its feature set and reward weights as the whole story.

In SRC the penetration is the fraction of vehicles a central controller writes
`DesSpeed` to. Under decentralized execution the same fraction are vehicles running the
policy themselves. The number and the actuation are identical; only where the policy is
evaluated moves, which is what makes the two arms comparable.

## Results so far

Seeds 1-10 train, 11-15 validate and select, 16-20 are read once.

| run | arrivals | speed km/h | return |
|---|---|---|---|
| no control | 3,455 | 24.01 | -796.63 |
| 30 episodes, empty edges as zeros | 3,270 | 20.47 | (different reward, not comparable) |
| 30 episodes, occupied edges only | 3,434 | 23.19 | -999.16 |

The first configuration trained against a reward whose penalty never fired, so its
numbers are not comparable to the others and are kept only to show what the
aggregation cost.

**The trained controller does not yet beat no control.** In the 30-episode run the best
checkpoint was the final episode and validation was still improving, so that run is
undertrained rather than converged. A 120-episode run is in progress.
