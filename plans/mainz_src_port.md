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

## Results

Seeds 1-10 train, 11-15 validate and select, 16-20 are read once. 100% penetration,
2,500 s episodes, 80 training episodes per arm.

| arm | return | arrivals | speed km/h |
|---|---|---|---|
| DSRC, the HERE observation | **-283.54** (+13.2%) | 1,582 (-0.9%) | **21.89** (+9.2%) |
| SRC, its original six features | -292.89 (+10.4%) | 1,559 (-2.4%) | 21.03 (+4.9%) |
| no control | -326.83 | **1,597** | 20.05 |

Percentages are against no control.

**Restricting the observation to what a vehicle can read costs nothing measurable.**
DSRC is ahead of SRC on all three columns. The four features that a traffic API cannot
return -- density, following gap, and the entry and exit counts -- carry no information
this controller was using. That is the result the extension needs, and it is the one
that makes decentralized execution possible at all.

**Neither arm reproduces the paper's +5% throughput.** Both improve the objective they
were trained on and both raise mean speed, but arrivals are slightly BELOW no control.
So on this port the reward and throughput are not aligned: a policy that improves
`-100 * 1[rho > 0.3] + 0.2 * speed` does not thereby serve more vehicles. The same
divergence was recorded on the earlier simulation leg, where arrivals peaked at a
different speed from the reward.

**What is not established.** No seed spread was computed for these means. The paper's
own no-control row carries +/- 19 on 2,227, which is +/-0.85%, so an arrivals difference
of 0.9% sits at that scale and nothing here resolves it. The reward and speed
differences are several times larger and consistent in direction across both arms.

The port's absolute offset from the paper stands: 1,597 no-control arrivals here against
2,227 published. Absolute comparability was judged unnecessary, so the arms are compared
against this port's own baseline rather than against the published table.

## Why the throughput gain is absent: there is no capacity drop

The mechanism SRC exploits is the capacity drop. Flow rises with density to a maximum
at a critical density and FALLS beyond it, so a controller that holds density below
critical recovers the flow that the collapse would have lost. If flow does not fall,
there is nothing to recover and no controller can gain throughput however well trained.

Measured with no control, sweeping the demand:

| offered veh/h | served veh/h | max segment density |
|---|---|---|
| 1,800 | 864 | 0.000 |
| 2,700 | 1,290 | 0.000 |
| 3,600 | 1,728 | 0.000 |
| 5,400 | 2,249 | 0.065 |
| 9,000 | 2,258 | 0.273 |
| 18,000 | 2,267 | 0.464 |

**From 5,400 to 18,000 veh/h offered, a factor of 3.3, served flow moves +0.8% while
the maximum density rises by a factor of 7.1.** Flow plateaus at a fixed bottleneck
rather than falling past a critical density. There is no capacity drop on this network,
so both trained arms raising speed and the reward while leaving arrivals unchanged is
the expected outcome rather than a training failure.

**The demand is far above the regime of interest.** 18,000 veh/h against a network that
discharges 2,260 is eight times oversaturation, and the entry queue then meters the
network for free: the interior never explores the density range where a drop would
appear. 18,000 is the top of the envelope the paper's text describes (11,000 to 18,000
summed over its four junctions), so it reads as a saturation setting rather than the
rush-hour profile.

**The bottleneck is faithful, not an artifact.** Eight single-lane links carry every
route, and at W99 with cc1 = 2.5 s one lane caps near 1,300 veh/h, which is about half
the observed 2,260. Lane counts were compared against the Vissim source across all 188
named edges: zero mismatches.

**A routing bug found while diagnosing this.** A Vissim routing decision sits ON a link
and its `linkSeq` is the path onward from it, so building routes from the sequence alone
omitted the entry link and inserted vehicles one edge downstream -- for three of the
eight entries, directly onto a single-lane edge. Including the entry link raised
in-network vehicles from 2,285 to 2,870 and cut the entry queue from 2,127 to 1,556. It
did not change discharge, which is itself evidence that the constraint is downstream of
the entries.

**The step size changes the sign and not the conclusion.** SRC's Vissim runs at
`SimRes = 1` and this port matched it, but SUMO's W99 at 1 s may not resolve the
stop-and-go oscillation a capacity drop comes from -- and this project retracted a 17.6%
throughput result once because a 1 s step manufactured it. Measured both ways:

| dt | offered veh/h | served veh/h | max density |
|---|---|---|---|
| 1.0 s | 5,400 | 2,249 | 0.065 |
| 1.0 s | 18,000 | **2,267** (+0.8%) | 0.464 |
| 0.1 s | 5,400 | 2,458 | 0.065 |
| 0.1 s | 18,000 | **2,428** (-1.2%) | 0.444 |

At 1 s flow RISES with density, so there is no capacity drop to speak of. At 0.1 s a
drop does appear and the sign flips, so the step size is real and 1 s is too coarse to
resolve the phenomenon. But the drop is **1.2%**, and the paper recovers **5%**
throughput. The headroom a controller could reclaim on this port is an order of
magnitude smaller than the gain being reproduced.

The finer step also raises the level by about 9%, from 2,249 to 2,458 veh/h, which
narrows the gap to the paper's 3,207 veh/h without closing it.

**So the port cannot demonstrate this mechanism at the scale claimed, and more training
will not change that.** The remaining explanation is that SUMO's W99 is a simplified
Wiedemann 99 which does not reproduce Vissim's capacity-drop behaviour even carrying the
paper's own cc1 to cc9. Before any further training on this network, the thing to
establish is whether SUMO's W99 can produce a capacity drop of the right magnitude on a
textbook single bottleneck at all. If it cannot, no amount of calibration on Mainz will
help and the choice is between a different simulator and a different claim.

## The fundamental diagram, measured properly

The coarse check above averaged flow across the surge and the drain and sampled a grid
that missed the onset, so it gave a lower bound rather than a measurement. Measured the
way `inverted_tree` was for task 100 -- dt 0.1, a fine grid through the onset, flow
counted only over a steady window inside the surge (600 to 1,200 s), three seeds:

| offered veh/h | density | speed km/h | served veh/h | served/offered |
|---|---|---|---|---|
| 2,400 | 0.035 | 55.6 | 1,938 | 0.81 |
| 3,000 | 0.044 | 53.9 | 2,294 | 0.76 |
| 3,600 | 0.053 | 51.8 | 2,538 | 0.70 |
| 4,200 | 0.064 | 49.0 | 2,678 | 0.64 |
| 4,800 | 0.074 | 46.5 | 2,750 | 0.57 |
| 5,400 | 0.087 | 43.8 | 2,834 | 0.52 |
| 6,000 | 0.098 | 41.3 | 2,844 | 0.47 |
| 7,200 | 0.118 | 38.3 | 2,836 | 0.39 |
| 9,000 | 0.131 | 36.6 | 2,856 | 0.32 |
| 12,000 | 0.142 | 35.1 | **2,858** | 0.24 |
| 18,000 | 0.143 | 35.0 | 2,790 | 0.15 |

**The capacity drop is 2.4%**, from a peak of 2,858 veh/h at 12,000 offered down to
2,790 at 18,000. `inverted_tree` under the same W99 calibration drops **24%** (task
100). So the drop is real but ten times smaller, and the earlier 1.2% was the same
finding through a blunter instrument rather than a different one.

**The network is insertion-limited, and that is why.** Doubling the offered demand from
9,000 to 18,000 moves the interior density by 9%, from 0.131 to 0.143, and served flow
by -2.3%. The entries cap admission near 2,850 veh/h and everything beyond that queues
outside, so the interior never becomes dense.

**The highest density reached anywhere in the surge is 0.143, which is 48% of the 0.300
threshold the controller exists to defend.** SRC's reward pays -100 for a super-segment
over critical density, and during the regime the mechanism targets that term cannot
fire at all. The congestion that does appear comes later, in the drain after demand
stops and vehicles already admitted accumulate -- which is not the phenomenon the
protocol is designed around.

**Throughput level, by contrast, nearly matches.** The plateau of about 2,850 veh/h is
within 11% of the paper's no-control 3,207 veh/h. At dt 1.0 it was 2,260, so the step
size accounts for most of the earlier level gap. It is the SHAPE of the curve, not its
height, that does not reproduce.

## What this means for the extension

A controller that holds density below a critical point can, on this network, recover at
most 2.4%. The paper reports 5%. So no training configuration, observation set or
penetration will reproduce the published gain here, and the two trained arms behaving
as they did is the predicted outcome rather than a failure to tune.

Three things could be true and they call for different work:

1. **The demand is the wrong operating point.** Every published rate is at or above
   12,000 veh/h offered against a 2,850 veh/h network. A scenario whose demand sits at
   the onset, around 3,600 to 5,400, would spend the episode near the critical point
   instead of queued outside it.
2. **SUMO's W99 is not Vissim's.** It is a simplified Wiedemann 99, and the parameters
   that the paper's appendix says control the capacity drop -- CC4 and CC5 for braking
   and acceleration tendencies, CC2 and CC6 for stop-and-go -- may not have the same
   effect in SUMO's implementation. `inverted_tree` gets 24% from them, so they do
   something; whether they do the same thing on a real network with single-lane
   constraints is untested.
3. **The bottleneck is the wrong kind.** `inverted_tree` is a purpose-built merge where
   congestion forms at a zipper junction. Mainz's constraint is eight single-lane links
   whose capacity is set by headway, and a fixed-capacity lane does not drop the way a
   merge does.
