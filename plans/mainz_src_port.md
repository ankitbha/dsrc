# The SRC Mainz network in SUMO, with a deployable observation

What this is, why each conversion decision was made, and what the runs have shown so
far. Written as the port progresses rather than after it, so the decisions are
recorded where they were taken. The sections below are in the order they were written;
this summary is the state as of the latest run.

## Result

Held-out seeds 16-20, Mainz, EIDM fleet, three-into-two lane drop at the exit, demand a
sustained 4,500 veh/h. Seeds 1-10 train, 11-15 select, 80 episodes, 100% penetration.

| arm | flow veh/h | arrivals | return | space-mean speed |
|---|---|---|---|---|
| DSRC, the HERE observation | **3,775** (+5.8%) | 1,997 | 182.5 | 40.81 |
| SRC, its original six features | 3,756 (+5.3%) | 1,991 | **193.8** | 40.82 |
| no control | 3,568 | 1,951 | 55.1 | **42.68** |

Paired on seed, DSRC gains **+207 +/- 92 veh/h** and SRC **+188 +/- 133**; both are
resolved against the seed spread, and DSRC is far steadier across seeds (+/-23 against
+/-109). The two arms differ by 0.5%, inside both bars.

**Restricting the observation to what a traffic API returns costs nothing, measured on a
network where throughput can move.** Both arms run slower than no control, 40.8 against
42.7 km/h, while serving more vehicles: they hold density below critical upstream of the
merge so it discharges near capacity instead of breaking down.

## What it took, and what is load-bearing

Three findings decided everything, each measured rather than assumed:

1. **A link's `q = k v` hill is not a capacity drop.** Krauss produces one too, at 15%,
   and Krauss has no capacity drop. The quantity that matters is served flow falling as
   offered demand rises.
2. **The SRC paper's W99 calibration has no capacity drop on SUMO.** Its queues discharge
   FASTER than free-flowing traffic at capacity, 1,980 veh/h against 1,899. No bottleneck
   of any kind, no start-up delay and no reaction time changed that, so every network
   tried served a constant and no controller could gain throughput.
3. **A bottleneck sets the LEVEL of throughput, not its dependence on density.** EIDM
   with a one-second reaction time has a 21% drop at an isolated lane drop; on Mainz it
   needed a merge at the exit as well, because link 218 ends into free outflow.

## Open

* One demand level only. The gain should be swept across demand before it is quoted as a
  property of the network.
* DSRC's selected checkpoint is its last episode, so it may not have converged.
* EIDM's parameters are SUMO defaults, not calibrated against measured traffic, and the
  three-into-two exit drop is a modelling choice of ours.
* `DENSITY_CRITICAL` is 0.178 in occupancy units, which is 0.308 of the measured jam
  density -- within 3% of SRC's published 0.3. Under W99 the same comparison gave 0.134.

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

## Testing hypothesis 1: the operating point

The published scenario surges at 18,000 veh/h for 1,200 s and then stops, so the
episode is half starved and half draining and the interior never sits anywhere. Holding
a rate for the whole 2,500 s instead, with no control:

| sustained veh/h | density at 600 s | at 1,500 s | at 2,500 s | segments over rho* | arrived | queued |
|---|---|---|---|---|---|---|
| 3,600 | 0.064 | 0.117 | 0.207 | 0 | 1,601 | 0 |
| 4,800 | 0.094 | 0.225 | 0.290 | 0 | 1,644 | 32 |
| **6,000** | **0.121** | **0.315** | **0.440** | **2** | **1,683** | **193** |
| 7,200 | 0.146 | 0.308 | 0.410 | 3 | 1,675 | 585 |
| 9,000 | 0.190 | 0.389 | 0.468 | 4 | 1,669 | 1,743 |

**6,000 veh/h sustained is the operating point.** Density crosses the critical 0.3
mid-episode rather than never or immediately, so a controller has an approach it can
act on; inflow continues throughout, so there is something to manage; and only 193
vehicles queue outside, so the controller's actions decide the outcome rather than the
entry queue. Under the published scenario the interior sat at 0.143 with 1,556 queued.

Arrivals also peak here and fall beyond it -- 1,683 at 6,000, 1,675 at 7,200, 1,669 at
9,000 -- which is the capacity drop appearing in the sustained regime.

**Training must run at dt 0.1.** At dt 1.0 served flow RISES with density, so there is
no drop to recover and the test would be vacuous whatever the demand. This is the same
step-size sensitivity that once manufactured a 17.6% result on this project, appearing
now as an effect the coarse step ERASES rather than creates.

## The no-control case, finalised

Everything before this measured inside a transient. Demand that stopped at 1,200 s left
the episode draining; demand sustained for 2,500 s left the density still climbing at
the end. Neither is a state a throughput number can be read from. Two 10,000 s runs with
demand sustained throughout, flow sampled per 200 s, no control:

| | 6,000 veh/h | 18,000 veh/h |
|---|---|---|
| mean density | 0.281 | 0.287 |
| max density | 0.577 | 0.550 |
| **steady-state flow** | **2,882 veh/h** | **2,858 veh/h** |
| vehicles in network | 3,549 | 3,578 |
| speed | 17.6 km/h | 17.0 km/h |
| queued outside | 1,174 -> 5,353 | 17,830 -> 37,856 |
| flow, first half vs second | 2,873 -> 2,891 | 2,850 -> 2,866 |

Averaged over the equilibrated stretch, t >= 5,000 s.

**The network equilibrates, and it equilibrates in a congested state.** Density settles
near 0.28 mean and 0.55 to 0.58 maximum, with five of twelve super-segments over the
critical 0.3, at 17 km/h. This is not a free-flowing network: it is jammed and stays
jammed for 5,000 s.

**Flow does not fall.** Within each run the second half matches the first to under 1%.
Across the two runs, tripling the demand moves flow by -0.8%.

**Higher demand does not even produce higher density.** Interior density, occupancy and
flow are the same at 18,000 veh/h as at 6,000; the only thing that changes is the queue
outside, which grows seven times longer. The network self-limits: admission is capped by
the network itself, so excess demand waits at the boundary and never reaches the
interior.

**So higher density does not mean lower flow here, at any demand, at any density the
network can reach, over 10,000 s of equilibrated congestion.** The network behaves as a
fixed-capacity server of about 2,870 veh/h with an unbounded entry queue.

That is the finding the whole port rests on, and it is now measured at equilibrium
rather than inside a transient. A controller that holds density below critical has
nothing to recover, because flow does not fall when density crosses critical; and it
cannot hold density down in any case, because the network already caps its own
admission.

## The fundamental diagram on a straight road, and what it says about the threshold

Ankit's instruction: a straight two-lane road, default cars, sweep demand, and the
flow-density curve must be a hill, because that shape has been measured on real roads
many times and a simulator that cannot produce it is misconfigured. It was not the
simulator.

**Three setup errors of mine, each of which alone prevented the curve.**

*Free outflow.* On a straight road that ends in free flow, demand above capacity queues
at the entry and the road itself never passes critical density -- the self-limiting
behaviour seen on Mainz. Every empirical diagram with a congested branch is measured
upstream of something holding traffic back. A signal at the far end does that without
adding a merge.

*Merge turbulence.* A lane drop was tried first. A section immediately upstream of a
bottleneck cannot exceed the BOTTLENECK's capacity, so it never shows its own, and it
sits in the merge's influence: speed fell to 67 km/h at a density of 6.6 veh/km/lane,
where vehicles are 150 m apart and cannot be interacting. Removed.

*Density from the wrong vehicles.* An early ring variant divided the REQUESTED vehicle
count by the ring length while measuring speed over the vehicles actually inserted,
reporting 275 veh/km/lane against a jam density of 200 and a flow of 19,546 veh/h/lane.

**With those fixed the curve is textbook**: free flow near 100 km/h up to the critical
density, a peak, a congested branch falling toward jam, and loading and unloading legs
showing the hysteresis real detector data shows.

**The calibration is wrong, and cc1 is the knob.**

| cc1 | capacity veh/h/lane | critical density veh/km/lane | congested branch reaches | fall from peak |
|---|---|---|---|---|
| **2.5, what `w99_calibrated` uses** | 1,057 | 13.8 | 44.5 | 31% |
| 1.8 | 1,348 | 16.8 | 75.7 | 58% |
| 1.4 | 1,596 | 19.3 | 86.8 | 65% |
| **1.1** | **1,754** | **20.0** | **117.3** | **82%** |
| Krauss, for comparison | 2,136 | 25.6 | 26.6 | 4% |
| a real highway lane | 2,000-2,400 | 20-30 | | |

`cc1 = 2.5` is the midpoint of the 2-3 s range the paper's appendix states, and it
gives a road with half of real capacity and a critical density of 13.8 where reality is
20 to 30. Krauss has realistic capacity and no congested branch, which is why this
project moved off it; W99 at cc1 = 1.1 has both.

## The threshold the controller defends is far past the critical density

SRC's reward pays -100 for a super-segment whose density exceeds 0.3 of jam. Its
density is `num_cars * 4.5 / lane-metres`, so jam is 222 veh/km/lane and the threshold
is **67 veh/km/lane**.

| cc1 | measured critical density | as a fraction of jam | SRC's threshold |
|---|---|---|---|
| 2.5 | 13.8 | 0.062 | 0.300 |
| 1.1 | 20.0 | 0.090 | 0.300 |

**The threshold sits three to five times above the density at which flow actually
starts to fall.** A controller trained on that reward is asked to defend a line deep
inside the congested branch, long after the flow it exists to protect has already
collapsed. Whatever else is true of the network, the objective cannot reward preventing
a capacity drop when its trigger fires well after the drop has happened.

That is an explanation for the absent throughput gain which is independent of the
network, the demand and the training, and it is testable directly: set the threshold at
the measured critical density rather than at 0.3.

## The exit, and the threshold

**The outflow was never free, and the backpressure is not in the published artifacts.**
Vehicles leave from whatever edge their route ends on, so that edge's capacity is the
cap. Two porting errors hid this: routes were built from `linkSeq`, which names neither
end of the path, so they omitted both the entry link and the destination link. With the
destination restored the route ends on link 218, three lanes at 60 km/h, which caps
outflow at 3,899 veh/h. That is above the 2,880 the network delivers, so it never binds
-- the exit absorbs everything offered, which is the substance of the objection even if
"free" overstates it.

The SRC paper describes a backpressure element at the exit. It is not in any published
layout: the conflict area on link 218 is PASSIVE in `Mainz20base`, and all four
signalised variants carry 68 conflict areas with 11 active, none touching 218. So any
exit constraint here is a stated choice of ours, not a reproduction.

**The attempt to add one is not yet sound and is recorded as unfinished.** A two-lane
exit road cut network discharge from 2,880 to about 1,000 veh/h, far below the 2,600 its
lane count implies, because a three-into-two merge loses much more than the ratio.
Replacing it with a three-lane road metered at 2,417 veh/h gave the same 1,000, so the
restriction comes from the junction that appears when the network is extended at all,
not from the rate chosen. Link 218 does show a bistable jump under it -- density goes
from 7.1 to 43.5 veh/km/lane between 900 and 1,500 veh/h offered while flow stays near
330 -- but flow never traces a peak, so this is not yet a fundamental diagram and the
number to fix is the junction, not the meter.

## The reward threshold, set to the measured critical density

Critical density is the density past which flow falls. Measured on a straight two-lane
road at Mainz's own 60 km/h limit, with the same W99 calibration and the signal method
that reproduced the curve:

| | measured |
|---|---|
| capacity | 1,000 veh/h/lane |
| **critical density** | **21.1 veh/km/lane** |
| speed there | 47 km/h |
| congested branch reaches | 38.1 veh/km/lane at 779 veh/h/lane |
| fall from the peak | 22% |

Jam density is 1000/4.5 = 222 veh/km/lane, so the critical density is **0.095 of jam**.
`DENSITY_CRITICAL` is now that, with SRC's 0.3 kept beside it for comparison.

SRC's 0.3 is 67 veh/km/lane, **three times past the density at which flow starts to
fall**. A reward thresholded there pays for congestion long after the capacity drop it
exists to prevent has happened, which is a reason the objective cannot reward
prevention that is independent of the network, the demand and the training.

## The exit backpressure, working

The junction was the bottleneck, and the produced net says so directly. Extending the
network past link 218 created junction `218-end`, and two things about it throttled the
road whatever meter rate was configured:

* **The movements crossed.** The connections fanned each of 218's three lanes into each
  of the exit road's three, so the junction carried nine internal lanes and a foe matrix
  in which movement 2 conflicts with six others. At a priority junction the conflicting
  movements yield to one another, so vehicles crossed one at a time.
* **Every movement was read as a turn.** The exit road ran due east while 218 arrives
  heading about 40 degrees north of east, so netconvert capped the internal lanes at
  7.62 to 11.95 m/s against 16.67 m/s on both sides of the junction.

The exit road now takes 218's own lane count and its own final heading, and lane i
connects only to lane i. Junction `218-end` then has three internal lanes, `foes="000"`,
and internal speeds of 16.67 m/s. Discharge measured at 6,000 veh/h offered:

| exit | discharge | segment 11 density |
|---|---|---|
| no meter | 2,874 veh/h | 0.075 of jam |
| 0.95 green | 2,442 veh/h | 0.144 |
| 0.62 green | 1,440 veh/h | 0.211 |

Before the fix all three were about 1,000 veh/h. The meter is now what restricts the
exit, and the rate is a number we choose.

### The fundamental diagram at the exit super-segment

`scripts/measure_fd_exit.py`. The instrument is the one that reproduced the curve on a
straight road: an intermittent downstream restriction rather than a narrowing, so the
section can run at its own capacity during green and fill during red; a demand rush that
rises and falls, so the queue grows back over the section and then clears; and 20 s bins,
which is what a loop detector reports. Density and speed are read on super-segment 11's
occupied edges, which are 218 and 31 -- 219 and 517 are the opposite carriageway and
hold no vehicle at any demand here, so counting their lane-metres would halve the
density. Flow is k times v.

At 0.85 green on a 90 s cycle, with a rush peaking at 4,200 veh/h over 5,400 s:

| | measured |
|---|---|
| peak flow | 826 veh/h/lane |
| **critical density** | **17.0 veh/km/lane** |
| speed there | 48.6 km/h |
| congested branch reaches | 45.3 veh/km/lane at 595 veh/h/lane |
| fall from the peak | 28% |

Flow rises linearly to a peak and falls beyond it, and speed falls monotonically from 59
to 14 km/h. The loading and unloading halves of the rush both reach 29.3 to 42.6
veh/km/lane and give mean flows of 693 and 685 veh/h/lane there, a gap of 1%, so the
congested branch is a property of the road and not a transient of the loading.

The shape holds across meter rates, and the peak approaches the road's own capacity as
the meter loosens rather than being set by it:

| green | peak flow | at density | fall past the peak |
|---|---|---|---|
| 0.60 | 735 veh/h/lane | 14.2 | 54% |
| 0.75 | 813 | 19.1 | 38% |
| 0.85 | 845 | 17.8 | 28% |
| 0.95 | 855 | 22.2 | 16% |

The critical density of 17.0 veh/km/lane on this super-segment is 19% below the 21.1
measured on the straight road and now in `DENSITY_CRITICAL`, which is the direction
expected: super-segment 11 contains a four-to-three lane drop between edges 31 and 218,
so its capacity is below an isolated straight road's. The threshold is not changed,
because it applies to all twelve super-segments and the straight-road measurement is the
one made independently of any particular segment's geometry. SRC's 0.3 of jam is 67
veh/km/lane, which on this segment is four times the density at which its flow peaks.

## Training on the throttled network

Both arms trained on seeds 1-10, selected by validation return on seeds 11-15, and read
seeds 16-20 once. 80 episodes, 2,500 s, 100% penetration, 0.5 s steps, throughput
counted from 600 s so the fill is excluded. The exit meter is at 0.85 green, and
`DENSITY_CRITICAL` is 0.095 rather than SRC's 0.3.

The reward still discriminates at the lower threshold: with no control, 5 of the 12
super-segments are over critical at 600 s, 10 at 1,200 s and 12 by 2,100 s, so the
congestion term carries signal for most of the episode and saturates only at the end.

Held-out seeds 16-20:

| arm | return | flow veh/h | arrivals | speed km/h |
|---|---|---|---|---|
| DSRC, the HERE observation | **-2,421.85** (+11.7%) | 2,150 (-0.3%) | 1,266 (-1.1%) | 14.32 (+9.4%) |
| SRC, its original six features | -2,432.82 (+11.3%) | 2,146 (-0.5%) | 1,243 (-2.9%) | **14.84** (+13.4%) |
| no control | -2,743.99 | **2,156** | **1,280** | 13.09 |

Percentages are against no control. Best checkpoints were episode 59 for DSRC and 54 for
SRC; both arms degraded after that, DSRC's flow falling from 2,150 at its best checkpoint
to 1,737 by episode 79, so selection on the validation seeds is doing work here.

**The deployable observation still costs nothing.** DSRC and SRC differ by 0.5% in
return, 0.2% in flow and 3.6% in speed, in opposite directions on the last two. The four
features a traffic API cannot return -- density, following gap, and the entry and exit
counts -- carry no information this controller uses. That is the same result the
unthrottled network gave, now with a capacity drop present in the network.

**Throughput is unchanged by control, and the throttle did not change that.** All three
arms serve between 2,146 and 2,156 veh/h, a spread of 0.5%, while their mean speeds span
13.09 to 14.84 km/h. Both arms improve the objective they were trained on and neither
serves more vehicles.

### Why the capacity drop is present and still not recoverable

The exit fundamental diagram shows the drop plainly: 826 veh/h/lane at 17.0 veh/km/lane
against 595 at 45.3. A controller holding the exit super-segment near the critical
density rather than letting it reach 45 would discharge about 39% more. That did not
happen, and the reason is the demand rather than the controller.

The published scenario offers 18,000 veh/h for 1,200 s, which is 6,000 vehicles. The
throttled network discharges about 2,150 veh/h, so about 1,490 vehicles can leave during
a 2,500 s episode. Demand exceeds capacity by roughly a factor of three from about 900 s
onward, and every arm ends the episode with a standing queue: arrivals are 1,243 to
1,280 against 6,000 offered. A policy that holds density below critical cannot do so when
demand is three times capacity. It can choose where the queue stands, not whether it
stands.

So the throttle did what it was for -- the network now has a capacity drop, measured on
its own exit -- but at this demand the drop is not avoidable and therefore not
recoverable. Testing whether the controller can recover it requires a demand near the
throttled capacity rather than three times it. That is a separate run and is not made
here.

## The exit meter is saturated, so throughput cannot respond to control

Two defects, found by asking how mean speed could rise while flow did not move.

**The reported speed was not a speed.** `metrics()["mean_speed_kmh"]` is an unweighted
mean over super-segments of an unweighted mean over each segment's occupied edges, read
at the final simulation step rather than averaged over the window. A segment holding
three vehicles counts as much as one holding four hundred. On the throttled network with
no control it reads 21.0 km/h where the vehicle-weighted speed is 11.9. The statistic is
what SRC's reward is built from and stays, but it is not a description of the traffic;
`space_mean_speed_kmh` and `mean_vehicles`, both averaged over the scored window, are.

**Throughput is set by the exit meter, not by the network.** Measured over 600-2,500 s
on seeds 16-20, with the checkpoints trained without the entry gate:

| arm | vehicles in net | space-mean speed | reported speed | k occupied | veh km/h | exit queue | flow |
|---|---|---|---|---|---|---|---|
| no control | 2,511 | 11.85 km/h | 21.00 | 47.9 | 26,700 | 48 | 2,035 |
| DSRC, gated entries | 2,119 | 12.99 km/h | 22.01 | 40.2 | 26,000 | 49 | 2,016 |

Vehicle-kilometres per hour agree within 3%, as they must when throughput and route
length agree, so the speed difference is a vehicle-count difference and nothing else:
16% fewer vehicles cover the same distance per hour, so each averages more speed. No
traffic moves faster.

The flow-density pair is the part that does not belong on a fundamental diagram. Both
arms sit far past the critical density of about 21 veh/km/lane, at 40.2 and 47.9, and
the diagram requires the lower density to carry the higher flow. It does not: 2,016
against 2,035. The exit-road queue explains it, at 48 and 49 vehicles in both arms. The
meter is saturated in every arm, and a saturated signal discharges at its own rate
whatever is upstream, so flow is clamped near 2,150 veh/h and the network only has to
deliver at least that. Flow is not a function of network density here.

**The cause is a choice made when the meter was set.** 0.85 green passes about 2,150
veh/h and the unmetered network delivers about 2,874, so the constraint was put BELOW
the network's own capacity. That makes the boundary the bottleneck and takes the
network's capacity drop out of play, which is why no arm can gain throughput. It is a
property of the scenario and not of the controller.

The exit constraint has to stay finite -- a free sink lets any number of vehicles leave
-- but it has to sit ABOVE the uncontrolled capacity, around 3,000 to 3,200 veh/h, so
that the binding limit is the capacity drop inside the network, where density is what
the controller acts on and where holding it near critical can recover flow.

## The gate: served flow does not depend on density, at any exit setting

The measurement that should have run before any training. No control, flat demand held
for the whole episode -- which is what training uses -- flow counted from 900 s, dt 0.5.
The earlier curve was traced under a rush ramping from a quarter of its peak and back,
where the exit meter alternately binds and releases; training then ran under a flat
demand where it is saturated throughout. A curve measured under one demand does not
license an experiment under another.

Exit unmetered, so the network's own capacity is the limit:

| offered veh/h | served veh/h | density veh/km/lane | space-mean speed |
|---|---|---|---|
| 1,800 | 1,786 | 2.3 | 55.7 |
| 3,000 | 2,864 | 4.5 | 48.7 |
| 3,600 | 2,883 | 6.5 | 36.9 |
| 6,000 | 2,883 | 15.3 | 16.1 |
| 9,000 | 2,894 | 21.8 | 10.8 |
| 18,000 | 2,891 | 23.3 | 9.8 |

Over the five points where demand exceeds what the network serves, served flow stays
between 2,853 and 2,894 veh/h, a range of 1.4%, while density runs 6.5 to 23.3
veh/km/lane and space-mean speed falls from 36.9 to 9.8 km/h. Metering the exit moves
the constant and changes nothing else: 2,403 to 2,438 at 0.98 green, 2,157 to 2,165 at
0.85.

**Served flow is a constant set by a saturated bottleneck, not a function of density.**
There is no capacity drop for a density-holding controller to prevent, so no arm
comparison on this port can show a throughput gain, and none of the four training runs
made in this leg was measuring anything else.

### The interior is deeply congested and it makes no difference

At 9,000 veh/h offered with the exit unmetered, the twenty slowest links run at 0.8 to
8.4 km/h against a 60 km/h limit, at densities of 43 to 125 veh/km/lane against a jam
density of 222. Links 53, 56 and 59 sit at 120 to 125 veh/km/lane moving at 0.8 km/h.
The network average of 23.3 hides this: it is taken over 188 edges, most of which the
demand never reaches.

So the interior does reach deep congestion, and served flow still does not fall. The
binding bottleneck discharges at the same rate whether or not a queue stands behind it.
The base network contains no traffic lights at all, so that bottleneck is a priority
junction, and SUMO's junction model keeps a vehicle out of a junction it cannot clear,
which is what prevents the blocking that costs a real network its throughput. Vissim's
conflict areas model exactly that blocking; the base layout carries one and it is
PASSIVE, while the four signalised variants carry 68 of which 11 are active.

## Why no exit setting, entry gate or junction parameter changes the answer

**Mainz as published has almost no conflicting movements.** Of 139 priority junctions in
the produced net, 2 have any conflicting movement at all; the rest are merges and
diverges where movements do not cross. That is consistent with the paper's own title,
which is about free flow road networks. Making junctions block therefore changes nothing,
because there is nothing to block: with `jmIgnoreKeepClearTime` at 0 s served flow is
2,869 to 2,891 veh/h and with SUMO's default it is 2,853 to 2,894.

**Throughput is the capacity of the single exit link.** Every route leaves through link
218, three lanes. Measured at 9,000 veh/h offered, 218 carries 954 veh/h/lane, and the
other high-throughput links sit at 836 to 969 veh/h/lane -- the same per-lane capacity
measured on the straight road, 1,000 veh/h/lane at 21.1 veh/km/lane. Three lanes at 955
is 2,865 veh/h, which is the served flow. The network funnels everything through one link
running at its capacity.

**An uncongested link's capacity does not drop.** 218 is at capacity but not congested,
because nothing downstream restricts it, so its discharge is a constant. The congestion
that does exist -- links 53, 56 and 59 at 120 to 125 veh/km/lane moving at 0.8 km/h -- is
a queue upstream of 218 that feeds it at exactly its capacity. Queue length varies with
demand; discharge does not.

That is the whole of it. The port is not broken: this network's throughput is a constant
by construction, and no controller acting on density can change a constant.

### What remains, and it needs a different demand rather than a different network

A link does have a capacity drop: the straight road falls from 1,000 veh/h/lane at 21.1
veh/km/lane to 779 at 38.1, a fall of 22%, and the exit super-segment falls 28%. So if
link 218 were pushed INTO congestion it would discharge about 780 veh/h/lane, or 2,340
veh/h against 2,865. The 18% difference is real and is exactly what SRC exists to
recover.

Flat demand cannot show it. Below 2,865 veh/h nothing congests; above it, the excess
queues upstream and 218 still runs at capacity. There is no flat demand at which 218
breaks down. A BURST can: demand above capacity for long enough to push 218 into the
congested branch, then falling back below capacity, and the question is whether discharge
returns to 2,865 or stays near 2,340. If it stays, the network has hysteresis, a
controller that prevents the breakdown recovers 18%, and the experiment has something to
measure. If it recovers immediately, it does not.

## The conflation at the root of this leg

Two different quantities were being called the fundamental diagram, and only one of them
bears on whether a controller can gain throughput.

**A link's `q = k v` curve.** Measure density and speed on a stretch of road and multiply.
As the road fills, speed falls faster than density rises, so the product falls and the
plot is a hill. This is what `measure_fd_straight.py` and `measure_fd_exit.py` produce,
and it is what was delivered when the request was to make the network behave like a
fundamental diagram. It is a correct measurement of a real relation.

**It is not evidence of a capacity drop.** Measured on the straight road at Mainz's
60 km/h with a 0.5 s step:

| fleet | peak flow | at density | fall past the peak |
|---|---|---|---|
| W99, the paper's calibration | 997 veh/h/lane | 21.1 veh/km/lane | 23% |
| Krauss, SUMO's default | 2,127 veh/h/lane | 43.0 veh/km/lane | 15% |

Krauss is the model this project recorded as having no capacity drop at all -- on
`inverted_tree` its served flow rose monotonically from 890 to 1,110 veh/h and a
perfect-information metering oracle could not beat inaction. It produces the hill anyway.
The hill is the shape of `v(k)`, which every car-following model has.

**The quantity that matters is the bottleneck's discharge falling once a queue stands
behind it**, which shows as SERVED FLOW falling as offered demand rises. That is what
`measure_mainz_fundamental_diagram.py` measures and what a controller holding density
below critical recovers. On Mainz it does not fall at any exit setting, entry policy or
junction parameter: 2,853 to 2,894 veh/h across a 3.6x density range. On `inverted_tree`
it does not fall either once the seed spread is computed.

So the exit backpressure did what was asked of it -- the exit super-segment traces a
proper hill, peak 826 veh/h/lane at 17.0, falling 28%, loading and unloading agreeing to
1% -- and that result is sound. It was the inference from it that was wrong: a link that
traces a hill does not imply a network whose throughput can be recovered.

## The fleet has no capacity drop, and that is the whole cause

Measured directly rather than inferred. Served flow through a two-into-one lane drop,
the restriction in place throughout, demand swept from below capacity to four times it:

| offered veh/h | 900 | 1,100 | 1,300 | 1,600 | 2,000 | 2,800 | 3,600 |
|---|---|---|---|---|---|---|---|
| W99, the paper's calibration | 892 | 960 | 932 | 956 | 976 | 956 | 952 |
| EIDM | 900 | 1,116 | 1,260 | 1,596 | 2,012 | 1,792 | 1,900 |
| **EIDM, 1 s reaction** | 896 | 1,116 | 1,252 | 1,600 | **2,064** | **1,536** | 1,560 |
| IDM | 896 | 1,108 | 1,268 | 1,596 | 1,928 | 1,764 | 1,788 |
| Krauss, sigma 0.9 | 892 | 1,104 | 1,288 | 1,596 | 1,596 | 1,608 | 1,600 |

Under the paper's W99 the bottleneck serves a flat 930 to 976 veh/h whatever is offered.
Adding start-up lost time does not change it: at `startupDelay` 0, 1 and 2 s the queued
range is 936-957, 927-960 and 933-948 veh/h. Nor does reaction time on its own under
W99, nor removing the restriction and releasing the queue.

W99'S QUEUES DISCHARGE FASTER THAN FREE FLOW, which is the opposite of reality. A
released W99 queue discharges at 1,980 veh/h against a free-flow capacity of 1,899, and
the gap widens with reaction time: at a 2 s action step, free-flow capacity falls to
1,604 while queue discharge stays at 1,920, so the queue is 19.7% FASTER. A discharging
W99 platoon is perfectly regular at the model's tightest headway, where free-flowing
traffic at capacity has spread in speeds and gaps. Real queue discharge is 5 to 20%
BELOW free-flow capacity, from start-up lost time and bounded acceleration.

That single fact explains every null in this leg. A bottleneck sets the LEVEL of
throughput; it does not make throughput fall with density. So Mainz's three-lane exit
link, `inverted_tree`'s merges, a lane drop on a straight road and a metered exit all
behave identically: served flow is a constant, and no density-holding controller can
recover a constant.

### EIDM with a reaction time has one, and it is resolved

Five seeds, two-standard-error bars, same lane drop:

| offered veh/h | 1,600 | 1,800 | 2,000 | 2,200 | 2,600 | 3,000 | 3,600 |
|---|---|---|---|---|---|---|---|
| served | 1,603 | 1,798 | **2,018** | 1,668 | 1,602 | 1,619 | 1,595 |
| +/- 2se | 10 | 29 | 26 | 161 | 42 | 98 | 91 |

Free-flow capacity 2,018 veh/h/lane, queue discharge about 1,600, a fall of 423 veh/h or
**21.0% against a combined bar of 95** -- resolved, not noise. Both numbers are also
closer to reality than W99's: real single-lane capacity is 2,000 to 2,400 veh/h and real
queue discharge 1,700 to 2,000, where W99 at CC1 = 2.5 gives 950.

The cost is the paper's Appendix A calibration, which is its own transcription of what
its Vissim runs used. Keeping it means keeping a fleet in which the mechanism the paper
depends on does not exist. `carFollowModel="EIDM"` with `actionStepLength="1.0"` is
SUMO's model for driver imperfection and reaction time, and its parameters here are
defaults rather than anything calibrated, which is a provenance question of its own.

## Result on the EIDM fleet with a lane-drop exit

The first configuration in this leg whose pre-training gate passed, and the first with a
throughput difference. Seeds 1-10 train, 11-15 validate and select, 16-20 read once. 80
episodes, 2,500 s, 100% penetration, 0.5 s steps, flow counted from 900 s. Demand is a
sustained 4,500 veh/h, just past the 3,900 at which the network breaks down, so there is
a drop to prevent rather than one that cannot be avoided.

| arm | flow veh/h | arrivals | return | space-mean speed | held at entry |
|---|---|---|---|---|---|
| DSRC, the HERE observation | **3,775** (+5.8%) | 1,997 | 182.5 | 40.81 | 51 |
| SRC, its original six features | 3,756 (+5.3%) | 1,991 | **193.8** | 40.82 | 50 |
| no control | 3,568 | 1,951 | 55.1 | **42.68** | 0 |

Per seed, and paired on seed because each arm ran the same traffic realisation:

| seed | 16 | 17 | 18 | 19 | 20 | mean | +/-2se |
|---|---|---|---|---|---|---|---|
| no control | 3,722 | 3,553 | 3,564 | 3,556 | 3,444 | 3,568 | 89 |
| DSRC | 3,760 | 3,756 | 3,793 | 3,811 | 3,753 | 3,775 | 23 |
| SRC | 3,849 | 3,707 | 3,564 | 3,827 | 3,836 | 3,756 | 109 |

DSRC gains **+207 +/- 92 veh/h**, SRC **+188 +/- 133**. Both are resolved against the
seed spread, DSRC comfortably and SRC narrowly.

**The deployable observation costs nothing, on a network where throughput can move.**
Every earlier version of this comparison was made where served flow was a constant, so
the two arms agreeing said little. Here throughput responds and they still agree: 3,775
against 3,756, a difference of 0.5% inside both bars. DSRC is also far steadier across
seeds, 23 against 109. The four features a traffic API cannot return -- density,
following gap, and the entry and exit counts -- carry no information this controller uses.

**The gain is throughput bought with speed.** Both arms run SLOWER than no control,
40.8 against 42.7 km/h, while serving more vehicles. That is the mechanism working as
described: holding density below critical upstream of the merge keeps the merge
discharging near capacity instead of breaking down to the congested branch. It is not an
artifact of admission control -- the entry gate held about 50 of 3,124 vehicles, against
1,750 in the earlier W99 runs where it was doing all the work.

**What is not established.** One demand level, chosen just past breakdown; the gain
should be swept across demand before it is quoted as a property of the network. DSRC's
selected checkpoint is its last episode, so it may not have converged. EIDM's parameters
are SUMO defaults rather than anything calibrated against measured traffic, and the
three-into-two exit drop is a modelling choice of ours.

## `inverted_tree` under EIDM

Asked whether the tree passes the same gate. Both variants do, and less strongly than
Mainz with a lane-drop exit. Five seeds, 1,500 s episodes at dt 0.5, flat sustained
demand, no control.

`inverted_tree`:

| offered | 900 | 1,200 | 1,500 | 1,800 | 2,100 | 2,400 | 3,000 | 3,600 |
|---|---|---|---|---|---|---|---|---|
| served | 900 | 1,198 | 1,478 | 1,793 | 1,728 | 1,759 | 1,665 | 1,859 |
| +/-2se | 1 | 4 | 18 | 5 | 11 | 71 | 178 | 87 |

Capacity is 1,793 +/- 5 veh/h at 1,800 offered, the last point where demand was met, and
the first saturated point serves 1,728 +/- 11: a fall of 65 veh/h, **3.6%, resolved**.
Past that the curve is not monotone -- 1,759, 1,665, then 1,859 with bars of 71 to 178 --
so only the first step past breakdown is readable.

`inverted_tree_bottleneck`, which has the final lane drop:

| offered | 900 | 1,200 | 1,500 | 1,800 | 2,100 | 2,400 | 3,000 | 3,600 |
|---|---|---|---|---|---|---|---|---|
| served | 901 | 1,199 | 1,161 | 1,207 | 1,140 | 1,224 | 1,148 | 1,201 |
| +/-2se | 2 | 5 | 62 | 20 | 20 | 29 | 21 | 16 |

Capacity 1,199 +/- 5, lowest past it 1,140 +/- 20: a fall of 59 veh/h, **4.9%,
resolved**.

Both capacities are lower bounds, because the grid steps from a fully served point
straight to a saturated one and the true peak lies between. Against Mainz with a
three-into-two exit drop at 11.3%, the tree's drop is a third the size, so Mainz remains
the topology to run the arms on.

Two instrument defects were fixed to get here. `SumoTopologyEnv._FOLLOWING_ATTRIBUTES` is
an allowlist and refused EIDM's `tau` and `actionStepLength` outright, so the tree could
not have been run with this fleet at all. And the verdict took the largest number in the
served column as capacity; once a network saturates, a later point can read higher than
the pre-breakdown one through noise, which put the peak past the breakdown and reported
that nothing was sampled beyond it. Capacity is now the best flow measured while demand
was still being met.
