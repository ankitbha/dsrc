# The simulation leg: what was measured, and why it is a null

One account of a result spread across task-list entries 92 to 134. It states what was
measured, what each measurement was controlled against, and exactly what is and is not
established. Sections below are in the order they were written; this summary is
current as of 2026-09-10 evening.

## Short version

**The mechanism this project posits -- in-stream AV speed modulation -- neither gains
when handed perfect information nor presents a learnable gradient, on this road.** Two
instruments built for different purposes agree:

- a perfect-information metering oracle serves **-0.8 and +0.8** more vehicles than
  no control, against a spread of about 16;
- the correlation between one AV's action and its own advantage is indistinguishable
  from zero: **+0.00 +/- 0.29** at a one-second lever and **-0.47 +/- 0.51** at a
  one-minute one, against an instrument that reads 17 to 33 times its null when a
  correlated advantage is supplied.

**The predecessor paper's own formulation was ported in full and is also a null at
the vehicle level.** Its two-term threshold reward, its 8.33/12.5/16.67 m/s speed
bins, one decision per simulated minute at gamma 0.9, and a corrected link-level
congestion signal every co-located AV sees identically. The paper's design does not
transfer to a vehicle-attached agent on this road.

**The objective is not the problem.** Paired over five seeds, the threshold reward
responds to fleet behaviour at every mix level and discriminates it at least as well
as counting arrivals does, so the null is not attributable to an unmovable or noisy
objective. Its entire resolved response is the speed term; the congestion-threshold
half, which is what pays for keeping a link below critical density, resolves at one
mix level of four and changes sign across them.

**Three things were NOT established when this section was first written. Two are now
measured and the third stands.**

1. **STANDS.** The oracle is one hand-written heuristic given perfect state, so it is
   a LOWER bound on the best controller and not an upper one. "No controller can gain
   on this network" is not proven and cannot be proven this way.
2. **RESOLVED.** Every arm that produced a clean null pays each agent the SAME reward, so those
   nulls cannot separate "the reward does not respond to individual actions" from
   "nothing responds to individual actions". Two arms have an unshared reward and BOTH
   HAVE NOW BEEN MEASURED: a per-agent local reward at the vehicle level reads a
   correlation of +0.0016 against the shared arm's -0.0042 at a standard error of
   0.0135, and a segment paid its own threshold term reads z -0.12 +/- 0.51. Neither
   separation appears, so the shared reward is not the reason.
3. **NULL, BUT AT A COARSE SENSITIVITY.** The paper's agent is a ROAD that persists for the episode; ours is a VEHICLE that
   makes 7.1 decisions and leaves. That difference is confounded with every lever
   measurement. The segment-attached agent that separates it reads null at both powers
   measured, -0.12 +/- 0.51 and -0.46 +/- 0.66. Scaled by the instrument's own
   only a z, which the calibration shows cannot separate a correlation of 0 from 0.1 on
   an arm this size.

**The instrument was rebuilt mid-investigation.** Its original permutation floor is
not a valid null when advantages are temporally correlated, and it made arms read
WORSE than random. The replacement resamples the action from the policy at the same
state; it has seven controls, and it reproduced the vehicle-level result at higher
precision, so the earlier work stands rather than falls. Readings taken under the old
floor were discarded rather than reported with a caveat.

## The environment is sound, and four defects had to be fixed to make it so

Each of these alone prevented any result. They are listed because the null means
nothing without them: a null on a broken environment is not a measurement.

| defect | what it did | fix |
|---|---|---|
| 1 s physics step | manufactured a 17.6% "throughput gain" that vanished at a converged step size | dt 0.1, and the result retracted (task 92) |
| unintended permanent yield at the merge | every edge had `priority="-1"`, so netconvert broke the tie by geometry and one approach was permanently minor | zipper junctions; capacity 1110 -> 1600 veh/h (task 98) |
| Krauss car-following | computes a collision-free safe speed exactly and recovers immediately, so there is no capacity drop and nothing to recover | the predecessor paper's calibrated Wiedemann-99, giving a 24% capacity drop (task 100) |
| single-lane approaches | a slow AV cannot be overtaken, so it is an obstruction and not a meter | two lanes on every approach (task 102) |

The oracle progression across those fixes is a sequence of removed harms and no
found benefit: -16.6 +/- 7.8, then +7.6 +/- 15.2 at congestion onset, then -21 to
-38 at the capacity peak, then -0.8 and +0.8 oversaturated with two lanes.

## The learning result, and the instrument that produced it

**The statistic.** With advantages normalised to unit standard deviation, the norm of
`d(policy_loss)/d(actor)` measures how much the advantage correlates with the action:
a term uncorrelated with the action cancels across the batch, an aligned one adds.
`policy_loss` itself says nothing, because at a probability ratio of 1 it is minus
the mean normalised advantage, which is zero by construction however informative the
advantage is.

**Its two controls, without which every reading is uninterpretable -- and the floor
had to be rebuilt.** The ceiling is a SYNTHETIC advantage built to correlate with the
action; it drives no vehicle and is not a controller, it exists to show the statistic
can move.

The floor was originally the same advantages permuted across the batch. **That is not
a valid null when advantages are temporally correlated**: it destroys the
action-advantage pairing AND each agent's temporal profile, so a smooth advantage
sequence becomes rough, cancels less against `grad log pi`, and the floor comes out
high. It was caught when two segment-level arms read a measured value systematically
BELOW their own null, which is that bias showing rather than data worse than random.

The valid null resamples an action from the policy at the SAME observation and keeps
the advantage, so states, advantages and their temporal structure all survive and only
the pairing breaks; under the score-function identity its expectation is zero. It is
tested (`tests/test_action_alignment.py`) against a signal it must see, noise it must
not, an autocorrelated-but-uninformative sequence, and a partial signal it must grade
-- none of which the permutation floor ever was.

**The result survived the repair.** On `mappo_sumo`, three seeds:

| null | mean z | ceiling |
|---|---|---|
| permutation floor | -0.10 +/- 0.42 | 25 to 39 |
| resampled action (valid) | **+0.00 +/- 0.29** | 31 to 33 |

Same conclusion, better precision. The table below therefore stands. Measured on the
shipped configuration, three seeds, 17,296 decisions:

| advantage | gradient norm | over the floor |
|---|---|---|
| as measured | 0.0499 | **0.784** |
| shuffled -- the floor | 0.0637 | 1.000 |
| action-correlated -- the ceiling | 1.5971 | **25.1** |

**Seven dimensions varied, none clearing the floor.** Three seeds each, floor and
ceiling on every arm:

| varied | range | best measured over floor |
|---|---|---|
| reward decomposition | team, neighbourhood, own-vehicle, both | 0.850 |
| action hold length | 1 s, 5 s, 20 s | 0.945 |
| discount horizon | 10 to 1000 decisions | within 1.1x |
| critic input | with and without privileged neighbourhood | within 1.05x |
| speed bin scaling | four schemes, binding share 6% to 29% | 1.020 |
| operating point | 900, 1200, 2400 veh/h | 1.269 |
| AV penetration | 0.25, 0.50, 1.00 | 1.403 |

At 100% penetration the policy commands the entire fleet -- 168 vehicles at 2400
veh/h -- and one agent's action still does not correlate with its own advantage.

**Those ratios are against ONE permutation, which is a single draw from the floor and
not the floor.** Estimated properly from 60 permutations per rollout, the floor's
standard deviation is 20 to 30% of its mean, so every ratio above is inside it. The
correct statistic is the z-score of the measured value in the permutation
distribution:

| arm | floor mean | floor sd | mean z over three seeds |
|---|---|---|---|
| `sumo_capacity_drop`, penetration 0.25 | 0.054 | 0.016 | -0.10 |
| `sumo_saturating`, penetration 0.25 | 0.088 | 0.025 | -0.01 |
| `sumo_saturating`, penetration 1.00 | 0.042 | 0.011 | +0.82 |
| `sumo_capacity_drop`, gamma 0.999 | 0.052 | 0.013 | +0.33 |

The largest is under one standard deviation. Penetration 1.00 was then run on ten
seeds: **mean z +0.630 +/- 0.379, which is 1.66 standard errors and does not clear
the project's two-standard-error bar.** Eight of the ten seeds are positive, one
reaches +2.36 and one -1.92. It is the single place in this investigation where "no
signal" might be wrong, and it is the arm to extend if the question is reopened.

It does not reopen the headroom question. A learning signal at 100% penetration would
say a policy could be trained, not that a trained policy would gain anything, and the
metering oracle still serves -0.8 and +0.8 more vehicles than no control. Both would
have to move. 100% penetration is also not a deployment operating point: every vehicle
on the road is controlled, against the 25% of the predecessor paper.

**The control on the instrument itself.** Every reading above was taken at a randomly
initialised actor, so a rising correlation with training would invalidate them. A
checkpoint 23 updates in reads 0.812 against a fresh actor's 0.848: indistinguishable.

## Two measurements that stand and do not reach the gradient

Both are real and neither changes the conclusion, which is worth stating so they are
not mistaken for support.

- **The reward counterfactual.** Holding one AV at 20 m/s for a 20 s window and
  repeating from the same seed at 30 m/s: the agent's own local reward moves 22.82,
  another agent's action moves it 1.28, a ratio of **17.8**. The team reward moves
  2.06 against a window total of 193.03, which is **1.07%**. So the per-agent reward
  is far better attributed than the team reward -- and neither survives into the
  advantage.
- **The critic regression.** Giving the centralized critic the agent's own
  neighbourhood raises out-of-sample R2 on the per-agent return from 0.808 to 0.886,
  and leaves it at 0.854 against 0.858 under the team reward, which is the control
  saying the improvement is about the per-agent term. The actor's gradient is
  unchanged.

## What was retracted, and by what

Recorded because the retractions are part of the result.

| claim | retracted by |
|---|---|
| AVs held at 10 m/s raise throughput 17.6% | the step-size sweep: the treatment was invariant to dt and only the control moved (task 92) |
| joint gradient clipping starved the actor | Adam is invariant to a uniform gradient rescale; 20 steps move a parameter 0.383268 unscaled and 0.383267 scaled by 0.0063 |
| the per-agent reward lowered the policy gradient; the horizon does not matter; privileged critic features do not help the gradient | all three were comparisons between two noise floors, taken before the floor was measured (task 105) |
| rescaling the speed bins would produce a gradient | three rescalings raise the binding share from 6% to 29% and leave the gradient at its floor; the recommended one is the worst (task 106) |
| the permutation floor is a valid null | two arms read systematically BELOW their own floor, which is the signature of a null not exchangeable with the data; replaced by resampling the action from the policy at the same state |
| every z-score error bar in the investigation | `seed_everything(0)` made the actor and critic weights bit-identical across seeds, so each bar came from ONE network draw; re-measured with the draw varying per seed |
| 7 of 9 segments run over critical density | inferred from a training score by assuming every segment ran at the vehicle-weighted network mean; measured directly at 1.5 of 9 (task 130) |
| one branch is congested and the other free | read off density ratios, but density is flow over speed, so a free-flowing link and an empty one look alike at fixed flow; both branches carried the same demand |
| the summed entropy declines monotonically | called from three updates; update 5 reversed it, and updates 7 and 10 reversed it again |
| the shared-reward segment arm reads consistently negative, and here is why | it reads +0.25 +/- 0.76 under the valid null with the initialisation redrawn per seed; the negativity was the instrument |
| the threshold objective is a noisier measure of behaviour than an arrival count | true unpaired and false paired: the ratio of each difference to its own bar is 2.79, 1.76, 1.75 and 2.63 for the reward against 2.91, 1.60, 1.29 and 0.85 for arrivals (task 133) |

## The pre-registered run

`configs/training/mappo_sumo.yaml`, seed 7, 25 updates of three episodes each, with
the gate fixed before the run. Criterion 1, summed entropy below 1.9775 of a 2.1972
maximum: the lowest reached is 2.1778. Criterion 2, the score trending up by more
than the variation between consecutive updates: it moves -0.1772 against a step
standard deviation of 0.2186. Criterion 3 as written carried no numeric threshold,
which is a defect in the pre-registration and is recorded as such rather than
resolved after the fact.

A randomly initialised actor already reads a joint modal share of 0.1378 against a
uniform 0.1111, and 23 updates take it to 0.1486, so most of the departure from
uniform is initialisation.

## SUPERSEDED IN SCOPE, 2026-09-10: this is a null about ONE formulation

Everything above was measured with a per-vehicle action taken once a second. Reading
the predecessor paper (arXiv:2506.11973) showed that is not the problem it solved.
Its agent sets a maximum speed for a 2-3 km super-segment once a MINUTE, from a state
of per-segment density, speed, gap, inflow and outflow, and rewards
`-alpha * 1[rho > rho*] + beta * v` -- two terms, not eleven. Its action is thousands
of times the lever, and it is a centralized controller with AVs as the compliance
mechanism, tested at 25 to 100% compliance.

**So the seven dimensions in the table above were all varied around a one-second
lever.** They do not bound what a macroscopic one does, in either direction. Ankit's
statement of it: each vehicle is changing decisions too quickly, nearby vehicles are
not matching, and too fine a control resolution devolves into noise.

**THE VEHICLE-LEVEL HALF OF THAT PORT IS NOW MEASURED, AND IT IS ALSO A NULL.**
`mappo_src` under the valid resampled-action null, three seeds: mean z
**-0.47 +/- 0.51** (per seed -1.33, +0.44, -0.53) against the one-second baseline's
+0.00 +/- 0.29. With a ceiling near 17 and a null standard deviation of 0.029 on a
null of 0.11, z is about 60 times the correlation AT THE CEILING. That slope does not
hold at small correlations -- z/c falls to 3.7 at c = 0.05, measured in task 137 --
so the bound from this arm is about 0.025 rather than the 0.017 an earlier version of
this paragraph gave. It remains a null from a sensitive instrument, not an insensitive
one, and the sensitivity is stated at the correlation it applies to.

So holding one action for a simulated minute -- with the paper's threshold reward, its
speed bins, its discount horizon, and a link-level congestion signal the actor can
see -- produces no more action-attributable signal than holding it for a second.

**One confound this cannot resolve.** The lever change carries a cost: at 60 s each
agent makes 7.1 decisions in its lifetime instead of 90.5, a thirteenfold loss of
temporal structure, because the agent is a vehicle that leaves rather than a road that
persists. A larger per-decision effect and a much shorter trajectory pull opposite
ways and this measurement cannot separate them. The segment-as-agent arms, where the
agent persists for the whole episode, are what can, and they have now reported: paid
its own per-segment reward, the road-attached agent reads -0.12 +/- 0.51 at 171
segment-decisions and -0.46 +/- 0.66 at 533. Neither separates from zero, so the
confound resolves in favour of neither side -- restoring the agent's temporal
structure does not recover a signal that the shorter trajectory was hiding.

`configs/training/mappo_src.yaml` ports the formulation. What it
changes and what was measured about each is task 113; the honest accounting of which
change is large is:

| change | size, measured |
|---|---|
| threshold reward | LARGE, but not as large as first claimed: the penalty fires on **1.50 of 9 segments** on average, measured directly by the fleet-mix sweep at p = 0 over a 600 s episode. The earlier figure of "about 7 of 9" was INFERRED from the training score using a bad approximation -- it assumed every segment runs at the network mean speed, but that is vehicle-weighted, so a nearly-empty free-flowing leaf contributes a segment mean of 25 m/s and almost nothing to the network mean |
| decision interval 60 s from 1 s | LARGE, and untested before this |
| `downstream_congestion_estimate` corrected to the link ahead, ungated | real: it was reading the ego link and vanishing whenever no AV was in range |
| speed bins 8.33/12.5/16.67 from 20/27/30 | SMALL: the band on which the three values differ is 6.2% against roughly 5%, because the median AV speed is 0.11 m/s and no command binds on a stopped vehicle |

## The structure underneath every null here: the reward is shared

Stated late, because it took the segment arms to make it obvious. **In every arm that
has produced a clean null, each agent is paid the SAME reward at each step.** The team
reward is a network aggregate; so is the threshold reward, summed over segments. With
a common reward the reward SEQUENCES are identical across agents, so the whole
cross-agent variation in the advantage comes from the critic's value estimates.

At the vehicle level that is not a defect, it is the configuration under test: a
shared reward that does not respond to an individual action is precisely the
credit-assignment problem. The measurement says so cleanly.

But it means a null on those arms cannot distinguish two things:

- the reward does not respond to individual actions;
- nothing responds to individual actions.

**Only an unshared reward separates them**, and exactly two arms have one: the
per-agent local reward of task 103, and the segment-as-agent arm paid its own term.

**THE FIRST HAS NOW BEEN MEASURED AND THE HYPOTHESIS IS WRONG.** Paired on identical
rollouts with the network draw varying, three seeds:

| arm | mean z | correlation between choosing `slow` and the advantage |
|---|---|---|
| team only, shared reward | -0.44 +/- 0.56 | **-0.0042** |
| local 0.5, UNSHARED reward | -0.81 +/- 0.14 | **+0.0016** |

At n about 5,500 the correlation's standard error is 0.0135, so both are noise and
they are indistinguishable from each other. **Giving each agent a reward that
genuinely differs from its neighbours' produces no more action-advantage alignment
than a shared one**, at sensitivity to a correlation of about 0.016.

So the shared reward is NOT the reason the advantage is silent. That was the best
remaining explanation and it is ruled out. What is left is the third item in the short
version -- the agent is a vehicle that makes 7.1 decisions and leaves, where the
paper's is a road that persists. The segment arm that tests it has since read null at
both powers run, and scaled by the instrument's own ceiling it would show a
correlation bound no tighter than about 0.1 (task 138).

**RETIRED, same day.** This section previously explained why the shared-reward
segment arm read consistently negative -- with a common reward its advantage is
essentially a function of the untrained critic, so the advantage and the action are
both functions of the same observation through independently initialised networks
(task 123). Re-measured against the resampled-action null with the network
initialisation redrawn per seed, that arm reads **+0.25 +/- 0.76** over three seeds
(+1.77, -0.61, -0.42), centred on zero. The -1.32 +/- 0.16 the explanation was built
for combined a fixed initialisation with the invalid permutation floor, so the
negativity was a property of the instrument and there is nothing left to explain. The
argument may still be correct; it no longer has an observation supporting it.

**THE SECOND HAS NOW BEEN MEASURED TOO, AND IT IS ALSO A NULL.** The segment-as-agent
arm paid each segment its own threshold term rather than the network sum -- verified a
genuine decomposition, in that a clear segment reads +1.000 and a jammed one -0.850,
summing to the network's +0.150 -- and was measured against the resampled-action null
with the network initialisation redrawn per seed:

| seed | segment-decisions | measured | null | z |
|---|---|---|---|---|
| 7 | 171 | 0.26741 | 0.30232 +/- 0.08000 | -0.44 |
| 17 | 171 | 0.24793 | 0.30957 +/- 0.07627 | -0.81 |
| 27 | 175 | 0.38380 | 0.31322 +/- 0.07969 | +0.89 |

Mean z **-0.12 +/- 0.51** over the three seeds, against an instrument ceiling of 6.2
to 6.6 on a supplied correlated advantage. So neither unshared-reward arm reads above
its null, and the reading is centred on the null rather than below it, which is what
the void permutation-floor arms did.

**CORRECTED, same day.** This paragraph put the arm's threshold at 0.15 from
2/sqrt(n), the sampling error of a correlation coefficient, which is the wrong formula
for an arm reporting a gradient-norm z. Scaled by the instrument's own ceiling the arm
reports only a z, and the calibration in task 138 shows that z cannot separate a
correlation of 0 from 0.1 on an arm this size. The
higher-power version runs three episodes per rollout instead of one, taking n to about
533 and buys no sensitivity, because the cross-seed standard
error rises with the per-seed sensitivity -- and still short of a clean test, which
needs the paper's five per-super-segment
observation fields and therefore a wider deployed contract. **It has now been
measured: -0.46 +/- 0.66 over three seeds (+0.86, -1.16, -1.09) at 532 to 533
segment-decisions**, and it buys no sensitivity, because the cross-seed standard error
rises along with the per-seed sensitivity. Tasks 129, 134 and 136.

## A property of the operating point that bounds every comparison here

The network is bistable. Five seeds of identical demand on an identical, structurally
symmetric road produce two regimes -- congestion in the middles with empty leaves, or
on the leaves with moderate middles -- and served trips range from 156 to 198, a 27%
spread. That is where the paired standard deviation of 17 to 32 arrivals comes from.
It belongs to the road, not to any controller, and it caps a five-seed comparison at
effects above roughly 8%. Task 114.

## The objective responds to behaviour; only its speed half does

The question prior to every gradient measurement is whether the objective can be moved
by behaviour at all. If it were flat, no learner could exploit it however well credit
were assigned, and the whole investigation would have been measuring the attribution
of a quantity with nothing to attribute. It is not flat.

Each AV independently issues the config's own `slow` command, 8.33 m/s, with
probability p at each decision and is otherwise released to SUMO's car-following.
Five seeds, each fixing both the traffic and the metering draw sequence, so the same
seed at two mixes differs only in the mix. Differences are from p=0 with
two-standard-error bars.

| p | d reward | d speed term | d penalty term | d arrivals |
|---|---|---|---|---|
| 0.25 | **-1.51** +/- 0.54 | **-1.24** +/- 0.72 | **-0.28** +/- 0.19 | **-25.4** +/- 8.7 |
| 0.5 | **-1.29** +/- 0.73 | **-1.47** +/- 0.82 | +0.18 +/- 0.37 | **-36.8** +/- 23.0 |
| 0.75 | **-1.70** +/- 0.97 | **-1.74** +/- 0.93 | +0.04 +/- 0.35 | **-32.0** +/- 24.7 |
| 1 | **-1.86** +/- 0.71 | **-1.72** +/- 0.83 | -0.14 +/- 0.50 | -27.8 +/- 32.8 |

Bold means the bar excludes zero. The measurement reproduces two independent
baselines before any comparison is drawn: its p=0 arrivals match the fleet-mix sweep's
reference seed for seed, 198, 194, 156, 179 and 196, and its p=0 reward over three
seeds matches the unpaired pass exactly.

**The objective resolves the behaviour change at every mix level, and at least as well
as counting arrivals does.** The ratio of each difference to its own bar is 2.79,
1.76, 1.75 and 2.63 for the reward against 2.91, 1.60, 1.29 and 0.85 for arrivals; at
p=1 the reward resolves the change and the arrival count does not. So no part of the
null is attributable to the objective being unmovable or noisy. It stays with the
mechanism and the credit assignment.

**Only one of the reward's two halves responds.** `congestion_penalty` is 1.0, so the
penalty term is exactly minus the count of segments over rho\* and the speed term is
the remainder. The speed term is resolved and negative at every level and is
essentially the whole difference. The penalty term resolves at one level of four and
changes sign across them. **The half of the objective designed to pay for keeping a
link below critical density -- the anticipatory behaviour the mechanism is supposed to
produce -- does not respond consistently to this behaviour, and the half that does is
the half that restates mean segment speed.**

**It tracks throughput in the mean and not per application.** Across the twenty
individual (mix, seed) treatment applications, the correlation between the change in
the objective and the change in arrivals is -0.035, with the speed term at -0.182 and
the penalty term at +0.318. The objective detects that metering happened; it does not
measure how much throughput it cost. At n=20 the two-standard-error bar on a
correlation is 0.485, so this excludes a tight relationship and not a moderate one.
The pooled correlation over all 25 cells is +0.582, but that mixes the shared
treatment trend with seed bistability and is the wrong statistic: two quantities that
both fall on average correlate positively whether or not their fluctuations are
related.

**Do not read this p against the fleet-mix sweep's p.** Both sweeps draw a Bernoulli
at each decision, at the same 1 s interval, on the same 6000-step episodes after the
same 3000-step fill -- their p=0 arms are identical, 198, 194, 156, 179 and 196 seed
for seed. What differs is the command. This one sets 8.33 m/s, a CEILING that SUMO's
car-following often already dominates: the network mean speed here is 1.9 to 2.6 m/s,
so the command frequently changes nothing. The fleet-mix sweep's `binding` scheme sets
0.6 times the vehicle's CURRENT speed, which always binds and compounds across
decisions. That is why the same nominal p costs 25 to 37 arrivals here and 56 to 147
there. The two measure different interventions and their p columns are not comparable.

**Two further limits.** The objective is maximised at p=0, by not metering at all, so within
this cut a policy that has learned nothing and a policy that has learned to leave the
fleet alone produce identical behaviour; improvement on this axis cannot demonstrate
learning. And the cut is one dimension, a uniform random command at a single speed
value, so it bounds what an unselective fleet can do to the objective rather than what
a selective policy could. The metering oracle covers the selective case with perfect
information and gains nothing.

## The explanations, and which survive

Every candidate for why the advantage is silent about the action, with what closed it.
The point of the table is that the eliminations are measurements rather than
arguments. Every row now has one. The last row's is still the weakest, but by less
than earlier text here claimed, and weaker than the row itself says: that arm reports
only a z, and a calibrated z cannot separate a correlation of 0 from 0.1 on an arm its
size (task 138). The other rows rest on directly measured correlations.

| explanation | status | what closed it |
|---|---|---|
| the reward does not respond to an individual action, because it is shared | **ruled out** | an unshared per-agent reward is equally silent: correlation +0.0016 against -0.0042 shared, standard error 0.0135 |
| the control resolution is too fine | **ruled out** | 60 s per decision reads -0.47 against 1 s at +0.00 |
| the speed bins do not bind | **ruled out** | 8.33 m/s binds on 21.4% of AV-steps against 10.0%, and changes nothing |
| the reward shape is wrong -- eleven weighted terms | **ruled out** | the paper's own two-term threshold reward gives the same result |
| the actor cannot see what the reward pays for | **ruled out** | it sees the exact density ratio of the link ahead, spread 0.282 across agents, agreeing with truth on 100% of observations |
| the critic cannot centre the advantage | **ruled out** | privileged neighbourhood features raise out-of-sample R2 from 0.808 to 0.886 and leave the gradient unchanged |
| penetration is too low | **ruled out** | 100% penetration, commanding the entire fleet, reads the same |
| the operating point has no headroom | **partly** | the metering oracle gains nothing, but it is one heuristic and a lower bound |
| **the agent is a VEHICLE that makes 7.1 decisions and leaves, where the paper's is a ROAD that persists** | **null, at a sensitivity too coarse to call it closed** | a segment paid its own reward reads -0.12 +/- 0.51 at 171 segment-decisions and -0.46 +/- 0.66 at 533, but these arms report only a z and the calibrated z cannot separate a correlation of 0 from 0.1 on an arm this size (task 138) |

**Every candidate now has a measurement against it, and the last one has two.** The
road-attached agent paid its own reward reads -0.12 +/- 0.51 at 171 segment-decisions
and -0.46 +/- 0.66 at 533. Both are centred on zero.

**How sensitive those arms are, corrected.** Earlier text here put their thresholds at
0.15 and 0.09, taken from 2/sqrt(n), the sampling error of a correlation coefficient.
That is the wrong formula: these arms report a gradient-norm z, not a correlation. The
instrument carries its own scale instead. Its ceiling advantage is an affine function
of the indicator of `slow`, so it has a correlation of exactly 1.0 with the action,
and the z it produces is what a correlation of 1.0 looks like on that batch: 20.8,
22.7 and 20.4 at the lower power and 30.6, 44.8 and 3.3 at the higher. Against the
observed cross-seed standard errors, a correlation of **0.048** and **0.050**
respectively would show at two standard errors.

So the arms are three times more sensitive than the retracted figures said, and **the
extra episodes bought no sensitivity**: per-seed sensitivity rose and the cross-seed
standard error rose with it, from 0.52 to 0.66. The third seed of the higher-power arm
is also badly conditioned, reading 3.3 where the others read 30.6 and 44.8, a
tenfold spread in instrument sensitivity across seeds of the same arm.

**THAT EXTRAPOLATION WAS LINEAR, THE CURVE IS NOT, AND THE CALIBRATED ANSWER IS THAT
THE z CANNOT DO THIS JOB.** Advantages were synthesised at known correlations, every
target reproduced to four decimals, with the noise component taken from the REAL
advantage so the curve passes through its own null:

| c | 0.00 | 0.02 | 0.05 | 0.10 | 0.20 | 0.35 | 0.50 | 1.00 |
|---|---|---|---|---|---|---|---|---|
| z | +0.24 | -0.03 | -0.25 | +0.03 | +1.90 | +5.35 | +8.97 | +22.02 |

z crosses 2 at c = 0.204, where linear extrapolation from the ceiling says 0.091. The
crossing is not the main point. Below c of about 0.2 the readings are non-monotone and
every one sits inside the null's own scatter, so **on an arm this size the gradient
norm cannot separate a correlation of 0 from one of 0.1.**

**This weakens the segment arms and not the vehicle arms.** The segment arms report
only a z, so their bound is of order 0.1 to 0.2 -- earlier text here said 0.15, then
0.05, then 0.07, each time correcting the conversion rather than the data. The vehicle
arms report `corr(slow, A)` directly, at two standard errors of 0.035, and a measured
correlation needs no conversion or calibration at all. The segment arm is being re-run
to report its correlation the same way. Tasks 137 and 138.

The conversion is load-bearing enough to be a tested instrument rather than a one-off:
`scripts/measure_alignment_calibration.py`, with `tests/test_alignment_calibration.py`
covering the construction it rests on -- that the synthesised advantage carries the
correlation it claims, that a noise vector already correlated with the action does not
leak into the target, and that an action column with no variance raises rather than
returning a zero correlation that would read as a measured null. Verified against a
control: removing the orthogonalisation fails seven of the eleven.

What remains untested is everything below about 0.1 on these arms, and a
clean test of the paper's own formulation rather than this approximation of it, which
needs its five per-super-segment observation fields and therefore widens the deployed
contract that the Jetson builder and the parity ledger both depend on.

## THE PORTED RUN, STOPPED AT UPDATE 13 OF 20, AND ITS GATE

Ankit stopped the run at update 13 on 2026-09-10. The gate was pre-registered at 20
updates, so criteria 1 and 2 are read on a SHORTER run than they were written for.
That cuts both ways and is recorded rather than argued: fewer updates give a trend less
chance to appear, and the trend that did appear points the wrong way.

| criterion | threshold | reading at update 13 | |
|---|---|---|---|
| 1. summed entropy falls | below 1.9775, of a 2.1972 maximum | lowest reached **2.1459** | fail |
| 2. score trends up | by more than the step-to-step variation | moved **-0.4679** against a step standard deviation of 0.2625 | fail |
| 3. action distribution leaves uniform | joint modal share above 0.20 | **0.1460** | fail |

**All three fail.** Criterion 3 is measured on the update-13 actor over 1,503
observations in the environment it trained in. Both heads sit near uniform: the speed
head at 0.3176 / 0.2994 / 0.3829 and the headway head at 0.2659 / 0.3529 / 0.3812,
summed entropy 2.1804 of a 2.1972 maximum. For scale, a randomly initialised actor
reads a joint modal share of 0.1378 and the earlier `mappo_sumo` run reached 0.1486
after 23 updates, so 0.1460 after 13 is initialisation drift rather than a learned
preference.

**The early stop does not rescue criterion 2, and the amount by which it does not is
computable.** For the criterion to pass at update 20, updates 14 to 20 would have had
to hold a score of -6.03. That is better than every update of the run except the
first, against a whole-run range of -6.811 to -5.777.

**Criterion 2 fails in the wrong direction, which is the part worth noting.** The
score did not stay flat; it declined, and by more than the update-to-update noise. The
curve: -5.777, -6.567, -6.446, -6.463, -6.462, -6.121, -6.353, -6.302, -6.304, -6.318,
-6.605, -6.731, -6.811.

**Entropy never approached its gate.** It fell 0.0439 over thirteen updates, from
2.1898 to 2.1459, against the 0.2197 fall the gate asks for, and it rose on four of
the twelve steps. An earlier claim here that the decline was monotone was retired
after update 5 reversed it, and updates 7 and 10 reversed it again.

## What was and was not built, stated exactly

Asked directly whether this implements super-segment control or only a coarse lever,
the answer is that the lever is coarse in time everywhere, coarse in space only in
arms that were never trained, and the STATE the controller acts on is a single
vehicle's kinematics throughout. Precisely:

| the paper | what was built | where |
|---|---|---|
| one decision per 60 s | one decision per 60 s | `mappo_src`, `decision_interval_s: 60.0` |
| one action per super-segment, applied to all vehicles on it | same, in the measurement arms only | `measure_segment_as_agent.py`, `segown_highpower.py` -- gradient measurements, never trained |
| | one action PER VEHICLE in the trained run | `mappo_src` is vehicle-attached; each AV samples from its own observation |
| super-segments of 2 to 3 km | links of 500 m (leaves), 600 m (middles and trunk), 300 m (bottleneck), nine of them | `configs/topology/inverted_tree.yaml` |
| state is per-super-segment density, speed, gap, inflow, outflow | state is one representative vehicle's 33-field local vector: `ego_speed`, `ego_headway_s`, `leader_gap`, `follower_gap`, lane gaps and so on | `LOCAL_OBS_FIELDS` in `src/rl/encoders.py` |

Of those 33 fields about five are link-level -- `downstream_congestion_estimate`,
`segment_target_speed`, `local_density_bin`, `local_mean_speed_bin`,
`local_queue_estimate`. The rest are per-vehicle kinematics.

**The paper's five fields already exist in this codebase.** `SEGMENT_FIELDS` carries
`density`, `mean_speed`, `queue_length`, `inflow` and `outflow` per segment, and the
CENTRALIZED CRITIC already consumes them. They are not routed to the actor. That is a
deliberate boundary rather than an oversight: the actor's input width is
`local_obs_dim()`, and the same width is pinned in the deployed contract that
`src/analysis/observation_parity.py` and the Jetson builder both check against. The
privileged-critic work was built specifically to add features without touching it.

**So the honest label for what was measured is "super-segment ACTION driven by a
single vehicle's OBSERVATION, on 500-to-600 m links".** That is not the paper's
controller, and no result here bounds the paper's controller. Widening the actor's
observation to the five per-segment fields is the one untested direction whose cost is
known: it breaks the deployed observation contract on both the Jetson and the phone.

## What this leaves

Four directions, of which the second has changed since it was first written here,
because one of its two halves has since been measured.

1. **Report the simulation leg as a null**, with the deployment carrying the
   feasibility claim as it already does. This is my recommendation.
2. **Change what the AVs can do** rather than how their reward is priced. Of the two
   candidates originally listed under this heading, the explicit junction meter is now
   measured and is not a direction: a perfect-information metering oracle serves 0.8
   fewer and 0.8 more vehicles than no control on the two demands, against a
   seed-to-seed spread of about 16 arrivals. Platoon coordination is untested and is
   what remains of this option.
3. **Test the paper's own formulation of the road-attached agent.** Raising the
   episode count is NOT the way: three episodes per rollout left the threshold where
   one episode did, because the cross-seed standard error rose along with
   the per-seed sensitivity (task 136). What is untested is the formulation itself.
   This arm draws its action from one representative vehicle's observation; the paper
   uses five per-super-segment fields, and adding them widens the deployed contract
   that the Jetson builder and the parity ledger depend on.
4. **Change the advantage estimator** to a counterfactual one. It is aimed exactly at
   the quantity that reads null and would probably raise the correlation.
5. **Move the congestion threshold, or establish that moving it does not help.** This
   is new, and it is the cheapest of the five: one sweep, not several days. The
   objective's anticipatory half does not respond to behaviour, and rho* = 0.3 of jam
   density fires on only 1.50 of 9 segments, so it is nearly inactive on this road.
   Either the threshold is misplaced and a placement exists that makes the term
   respond, or nothing the lever does moves any threshold. Those call for different
   work and the measurement separates them.

**On the order.** 5 comes first on cost alone: it is one sweep and it can retire an
option rather than open one. After that, an estimator that learns better is worth
building only after something shows headroom to reach, and oracles are the cheap way
to look for headroom, so 4 is last. For the same reason 3 comes before 4: it asks whether the mechanism is learnable
at the granularity the paper used, before spending on a better learner at the
granularity this project uses.

**The decision is Ankit's and is open.** Nothing further should be built until he
answers, because 2, 3 and 4 are each several days and they are alternatives rather
than a sequence. 5 is running now, since it costs one sweep and its answer bears on
whether the others are worth starting.
