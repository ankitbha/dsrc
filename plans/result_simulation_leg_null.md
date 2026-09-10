# The simulation leg: what was measured, and why it is a null

One account of a result currently spread across task-list entries 92 to 109. It
states what was measured, what each measurement was controlled against, and exactly
what is and is not established.

## Short version

**The mechanism this project posits -- in-stream AV speed modulation -- neither gains
when handed perfect information nor presents a learnable gradient, on this road, at
every operating point and penetration tried.** Two instruments built for different
purposes agree:

- a perfect-information metering oracle serves **-0.8 and +0.8** more vehicles than
  no control, against a spread of about 16;
- the correlation between one AV's action and its own advantage is **about 1%**,
  where the same instrument reads 14 to 80 times its noise floor when a correlated
  advantage is supplied.

**What is NOT established.** The oracle is one hand-written heuristic given perfect
state, so it is a lower bound on what the best controller could do and not an upper
bound. "No controller can gain on this network" is not proven and cannot be proven
this way.

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
null of 0.11, expected z is about 60 times the true action-advantage correlation, so
the arm resolves about 0.03 and bounds the correlation within +/- 0.017 of zero. This
is a null from a sensitive instrument, not an insensitive one.

So holding one action for a simulated minute -- with the paper's threshold reward, its
speed bins, its discount horizon, and a link-level congestion signal the actor can
see -- produces no more action-attributable signal than holding it for a second.

**One confound this cannot resolve.** The lever change carries a cost: at 60 s each
agent makes 7.1 decisions in its lifetime instead of 90.5, a thirteenfold loss of
temporal structure, because the agent is a vehicle that leaves rather than a road that
persists. A larger per-decision effect and a much shorter trajectory pull opposite
ways and this measurement cannot separate them. The segment-as-agent arms, where the
agent persists for the whole episode, are what can.

`configs/training/mappo_src.yaml` ports the formulation. What it
changes and what was measured about each is task 113; the honest accounting of which
change is large is:

| change | size, measured |
|---|---|
| threshold reward | LARGE: fires on about 7 of 9 segments, range about -7 to +1.3, against an eleven-term reward that sat near zero |
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
The first was measured only against the permutation floor that has since been shown
invalid -- its reading was one of the four results retracted -- and the second had not
been run. Both are now queued against the valid null. They are the arms that decide
whether an unshared reward produces signal where a shared one cannot.

It also explains why the shared-reward segment arm reads consistently negative:
with a common reward its advantage is essentially a function of the untrained critic,
and both the advantage and the action are then functions of the same observation
through independently initialised networks. Task 123.

## A property of the operating point that bounds every comparison here

The network is bistable. Five seeds of identical demand on an identical, structurally
symmetric road produce two regimes -- congestion in the middles with empty leaves, or
on the leaves with moderate middles -- and served trips range from 156 to 198, a 27%
spread. That is where the paired standard deviation of 17 to 32 arrivals comes from.
It belongs to the road, not to any controller, and it caps a five-seed comparison at
effects above roughly 8%. Task 114.

## What this leaves

1. **Report the simulation leg as a null**, with the deployment carrying the
   feasibility claim as it already does.
2. **Change what the AVs can do** rather than how their reward is priced: platoon
   coordination, or an explicit meter at the junction rather than in-stream
   vehicles. Penetration is measured and does not do it.
3. **Change the advantage estimator** to a counterfactual one, which is precisely
   targeted at the measurement above and would probably raise the correlation -- and
   would fix the learning of a mechanism that gains nothing when handed perfect
   information.

The order matters: an estimator that learns better is worth building after an oracle
shows headroom to reach, and oracles are the cheap way to look for headroom.
