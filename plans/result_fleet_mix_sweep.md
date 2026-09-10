# Does the network outcome depend on the fleet-wide action mix?

Ankit's objection, and it was the right one: a shared policy can represent the
centralized behaviour, so if a centralized controller helps, the behaviour is in the
search space and the problem is the search rather than the space. Everything measured
before this was about a single agent's single decision. This measures the fleet.

## Why the policy cannot explore this dimension, measured

Each agent samples independently from the shared policy, so the fleet-wide mix
concentrates. Measured over five rollouts at 39 agents:

- per-step fraction choosing `slow`: standard deviation **0.0714**
- rollout-mean mix across five seeds: **0.3073 +/- 0.0082**

**The gradient only ever sees fleet configurations between about 0.291 and 0.324**, a
three-point window, against the 0 to 1 this sweep covers. Each agent's share of a
fleet-level signal is `dJ/dp / N`, which at N = 39 is what sits below the gradient
noise floor. The estimator accounts for parameter sharing correctly -- the gradient is
summed over all agents with respect to the shared parameters -- and it still cannot
explore the fleet dimension, because independent sampling concentrates it.

## The design

At every decision each AV independently issues a command with probability `p` and
otherwise RELEASES to SUMO's car-following, so `p = 0` is exactly the uncommanded
fleet and `p = 1` is every AV commanded at every decision. Without the release a
vehicle commanded once stays commanded and `p` would mean "fraction ever commanded".
Penetration is fixed in every arm, so every arm at a seed sees identical traffic and
they differ only in what is commanded. `sumo_capacity_drop`, 600 s episodes after a
300 s warm-up, dt 0.1, decisions at 1 Hz, five seeds, paired.

## Result: speed metering, and there is no positive region

Reference 184.6 arrivals. The `+/-` is one standard error of the paired difference.

| command | p | arrivals | paired vs p=0 | verdict |
|---|---|---|---|---|
| `shipped`: max(12, allowed - 10), binds on 6% of decisions | 0.25 | 176.2 | -8.4 +/- 6.2 | no effect |
| | 0.50 | 185.2 | +0.6 +/- 14.2 | no effect |
| | 0.75 | 171.4 | -13.2 +/- 12.5 | no effect |
| | 1.00 | 169.2 | -15.4 +/- 14.0 | no effect |
| `binding`: 0.6x the vehicle's own speed, binds on 29% | 0.25 | 128.2 | **-56.4 +/- 15.1** | real |
| | 0.50 | 66.6 | **-118.0 +/- 12.7** | real |
| | 0.75 | 37.8 | **-146.8 +/- 9.6** | real |
| | 1.00 | 45.8 | **-138.8 +/- 11.8** | real |

**J(p) is monotone decreasing.** The command that barely bites is flat within noise
with a downward trend; the command that bites destroys throughput in proportion to how
often it is issued. The two definitions are both present because a flat result under
a command that does nothing cannot be told from a flat objective.

**So this is not "findable but not by PPO".** A sweep across the whole fleet-mix
range, which PPO structurally cannot reach, finds no speed-metering configuration that
beats doing nothing.

It also settles the bin-rescale question retracted earlier from the other direction:
bins that bind make things worse, not better.

## What the actions CAN do, since "the actions do nothing" was too strong

Both heads reach the world; the earlier phrasing was wrong.

| head | effect | measured |
|---|---|---|
| speed | acts, but its three values are equivalent on 97% of decisions at this operating point | mean AV speed 7.33 m/s against bins of 20, 27 and 30 |
| headway | acts at any speed. `setTau` reaches W99 although W99 follows on `cc1`, which was not obvious | mean AV gap 78.05 m -> 83.51 m, arrivals 42 -> 40, paired on one seed |

## Still running

`headway` (tau 3.0) and `harmonize` (command the segment's prevailing speed, so a fast
vehicle slows and a slow one speeds up) across the same mix range. Harmonization is
the mechanism the ring-road results damp stop-and-go waves with, and no arm above
tests it: every one of them only ever slows a vehicle.

A first look at harmonization on one seed is neutral on throughput -- 42, 40, 42
arrivals at p = 0, 0.5, 1.0 -- and RAISES jerk from 0.390 to 1.041, because the
command is a hard `setSpeed` to the segment mean once a second and therefore steps
rather than smooths. That is a limitation of this implementation of harmonization, not
a measurement of harmonization.

## Two defects in this instrument, both caught before they reached a result

- **An unrecognised scheme name fell through to the binding command.** `harmonize`
  ran as `binding` under a new label and produced three numbers that looked like a
  measurement of something never executed. An unknown scheme now raises. The guard was
  then run against a control and the first control was inconclusive -- with a 60-step
  warm-up no agent exists, so the command branch is never reached and nothing raises.
- **`harmonize` was missing from `SCHEMES`**, so the queued batch would have skipped
  it silently and the sweep would have reported three schemes where four were
  intended.
