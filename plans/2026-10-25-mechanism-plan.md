# Queue release, camera-opened gate, camera-only arm and driver response (2026-10-25)

## Request

Ankit, 2026-10-25, after the advisor's RA-L review. From the review's Major Concern 5: treat the
cost of intervening when no jam would form, and the extra jam at partial compliance, as a
design problem; add a mechanism that limits unnecessary interventions and stops slowing once a
queue has formed; select it on seeds 11-15; re-run every number. From Major Concern 2: report
the full ablation, including a camera-only arm (C1). From Major Concern 1: model driver
response (C2). Title and abstract follow the advisor's text. Points 1-3 of the review (approach
roads, background on the earlier paper, second network / driver study / ranging) get a rebuttal
file instead of changes. dsrc validation stays stopped (2026-10-18): no validator round.

## 1. The mechanism

Both parts are the two items the paper's conclusion already proposed as future work.

**R, queue release.** "Switch the correction off while the camera sees the road ahead
discharging." A vehicle's road ahead is discharging when its occupancy estimate is at or above
the critical occupancy and its 30 s trend is negative: `k_hat >= k_c` and `s_hat < 0`. While
that holds the rule cannot switch the vehicle on, and a vehicle already on is released through
the existing hysteresis (at least 30 s on, two release ticks). No new parameter.

**Camera-opened gate.** "Let the camera's density estimate open the feed gate instead of
waiting for the feed." With a feed gate `gate_J`, a vehicle's gate is open when its applied
feed version has `gate_signal >= gate_J` OR its own `k_hat >= kappa * k_c`. One parameter,
`kappa`. A camera gate without a feed gate is refused (there is nothing for it to open).

`gate_J` stays at 0.5, block N's best gated candidate; it is not re-selected.

## 2. Selection (block N2), stated before any N2 episode runs

Seeds 11-15, approach boundary, 3,600 s, scenarios B5, B7.5, B9, I_exit, I_mid, BG, `m_on` 0.

New candidates (all the rule at `m_on` 0):

| Label | R | feed gate | kappa |
|---|---|---|---|
| m0_r | yes | none | none |
| m0_r_g0.5 | yes | 0.5 | none |
| m0_r_g0.5_c1 | yes | 0.5 | 1.0 |
| m0_r_g0.5_c0.8 | yes | 0.5 | 0.8 |
| m0_g0.5_c1 | no | 0.5 | 1.0 |
| m0_g0.5_c0.8 | no | 0.5 | 0.8 |

Block N2 also re-runs no control and plan alone on the same seeds. Its summary refuses to select
unless their jam time and flow equal block N's records seed by seed; that is the check that the
code change left the existing arms unchanged. Block N's five candidates (`m0`, `m0_g0.5`,
`m0_g1`, `m0_g1.5`, `m0_g2`) join the pool from `N.jsonl`.

**Rule.** A candidate with any collision, collision teleport or write above the plan is
excluded. A candidate is eligible only if, with no trigger (BG), it serves at least plan
alone's flow minus 1% of no control's flow: the correction may cost at most 1% of throughput
beyond the plan's own cost when no jam would form. Among eligible candidates the lowest
criterion C (jam plan section 5.7) wins; candidates within C_TIE = 0.02 of it are ordered by BG
served flow, highest first. If no candidate is eligible, nothing is selected and the result
goes to Ankit.

**Disclosure.** This rule was written after block N and after the evaluation-seed runs of the
gate at 0.5 had been seen (Section VI-F of the earlier draft). On block N's selection-seed
numbers the ungated rule (`m0`) fails the constraint (BG 3,308.3 against plan 3,442.1 minus
35.6) and the gate at 0.5 meets it (3,432.8).

## 3. C1, the camera layer alone

The selected correction (its R setting) on a plan of 1.0 everywhere: no feed plan and no feed
gate, so the camera is the only input. `JamArm.no_plan`. Variant suffix `_local`. Run in block
M on its 7 scenarios, seeds 16-30.

## 4. C2, driver response

A driver turns the shown advisory into the vehicle's speed cap. Per controlled vehicle, drawn
from `default_rng((seed + 7000, crc32(vehicle id)))`, so a driver's traits do not depend on
the arm or on the order vehicles appear, and arms stay paired by seed:

- reaction time `r ~ lognormal(median, sigma_log)`: a change of the shown speed reaches the cap
  `r` seconds later (on the 1 s tick); a newer change replaces a pending one;
- compliance `e ~ Normal(mean, sd)`, clipped to [-0.1, 0.5]: the driver aims at
  `shown x v_limit x (1 + e)`;
- deceleration: a lower cap is approached at `a` m/s^2 (the cap falls by `a` per second);
  a higher cap applies at once, and car-following limits the acceleration and still brakes
  for the leader.

| Setting | median r (s) | sigma_log | mean e | sd e | a (m/s^2) |
|---|---|---|---|---|---|
| nominal | 2.0 | 0.4 | +0.05 | 0.05 | 1.5 |
| lax | 4.0 | 0.5 | +0.15 | 0.10 | 1.0 |

Provenance: tier 3, assumed. The deployment recorded no driver acting on an advisory, and no
field measurement of response to an in-vehicle speed advisory was found. The medians bracket
the 2.5 s perception-reaction time used in highway design; 1.5 m/s^2 is the deceleration the
local layer's horizon `tau_l` already assumes.

Block D: plan alone and the selected system, B9 and I_exit, seeds 16-30, 7,200 s, both
settings. 120 episodes.

## 5. Evaluation matrix (seeds 16-30, approach boundary)

| Block | What runs | Episodes |
|---|---|---|
| M | selected system, its oracle (true occupancy, same mechanism), C1 | up to 315 |
| L | selected system at lag 0, 2, 4 | 90 |
| K | selected system at camera bias 0.8, 1.25 and pair spread 0.5 | 90 |
| P | selected system at 0.5, 0.75 | 60 |
| sweep | selected system at 0.55-0.95 | 240 |
| D | section 4 | 120 |

Arms already in the committed records are resumed, not re-run. If block N2 selects one of
block N's candidates, its existing M and P records stand.

## 6. Decisions taken without asking (for Ankit)

1. R and the camera gate are the conclusion's two future-work items, defined as above.
2. The no-trigger constraint (1% beyond plan alone) is the selection's new rule.
3. `gate_J` fixed at 0.5; `m_on` stays 0; `tau_l` stays 10 s.
4. C1 has no gate; C2's parameters are assumed (tier 3) and run at two settings.
