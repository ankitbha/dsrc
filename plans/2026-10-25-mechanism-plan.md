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

## Block N2 result (2026-10-25, 12:20)

240 episodes at `9f2ffb6`, seeds 11-15, 0 collisions, 0 collision teleports, 0 writes above
the plan. No control and plan alone equal block N's records on every seed. Floor with no
trigger: 3,406.5 veh/h (plan alone 3,442.1 minus 1% of no control's 3,561.1).

| Candidate | C | BG flow (veh/h) | Meets floor |
|---|---|---|---|
| m0_r | -0.1353 | 3,299.7 | no |
| m0 | -0.1334 | 3,308.3 | no |
| m0_g0.5_c1 | -0.1279 | 3,365.6 | no |
| m0_r_g0.5_c0.8 | -0.1263 | 3,317.1 | no |
| m0_r_g0.5_c1 | -0.1255 | 3,371.7 | no |
| m0_g0.5 | -0.1132 | 3,432.8 | yes |
| m0_g0.5_c0.8 | -0.1113 | 3,316.5 | no |
| m0_g1 | -0.1102 | 3,436.8 | yes |
| m0_r_g0.5 | -0.1093 | 3,429.3 | yes |
| m0_g1.5 | -0.0923 | 3,438.1 | yes |
| m0_g2 | -0.0775 | 3,442.9 | yes |

**Selected: `m0_g1`, the feed gate at 1.0, without the queue release or the camera gate.**
`m0_g0.5`, `m0_g1` and `m0_r_g0.5` are within C_TIE of the lowest eligible C; the gate at 1.0
serves the most with no trigger.

- The camera gate fails the floor at both kappa: with no trigger, the camera's occupancy
  reaches 0.8 or 1.0 k_c often enough to reopen most of the correction's cost (the shown speed
  changed on 7.3-14.7% of BG vehicle-ticks, against 0.4-0.6% for the feed gates).
- The queue release changes C by less than C_TIE at full penetration (-0.1353 against -0.1334
  without a gate, -0.1093 against -0.1132 with the gate at 0.5). It was built for partial
  compliance, which the selection does not test, so block P also runs the selected system with
  the release added (`--release-candidate`, recorded as not selected) at 0.5 and 0.75.
- Disclosure, as in section 2: the gate at 0.5's evaluation-seed results were seen before the
  floor was written; the gate at 1.0 had not been run on evaluation seeds.

## Evaluation result (2026-10-25, 15:12)

975 episodes at `7378d16` from a clean worktree, merged into `results/jam_network/` (M, L, P,
K appended; N2 and D new; sweep dirs appended). Every committed record file is an unchanged
prefix of its replacement. Over all 3,000 controlled evaluation runs: 0 collisions, 0 collision
teleports, 0 writes above the plan, 0 speed-cap mismatches, no approach-short flag; block D's
26.8 million driver caps read back with 0 mismatches. Run first at 9 in parallel, then paused
and resumed at 6 after Ankit asked not to overload the machine (CPU was 99% busy); a watchdog
logged memory at about 50% free throughout and never paused. Figures regenerate byte-identical
from `results/jam_network` with `--mechanism-selection results/jam_network/N2_summary.json`.

Ablation (five triggers, 75 runs; `results/jam_network/ablation.json`):

| Arm | Jam-free | Jam | Throughput | Travel time | No trigger |
|---|---|---|---|---|---|
| No control | 1 | 130.8 | 3,456 veh/h | 1,622 veh-h | 3,546 veh/h |
| Feed lookahead | 46 | 11.6 | -5.4% | +11.9% | -20.8% |
| Plan only | 39 | 22.1 | +2.6% | -10.0% | -2.9% |
| Camera only (`m0_local`) | 5 | 95.5 | +1.3% | -1.1% | -0.9% |
| No gate (`m0`) | 51 | 4.4 | +2.5% | -6.0% | -6.9% |
| **System (`m0_g1`)** | **56** | **3.9** | **+3.9%** | **-12.6%** | **-3.0%** |
| True occupancy (`tau10_m0_R60_g1`) | 58 | 3.4 | +4.3% | -14.3% | -3.0% |

- Prevented 55 of the 74 runs that jam under no control; exit block 137.1 -> 11.1 (no gate 8.1;
  gated minus ungated +3.0 +- 0.9). Against the lookahead -7.8 +- 1.3 (less after every trigger).
- 50% following, B9: +13.0 +- 15.9 over no control (ungated +29.5 +- 24.6), +32.6 +- 17.2 over
  plan alone. Release candidate `m0_r_g1` at 0.5: +13.1 +- 21.3 against the system on B9,
  -10.0 +- 24.4 on I_exit; at 0.75 +13.2 +- 21.2 and +0.1 +- 2.7. The release does not reduce the
  50% deficit.
- Driver response (block D, `driver_table.json`): B9 system 3.7 (exact) / 11.1 (nominal) / 46.7
  (lax), I_exit 11.1 / 12.3 / 13.7; below no control on every seed in every setting, below plan
  alone under the same drivers.
- Camera scale 0.8 / 1.25: within 1.8 segment-minutes of scale 1, below the lookahead on both
  triggers (the ungated rule was +25.0 +- 15.9 above it at 0.8 on B9).
- Feed delay: on B9 at D = 189 and 368 s the system is within noise of the lookahead
  (+2.9 +- 10.8, -0.8 +- 5.3) and above the ungated rule (14.5 and 17.5 against 4.9 and 6.8): the
  gate waits for the feed. On I_exit it is below the lookahead at every delay.
