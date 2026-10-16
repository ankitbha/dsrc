# Feed agreement: the plan for the dsrc paper's algorithmic component

Written 2026-10-15 by the planning agent, on branch `feed-agreement`, against commit
`2c01189`. Every design decision below was taken by recommendation without asking; the
decisions table in the short version records each one, and the open items list the
decisions where the planner had no real preference or where the code contradicted the
brief. Nothing in this plan has been implemented or run, except the one timed smoke
episode and the read-only analyses recorded in the appendix.

## Short version

### Summary

The paper needs an algorithmic component beyond SRC. The component is a rule by which
vehicles that read a public traffic feed at different times act on the same feed
snapshot, so that a fleet of vehicles running one deterministic policy coordinates
without a coordinator. This plan specifies that rule (Algorithm 2, the version-keyed
rule), the lemma that bounds the fraction of vehicles acting on a different snapshot, a
polling experiment that measures the feed's update process, the SUMO changes that
simulate per-vehicle snapshots on the Mainz port, the drain-time metric, the
minimum-penetration certificate, the experiment matrix, the figures, the validator's
attack points and the risks.

The grounding facts, all measured on 2026-10-15 and recorded in the appendix:

- The drive logs do carry a wall-clock anchor (`metadata.jsonl` has `t_wall` and `t_mono`
  on every line), so the publication-to-receipt lag can be computed from the drives. Over
  the 60 versions in 123 responses, the first-receipt lag (receipt time of the first
  response carrying a version, minus that version's `sourceUpdated`) has a minimum of
  33.8 s, a median of 62.8 s and a maximum of 109.7 s. The drives queried every 16.7 s or
  every 60 s, so the first-receipt lag exceeds the availability lag by up to one query
  interval. On consecutive versions the bracketing intervals do not overlap, so the
  availability lag varied by at least 10 s between consecutive versions on 2026-09-08.
- The feed publishes on a 60 s cycle: 48 of 59 gaps are 60 s, 4 are 26 s, 4 are 34 s and 1
  is 120 s; the second-of-minute of `sourceUpdated` is :36 on 38 versions and :02 on 22.
- One controlled 2,500 s episode (the `here` arm, seed 16, 0.5 s step) takes 6.0 s of wall
  time with libsumo and reproduces the committed per-seed flow, 3,751.11 veh/h, exactly.
  The whole experiment matrix below is about 1,500 episodes of 2,500 s and 62 episodes of
  9,000 to 12,000 s, which is about 5 hours of single-process compute.
- The HERE key in `phone/local.properties` is shared with Nash production
  (`plans/plan_task21_here_client.md`, item 2), and the account's monthly allowance is not
  recorded anywhere in the repository. The polling run is therefore sized at 5,760
  requests at most, with a hard cap in the script, and needs the author's confirmation
  against the account before it runs.

### Scope boundary

In scope: the rule and lemma; the polling script and its analysis; changes to
`src/sumo/mainz.py`, `scripts/build_mainz_scenario.py` and `scripts/train_mainz_src.py`'s
`SCENARIOS`; new modules for the feed process, the drain metric and the gain
certificate; a runner with checkpoint, resume and pause; figure scripts; tests.

Out of scope: retraining either checkpoint (the existing `mainz_here_best.pt` is
evaluated; `mainz_src_best.pt` is not evaluated in any new block); any change under
`paper/`; any drive; any change to the reward, the action set or the network; the
`inverted_tree` scenarios.

### Open items

Each item names a decision taken by recommendation where the planner had no strong
preference, or a place where the code or data contradicted the brief.

1. **Polling budget against a production key.** The key is shared with Nash production
   and the allowance is unknown. The plan uses 5,760 requests at most (two sessions of
   2 hours at one request per 5 s per area, two areas). The author must confirm the budget
   before the run. If the allowance is smaller, one session (2,880 requests) still gives
   120 versions per area.
2. **"Westfield NJ" does not match the logged coordinates.** The brief and
   `plans/paper_deploying_self_regulating_cars.md` call the drive area Westfield NJ. The
   logged HERE query centres span 40.33 to 40.50 N and -74.65 to -74.40 W; Westfield is
   at 40.66 N, -74.35 W. The plan polls the centroid of the run with the most responses,
   40.3525 N, -74.6239 W. The author should confirm the label before the paper names it.
3. **Lag jitter may exceed what the rule's no-skip condition tolerates.** The lemma's
   zero-disagreement case needs the query period `P_q` to satisfy
   `P_q <= g_min - J`, where `g_min` is the shortest publication gap (26 s measured) and
   `J` the lag jitter. The drives bound `J` from below at 10 s. If the polling run finds
   `J > 16 s`, the version-skip term of the lemma (part c) is nonzero at `P_q = 10 s`,
   and the paper either lowers `P_q` to 5 s or reports the skip term.
4. **Burst scenario numbers.** Background 3,650 veh/h, burst 5,000 veh/h from 900 s to
   1,500 s, cap 12,000 s, chosen from a point-queue reading of the published served
   flows. A two-episode pilot (section 5, block D) decides whether the burst breaks no
   control down and whether the controlled arm drains before the cap, with stated
   fall-backs. The planner had no strong preference between 5,000 and 6,000 veh/h.
5. **Disagreement injection draws a fresh late set at every decision.** The alternative, a
   persistent set of late vehicles, models a vehicle with a permanently bad schedule and
   is not run. Weak preference.
6. **The entry gate holds only controlled vehicles at partial penetration.** The gate
   stands in for the policy acting on the unsimulated link above each entry, so a vehicle
   that does not run the policy is not held by it. The author may prefer to gate every
   vehicle; the mechanism is one flag.
7. **Pulse size for the gain certificate.** 60 vehicles over 60 s at one site, with a
   30-vehicle linearity check on three seeds. Weak preference; the linearity check
   decides whether 60 is inside the linear range.
8. **Mainz coordinates are derived, not verified.** The net file has no projection
   (`projParameter="!"`). The coordinates come from the Vissim file's `refPointMap`
   (read as Web Mercator) and `refPointNet`; the exit link 218 falls at 49.9312 N,
   8.3091 E and the network's box at 49.908 to 49.964 N, 8.151 to 8.317 E. The
   implementer must check one response's road names against a map before the polling
   session.
9. **`build_mainz_scenario.py --av-fraction` cannot express 75%.** It controls vehicle
   `index % round(1/p) == 0`; `round(1/0.75) = 1`, so 0.75 produces 100%. The plan moves
   penetration to run time and does not use the flag; the defect is recorded, not fixed.
10. **Hold period 120 s** for the drain condition, with 60 s and 300 s as sensitivities.
    Weak preference.
11. **Drive-log lag numbers depend on the Jetson's wall clock.** The phone's wall clock
    differed from the Jetson's by 1.15 to 1.18 s on the three runs with HERE responses,
    and which one is right is unknown. The polling run from an NTP-synced laptop settles
    the lag; the drive numbers are quoted with a 1.2 s uncertainty.
12. **SUMO's pending insertions count as "outside the simulation".** The brief names only
    the gate's held count in the drain condition. In this environment a vehicle SUMO
    cannot insert waits as a pending vehicle (`--max-depart-delay -1`), which is also
    latent demand outside the simulation and exists in every arm, so the plan requires
    both counts to be zero. This is a widening of the brief's condition, recorded here.

### Decisions table

All taken by recommendation. "Real" means the planner had a reasoned preference; "weak"
means the alternative is defensible and the item is listed above.

| # | Decision | Taken | Alternatives not taken | Preference |
|---|---|---|---|---|
| D1 | Agreement rule for the paper | Version-keyed rule, `P_q = 10 s`, `D = L_bar + P_q` with `L_bar` the largest first-receipt lag in the polling artefact | Fixed-phase rule; no rule | Real |
| D2 | Disagreement measure | Instantaneous fraction of controlled vehicles not on the plurality applied version; time-averaged over the scored window; command-level fraction recorded beside it | Fraction not on the newest available version | Real |
| D3 | Reference arms for flow | The published `here` arm (snapshots at the decision instants, zero lag) and a `synchronous` arm (every vehicle switches at the availability instant) | `here` only | Real |
| D4 | Feed process in the rule arms | Replay of the polling artefact's (gap, lag) sequence from a seeded start; drive-log placeholder until the artefact exists, with its sha256 in every record | Synthetic cadence with Gaussian lag | Real |
| D5 | Disagreement injection | Fresh random subset of fraction `delta` at each decision, applying the previous decision's action vector for the whole interval | Persistent subset | Weak (item 5) |
| D6 | Staleness injection | Snapshot ring buffer at 10 s; every vehicle applies the policy on the snapshot `s` seconds old; `s` in {30, 60, 120, 180, 300} | Delay the whole decision clock | Real |
| D7 | Penetration mechanism | Run time: vehicle with schedule index `i` is controlled iff `floor((i+1)p) - floor(ip) == 1`; the route files are unchanged | Builder's `--av-fraction` (defective at 0.75) | Real |
| D8 | Gate at partial penetration | Holds only controlled vehicles; uncontrolled vehicles are inserted at their departure time | Gate every vehicle | Weak (item 6) |
| D9 | Drain condition | Every super-segment out of the jam state for the hold period AND gate-held count zero AND SUMO-pending count zero | Held count only | Real (item 12) |
| D10 | Jam threshold | `(s1 + s2)/4` from a three-piece piecewise-linear fit to the CDF of super-segment space-mean speeds pooled over the no-control arm's episodes of the scenario; sensitivities: per-segment fit, fixed 30 km/h | Density above 0.178 | Real |
| D11 | Jam sampling and hold | Jam indicator at 10 s samples; hold period 120 s; sensitivities 60 s and 300 s, computed post hoc from the same traces | Decision-interval sampling | Weak (item 10) |
| D12 | Drain scenario (a) | Builder's default demand (18,000 veh/h for 1,200 s, 6,000 vehicles), cap 9,000 s, clock starts at 1,200 s | Longer cap | Real |
| D13 | Drain scenario (b) | Background 3,650 veh/h, burst 5,000 veh/h on [900, 1,500) s, cap 12,000 s, clock starts at 1,500 s; pilot with fall-backs | 6,000 veh/h burst; 3,600 veh/h background | Weak (item 4) |
| D14 | Arms in the drain block | `no_control` and `here` | Add `src` | Real |
| D15 | Certificate state variable | Super-segment density deviation from an unperturbed run of the same seed (ground truth, exact from counts) | Speed deviation | Real |
| D16 | Certificate perturbation | 60 extra vehicles over 60 s starting at 1,200 s, inserted at the most upstream edge of the site segment on a traversing route, with suffix routes; sites: all 12 segments; linearity check with 30 vehicles on 3 seeds | Entry-link pulses only (covers 4 of 12 segments) | Weak on size (item 7) |
| D17 | Gain estimation | Per seed, least squares of 10 s density increments on the 12 current deviations over 600 s after the pulse, pooled over sites; `a_e = -coef_ee`, `b_ej = abs(coef_ej)` | Exponential decay fit of the self-response | Real |
| D18 | Confidence bounds | One-sided 95% t bounds over 15 seeds: `a_LCB = mean - t se`, `b_UCB = max(0, mean + t se)`; `rho_UCB = rho(A_LCB^-1 B_UCB)`; no certificate if any `a_LCB <= 0` | Bootstrap | Real |
| D19 | Penetration levels | {1.0, 0.75, 0.5, 0.25, 0.1}, seeds 16-30, paired | Add 0.9 | Real |
| D20 | Seeds and pairing | 16-30 for every block, paired on seed; no retraining | Fewer seeds for the certificate | Real |
| D21 | Polling areas and schedule | NJ centroid 40.3525 N, -74.6239 W and Mainz exit 49.9312 N, 8.3091 E, circles of 500 m, one request per 5 s per area, two 2 h sessions (afternoon and night), hard cap 5,760, stop at the first 429, no retry | 2 s interval; one area | Weak on budget (item 1) |
| D22 | Polling storage | Every request: send and receive wall times, status, `Date`, `Age`, `Cache-Control`, `ETag`, `sourceUpdated`, body sha256, body size, round-trip time; body kept for the first response of each new version per area | Every body | Real |
| D23 | Lag estimator | Per version: upper bound = first receipt minus `sourceUpdated`; lower bound = previous query's send time minus `sourceUpdated`; `L_bar` = max upper bound, `L_low` = min lower bound, `J = L_bar - L_low` | Point estimate | Real |
| D24 | Fixed-phase parameter `psi` | The second-of-minute farthest from the support of the availability phase measured by polling | A fixed :50 | Real |
| D25 | F1 check | Fit flow loss against `delta` and against `s` as a line through the origin; report slope and whether a quadratic term is resolved | Visual comparison | Real |
| D26 | Numerical dependencies | None added: three-piece CDF fit by numpy grid search; Kaplan-Meier written directly | Install `pwlf`, `lifelines`, `scipy` | Real |
| D27 | Runner | One process, sequential episodes, one JSONL line per episode, resume by key, pause sentinel checked between episodes, environment recorded in the header | Parallel workers | Real |
| D28 | Figures | Three result figures and one table, allocated in section 6 | Four figures | Real |
| D29 | Trace storage | Per-10 s traces under `outputs/feed/` (gitignored); summaries under `results/feed/` | Traces in `results/` | Real |

## 1. The agreement rules and the lemma

### 1.1 Setting

The feed publishes versions `k = 1, 2, ...`, each stamped `T_k` (`sourceUpdated`), with
`T_{k+1} - T_k >= g`. Version `k` is first returned by a query at the availability
instant `A_k = T_k + L_k`, where the availability lag `L_k` lies in `[L_low, L_bar]`;
write `J = L_bar - L_low` for the lag jitter and assume `g > J`, so availability instants
are in publication order. A query at time `t` returns version `k(t) = max{k : A_k <= t}`.

A fleet of `N` controlled vehicles reads the feed. Vehicle `i`'s query instants are
`P_q`-dense: every interval of length `P_q` contains at least one of them. The periodic
schedule `{phi_i + m P_q}` with phase `phi_i` in `[0, P_q)` is the case the simulation
uses; the lemma's part (a) holds for any `P_q`-dense schedule. Clocks are GPS-disciplined
and within `epsilon` of true time, with `epsilon` far below 1 s.

Every vehicle holds the same policy `pi` (the `here` checkpoint) and computes, from a
snapshot `x_k` (the 12 x 5 observation carried by version `k`), the action vector
`a_k = pi(x_k)`, one speed fraction per super-segment. A vehicle on super-segment `e`
applying version `k` commands the fraction `a_k[e]`.

Two quantities describe a rule:

- **Disagreement** `delta(t)`: the fraction of controlled vehicles whose applied version
  differs from the plurality applied version at `t`. Its time average over the scored
  window is the number the paper reports. The command-level fraction, meaning vehicles
  whose commanded fraction on their own segment differs from the plurality command on
  that segment, is recorded beside it; it is at most `delta(t)` and is the quantity the
  flow responds to.
- **Age** of an applied snapshot at `t`: `t - T_k` for the version `k` the vehicle is
  applying. The fleet mean age, time-averaged, is the staleness the rule imposes.

### 1.2 The three rules

**No rule (the deployed rig as driven).** Vehicle `i` queries at `{phi_i + m P_q}` and
applies the returned version on receipt. With `P_q = 60 s` this is what the rig did on
2026-09-08. For a periodic schedule with uniform phases and `P_q <= g`, the fraction of
vehicles not yet on the newest available version `tau` seconds after `A_k` is
`1 - tau/P_q` for `tau < P_q`, so the fraction outside the plurality is
`min(tau/P_q, 1 - tau/P_q)`, which averages `P_q/(4g)` over a cycle and peaks at 1/2.
With `P_q = g = 60 s` the average is 25% and the peak 50%; with `P_q = 10 s` the average
is 4.2%. The mean age at application is `L + P_q/2` and the time-averaged mean age is
about `L + P_q/2 + g/2`.

**Fixed-phase rule, parameter `psi`.** Every vehicle queries once per minute at wall-clock
second `psi` and applies on receipt. Requests complete within `[psi - epsilon,
psi + epsilon + r_bar]`, where `r_bar` is the largest request round-trip time (measured
by the polling run). In a minute whose availability instant falls outside that window,
every vehicle receives the same version and `delta = 0`; in a minute whose availability
instant falls inside it, the vehicles whose requests completed before `A_k` hold the
previous version for the whole minute. The expected time-averaged disagreement is the
probability that the availability phase `(T_k + L_k) mod 60` falls in the window, times
the expected split. `psi` is chosen as the second-of-minute farthest from the measured
support of the availability phase (D24). The rule's costs are that every vehicle's
request lands in the same second, that a 26 s gap makes it skip a version, and that a
publication-phase switch (:02 to :36 or back) moves the availability phase by 26 or 34 s
and can move it into the window.

**Version-keyed rule, parameter `D` (Algorithm 2, recommended).** Vehicle `i` queries at
`{phi_i + m P_q}`. On fetching version `k` at query time `q`, it computes `a_k = pi(x_k)`
and applies `a_k` from the switch instant `S_k = T_k + D` if `q <= S_k`, else from `q`.
It replaces `a_k` only when a later version reaches its own switch instant. The
designated version at `t` is `kappa(t) = max{k : S_k <= t}`. Every vehicle keys its switch
to the feed's own stamp, so no vehicle needs to know `psi`, the publication phase or any
other vehicle's schedule. `D` trades staleness (the applied snapshot is at least `D` old)
against disagreement (a vehicle that has not fetched `k` by `S_k` is late).

Printable statement of the rule:

> Algorithm 2 (version-keyed agreement). Each vehicle queries the feed every `P_q`
> seconds at its own phase. On receiving a response stamped `T`, it computes the action
> vector from that response and schedules it to take effect at `T + D`; if `T + D` has
> already passed it takes effect immediately. A scheduled action is replaced only by the
> action of a later stamp at that stamp's own `T + D`.

### 1.3 The lemma

**Lemma 1 (feed agreement).** Under the setting of 1.1 and the version-keyed rule:

(a) If `P_q <= g - J` and `D >= L_bar + P_q`, then every vehicle applies `a_k` on exactly
`[T_k + D, T_{k+1} + D)`, up to the clock error `epsilon`. Hence `delta(t) = 0` for all
`t`, and the age of every applied snapshot lies in `[D, D + (T_{k+1} - T_k)]`.

(b) If `P_q <= g - J` and `L_bar <= D < L_bar + P_q`, and the query phases are
independent and uniform on `[0, P_q)`, then for `0 <= tau < P_q - (D - L_bar)` the
expected fraction of vehicles not applying `a_k` at time `T_k + D + tau` is at most
`1 - (D - L_bar + tau)/P_q`, and it is zero for larger `tau`. Averaged over the cycle
`[T_k + D, T_{k+1} + D)` of length at least `g`, the expected disagreement is at most
`(P_q - D + L_bar)^2 / (2 P_q g)`.

(c) If `P_q > g - J`, a vehicle can skip version `k` by querying before `A_k` and next
after `A_{k+1}`. The expected skipping fraction is at most `1 - (g - J)/P_q`, and a
skipping vehicle disagrees throughout `[T_k + D, T_{k+1} + D)`. This term adds to the
bounds in (a) and (b).

**Proof sketch.** Let `W_k = [A_k, A_{k+1})` be the interval during which a query returns
version `k`. (a) The interval `[A_k, A_k + P_q]` has length `P_q`, ends by
`T_k + L_bar + P_q <= S_k`, and lies inside `W_k` because
`A_{k+1} - A_k >= g - J >= P_q`. `P_q`-density puts a query of every vehicle in it; that
query returns `k`; so every vehicle holds `a_k` before `S_k` and, by the rule, applies it
at `S_k`. No vehicle applies `a_{k+1}` before `S_{k+1} > S_k`. The age statement is
`t - T_k` on `[S_k, S_{k+1})`. (b) A vehicle is not yet applying `a_k` at `S_k + tau` iff
it has no query in `[A_k, S_k + tau]`, an interval of length `D - L_k + tau >=
D - L_bar + tau`; a periodic schedule with uniform phase has no query in an interval of
length `w < P_q` with probability `1 - w/P_q`. Integrating `1 - (D - L_bar + tau)/P_q`
over `tau` from 0 to `P_q - (D - L_bar)` and dividing by the cycle length gives the
average. (c) Skipping is the event of no query in `W_k`, whose length is at least
`g - J`; the same probability formula applies. The plurality-based `delta(t)` is at most
the fraction not applying `a_{kappa(t)}`, so every bound above bounds `delta`. QED.

Six-line form for the paper: state (a) and (b) only, with (c) as a sentence in the
remark: "when the shortest gap between publications is less than the query period plus
the lag jitter, a vehicle can miss a version, and the fraction that does is at most
`1 - (g - J)/P_q`."

### 1.4 What the measured feed implies for the parameters

From the drives: `g_min = 26 s` (4 of 59 gaps), `J >= 10 s`, `L_bar <= 110 s` (first
receipt, an upper bound). With `P_q = 10 s`, part (a) needs `J <= 16 s`; the polling run
decides this (open item 3). `D = L_bar + 10 s` makes the applied snapshot between `D` and
`D + 60 s` old. If `L_bar` is near the drive maximum, that is 120 to 180 s of staleness,
which is why the staleness sweep in block B runs to 300 s: the paper must report what
that age costs in throughput, and the figure of disagreement against `D` lets the author
choose a looser `D`.

Recommendation (D1): the version-keyed rule, because it needs no parameter tied to the
feed's phase, keeps zero disagreement under phase switches, spreads the fleet's requests
over the cycle, and has a proof. The fixed-phase rule has lower staleness when the
availability phase is stable and lies away from `psi`, and the experiment in block A
quantifies both; the paper reports both and recommends V unless the polling run shows a
lag jitter below 5 s and no phase switch over both sessions, in which case the author
may prefer F for its lower staleness and the plan's figures carry the comparison either
way.

## 2. The polling experiment

Purpose: measure the feed's update process, meaning cadence, phase, availability lag
and its jitter, and the request round-trip time, from a stationary laptop with an
NTP-synced clock, in the two areas the paper names.

### 2.1 Areas

| area | centre | radius | why |
|---|---|---|---|
| `nj` | 40.3525 N, -74.6239 W (centroid of the 93 query centres of `run_20260908_170849`) | 500 m | same area and radius as the drives, so the drive-log and polling lags are comparable |
| `mainz` | 49.9312 N, 8.3091 E (midpoint of edge 218, the exit link on every route, through the Vissim georeference) | 500 m | the network the policy is trained on; checks that HERE covers it and whether the European feed's process differs |

One additional request per session at `mainz` with a 7,000 m radius, body stored, to
confirm the network's roads are returned (the network box is 49.908 to 49.964 N, 8.151
to 8.317 E; a 7 km circle at the exit covers its eastern two thirds, and a second 7 km
request at entry 289's position, 49.9600 N, 8.1529 E, covers the west). These two bodies
are the coverage check for open item 8.

### 2.2 Schedule and budget

- Interval 5 s per area, areas interleaved 2.5 s apart. Lag resolution is 5 s, which is
  8% of the 60 s cycle; the first-receipt estimator is an upper bound on the availability
  lag and the previous query's send time a lower bound, so the 5 s quantisation widens
  the bracket in the direction that makes `D` larger, never smaller.
- Two sessions of 2 hours, one in the afternoon and one at night, local time of the
  laptop, to see whether cadence or lag differ by time of day. 1,440 requests per area per
  session, 2,880 per session, 5,760 in total, plus 4 coverage requests.
- Hard cap: `--max-requests`, default 2,900 per session; the script stops at the first
  HTTP 429, at the first 5 consecutive non-200 responses, or at the cap, and never
  retries. A 10-minute dry session (`--duration-s 600`) runs first and its 429 count is
  checked before the 2 h session starts.

### 2.3 Key handling

The script reads `here.apiKey` from `phone/local.properties` at start, keeps it in one
local variable, passes it only as the `apiKey` entry of the `requests` params dict, and
builds the recorded `request_url` without it. Every string the script writes passes
through `redact(text, key)`, which replaces the key with `REDACTED`; exception messages
from `requests` can embed the full URL, so they are redacted before logging. Before exit,
the script re-reads every file it wrote and raises if the key appears in any of them,
deleting the file first. Tests use a fake key and exercise the redaction, the self-check
and the URL construction; no test touches the network.

### 2.4 Records

One JSONL per session, `outputs/feed/polling/{session_id}/requests.jsonl`, with per
request: `seq`, `area`, `t_send_wall`, `t_send_mono`, `t_recv_wall`, `t_recv_mono`,
`status`, response headers `Date`, `Age`, `Cache-Control`, `ETag`, `Last-Modified` when
present, `sourceUpdated`, `body_sha256`, `body_bytes`, `rtt_s`, `request_url` (no key),
`body_file` (set only for the first response of each new `sourceUpdated` per area, stored
under `bodies/`). A header line records the laptop's clock check: the offset between
`t_recv_wall` and the server's `Date` header on the first response (1 s resolution), the
output of `sntp -t 5 time.apple.com` if available, platform, Python version and the git
SHA of the script's tree.

### 2.5 Analysis (`scripts/analyze_feed_process.py`)

Per area and session, from the records:

- Versions: distinct `sourceUpdated` in order; gaps `T_{k+1} - T_k`; phase
  `T_k mod 60`; counts of each gap value and each phase.
- Lag bracket per version: upper `L_up_k = t_recv_wall(first response carrying k) -
  T_k`; lower `L_lo_k = t_send_wall(last query before that one) - T_k`. `L_bar` is the
  maximum of `L_up`, `L_low` the minimum of `L_lo`, `J = L_bar - L_low`. Also the
  distribution of `L_up` (min, p50, p90, max) for the figure.
- Availability phase support: the set of `(T_k + L_up_k) mod 60`, with the margin of
  every candidate `psi` to that support; `psi*` is the arg-max (D24).
- Round-trip time: p50, p95, max of `rtt_s`, which sets `r_bar` for the fixed-phase
  window.
- `Age` header statistics, because a CDN age explains part of the lag if present.
- The same analysis on the drive logs through the `metadata.jsonl` wall-clock anchor
  (the appendix has the numbers), written to the same artefact with
  `source: "drive_2026-09-08"`.

The artefact `results/feed/feed_process.json` carries all of the above plus the (gap,
`L_up`) sequence per session, which block A replays, and its sha256 is recorded by every
SUMO run that reads it.

## 3. The SUMO changes

### 3.1 Per-vehicle feed versions (`src/sumo/feed.py`, new)

- `FeedProcess`: a sequence of publication instants and availability lags in simulation
  time. Two constructors: `aligned(decision_instants)` (publication at the env's decision
  instants, lag 0: the published arm) and `from_artefact(feed_process.json, session,
  seed, start_s)` (replay of the recorded (gap, `L_up`) pairs from a seeded start index,
  publication instants `T_1 = start_s`, `T_{k+1} = T_k + gap_k`). A synthetic
  constructor exists for tests only.
- `Snapshot`: `(k, T_k, A_k, x_k, a_k)`. The env takes `x_k = observe()` at `T_k` and the
  runner computes `a_k = greedy_actions(model, x_k)` once per version, so the policy is
  evaluated per snapshot and vehicles sharing a version share the vector.
- `VehicleFeedState`: per controlled vehicle, its query phase `phi_i` (drawn from
  `numpy.random.default_rng(seed + 1000)` at insertion, independent of SUMO's seed so the
  traffic realisation is unchanged), the version it holds, the version it applies and the
  instant it last switched.
- Rules as classes with one method `applied_version(vehicle_state, feed, t)`:
  `NoRule(P_q)`, `FixedPhase(psi, P_q=60, r_bar)`, `VersionKeyed(P_q, D)`,
  `Synchronous()` (switch at `A_k`). Each is a pure function of the vehicle's query
  instants and the process, so it is unit-tested without SUMO.
- `lemma_bound(D, L_bar, P_q, g, tau)` and `lemma_cycle_average(D, L_bar, P_q, g)`:
  the expressions of Lemma 1 (b), used by the figure and by a test that the simulated
  fleet respects them.
- `disagreement(applied_versions_by_vehicle)` and `command_disagreement(applied, segment
  of each vehicle, action vectors)`: the two fractions of 1.1.

### 3.2 Environment changes (`src/sumo/mainz.py`)

- Penetration: `MainzEnv(..., penetration=1.0)` and `controlled(vehicle_id) -> bool`,
  implemented by the floor rule of D7 over the schedule's departure index (the schedule
  rows are already in departure order). `_command` and `release` test `controlled(id)`
  instead of `getTypeID == "av"`; at `penetration=1.0` the two tests agree on every
  vehicle, which the equivalence sanity test checks.
- Gate: `_admit` inserts an uncontrolled vehicle at its departure time without the
  density check (D8) and keeps the per-entry order among controlled vehicles only.
  `held_at_entry()` counts controlled vehicles only; `pending()` returns
  `len(getPendingVehicles())`.
- Ticked advance: `advance_ticks(seconds, tick_s=1.0, on_tick=callback)` steps the
  simulation in 1 s ticks (two 0.5 s steps) and calls `on_tick(t)` after each, so the
  runner can take snapshots at publication instants, switch vehicles at their own
  instants and record traces at 10 s. `step()` is unchanged and remains the published
  arm's path.
- `command_vehicles(mapping: dict[vehicle_id, fraction])`: writes `setSpeed` for the
  listed vehicles only, with the target `fraction * free_flow_kmh / 3.6` of the edge the
  vehicle is on. Vehicles not listed keep their last command, exactly as a vehicle
  changing segment between decisions does today.
- `segment_state()`: per super-segment vehicle-weighted space-mean speed (km/h), density,
  count, from one `_per_edge()` read, for the 10 s traces. The drain metric reads this
  speed and never `densities()`.
- Pulse injection (`inject_pulse(site, n, start_s, duration_s, seed)`): adds a route
  whose edges are the suffix of a traversing route from the site segment's most upstream
  edge to the exit (`route.add`), then inserts `n` vehicles evenly over `duration_s` with
  ids `pulse_{site}_{j}`, controlled by the same floor rule continued over `j`. Sites and
  their edges come from `pulse_sites(segments, routes)` in `src/analysis/gains.py`, which
  walks each route's segment sequence (0-1-2-3-4-5-...-11; entries at 2, 5 and 8 join the
  chain) and picks, per segment, the first edge of that segment on the longest route
  through it.

### 3.3 The feed-aware episode (`src/sumo/feed_episode.py`, new)

`run_feed_episode(model, seed, scenario, rule_or_injection, feed_process, penetration,
duration_s, trace_every_s=10, pulse=None, record=...) -> dict` runs one episode through
`advance_ticks` and returns the metrics of `run_episode` plus: time-averaged `delta` and
command-level `delta`, mean snapshot age, per-version switch counts, the number of
`setSpeed` writes, and the trace (per 10 s: segment speeds, densities, counts, held,
pending, running, arrived, applied-version histogram, `delta`, command `delta`).

Modes, all through the same code path:

- `published`: `FeedProcess.aligned`, `Synchronous` rule, lag 0. Must reproduce the
  `here` arm's committed per-seed numbers exactly (3,751.11 veh/h on seed 16).
- `synchronous`: artefact process, every vehicle switches at `A_k`.
- `no_rule_60`, `no_rule_10`, `fixed_phase`, `version_keyed(D)`: artefact process.
- `inject_delta(delta)`: aligned process; at each decision instant a fresh subset of
  controlled vehicles of fraction `delta` (seeded) applies `a_{k-1}` for the whole
  interval; the rest apply `a_k`.
- `inject_stale(s)`: aligned process with a 10 s snapshot ring; at each decision instant
  every vehicle applies `pi(x(t - s))`.
- `penetration(p)`: `published` mode with `penetration=p`.
- `pulse(site, n)`: `penetration(p)` mode plus `inject_pulse` at 1,200 s for 60 s.

### 3.4 Scenario builder (`scripts/build_mainz_scenario.py`)

- `--out-name NAME`: write `NAME.rou.xml`, `NAME_routes.rou.xml`, `NAME_schedule.json`
  beside the shared `mainz.net.xml` instead of the `mainz*` names.
- `--demand-only`: skip `netconvert` and the exit constraint; reuse `mainz.net.xml`
  unchanged (its sha256 is asserted equal before and after).
- `--demand-burst START END SCALE`: on top of `--demand-duration-s D --demand-scale S`,
  replace every entry's rate on `[START, END)` by `opening volume x SCALE`, so the
  intervals per entry become `[0, START) at S`, `[START, END) at SCALE`, `[END, D) at S`.
- Scenario (a) is the builder's default demand with `--duration-s 9000 --out-name
  mainz_surge --demand-only` (departures stop at 1,200 s by the `.inpx` intervals).
  Scenario (b) is `--demand-duration-s 12000 --demand-scale 0.202778 --demand-burst 900
  1500 0.277778 --duration-s 12000 --out-name mainz_burst --demand-only`
  (3,650 / 18,000 and 5,000 / 18,000).
- `SCENARIOS` in `scripts/train_mainz_src.py` gains `mainz_surge` and `mainz_burst`
  entries pointing at those files and the shared net.
- The committed sustained scenario files are not regenerated and not touched.

### 3.5 The drain-time metric (`src/sumo/drain.py`, new)

Inputs: one trace per episode (10 s samples), the scenario's burst end `t_0` (1,200 s for
(a), 1,500 s for (b)), the cap, the hold period `H`, and a threshold rule.

- `fit_three_piece_cdf(speeds) -> (s1, s2)`: evaluate the empirical CDF on a 0.5 km/h
  grid from 0 to the maximum speed; for every pair of breakpoints on the grid with
  `s1 < s2`, fit the continuous three-piece linear function by least squares (a linear
  problem in four coefficients given the breakpoints); return the pair with the smallest
  residual. Numpy only. The fit is made on the pooled super-segment space-mean speeds of
  the scenario's no-control episodes, samples from `t_0 - 600 s` to the cap, segments
  with at least one vehicle. Threshold `v_jam = (s1 + s2)/4`, following the 2023 paper's
  slowdown-jam rule. Sensitivities: per-segment fits, falling back to the pooled
  threshold for a segment whose fit has `s2 < 1.5 s1`; and a fixed 30 km/h, which is the
  policy's lowest speed action. All three thresholds are written to the artefact with the
  fitted `s1`, `s2` and the residual, and every drain result is reported under each.
- `jam_indicator(trace, v_jam)`: segment `e` is jammed at sample `t` iff its count is
  positive and its space-mean speed is below `v_jam`.
- `drain_time(trace, t_0, H, cap, v_jam)`: the smallest `t_d >= t_0 + H` such that no
  segment is jammed at any sample in `[t_d - H, t_d]`, `held(t_d) = 0` and
  `pending(t_d) = 0`; returns `(t_d - t_0, drained=True)`, or `(cap - t_0,
  drained=False)` when no such `t_d <= cap` exists. Censored runs are kept with
  `drained=False`.
- `onset(trace, t_0, v_jam)`: first jammed sample at or after the burst start (900 s for
  (b), 0 s for (a)); `None` if no sample is jammed (reported as the fraction with an
  onset).
- `jam_segment_minutes(trace, v_jam)`: `10 s x (number of jammed (segment, sample)
  pairs) / 60`, from the burst start to the cap, a lower bound for censored runs and
  labelled as such.
- `kaplan_meier(times, drained)`: the product-limit survival estimate with censoring at
  the cap; the median is the first time at which the estimate is at or below 0.5 and is
  reported as "> cap" when the estimate never reaches 0.5. Fraction drained, per-seed
  paired differences on seeds where both arms drained, and the KM curves per arm are the
  outputs.
- The module never imports `DENSITY_CRITICAL` or calls `densities()`; a test greps its
  source for both names.

### 3.6 The minimum-penetration certificate (`src/analysis/gains.py`, new)

Model: the sampled comparison system of SNS Assumption 3 and Proposition 1 at the 10 s
sampling of the traces, `y(t + dt) - y(t) ~= dt (-A y(t) + B y(t))` with `A =
diag(a_e)`, `B = [b_ej]`, `b_ee = 0`, `b_ej >= 0`, `y_e(t) = |rho_e^pulse(t) -
rho_e^base(t)|` the absolute density deviation of segment `e` between a pulsed episode
and the unperturbed episode of the same seed and penetration (the F3 reading: perturb one
super-segment's inflow, record how it and the others respond under the trained policy).

- Perturbation: 60 vehicles over 60 s at `t = 1,200 s` at site `e` (one of 12), the
  controlled regime after the throughput window opens. Deviations are read from 1,200 s
  to 1,800 s (60 samples).
- Estimation per seed: stack the 12 sites' episodes; for each segment `e`, regress the
  increments `y_e(t + 10) - y_e(t)` on the 12 current deviations `y_j(t)` (no intercept);
  `a_e = -coef_ee / 10 s`, `b_ej = |coef_ej| / 10 s`. 15 seeds give 15 estimates of each
  of the 144 coefficients.
- Confidence bounds over seeds (D18): `a_LCB_e = mean - t_{0.95,14} se`,
  `b_UCB_ej = max(0, mean + t_{0.95,14} se)` on `|coef|`. The matrix
  `M = A_LCB^-1 B_UCB` has entries `b_UCB_ej / a_LCB_e` and zero diagonal.
  `rho_UCB = rho(M)` is an upper confidence bound on `rho(A^-1 B)` because the spectral
  radius of a nonnegative matrix is monotone non-decreasing in every entry, and every
  entry of `M` is an upper bound on the corresponding entry of `A^-1 B` with the stated
  confidence (a union over 144 one-sided bounds, which the paper states; a Bonferroni
  level is also reported). If any `a_LCB_e <= 0` the certificate is undefined at that
  penetration and the artefact says so.
- Point estimate `rho_hat = rho(A_hat^-1 B_hat)` reported beside the bound.
- Penetration levels {1.0, 0.75, 0.5, 0.25, 0.1}; `p_min` is the smallest level with
  `rho_UCB < 1`, with the level below it reported as the first level where the bound is
  not met. Throughput at the same levels comes from block C.
- Linearity check: 30-vehicle pulses on seeds 16-18 at every level and site; the ratio
  of peak self-deviation (60-vehicle over 30-vehicle) is reported; a ratio outside
  [1.6, 2.4] is stated in the paper as a limit of the linearisation.
- Limits to state in the paper: the gains are local to the operating point (sustained
  4,500 veh/h, `t` in [1,200, 1,800] s), statistical (15 seeds), and the argmax policy
  makes the closed loop piecewise, so Proposition 1's differentiability holds between
  action switches and the regression averages over switches.

### 3.7 Recording and runner (`scripts/run_feed_experiments.py`, new)

- Blocks `A` to `E` of section 5, selectable; every episode is identified by the key
  `(block, arm, variant, seed)`.
- One JSONL per block under `results/feed/{block}.jsonl`, appended and flushed after
  every episode; traces under `outputs/feed/traces/{block}/{arm}_{variant}_{seed}.json`.
  The summary `results/feed/{block}_summary.json` is rebuilt from the JSONL at the end of
  a run or on `--summarize-only`, written whole-then-rename as the existing evaluator
  does.
- Header record: git SHA and dirty flag (via `_provenance`), `sumo --version` first line,
  `libsumo.__version__`, Python version, platform, the command line, the config, the
  checkpoint path and sha256, the feed-process artefact path and sha256, the scenario
  files' sha256s, and the start time. A second provenance sample is written at the end
  and compared, as `evaluate_mainz_checkpoints.py` does.
- Resume: `--resume` reads the block's JSONL and skips every key present.
- Pause: between episodes the runner checks `--pause-file` (default
  `results/feed/PAUSE`); when it exists the runner writes `results/feed/{block}_state.json`
  with the next key and exits 0 with the line `paused before <key>`. Removing the file and
  re-running with `--resume` continues.
- Sequential, one process, one environment at a time (libsumo is process-global).

## 4. Step table for the implementer

Test-driven: write the unit tests first, then the code, then the sanity test. Unit
tests check concrete inputs and outputs without SUMO. Sanity tests run a short SUMO
episode and check expected behaviour; each sanity test file runs in its own pytest
process (`.venv/bin/python -m pytest -q tests/<file> -p no:cacheprovider`), never with
`-n`. The implementer runs smoke episodes only (one short episode per block, seed 16) and
does not run the experiments.

| step | files | functions | unit tests (concrete I/O) | sanity tests (SUMO) |
|---|---|---|---|---|
| S1 builder options | `scripts/build_mainz_scenario.py`; `scripts/train_mainz_src.py` (`SCENARIOS`) | `burst(volumes, start, end, scale)`; `--out-name`; `--demand-only` | `burst` on a two-entry fixture gives the three intervals with the stated rates; departure count of a 1,000 s background at 3,600 veh/h with a 100 s burst at 7,200 veh/h is 1,100; `--out-name` writes the three named files and nothing named `mainz.rou.xml`; `--demand-only` leaves `mainz.net.xml`'s sha256 unchanged | build `mainz_burst` and `mainz_surge`; a 400 s no-control episode on each starts and inserts vehicles |
| S2 env penetration and gate | `src/sumo/mainz.py` | `controlled`, `_admit`, `held_at_entry`, `pending`, `command_vehicles`, `segment_state`, `advance_ticks` | floor rule: `p=1` controls all 3,124, `p=0` none, `p=0.5` exactly 1,562, `p=0.75` exactly 2,343, consecutive controlled indices never more than `ceil(1/p)` apart; `segment_state` on a stubbed `_per_edge` weights speed by count | 400 s episode at `p=0.5`: `setSpeed` writes go to exactly the controlled ids; uncontrolled vehicles are never in `held_at_entry`; `advance_ticks(60)` with a no-op callback reproduces `step()`'s arrivals and flow on seed 16 at 600 s exactly |
| S3 feed process and rules | `src/sumo/feed.py` | `FeedProcess.aligned/from_artefact`, rules, `lemma_bound`, `disagreement`, `command_disagreement` | synthetic process (`g=60`, `L` in [40, 50], `P_q=10`, `D=60`): 1,000 vehicles with uniform phases give `delta=0` at every 1 s instant; `D=50` gives a late fraction within 0.03 of the bound at `tau` in {0, 5}; `NoRule(60)` averages 0.25 within 0.02 and peaks at 0.5; `FixedPhase(psi)` is 0 with `psi` outside the window and splits inside it; `from_artefact` replays the recorded gaps in order from the seeded start; `command_disagreement` on two vehicles with different versions but equal commands is 0 | none |
| S4 feed-aware episode | `src/sumo/feed_episode.py` | `run_feed_episode`, modes of 3.3 | per-tick bookkeeping on a fake env: a vehicle switching at `S_k` is commanded once; `inject_delta(0.3)` marks 30% of 100 fake vehicles late per decision and they apply `a_{k-1}`; `inject_stale(60)` selects the ring entry 60 s old | `published` mode on seed 16, 2,500 s, reproduces the committed `here` flow 3,751.1111 and arrivals 1,989 exactly (6 s); `inject_delta(1.0)` on seed 16 at 1,200 s differs from `published` in flow; `inject_stale(60)` equals a direct "everyone applies the previous decision's vector" implementation at 1,200 s; `version_keyed` with the placeholder process reports `delta=0` and `no_rule_60` reports `delta > 0.15` at 1,200 s |
| S5 drain module | `src/sumo/drain.py` | `fit_three_piece_cdf`, `jam_indicator`, `drain_time`, `onset`, `jam_segment_minutes`, `kaplan_meier` | bimodal synthetic speeds (modes 15 and 50 km/h) give `s1` in [12, 22] and `s2` in [40, 55]; threshold below `s1`; a hand-built trace with jams ending at 2,000 s, held reaching 0 at 2,300 s, `H=120` gives drain 2,300 - t_0; the same trace with pending 1 until 2,600 s gives 2,600 - t_0; a trace jammed at the cap is censored with `drained=False`; KM on (10, 20, 30+, 40) with the 30 censored matches the hand values 0.75, 0.5, 0.5, 0.0 at those times and median 20; the module source contains neither `DENSITY_CRITICAL` nor `densities(` | none |
| S6 gains module | `src/analysis/gains.py` | `pulse_sites`, `deviations`, `fit_gains`, `confidence_bounds`, `rho_upper_bound`, `min_penetration` | a synthetic linear system with known `a` and `B` simulated for 60 samples recovers both within 10%; lowering any `a_e` never lowers `rho`; raising any `b_ej` never lowers `rho`; `rho_UCB >= rho_hat` on 100 random instances; a case with `a_LCB <= 0` returns "no certificate"; `pulse_sites` on the committed segments and routes returns 12 sites whose edges belong to their segments | none |
| S7 pulse injection | `src/sumo/mainz.py` | `inject_pulse` | suffix route for site 7 on `r277` starts with segment 7's first edge and ends with `exit_tail` | 600 s episode, pulse of 20 vehicles at 350 s: arrivals exceed the unpulsed run of the same seed by 20 at 600 s, or `running + pending` accounts for the remainder; the pulse ids are controlled per the floor rule |
| S8 polling script and analysis | `scripts/poll_here_feed.py`, `scripts/analyze_feed_process.py` | `read_key`, `build_request`, `redact`, `poll`, `self_check`, `analyze_records`, `analyze_drive_logs` | fake key in a temp properties file is read; `build_request` puts the key only in params and the recorded URL lacks it; `redact` removes the key from a URL-bearing exception string; `self_check` raises and deletes a file containing the key; `--max-requests 3` stops after 3 with a stubbed session; a stubbed 429 stops; `analyze_records` on a synthetic record set with gaps 60/26/34 and lag 45 recovers them and `J=0`; drive-log analysis reproduces the appendix numbers (60 versions, min 33.8 s, max 109.7 s) | none; no network in tests |
| S9 runner | `scripts/run_feed_experiments.py` | `run_block`, `resume_keys`, `check_pause`, `header`, `summarize` | with stubbed episodes: resume skips the 2 keys in a pre-written JSONL and runs the other 4; a pause file present before episode 3 exits 0 after episode 2 with the state file naming episode 3; the header carries SUMO version, libsumo version, checkpoint sha256 and feed-process sha256; arms share the identical seed list | one episode of each block on seed 16 at 600 s writes a parseable JSONL line, a trace and a summary |
| S10 figures | `scripts/figures/feed_figures.py` | `fig_feed_and_rule`, `fig_flow_consequence`, `fig_drain`, `table_certificate_and_rules` | each function takes loaded JSON and returns the plotted series; a test checks the series equal the JSON's values; a missing artefact raises | none |
| S11 smoke | — | — | — | the S9 smoke, once per block, recorded in the implementation record with wall times |

Every new test file is separate; the existing `tests/test_mainz_port.py` and
`tests/test_mainz_paired_evaluation.py` must still pass unchanged.

## 5. The experiment matrix

All blocks use the `here` checkpoint, seeds 16 to 30 paired on seed, step 0.5 s, warm-up
300 s, flow window from 900 s, 2,500 s episodes unless stated. The measured cost is 6.0 s
per published 2,500 s episode (appendix); the per-tick path adds edge scans, so the
budget per 2,500 s episode is 10 s. Drain episodes are 9,000 and 12,000 s with up to
several thousand pending vehicles; the budget is 60 s each, an extrapolation.

| block | arms and variants | episodes | budget |
|---|---|---|---|
| A rules | `published`, `synchronous`, `no_rule_60`, `no_rule_10`, `fixed_phase(psi*)`, `version_keyed(D)` for `D - L_bar` in {-20, -5, 0, +5, +10} s with `P_q=10`; artefact process | 10 x 15 = 150 | 25 min |
| B injection | `inject_delta` for `delta` in {0.1, 0.25, 0.5, 0.75}; `inject_stale` for `s` in {30, 60, 120, 180, 300} | 9 x 15 = 135 | 23 min |
| C penetration | `penetration(p)` for `p` in {0.75, 0.5, 0.25, 0.1} (`p=1` is `published`; `p=0` is `no_control` from the committed JSONL) | 4 x 15 = 60 | 10 min |
| D drain | scenarios `mainz_surge` (cap 9,000 s) and `mainz_burst` (cap 12,000 s) x {`no_control`, `here`}; pilot first: seed 16 on `mainz_burst`, both arms | 60 + 2 | 65 min |
| E certificate | `pulse(site, 60)` for 12 sites x 5 levels x 15 seeds; `pulse(site, 30)` for 12 sites x 5 levels x seeds 16-18; unperturbed baselines reused from C and `published` | 900 + 180 = 1,080 | 3.0 h |
| total | | 1,427 | about 5 h, single process |

Block ordering: the polling run first (two 2 h sessions, idle laptop), then A (needs the
artefact), then B, C, D, E. B to E do not depend on the polling artefact; if the artefact
is delayed, run B to E first and A last.

Pilot rule for block D(b): run seed 16 for both arms; the burst breaks no control down if
its served flow over [1,500, 3,000] s is below 3,650 veh/h and its pending count at
3,000 s exceeds its pending count at 1,500 s; if not, re-build with burst 6,000 veh/h and
cap 16,000 s and repeat once. The controlled arm must reach held + pending = 0 before
the cap; if not, re-build with background 3,600 veh/h and repeat once. The pilot's
outcome and the final parameters are recorded in the block's header.

Block A's `L_bar` and `psi*` are read from `feed_process.json`; the five `D` values are
written to the config as absolute seconds. Block A's time-averaged `delta` per arm,
mean age and flow are the table rows; the five `version_keyed` arms give the measured
`delta(D)` curve against Lemma 1 (b).

Checkpoint, resume and pause: section 3.7. Environment recorded at run time: section
3.7's header. Nothing retrains.

## 6. Artefacts and figures

Artefacts, all under `results/feed/` unless stated: `feed_process.json` (section 2.5);
`A.jsonl`, `A_summary.json`, ..., `E_summary.json`; `drain_thresholds.json` (fitted
`s1`, `s2`, residuals, the three thresholds per scenario); `certificate.json` (per
level: `a_hat`, `B_hat`, `a_LCB`, `B_UCB`, `rho_hat`, `rho_UCB`, `p_min`, linearity
ratios); figures as PDF under `results/feed/figures/`; traces under `outputs/feed/`.

Three result figures and one table, for a 6-page letter:

| item | content | result it carries |
|---|---|---|
| Fig. 1 | (a) the measured update process: `sourceUpdated` phase against time with the availability-lag bracket per version, both areas, both sessions; (b) disagreement against `D - L_bar` for the version-keyed rule: Lemma 1 (b) as a line, block A's measured points, with `no_rule_60`, `no_rule_10` and `fixed_phase` as horizontal reference marks | additions 1 and 2: the feed's cadence, phase switching and lag; the bound and its agreement with simulation |
| Fig. 2 | (a) paired throughput loss against injected disagreement `delta`, with the penetration curve plotted at `1 - p` on the same axis and the F1 line through the origin; (b) paired throughput loss against staleness `s` with its line | addition 3: flow consequence; the "a disagreeing vehicle acts like a non-compliant one" reading, confirmed or refuted by whether the two curves coincide |
| Fig. 3 | Kaplan-Meier drain curves per arm for scenarios (a) and (b), censoring marks at the cap, fraction drained and KM median annotated, under the primary threshold; sensitivities as a one-line note | the drain-time metric |
| Table 1 | block A rows: rule, time-averaged `delta`, command `delta`, mean age, paired flow gain against `published`; block C and E rows: penetration, paired flow gain, `rho_hat`, `rho_UCB`, certificate yes/no, `p_min` marked | the rule comparison and addition 4 |

## 7. Validator attack points

Checks most likely to catch a wrong reported number, each with the concrete assertion:

1. **The per-vehicle path is the published arm.** `published` mode on seeds 16-18 equals
   the committed `mainz_paired_seeds.jsonl` `here` rows in flow and arrivals exactly.
2. **Disagreement is injected and changes commands.** For every `inject_delta` episode,
   the recorded command-level `delta` is positive on at least one decision, and
   `inject_delta(1.0)` on seed 16 differs from `published` in flow. The artefact's
   `setSpeed` write counts differ between late and on-time vehicles.
3. **Staleness is injected.** `inject_stale(s)`'s action vectors at decision `k` equal
   `greedy_actions(model, x(t_k - s))` recomputed from the stored snapshots.
4. **The rule arms use the measured process.** Block A's header records the sha256 of
   `feed_process.json` and the session replayed; the placeholder process is refused
   unless `--allow-placeholder` is passed, and the summary carries the flag.
5. **Lemma conditions are checked, not assumed.** Block A's summary records `g_min`, `J`,
   `P_q` and whether `P_q <= g_min - J`; if false, the version-skip term is reported and
   `delta=0` is not claimed.
6. **Held and pending vehicles are counted.** Every drain trace has both series; the
   drain time of a trace whose pending series is forced to 1 moves to the cap; the
   no-control arm has `held = 0` on every sample and pending positive after the burst in
   scenario (b).
7. **The jam threshold is not the reward's.** `src/sumo/drain.py` contains neither
   `DENSITY_CRITICAL` nor `densities(`; the fitted `v_jam` is a speed in km/h recorded
   with `s1` and `s2`; recomputing the drain times under the fixed 30 km/h threshold
   changes at least one drain time, proving the threshold is live.
8. **Censored runs are kept.** The KM denominator equals the number of seeds; the
   fraction drained is `drained / 15`; a censored run appears in the per-seed table with
   `drained=False` and the cap.
9. **The clock starts at the same instant for every arm.** `t_0` in the summary is one
   number per scenario, not per arm.
10. **The bound's direction.** `rho_UCB >= rho_hat` at every level; replacing `a_LCB` by
    `a_hat` lowers or keeps `rho`; replacing `b_UCB` by `b_hat` lowers or keeps `rho`.
11. **The pulse is the stated size.** Pulsed and unperturbed episodes of the same seed
    differ in total inserted vehicles by exactly 60 (or 30).
12. **Penetration is what it says.** At `p = 0.5` the number of distinct ids ever
    commanded is 1,562 plus the controlled pulse vehicles; at `p = 0.1`, 312.
13. **The key never reaches disk.** Run `grep -rF "<key>"` over `outputs/feed/polling/`
    and `results/feed/` after the polling run (the validator reads the key from
    `local.properties` into a shell variable and never echoes it); zero matches. Also the
    script's own self-check log line.
14. **Pairing.** Every block's seed list is identical across arms; the paired bar is
    `2 stdev(d)/sqrt(n)` over per-seed differences, with `n = 15`.
15. **Figures carry artefact numbers.** Each plotted series is reproduced from the JSON by
    the figure module's own function; no literal in the figure script matches a result.
16. **Timing.** The header's per-episode wall times are within a factor of 2 of the
    budgets; a block whose episodes take far longer than budget is a sign the per-tick
    path is doing per-vehicle `getRoadID` calls.

## 8. Risks

| risk | what it would do to the paper's claims | mitigation in this plan |
|---|---|---|
| The polling budget is refused or the allowance is small | No measured lag or jitter; Lemma 1's parameters come from the drives only, with the lag an upper bound at 16.7 or 60 s resolution and jitter a lower bound | One session of 2,880 requests still gives 120 versions per area; the drive-log analysis stands as the fall-back and is labelled as such |
| Lag jitter `J` exceeds 16 s | Part (a) does not apply at `P_q = 10 s`; the version-skip term is nonzero and the rule's zero-disagreement claim weakens to a bound | `P_q = 5 s` doubles the query rate; part (c) is stated; block A measures the skip term |
| `L_bar` is large (over 100 s) | The version-keyed rule applies snapshots 2 to 3 minutes old and the staleness sweep shows a throughput cost comparable to the disagreement it removes | The `delta(D)` figure lets the paper recommend a `D` with a stated nonzero disagreement; the fixed-phase rule is reported as the lower-staleness alternative |
| Command-level disagreement is near zero because action vectors change rarely | The flow consequence of snapshot disagreement is small and the F1 linearity check is under-powered | Report command-level `delta` beside snapshot `delta`; state that the rule matters only on the decisions where the action changes, and report how many decisions those are |
| The burst does not break no control down, or the controlled arm never drains | Scenario (b)'s "controlled arms drain and no control does not" prediction is not testable at those parameters | The pilot and its two fall-backs; scenario (a) does not depend on the band |
| Drain times are censored in both arms under scenario (b) | Only fractions drained are reportable; medians are "> cap" | KM and fraction drained are the reported quantities by design; the cap is 12,000 s |
| The speed CDF is not bimodal in SUMO | `(s1 + s2)/4` is ill-defined and the primary threshold is arbitrary | The fit's residual and the per-segment check are recorded; the fixed 30 km/h sensitivity is reported in every case |
| The regression gives `a_LCB <= 0` at every level | No certificate; the minimum penetration cannot be stated | Report `rho_hat` and the point at which `rho_hat < 1`, labelled as a point estimate without a bound; raise the pulse to 90 vehicles on 3 seeds to test whether the response is below the noise |
| The closed loop is strongly nonlinear at a 60-vehicle pulse | The gains describe a larger perturbation than the comparison system assumes | The 30-vehicle linearity check; the paper states the ratio |
| Vehicles pending in SUMO are an artefact of insertion rather than latent demand | The drain condition is stricter than the physical one and over-states drain times for no control | Pending is latent demand in this environment by construction (`--max-depart-delay -1`) and is reported as a separate series so a reader can recompute drain times without it |
| Determinism breaks under the per-tick path | The equivalence check in attack point 1 fails and no feed number is comparable to the published ones | The check runs before any block; the per-tick path uses the same `setSpeed` calls at the same instants in `published` mode |
| Compute exceeds the budget | Blocks are cut | Blocks resume from their JSONL; E can be run on 10 seeds with the bars widening by `sqrt(1.5)` |

## Appendix: measurements made while planning (2026-10-15)

All read-only, scratch scripts under the session scratchpad `dsrc_feed/`.

**Drive-log lag.** Anchor: median of `t_wall - t_mono` over `metadata.jsonl` lines
(3,392 and 3,514 anchors; one line in each run is 10 s off, the rest within 50 ms).
Conversion of `received_t_mono` to wall time, minus `sourceUpdated`:

| run | responses | versions | query interval | first-receipt lag per version |
|---|---|---|---|---|
| `run_20260908_170849` | 93 | 30 | 16.7 s | 33.8 to 67.1 s, cycling with the query phase |
| `run_20260908_183538` | 2 | 2 | 60 s | 99.7, 99.8 s |
| `run_20260908_190548` | 28 | 28 | 60 s | 49.8 to 109.7 s, 75.7 to 76.0 s on 24 consecutive versions |

Over 60 versions: minimum 33.8 s, median 62.8 s, p90 75.9 s, maximum 109.7 s. On
`run_20260908_170849`, version 21:13:36Z was first returned 67.1 s after its stamp and
the query 16.7 s earlier returned the previous version, so its availability lag lies in
(50.4, 67.1] s; the next version, 21:14:36Z, lies in (23.6, 40.3] s; the intervals do
not overlap, so the lag differed by at least 10.1 s between consecutive versions. The
phone's wall clock differed from the Jetson's by 1.176 and 1.145 s on the two long runs
(`summary.json`, `phone/wall_clock_offset_s`).

**Cadence.** 60 versions; gaps 60 s x 48, 26 s x 4, 34 s x 4, 120 s x 1; phases :36 x 38,
:02 x 22; one switch :36 to :02 and back within `run_20260908_170849` and one :02 to :36
within `run_20260908_190548`.

**Smoke episode.** `here` checkpoint, seed 16, 2,500 s, 0.5 s step, gate on, libsumo
1.27.1, Python 3.12.14, commit `2c01189` (dirty: `paper/` edits only): 6.0 s wall for
the episode, 9.3 s for the process; flow 3,751.11 veh/h, arrivals 1,989, held 56,
pending 0, which equals the committed `here` row for seed 16 in
`results/evaluation/mainz_paired_seeds.jsonl`. The committed 60-episode evaluation took
403 s, 6.7 s per episode. Segment-decision space-mean speeds in the controlled arm: min
16.8, p10 32.2, median 47.3, max 58.6 km/h; the exit segment 11 reached density 0.146.

**Coordinates.** Vissim `refPointMap` (914742.74, 6433239.88) read as Web Mercator gives
49.9247 N, 8.2173 E; with `refPointNet` (-980.41, -1271.32), the SUMO `netOffset`
(5760.10, 3083.28) and the Mercator scale factor 1.5533 at that latitude, edge 218's
midpoint is 49.9312 N, 8.3091 E, entry 289's is 49.9600 N, 8.1529 E, and the network's
box is 49.908 to 49.964 N, 8.151 to 8.317 E. Drive query centres: `run_20260908_170849`
40.332 to 40.393 N, -74.655 to -74.564 W (centroid 40.3525, -74.6239);
`run_20260908_190548` 40.327 to 40.501 N, -74.646 to -74.402 W.

**Environment.** Venv has numpy 2.5.2, pandas 3.0.5, matplotlib 3.11.1, requests 2.34.2,
torch 2.14.0, pytest 9.1.1, eclipse-sumo/libsumo/traci/sumolib 1.27.1; no scipy, pwlf or
lifelines. Entry links: 277 and 289 in super-segment 0, 231 and 255 in 2, 140 and 182 in
5, 293 and 295 in 8; every route runs the chain to segment 11 and the exit.

## Amendment 1 (2026-10-15, after validation round 1; orchestrator)

**Why.** Validation round 1 (`plans/2026-10-15-feed-agreement-validation.md`) found that
controlled vehicles rear-end stopped vehicles and SUMO teleports the follower (R1-7). The
orchestrator reproduced it independently (`scratchpad/dsrc_feed/orch/collision_check.py`): on
seeds 16, 17 and 18 the committed controlled arm has 76, 96 and 113 collision teleports and no
control has 0; with `--collision.action warn` the controlled arm serves 3,606.7, 3,602.2 and
3,626.7 veh/h against no control's 3,722.2, 3,553.3 and 3,564.4, so the paired gain on these
seeds falls from a mean of +159 veh/h to -2 veh/h. Applying the advisory as a maximum-speed
cap (`setMaxSpeed`, the semantics of Vissim's `DesSpeed`, which SRC used) gives 0 collisions
and 0 teleports on all three seeds (`orch/maxspeed_check.py`); the existing checkpoint then
serves 3,635.6, 3,451.1 and 3,606.7 veh/h. The validator's alternative, an EIDM action step of
0.5 s, removes collisions but changes the human model the capacity drop rests on (no control
rises to 3,846.7 veh/h on seed 16), so it is not adopted. The committed +230 veh/h result
therefore depends on collision teleports, and every comparison below is re-based on a
collision-free actuation and retrained checkpoints.

**Changes, superseding the sections they name.**

- **A1 Actuation (all sections).** Every path that applies an advisory (`MainzEnv._command`,
  `command_vehicles`, and the feed modes) sets the vehicle's maximum speed with
  `setMaxSpeed(target)`; `release` restores the vehicle type's maximum speed. `setSpeed` is no
  longer used to apply advice. Every episode record and every 10 s trace carries the
  colliding-vehicle count (`simulation.getCollidingVehiclesNumber`, summed over steps) and the
  collision-teleport count (`simulation.getStartingTeleportNumber`); every block summary
  reports the number of controlled episodes with a nonzero count. Test: a controlled
  2,500 s episode on seed 16 under A1 records 0 collisions.
- **A2 Retraining (scope boundary, section 5).** Both checkpoints are retrained under A1 with
  `scripts/train_mainz_src.py` and the published protocol: 80 episodes, 2,500 s, step 0.5 s,
  flow window from 900 s, train seeds 1-10, select on 11-15, all other options at their
  defaults, `--features here` and `--features src`. The results go to new files
  (`results/checkpoints/mainz_here_cap_best.pt`, `mainz_src_cap_best.pt`, with their result and
  log files); the committed checkpoints stay. The four-arm paired evaluation on seeds 16-30 is
  re-run under A1 into `results/evaluation/mainz_paired_seeds_cap.{json,jsonl}`. Both run at the
  experiment stage, before any block, and every block uses the `here` cap checkpoint. Attack
  point 1 becomes: per-tick `published` mode equals the `step()` path under A1 on seeds 16-18
  exactly. The implementer adds whatever option the evaluator needs to read the cap
  checkpoints and to write the new evaluation file, and smoke-tests one training episode
  under A1.
- **A3 Drain rule (section 3.5, R1-4).** A run is drained at `t_d` only if, in addition to the
  plan's condition, no later sample up to the cap starts a jam spell or a waiting spell
  (held + pending > 0) lasting at least `H`; otherwise it is censored at the cap. Test: a
  trace clear at `t_0` that jams from `t_0 + 200 s` to the cap with pending is censored.
- **A4 Threshold (section 3.5, D10, R1-5).** The primary `v_jam` is one network-level
  threshold fitted on the sustained `mainz` scenario's `published` and `no_control` episodes,
  seeds 16-30, samples 300 to 2,500 s, under A1, and applied to both drain scenarios. Any fit,
  pooled or per segment, with `s2 >= 3 s1` (the 2023 paper's condition) is refused and the
  refusal recorded. The per-scenario no-control fit and the fixed 30 km/h remain reported
  sensitivities. Test: the surge-like pool (knots 3.5 and 25.0 km/h) is refused.
- **A5 Pilot rule (section 5, R1-6).** No control "breaks down" in scenario (b) if it has a
  jam spell of at least `H` after `t_0` under A4's threshold. The controlled arm must drain
  under A3 before the cap. The fall-backs are unchanged.
- **A6 Block B axis (section 3.3, Fig. 2, D25, R1-3).** The runner records, per second, the
  fraction of controlled vehicles on segment edges that apply the previous snapshot.
  `DELTAS = (0.05, 0.1, 0.25, 0.5)`. The F1 fit and Fig. 2(a) use the measured time-averaged
  fraction on the x-axis, labelled "fraction of vehicles on the previous snapshot".
- **A7 `inject_delta` writes (section 3.3, R1-2).** At each decision every scanned controlled
  vehicle is written its target version's fraction on its current edge, late or not.
- **A8 Pulses (section 3.6, R1-8, R1-9).** Each pulse vehicle is added to SUMO at its own
  departure tick, so pulsed and unpulsed runs are identical before the pulse (test: identical
  through 1,200 s). The regression window starts when the injection ends. The certificate
  artefact reports a noise floor as the validator's R1-8 entry proposes. The pulse-size check
  counts vehicles in teleport.
- **A9 Replay (section 3.1, R1-10).** Replay start indices are chosen so the replay never
  wraps; a test checks that every replayed stamp keeps its recorded second of the minute.
- **A10 Tests (R1-14).** The tests listed in the validation file's R1-14 entry are added so
  that the 16 surviving mutants are killed.
- **A11 Documented limits (R1-11, R1-12).** Command-level disagreement is about 20 times
  smaller than version-level disagreement because the action vector changes on 8 to 13 of 37
  decisions; the drive-log lower lag bound overstates `L_low` by up to one round trip. Both
  are stated in the paper.
