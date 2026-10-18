# Approach roads: replacing the entry gate with a network that extends upstream

Written 2026-10-17 by the planning agent, on branch `approach-roads` at `4f4720a` (branched from
`jam-avoidance`). Every design decision below was taken by recommendation without asking; the
decisions table records each one, and the open items list the decisions where the planner had no
real preference and the places where the code or the measurements contradicted the brief. Nothing
in this plan has been implemented. The appendix records the measurements and pilots made while
planning. All of them ran from the session scratchpad
(`scratchpad/dsrc_jam/approach_plan/`), on training seeds 1 and 2 only, and wrote nothing into the
repository.

## Short version

### Summary

The jam-avoidance program evaluates its controllers on the Mainz SUMO port with an entry gate. A
controlled vehicle whose entry link is at critical density waits off the map, and under no control
SUMO's own insertion backlog also waits off the map. Neither queue is on a road, so the jam metric
never sees it. Block M's two-layer controller on the 9,000 veh/h burst held a mean of 6,836
vehicle-minutes off the map per episode, and no control kept a mean of 5,208 vehicle-minutes in
SUMO's backlog. The author asked for the network to be simulated as if it extended much farther
back, so that every queue stands on a road where the metric and the controllers can see it. This
plan specifies that boundary, the changes that let the existing controllers act on it without
retraining or re-selection, the checks that the boundary behaves as intended, and the re-run of
blocks M, L, P and K into `results/jam_approach/`.

The boundary is one straight 3 km road in front of each of the eight entry links, built by the
same netconvert patch mechanism the builder already uses for the exit road. Each road has its
entry link's lanes, lane widths and speed limit, and is split into three 1 km edges, each of which
is a new super-segment, so the trace gains 24 approach super-segments beside the original 12. Every
vehicle is inserted at the far end by SUMO, with no gate in any arm. Its departure is moved earlier
by its approach's measured free-flow travel time, so that under free flow it reaches its entry link
at its original scheduled time. Every scenario time is shifted by a global offset of `W = 240 s`,
which is a multiple of 60 s, so the feed replay, the decision instants, the burst windows and the
incident times all keep their meaning relative to the network. The plan's Q-network still reads
only the original 12 x 5 state. A vehicle on an approach is shown the plan's fraction for the entry
super-segment that approach feeds. The camera correction, its sensing pass, the oracle and the feed
lookahead act on approach vehicles, and the lookahead reads the simulated feed's speed and jam
factor on the approach super-segments.

Five measurements made while planning shape the design.

- **A 3 km approach holds every queue measured in 21 pilot episodes.** Across no control, the
  plan, the lookahead, the rule correction and the oracle on the 9,000 veh/h burst (B9), the exit
  block and the sustained 4,500 veh/h demand, the longest contiguous queue on an approach was
  799 m, and SUMO's insertion backlog stayed at zero in every episode. For comparison, the largest
  number of vehicles the gate held at one instant in the gated runs was 207, which would occupy
  1,553 m of one lane at SUMO's 7.5 m per stopped vehicle (5 m length plus the 2.5 m minimum gap).
- **The approach does not remove the storage of vehicles upstream of the network. It moves that
  storage onto a road, where its cost can be measured.** Under the plan at full penetration on B9
  (training seed 1), the approaches held on average 107 more vehicles than they hold under free
  flow, which is 11,252 vehicle-minutes over the scored window. The reason is that the plan shows
  approach vehicles the entry segment's fraction, often 0.5, so they drive 3 km at 30 km/h instead
  of 60 km/h. Because 30 km/h is above the 19 km/h jam threshold, the jam metric does not count
  these vehicles. The plan therefore adds two measures that do count them: the approach excess
  vehicle-minutes and the system excess vehicle-minutes (section 5).
- **Free-flow travel time over 3 km differs between vehicles, so arrivals at the entry links are
  smoothed rather than exact.** SUMO gives each vehicle its own desired speed, with a 10% spread
  by default. After the per-entry shift, arrivals lag the schedule by -14 to +89 s (standard
  deviation 12 to 14 s), and the count of arrivals in each 300 s window matches the schedule to
  within 5 vehicles at every entry. On the one-lane approach to entry 255, vehicles form platoons
  behind slower ones at burst rates, so its share of the B9 burst arrives a mean of 40.5 s late.
- **One pilot suggests that the deficit at penetration 0.5 is not caused only by the gate.** On
  training seed 1, the rule correction at penetration 0.5 on B9 had 253.5 segment-minutes of jam
  on the original 12 segments, against 221.8 for no control, and held a mean of 1,226 vehicles
  in the network from the burst's start to the cap, against 1,115 for no control. This is one seed and is not a result, but the author
  should know it before reading block P (open item 1).
- **Cost.** A 7,200 s episode on the 3 km boundary took 1.15 to 1.23 times as long as the gated
  block M mean for the same arm and scenario. Blocks M, L, P and K are therefore estimated at
  12.1 hours in one process, plus about 30 minutes for a qualification block on training seeds
  that runs first (section 7).

### Scope boundary

In scope: the approach roads and their netconvert patch in the builder; the approach versions of
the five demand files (BG, B7.5, B9, B5 and S45); the travel-time calibration; the changes to
`MainzEnv`, the feed engine, the jam engine, the sensing pass, the lookahead and the predictor's
feature function that let the existing arms act on approach vehicles; the clock offset; the jam
metrics split into all, network and approach parts, together with the two storage measures; the
runner's `--boundary approach` mode and a qualification block Q; the comparison script; tests;
the experiment run of Q, M, L, P and K.

Out of scope: any change to the controllers' logic, thresholds, Q-network, predictor or camera
model; re-selection on the new boundary (blocks G, T and S are not re-run, and the runner refuses
them on the approach boundary); any change to the original 12 super-segments, to
`data/mainz/mainz.net.xml` or to the committed demand files; figures and the paper (the summaries
carry every number the author needs to decide what goes in); anything under `paper/`;
`scripts/poll_here_feed.py`, `scripts/analyze_feed_process.py`; any HERE request.

### Open items

1. **The deficit at penetration 0.5 may not be the gate's.** The author's diagnosis is that at
   penetration 0.5 the gate lets the network hold more vehicles than under no control. In one
   pilot on the approach boundary (training seed 1, B9), the rule correction at 0.5 still held
   more vehicles in the network than no control (a mean of 1,226 against 1,115 from the burst's
   start to the cap) and had more network jam (253.5 against 221.8 segment-minutes). If block P confirms this, the
   cause lies in the mixed fleet itself and not in the gate. The plan changes nothing because of
   this; it is listed so that the author reads block P with it in mind.
2. **The storage on the approaches is not in the jam metric.** Controlled arms slow approach
   vehicles to 30 or 45 km/h, which stores vehicles upstream much as the gate did. In the B9
   pilot under the plan this was 11,252 vehicle-minutes, against the gated plan's block M mean of
   9,035 held vehicle-minutes. The plan reports the approach excess vehicle-minutes and the system
   excess vehicle-minutes for every episode and pairs them across arms (section 5.3). Whether the
   paper's main comparison should charge this storage is the author's decision.
3. **The burst on the one-lane approach arrives late.** The shift is calibrated at background
   rates. At the B9 burst rate of 744 veh/h on the single lane of approach 255, vehicles form
   platoons behind slower vehicles, and the burst there arrives a mean of 40.5 s later than
   scheduled. Six of the other seven entries stay within 3.0 s; entry 295's 37.3 s comes from a
   queue on its approach, which is the effect being studied. Entry 255 carries 62 of the 748
   vehicles scheduled to reach the entry links in the burst window. The alternative is a
   rate-dependent shift calibrated separately on each burst file. Weak preference for the single
   shift.
4. **Each approach super-segment is 1 km long.** Jam time is counted in segment-minutes, so the
   length of the approach super-segments sets how much a given approach queue adds. At 1 km, a
   queue of up to 1 km adds one segment-minute per minute, which is close to the original
   segments' 0.13 to 2.3 km along-route lengths. The alternative is one 3 km super-segment per
   approach, which detects only queues longer than about 250 m (section 1.3). Weak preference.
5. **Take-the-slower and the learned correction on approach vehicles.** Take-the-slower re-runs
   the Q-network with the entry segment's row replaced by the camera's leader speed. The learned
   correction reads its "own" and "next" feed features from the approach super-segment and the
   next one along the route, and reads its plan fractions through the entry segment (section
   4.4). Neither arm is one of the paper's three. Weak preference for both.
6. **"Free-flow travel time" is read as the measured mean, not as length over speed limit.**
   3,000 m at 16.67 m/s is 180 s, but the measured means are 188.7 to 194.7 s, because SUMO's
   default speed spread gives half the vehicles a desired speed below the limit, while the vehicle
   type's 16.67 m/s maximum caps the other half. Shifting by 180 s would make the mean arrival 9 to
   15 s late at every entry.
7. **The brief's "up to 6,836 vehicle-minutes held per B9 episode" is the two-layer controller's
   mean, not its maximum.** Block M's B9 episodes held a mean of 6,836 and a maximum of 9,237
   vehicle-minutes under the rule correction, and a maximum of 10,530 under the plan alone.
   Nothing in the plan depends on which number is quoted.
8. **A qualification block on training seeds runs before block M.** The brief did not ask for
   it. It runs every block M arm, the plan and the rule at penetration 0.5, and the lookahead at
   zero lag, on B9, the exit block and S45, on training seeds 1 and 2 (60 episodes, about 31
   minutes). It exists so that an approach that is too short is found before 12 hours are spent
   (section 7.1).
9. **Isolated slow vehicles near the insertion point were not diagnosed.** Under the plan and the
   lookahead at penetration 1.0, 11 to 18 of the 744 samples in each 7,440 s pilot showed a
   controlled vehicle within 200 m of an approach's far end driving at 4.17 to 4.79 m/s, 2.5 to
   16.5 s after its insertion, with a 30 km/h cap, about half of them with no leader within
   100 m. They are not connected to the entry queue, they never caused an insertion backlog, and
   the queue measure excludes them by construction (section 5.4). The trace records them per
   approach so the validator can see them.

### Decisions table

All taken by recommendation. "Real" means the planner had a reasoned preference. "Weak" means the
alternative is defensible, and the matching open item is named.

| # | Decision | Taken | Alternatives not taken | Preference |
|---|---|---|---|---|
| A1 | Layout | One straight approach per entry link (eight), from the entry link's start node backward along the heading of the entry link's first shape segment; the last approach edge ends at the existing node `<entry>-begin` | A shared feeder road; curved roads | Real |
| A2 | Length | 3,000 m per approach | 2 km (less dispersion, 2.5 times the longest pilot queue); 5 km (sd of the arrival lag 22 s) | Real |
| A3 | Edges and super-segments | Three 1,000 m edges `a<entry>_0` (far end) to `a<entry>_2`; each edge is one approach super-segment, index `12 + 3j + i` with entry order j = 277, 289, 231, 255, 140, 182, 293, 295 and i = 0 at the far end; 24 approach super-segments in `data/mainz/mainz_approach_segments.json`; `mainz_segments.json` unchanged | One 3 km super-segment per approach | Weak (item 4) |
| A4 | Lanes, widths, speed | Copied from the entry link (per-lane widths included); priority nodes; explicit connections lane i to lane i | Uniform 3.75 m lanes | Real |
| A5 | Build | `add_approaches` in `scripts/build_mainz_scenario.py`, the same netconvert patch mechanism as `add_exit_constraint`, writing the new file `data/mainz/mainz_approach.net.xml` from the unchanged `mainz.net.xml`; patch inputs `data/mainz/approach.{nod,edg,con}.xml`; sha256 of both nets in the approach metadata and in every header | Rebuild from the `.inpx` | Real |
| A6 | Departures | Every schedule row keeps its id, type, route and position; far-end departure = scheduled departure + `W` - `T_A(entry)`; routes prefixed by the three approach edges; files `<name>_A.rou.xml`, `<name>_A_routes.rou.xml`, `<name>_A_schedule.json` for `mainz_bg`, `mainz_b7500`, `mainz_b9000`, `mainz_burst`, `mainz` | Re-sort the schedule by far-end departure (would change the controlled set at partial penetration) | Real |
| A7 | `T_A` | Per entry, the mean time from far-end departure to the first 0.5 s step on the entry link, under BG and no control on training seed 1; verified on seed 2 | 180 s from length over limit | Real (item 6) |
| A8 | Burst insertion | At the far end with the same shift as every other row | Part-way along the approach | Real; the cost is item 3 |
| A9 | Clock | `W = 60 ceil(max T_A / 60)`, expected 240 s; every scenario time + `W`; the feed replay's base stamp + `W`; warm-up `300 + W` | Negative SUMO begin time | Real |
| A10 | Gate | Removed in every arm: `gate_entries=False`, the full route file, SUMO inserts; `held` is 0 by construction and asserted; `pending` recorded | Keep the gate for controlled arms | Real |
| A11 | What an approach vehicle is shown | The plan's fraction for the entry super-segment (`plan_row`); the Q-network input stays 12 x 5 and is asserted | Free flow on approaches | Real |
| A12 | Feed on approaches | At each version's snapshot, `observe_approaches()` gives speed, free flow, jam factor, lanes and length for the 24 approach super-segments, by the same formula as `observe()`; read only by the lookahead and the learned arm | No feed on approaches | Real |
| A13 | Lookahead on approaches | Downstream chain per super-segment: the later approach super-segments, then the entry segment, then the following original segments; 0.5 when the first element of the chain is at or above `J_th`, 0.75 when the second is; on the original 12 identical to the current index rule | Lookahead off on approaches | Real |
| A14 | Camera, sensing, oracle | Egos on approach edges are sensed; the lane pass scans approach edges; an approach edge's route successor is the next approach edge or the entry link | Sensing on the original edges only | Real |
| A15 | Take-the-slower on approaches | Replaces the entry segment's row with the camera's leader speed | Off on approaches | Weak (item 5) |
| A16 | Learned arm on approaches | Own and next feed features from the approach super-segment and the first element of its chain; plan fractions through `plan_row`; geometry from the approach edge | Features of the entry segment | Weak (item 5) |
| A17 | Thresholds | `results/jam/S_summary.json` (sha256 `5bce02db...`), `onset_predictor.json` (`3b2256cd...`), `camera_error_model.json` (`fdf527cd...`), checkpoint (`941ac210...`) unchanged; no re-selection | Re-select on the approach boundary | Real (brief) |
| A18 | Jam metrics | Over all 36 super-segments as primary; the network part (0 to 11) and the approach part (12 to 35) reported beside it; drain requires `pending` 0; served flow at the network's exit as before | Network only | Real (brief) |
| A19 | Storage measures | Approach excess vehicle-minutes and system excess vehicle-minutes (section 5.3) in every record and summary | Report jam time only | Real |
| A20 | Queue measure and flag | Contiguous queue from the entry link of vehicles below 19.0 km/h, consecutive members at most 100 m apart, per approach every 10 s; an episode is flagged `approach_short` when any queue exceeds 2,000 m or `pending` is ever above 0 | Furthest slow vehicle (a pilot showed isolated slow vehicles at 4.99 km) | Real on the flag; weak on the 100 m gap |
| A21 | Qualification | Block Q on training seeds 1-2 before block M (section 7.1); on a flag, rebuild at 5,000 m, re-calibrate and re-run Q | Start block M directly | Real (item 8) |
| A22 | Matrix | Blocks M, L, P, K on seeds 16-30 with the gated program's arms, variants, scenarios and thresholds; results to `results/jam_approach/`, traces to `outputs/jam_approach/` | A reduced matrix | Real (brief) |
| A23 | Regression | On the gated boundary the refactored code reproduces committed block M records exactly | Trust the refactor | Real |
| A24 | Comparison | `scripts/compare_boundaries.py` pairs each approach row with its gated row by seed | Compare in prose | Real |
| A25 | Provenance | Run from a clean detached worktree at the implementation's final commit; results in the gitignored `outputs/jam_approach/run/`, copied to `results/jam_approach/` and committed when the chain ends, as the gated run was | Run in the working tree | Real |

## 1. The approach roads

### 1.1 Why a road and not a larger gate

The gate stood for the policy acting on a link above the entry that the scenario did not
simulate. Simulating that link removes the need for a stand-in. Once the link exists, a held
controlled vehicle becomes a slowed vehicle on a road, an uncontrolled vehicle behind it on a
one-lane road is slowed by car-following instead of passing it off the map, and a queue that spills
back from an entry link stands on road that the feed, the cameras and the jam metric all read. One
road per entry is the smallest layout that does this without adding a merge. Every entry link
starts at a dead-end node with no incoming edge (checked for all eight), so an approach attached
there joins nothing else.

### 1.2 Length

The length has to be long enough that the largest queue in any arm never reaches the insertion
point, because a queue that reaches the far end turns back into an off-map backlog in SUMO's
pending list, which is the thing the approach exists to remove. Two estimates bound it.

- **From the gated runs.** The largest number of vehicles held at one instant in any gated
  episode was 207 (block K, the rule correction at distance scale 0.8, B9). The largest SUMO
  backlog under no control was 107 (block M, B9). A stopped SUMO vehicle occupies 7.5 m of lane,
  so 207 vehicles in a single lane would be 1,553 m long. That is a worst case, because the held
  vehicles were spread over eight entries with one to three lanes each.
- **From pilots on a 3 km trial build (appendix A3).** In 21 episodes on training seeds 1 and 2,
  covering no control, the plan at penetration 1.0 and 0.5, the lookahead, the rule correction and
  the oracle on B9, the exit block and S45, the longest contiguous queue on any approach was
  799 m. The insertion backlog was 0 in every episode.

3,000 m is 1.9 times the single-lane worst case and 3.75 times the longest pilot queue. A longer
road costs fidelity in arrival times, because the spread of travel times grows with length: the
standard deviation of the arrival lag was 12 to 14 s at 3 km and 20 to 24 s at 5 km. Because no
estimate can cover every arm in advance, the runner also records each approach's queue and the
backlog in every episode, flags an episode whose queue reaches the insertion segment (section 5.4),
and block Q tests the worst arms before block M starts (section 7.1).

### 1.3 Edges and super-segments

Each approach is three 1,000 m edges, `a<entry>_0` at the far end, then `a<entry>_1`, then
`a<entry>_2`, which ends at `<entry>-begin`. The nodes are `a<entry>_n0` to `a<entry>_n2`, placed
at 3,000, 2,000 and 1,000 m before the entry link's start. Each edge is its own super-segment. The
entry order j = 0 to 7 is 277, 289, 231, 255, 140, 182, 293, 295, and approach edge i of entry j
has super-segment index `12 + 3j + i`. The 24 approach super-segments are written, in index
order, to `data/mainz/mainz_approach_segments.json`, and `mainz_segments.json` is not touched.

The 1 km length is chosen for two reasons. First, the jam indicator uses the vehicle-weighted
space-mean speed of a super-segment, and queued vehicles stand close together, so a short queue
dominates the mean. A calculation for a two-lane approach at 496 veh/h, with queued vehicles
stopped at 7.5 m per vehicle per lane and the rest at 57 km/h, gives a mean below 19 km/h for a
queue of 100 m on a 1 km super-segment, but needs a queue of about 250 m on a single 3 km
super-segment. Second, jam time counts segment-minutes, and at 1 km an approach super-segment is
close to the original segments' 0.13 to 2.3 km along-route lengths, so a minute of approach jam
weighs about what a minute of network jam weighs.

Each approach has its entry link's lane count, per-lane widths and speed limit. The lane counts
are 2, 3, 2, 1, 3, 3, 2, 2 lanes for entries 277, 289, 231, 255, 140, 182, 293, 295, and every
speed limit is 16.67 m/s. Per-lane widths matter because the camera's lateral offsets are read
from them; approach 295, for example, has lanes of 3.00 and 3.25 m. The nodes are priority nodes,
and the connections are given explicitly, lane i to lane i, as the exit patch does. On the trial
build netconvert produced internal lanes of 0.1 m between approach edges and of 1.75 to 5.63 m at
the entry nodes.

### 1.4 How they are added to the net

`add_approaches(source_net, out_net, length_m=3000, segment_m=1000)` in
`scripts/build_mainz_scenario.py` writes `data/mainz/approach.nod.xml`, `approach.edg.xml` and
`approach.con.xml`. It then runs netconvert with `--sumo-net-file data/mainz/mainz.net.xml` and the
three files, writes `data/mainz/mainz_approach.net.xml`, and removes the build timestamp with
`drop_build_time`. This is the same mechanism `add_exit_constraint` uses. The pure part,
`approach_geometry(net, entries, length_m, segment_m)`, returns the node, edge and connection
records and the metadata, so it can be tested without netconvert. The CLI gains `--approach`,
`--approach-length-m` and `--approach-segment-m`. `mainz.net.xml`'s sha256 (`037de791...`) is
checked before and after the build, and the build refuses to write when it changes.

On the trial build, every one of the 414 original edges kept its lane count, lane lengths, speeds
and shapes, and every one of the 767 original connections was unchanged. The only changes were the
eight entry start nodes, which became priority junctions instead of dead ends, and one lane shape
of edge 289, which netconvert rewrote as "-0.00" where the original had "0.00". The test compares
coordinates as numbers, so this formatting difference does not count as a change.

The builder writes `data/mainz/mainz_approach.json`, which holds, per entry: the edges, lanes,
widths, speed and edge lengths; the super-segment indices; the entry's super-segment; `plan_row`
for all 36 indices; the downstream chain for all 36 indices; `T_A` and `W` (added in the second
pass, section 2.2); the sha256 of both nets; and the approach length and edge length.

## 2. Demand, insertion and the clock

### 2.1 Departures at the far end

The schedule of record is the gated one, so the approach demand is derived from the committed
schedule files rather than rebuilt from the `.inpx`. For each of `mainz_bg`, `mainz_b7500`,
`mainz_b9000`, `mainz_burst` and `mainz`, `write_approach_demand(name, T_A, W)` reads
`<name>_schedule.json` and `<name>_routes.rou.xml`. It then writes three files.

- `<name>_A_schedule.json`, with the rows in their original order. Each row keeps `id`, `type`,
  `route` and `entry`; `depart` becomes the far-end departure `depart + W - T_A(entry)`; and
  `entry_time = depart + W` is added. Keeping the order keeps every vehicle's position, and the
  position decides whether the vehicle is controlled (`controlled_index`) and which query phase it
  draws (`draw_phases`). The controlled set and the phases are therefore the gated runs' at every
  penetration.
- `<name>_A.rou.xml`, with every vehicle sorted by far-end departure (SUMO reads route files in
  departure order), `departSpeed="max"` and `departLane="best"`, as before.
- `<name>_A_routes.rou.xml`, with the vehicle types and the eight routes, each prefixed by its
  three approach edges.

A row whose far-end departure would be negative is refused. Each file set is registered in
`scripts/train_mainz_src.py` `SCENARIOS` as `<name>_A`, with `net` set to
`data/mainz/mainz_approach.net.xml` and an `approach` key naming `data/mainz/mainz_approach.json`.

### 2.2 The travel time `T_A` and the two-pass build

The first pass builds the net and provisional demand with `T_A = 180 s` for every entry.
`scripts/calibrate_approach_travel.py` then runs BG under no control on training seed 1 for
7,200 s plus `W`. It records, at every 0.5 s step, the first step at which each vehicle is on its
entry link, and writes `data/mainz/mainz_approach_travel.json` with, per entry, the number of
vehicles and the mean, median, standard deviation and 5th and 95th percentiles of the travel time
from far-end departure to the entry link, together with the seed and the sha256 of the files it
ran. The second pass rebuilds the demand files with the per-entry means as `T_A` and writes `T_A`
and `W` into `mainz_approach.json`. The pilot means were 188.7 to 190.3 s on the seven multi-lane
approaches and 194.7 s on the one-lane approach 255 (1 s polling, appendix A2). The verification on
seed 2 is a boundary check (section 6).

The mean is used, not the median, because the check compares arrival rates. With the mean shift,
the mean arrival lag is close to zero (seed 2: -1.46 to +1.21 s per entry), so the cumulative
arrival curve at each entry follows the schedule's.

### 2.3 Burst vehicles enter at the far end

Burst rows are shifted by the same rule as every other row, and nothing is inserted part-way along
an approach. There are three reasons.

1. A burst vehicle then enters the map the way every other vehicle does, so it passes the approach
   super-segments that the feed reports and the cameras watch, and a slowed controlled vehicle on
   a one-lane approach holds back burst vehicles behind it.
2. Inserting part-way puts a vehicle into a moving stream, where SUMO needs a safe gap at the
   insertion position. When there is no gap, SUMO delays the vehicle into its pending list. That
   is an off-map backlog part-way along the road, which is what the approach exists to remove.
3. The pilot did not show the problem that part-way insertion would solve. The largest burst
   rate per lane is 744 veh/h, on the single lane of approach 255, against the 1,877 veh/h per
   lane that the EIDM fleet carries at critical density on this network (`mainz.py`). Under no
   control on B9, seed 1, the burst-window arrivals at the entry links matched the schedule to
   within 4% at seven of the eight entries (0% at entry 255). Entry 295 received 25% fewer,
   because a queue from segment 8 stood on its approach, and that queue is the effect being
   studied.

The cost is that entry 255's share of the burst arrives a mean of 40.5 s late (open item 3).

### 2.4 The clock

Every time the program uses is moved later by `W`, so that a time means the same thing relative
to the network as it did on the gated boundary. `W` must be a multiple of 60 s. The feed replay
draws its start index from the versions whose remaining gaps cover the episode, and it moves its
first stamp to the recorded second of the minute (`FeedProcess.from_artefact`). When both the
replay's base stamp and its end move by a multiple of 60 s, the same start index is drawn and
every stamp moves by exactly `W`. A multiple of 60 s is also a multiple of the 10 s query period
`P_q`, the 5 s sensing tick and the 10 s trace spacing, so all of these stay aligned. The rule is
`W = 60 ceil(max_entry T_A / 60)`, which gives 240 s for the pilot's `T_A` (maximum 194.7 s). The
first far-end departure is then at about 48 s.

| quantity | gated | approach (W = 240 s) |
|---|---|---|
| warm-up end, control start, first decision | 300 s | 540 s |
| decision instants | 300, 360, ... | 540, 600, ... |
| feed replay base stamp (`FEED_START_S`) | 60 s | 300 s |
| flow window start (`window_start_s`) | 900 s | 1,140 s |
| scenario clock start (`start_s`) | 900 s | 1,140 s |
| perturbation end (`end_s`) | 1,500 (B5), 1,200 (B7.5, B9, I_exit, I_mid), 900 (BG, S45) | 1,740; 1,440; 1,140 |
| incident request | 900 s, standing 300 s from insertion | 1,140 s, standing 300 s from insertion |
| caps | 7,200; 3,600 (BG); 2,500 (S45) | 7,440; 3,840; 2,740 |
| selection window of `jam_time_min_900_3600` | [900, 3,600] | [1,140, 3,840] |
| precursor baseline of `leads` | [600, 900) | [840, 1,140) |
| smoke cap, smoke window | 600, 300 | 840, 540 |

The scenario table `jam_episode.SCENARIOS` keeps its gated times. A `Boundary` object carries `W`,
and `run_jam_episode` adds it in one place. Traces stay on SUMO's clock, and every record carries
`W`, so that `t - W` is the gated clock. Summaries report onset and drain relative to the scenario
start, as before.

## 3. No gate

The approach replaces what the gate stood for, so the gate is removed in every arm, including the
controlled ones. `_env_for` builds `MainzEnv` with `gate_entries=False` and the full route file
(`routes`, not `gated_routes`) for every arm on the approach boundary. SUMO then inserts every
vehicle at the far end at its scheduled time. `_load_schedule` leaves the per-entry queues empty,
so `_admit` only places incidents, `held_at_entry()` returns 0 and `metrics()["admitted"]` returns
-1. The episode asserts `held == 0` at every trace sample. `MainzEnv` still reads the schedule file
for the departure positions, so `controlled()` is unchanged. `pending()`, SUMO's insertion
backlog with incident vehicles left out, is recorded at every 10 s sample and is expected to be 0.
Any positive value flags the episode (section 5.4).

## 4. Control on the approaches with the existing methods

### 4.1 The index space

The env and the engines use one index space of 36 super-segments: 0 to 11 are the original
segments and 12 to 35 are the approach super-segments. Two tables from `mainz_approach.json`
translate between this space and what the controllers already understand.

- `plan_row[s]` is the Q-network row that super-segment `s` reads. It is `s` for `s < 12`, and the
  entry segment for an approach super-segment: 0 for 277 and 289, 2 for 231 and 255, 5 for 140
  and 182, and 8 for 293 and 295.
- `downstream[s]` is the ordered list of super-segments a vehicle on `s` passes next. For `s < 12`
  it is `[s+1, ..., 11]`, which holds because every route visits the original segments in
  increasing, contiguous order (checked for all eight routes). For approach super-segment
  `12 + 3j + i` it is the later approach super-segments of the same entry, then the entry segment,
  then the original segments after it. For example, `a277_0` (index 12) has `[13, 14, 0, 1, ..., 11]`
  and `a293_2` (index 32) has `[8, 9, 10, 11]`.

With no approach file, both tables are the identity on 0 to 11 and the env behaves exactly as it
does now.

### 4.2 The plan

The Q-network is not changed and still reads `observe()`, which stays 12 x 5 and is computed from
the original 12 segments only. The engine asserts the shape at every snapshot. A controlled vehicle
on approach super-segment `s` is written `a_k[plan_row[s]]` times its edge's free-flow speed, at
every 5 s refresh and at every version switch, exactly as a vehicle on the entry segment is. In
code, `FeedEngine._fraction(version, s)` reads `vectors[version][plan_row[s]]`, and
`FeedEngine._measure` maps segments through `plan_row` before `command_disagreement`. Without that
second change, `command_disagreement` would index the action vector at 12 or above, and in
rule mode it runs at every tick of the scored window.

### 4.3 The feed lookahead

At each version's snapshot, `JamEngine._snapshot` also stores `env.observe_approaches()`, a 24 x 5
array of speed, free flow, jam factor, lanes and length computed by `observe()`'s formula for the
approach super-segments. It is taken at the same tick as the 12 x 5 state, so the two arrays
describe the same instant. The real feed covers these roads, so the lookahead may read them. The
Q-network never reads them. For a vehicle on super-segment `s` applying version `k`, with
`J = concat(states[k][:, 2], approach_states[k][:, 2])` and `c = downstream[s]`, the lookahead
shows 0.5 when `c` is non-empty and `J[c[0]] >= J_th`, 0.75 when `c` has a second element and
`J[c[1]] >= J_th`, and 1.0 otherwise. `J_th = 2` is read from `S_summary.json`.
`LookaheadRule.fraction_chain` implements this. On the original 12 it must equal the existing index
rule for every jam-factor vector, and a unit test checks this on 1,000 random vectors.

The lookahead already feeds on itself inside the network. A segment it slows reads a jam factor of
5 at 30 km/h, which makes the segment upstream of it slow down too. On the approaches the same
effect continues for up to three more super-segments. This is the existing method applied to the
extended road, and its cost appears in the storage measures.

### 4.4 The corrections, the camera and the oracle

The sensing pass scans every lane of every edge in all 36 super-segments and of their route
successors (`SnapshotBuilder` reads `env.all_segments`). `JamEngine._segment_of` covers the
approach edges, so controlled vehicles on approach edges are egos. The successor map comes from the
approach route file, so a camera near the end of an approach edge sees across the junction onto the
next approach edge or onto the entry link. As before, internal junction lanes are not scanned.

- **Rule correction and oracle.** These do not depend on the segment. An approach vehicle's shown
  fraction is `min(a_k[plan_row[s]], step_down(a_k[plan_row[s]]))` while active, and the oracle
  reads its true occupancy in the same 60 m window as the camera.
- **Take-the-slower.** The re-run replaces row `plan_row[s]`, the entry segment, with the camera's
  mean leader speed, and shows `min(a_k[plan_row[s]], a'[plan_row[s]])` (open item 5).
- **Learned correction.** `engine_features` uses `s` and `n = downstream[s][0]` (or `s` when the
  chain is empty, which reproduces `min(e + 1, 11)` on the original 12). The feed's speed and jam
  factor for the own and next super-segment come from the extended 36-row snapshot, the plan's
  fractions come through `plan_row`, and the segment length, lane count and remaining distance come
  from the vehicle's own edge. A unit test checks that the function returns the current feature
  vector for every original segment (open item 5).

Every write still goes through `shown_fraction`, the counter of writes above the plan still covers
every write, and `_check` reads the cap back on approach edges too.

### 4.5 What stays unchanged

The thresholds and models are the selected ones: lookahead `J_th = 2`, rule `m_on = 0`, learned
`theta = 0.75`, best model the rule correction, from `results/jam/S_summary.json` (sha256
`5bce02db2330c63ac873c306988c6c961ef59dadb1a9703b65da9f07c016dc05`). The predictor is
`results/jam/onset_predictor.json` (`3b2256cd...`), the camera model is
`results/jam/camera_error_model.json` (`fdf527cd...`), the feed artefact and session are
`results/jam/feed_process_afternoon.json` and `afternoon_20261016_mainz` (`8f5430bf...`), and the
checkpoint is `mainz_here_best.pt` (`941ac210...`). `read_selection` already refuses a selection
built on another feed artefact, camera model or predictor. On the approach boundary the runner
also requires the selection file's sha256 to equal the value above. These were selected on the
gated boundary, and selecting again on the approach boundary is out of scope. Every summary states
this.

## 5. Metrics

### 5.1 The trace

Every 10 s the trace records the 36-column per-super-segment speed, density and count, and the
36-column plan mean, shown mean, active share, `k_hat_mean` and `k_true_mean`. It also records
`held` (always 0), `pending`, `running`, `arrived`, and the collision counters. Three groups of
fields are new: `on_approach` (vehicles on approach edges); `queue_m`, eight values in entry order
(section 5.4); and `slow_near_insertion`, eight counts of vehicles below 19 km/h within 200 m of
each far end.

### 5.2 Jam time, onset, prevented and drain

`jam_metrics.episode_metrics` gains a `columns` argument and an `offset_s` argument (`W`). The
runner calls it three times per episode and stores the results as `jam` (all 36), `jam_network`
(0 to 11) and `jam_approach` (12 to 35), and does the same for the 30 km/h sensitivity. The
definitions do not change: the 19.0 km/h threshold read from `results/feed/drain_thresholds.json`,
120 s spells, onset at the first such spell at or after the clock start, prevented meaning no such
spell by the cap, and jam time in segment-minutes. Jam time is additive over columns, so `all =
network + approach` holds exactly, and a test checks this. Drain uses `drain.drain_time`
unchanged. It requires every one of the 36 super-segments to be clear for 120 s and `held +
pending` to be 0. With `held` always 0, that means `pending` must be 0. `leads` is computed on the
all-segment onset (primary) and on the network onset, with the baseline shifted by `W`. Served flow
is the arrivals per hour at the network's exit (the end of `exit_tail`) over the flow window, as
before. Routes still end there, so the measurement point does not move.

### 5.3 The storage measures

Jam time does not charge a vehicle that is slowed to 30 km/h, and on the gated boundary held
vehicles were not charged either. Two measures charge them.

- **Approach excess vehicle-minutes**: the integral over `[start_s + W, cap + W]` of `on_approach(t)
  - N_ff(t)`, where `N_ff(t)` is the number of vehicles whose far-end departure is at or before
  `t` and whose departure plus `T_A(entry)` is after `t`. `N_ff` is the free-flow occupancy of the
  approaches, which follows from the schedule alone. Under no control the integral is close to zero
  (pilot: -54 on BG, +18 on B9 and -132 on the exit block). Under the plan on B9 it was 11,252. It is the approach
  boundary's counterpart of the gated boundary's held vehicle-minutes.
- **System excess vehicle-minutes**: the integral over the same window of `running + held +
  pending - N_ff`, where `N_ff = 0` on the gated boundary. Demand is the same in every arm of a
  scenario, so a lower value means fewer vehicle-minutes spent in the network, on the approaches
  above free flow, and waiting to enter. Because the definition is the same on both boundaries, the
  comparison script can put the two side by side.

### 5.4 The queue on each approach, and the flag

`env.approach_report(v_queue)` is called at every trace sample with `v_queue` = 19.0 km/h. For
each approach it lists the vehicles below `v_queue` with their distance upstream of the entry
link's start, sorts them by distance, and walks outward from 0: the queue ends at the first
vehicle more than 100 m beyond the previous member (or beyond 0, for the first one). `queue_m` is
the distance of the last member. The contiguity rule exists because a pilot that took the furthest
slow vehicle reported queues of 4.99 km. Those were isolated vehicles near the insertion point
(open item 9), and only a 200 m stretch next to the entry was an actual queue. The episode record
carries `queue_max_m` per approach and `pending_max`. It also carries `approach_short = True` when
any `queue_m` exceeds 2,000 m, that is, when the queue reaches the insertion super-segment, or when
`pending` is ever above 0. Summaries count flagged episodes per row.

## 6. Checks that the boundary does what it should

| check | how | tolerance | pilot value |
|---|---|---|---|
| Arrivals reproduce the schedule under free flow | BG, no control, training seeds 1 and 2, step-level entry crossing | per entry, mean of (arrival - `entry_time`) within +/-3 s; per 300 s window on `[W, cap - 300 s]`, absolute difference from the scheduled count at most max(6, 0.15 x scheduled); every vehicle whose `entry_time` is at or before `cap - 300 s` reaches its entry link | seed 2: mean -1.46 to +1.21 s; largest window difference 5 vehicles; 0 of 6,848 missing |
| The burst reaches the entries | B9, no control, seeds 1 and 2 | per entry whose approach shows no queue in `[900 + W, 1,200 + W)`, arrivals in that window within 10% of the scheduled count; all eight recorded with mean lags | seed 1: seven entries within 4% (255 at 0%, with a 40.5 s mean lag); 295 -25%, with a queue on its approach |
| Car-following holds an uncontrolled vehicle behind a slowed controlled one on a one-lane approach | SUMO sanity test on `a255_0`: an `av` leader inserted at 0 s and capped at 0.5 x 16.67 m/s at 5 s; a `human` follower inserted at 6 s at maximum speed | from 30 s after insertion to 90 s, follower speed within 0.5 m/s of the leader's; gap at least 2.5 m; the follower never passes | follower 8.19 to 8.60 m/s against 8.34 m/s from 20 s on, front positions about 15 m apart |
| No collisions | every episode of every block and test, all arms including no control | colliding vehicles 0, collision teleports 0 | 0 in all 23 pilot episodes |
| The original network is unchanged | parse `mainz.net.xml` and `mainz_approach.net.xml` | every original edge's lanes, lengths, speeds, widths and shape coordinates equal as numbers; every original connection equal; only the eight `<entry>-begin` nodes changed; `mainz.net.xml` sha256 `037de791...` and `mainz_segments.json` sha256 `a6f82397...` unchanged | met on the trial build |
| The camera model and the models are unchanged | sha256 | as in section 4.5 | not applicable |
| The gate is gone | episode records and traces | `held` 0 at every sample, `admitted` -1, `gate_entries` False recorded in every record | not applicable |
| The controlled set and phases are the gated runs' | approach schedule against gated schedule | identical `controlled_index` sets at penetration 0.5, 0.75 and 1.0 for all five demand files | not applicable |
| The feed replay is the gated replay moved by `W` | `FeedProcess.from_artefact` at base 60 and at 60 + W, seeds 16-30, end 7,380 and 7,620 | stamps differ by exactly `W`; lags, start index and wraps equal | not applicable |

## 7. The experiment matrix

### 7.1 Order, with the qualification first

1. **Calibration.** Section 2.2, training seed 1, then the verification checks on seeds 1 and 2.
   About 1 minute.
2. **Block Q, qualification of the boundary.** Training seeds 1 and 2. It runs every block M arm
   at penetration 1.0 (no control, lookahead `J2`, plan, take-the-slower, rule `m0`, learned
   `th0.75`, oracle at the camera's 60 m window), the plan and the rule at penetration 0.5, and
   the lookahead at `m = 0`. The lookahead at zero lag held the most vehicles in the network in
   block L. The scenarios are B9 and I_exit (cap 7,200 s) and S45 (2,500 s), all plus `W`. That is
   10 settings x 3 scenarios x 2 seeds = 60 episodes, about 31 minutes. **Pass rule:** no episode
   is flagged `approach_short`, and no episode has a collision. **On failure,** the orchestrator
   rebuilds at 5,000 m without asking (`--approach-length-m 5000`; `W` re-derived, 360 s expected),
   re-runs the calibration and block Q, and records the failure. Block Q chooses no threshold and
   no arm.
3. **Blocks M, L, P, K** on seeds 16-30, in one process chain, with the gated program's arms,
   variants and scenarios (`arm_list` unchanged) and the thresholds of section 4.5. A flagged
   episode is kept and reported. If any episode in a block is flagged, the orchestrator stops the
   chain after that block and reports to the author, because the remedy is to lengthen the
   approach, and that changes every block.
4. **Comparison.** `scripts/compare_boundaries.py --gated results/jam --approach
   results/jam_approach` writes `results/jam_approach/boundary_comparison.json`. Per block, arm,
   variant and scenario it contains: the gated network jam time; the approach boundary's jam time
   for all, network and approach parts; prevented counts; served flow; held plus pending
   vehicle-minutes on the gated boundary against approach excess plus pending vehicle-minutes; and
   system excess vehicle-minutes on both. Differences between the boundaries are paired by seed.
   The paper's three-arm contrasts on the approach boundary (the two-layer controller against no
   control and against the lookahead) are paired over seeds 16-30 for jam time (all and network),
   served flow and system excess vehicle-minutes.

### 7.2 Matrix and compute

The per-episode wall times are from the 3 km pilots with sensing on approach egos, compared with
block M's gated means for the same arm and scenario: B9 no control with passive camera 37.0 s
against 30.7 s; rule 42.8 against 35.8; oracle at 60 m 37.8 against 30.8; I_exit rule 35.7 against
29.5; S45 rule 13.2 against 11.5. The ratios are 1.15 to 1.23, and the budget uses 1.25 because the
pilot lacked the queue sampler and the approach feed rows.

| block | arms and variants | scenarios | seeds | episodes | gated wall | budget |
|---|---|---|---|---|---|---|
| Q | 10 settings (section 7.1) | B9, I_exit, S45 | 1-2 | 60 | none | 0.5 h |
| M | no control, lookahead J2, plan, take-the-slower, rule m0, learned th0.75, oracle 60 m; oracle 200 m on B9 and I_exit | B5, B7.5, B9, I_exit, I_mid, BG, S45 | 16-30 | 765 | 5.40 h | 6.75 h |
| L | lookahead J2, plan, rule m0; `m` in {0, 2, 4} | B9, I_exit | 16-30 | 270 | 2.46 h | 3.08 h |
| P | plan, rule m0; penetration 0.75, 0.5 | B9, I_exit | 16-30 | 120 | 1.02 h | 1.28 h |
| K | rule m0; `b` 0.8, 1.25; pair spread x 0.5 | B9, I_exit | 16-30 | 90 | 0.82 h | 1.03 h |
| total | | | | 1,305 | | 12.6 h |

If the measured times exceed the budget by more than 20% after the first 30 episodes of block M,
the orchestrator cuts in the gated program's order: block K, then penetration 0.75 in block P, then
S45 in block M, then `m = 0` and then `m = 4` in block L. Each cut is recorded in the summary.
Each trace sample holds eight per-segment lists, which grow from 12 to 36 values, so traces are
about three times the gated size: about 3.3 GB for M to K against the gated 1.1 GB, in the
gitignored `outputs/jam_approach/` (705 GB free on the disk).

### 7.3 Provenance, resume and pause

The run starts from a clean detached worktree at the implementation's final commit, so headers
carry the commit and `dirty: false`. Results go to `outputs/jam_approach/run/` (gitignored), with
the pause file `outputs/jam_approach/run/PAUSE`. When the chain ends, the results are copied to
`results/jam_approach/` and committed, as the gated run's were. Nothing is ever written into
`results/jam/`. Each header adds the following to what the gated headers carried: `boundary`,
`W`, the approach length and edge length, `T_A`, and the sha256 of `mainz_approach.net.xml`,
`mainz_approach.json`, `mainz_approach_segments.json`, `mainz_approach_travel.json` and every
`_A` demand file. `header_identity` gains a fifth element, the approach net's sha256 (None on the
gated boundary, so existing gated files keep their identity). `--resume` therefore refuses to mix
boundaries in one JSONL. `--resume` and the pause behave as before.

## 8. Step table for the implementer

Test-driven: unit tests first (concrete inputs and outputs, no SUMO), then code, then sanity tests
(short SUMO episodes). Each test file runs in its own process with
`PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -B -m pytest -q tests/<file> -p no:cacheprovider`,
never with `-n`. Check `memory_pressure` before a SUMO test. Smoke episodes use seed 1 and an
840 s cap (600 s plus `W`). The implementer runs no experiment block. The existing test files must
pass unchanged.

| step | files | functions | unit tests (concrete I/O) | sanity tests (SUMO) |
|---|---|---|---|---|
| S0 files not to edit | `paper/`, `scripts/poll_here_feed.py`, `scripts/analyze_feed_process.py`, `data/mainz/mainz.net.xml`, `mainz_segments.json`, every committed demand file, everything under `results/jam/` | none | none | none |
| S1 approach net | `scripts/build_mainz_scenario.py` | `approach_geometry`, `add_approaches`, `--approach`, `--approach-length-m`, `--approach-segment-m` | a synthetic entry at (0, 0) heading +x gives nodes at x = -3,000, -2,000, -1,000; edges `a<e>_0..2` with the entry's lanes, per-lane widths and speed; connections lane i to lane i; indices `12 + 3j + i`; `plan_row` and `downstream` match the two examples of section 4.1; a negative length is refused | built net: the network-unchanged check of section 6; each approach edge 1,000 +/- 0.05 m; every approach route connected (`check_connected`); `mainz.net.xml` sha256 unchanged |
| S2 approach demand | `scripts/build_mainz_scenario.py`, `scripts/train_mainz_src.py` (`SCENARIOS`) | `approach_rows`, `write_approach_demand`, `approach_offset` | far-end departure = depart + W - T_A; order and ids kept; a negative departure is refused; `approach_offset(194.67) = 240`, `approach_offset(240.5) = 300`; `.rou.xml` sorted by departure; controlled sets equal the gated ones at 0.5, 0.75 and 1.0 for all five files; row counts 7,144, 7,468, 7,594, 12,144, 3,124 | each `_A` file loads in SUMO; a 840 s BG no-control episode inserts at the far ends with `pending` 0 |
| S3 calibration | `scripts/calibrate_approach_travel.py` | `entry_crossings`, `travel_stats` | crossing times from a synthetic per-step road-id sequence; mean, median and percentiles on a hand list | seed 1 calibration writes `mainz_approach_travel.json`; the second pass rebuilds the files; the arrival check of section 6 passes on seeds 1 and 2; the burst check on seeds 1 and 2 |
| S4 env | `src/sumo/mainz.py` | `MainzEnv(approach_file=...)`, `approach_segments`, `all_segments`, `plan_row`, `downstream`, `observe_approaches`, `approach_report`, `contiguous_queue` | `contiguous_queue([0, 50, 140, 300, 2500], gap=100)` = 140; empty = 0; a single vehicle at 120 m = 0; with no approach file every method returns what it returns now, with the same shapes and the same values on a fake | 840 s plan episode: `observe()` is 12 x 5; `segment_state` has 36 columns; approach vehicles are listed by `controlled_on_segments` with indices 12-35; `held` 0 throughout; the one-lane follower test of section 6 |
| S5 engines | `src/sumo/feed_episode.py`, `src/sumo/jam_episode.py`, `src/sumo/correction.py`, `src/sumo/sensing.py`, `src/rl/onset_predictor.py` | `Boundary`, `GATED`, `approach_boundary`, `_env_for(boundary)`, `run_jam_episode(..., boundary=)`, `FeedEngine._fraction` and `_measure` via `plan_row`, `JamEngine._snapshot`, `_writes`, `_decide`, `_record_jam`, `LookaheadRule.fraction_chain`, `SnapshotBuilder` over `all_segments`, `engine_features` via `plan_row` and `downstream` | `fraction_chain` equals `fractions` on the 12 for 1,000 random vectors; with every approach super-segment below `J_th`, an approach vehicle on `a140_2` is shown 0.5 when segment 5 is at or above `J_th`, and one on `a140_1` is shown 0.75; `_fraction(k, 27)` = `vectors[k][5]`; the feed replay at base 300 equals the gated replay plus 240 for seeds 16-30; `engine_features` on the 12 equals the current function on a stub engine | **regression on the gated boundary**, through the runner's episode function with the committed checkpoint, feed artefact, camera model and predictor: `M/rule/m0/B9/16` gives flow 3,676.0 veh/h, 6,685 arrivals, 5.6667 jammed segment-minutes, 1,215,428 writes, active share 0.309509...; `M/no_control/-/B9/16` gives 3,509.14 veh/h, 6,442 arrivals, 185.5 segment-minutes and 6,374.0 pending vehicle-minutes, all exactly; approach boundary, 840 s B9 smoke per arm: 0 collisions, 0 writes above the plan, `pending` 0, the write log shows approach vehicles written `a_k[plan_row]` (rule and lookahead their own), camera egos include approach vehicles |
| S6 metrics | `src/analysis/jam_metrics.py` | `select_columns`, `episode_metrics(columns=, offset_s=)`, `free_flow_occupancy`, `approach_excess_minutes`, `system_excess_minutes`, `queue_flags` | on a hand trace with 36 columns, jam time all = network + approach; a jam only on column 20 gives onset on 20 and `prevented` False for all, True for the network part; `N_ff` on a three-vehicle schedule; the flag fires at 2,001 m and at `pending` 1 and not at 1,999 m | none |
| S7 runner | `scripts/run_jam_experiments.py` | `--boundary`, block Q, `BLOCK_SEEDS["Q"] = (1, 2)`, headers, `header_identity`, `summarize_eval` with three parts, storage, flags, collisions over all arms | Q refuses seed 3; M refuses seed 1; G, T, S refused on the approach boundary; Q refused on the gated boundary; a selection whose sha256 is not `5bce02db...` is refused on the approach boundary; resume refuses a JSONL whose headers ran the gated boundary; a stub block summarises the three parts and counts flags | `--smoke --boundary approach --block Q M L P K` runs one 840 s episode per arm into `outputs/jam_approach/smoke/` |
| S8 comparison | `scripts/compare_boundaries.py` | `pair_rows`, `main` | stub gated and approach rows pair by seed; a missing seed is listed, not dropped silently | none |
| S9 record | `plans/implementation_records.md` (append) | none | none | the S7 smoke per block, the S3 calibration and checks, with wall times |

## 9. Validator attack points

1. **The gate is really gone.** Every record has `gate_entries` False, `held` 0 at every trace
   sample and `admitted` -1. No vehicle is added by `_admit` except incidents. A mutant that builds
   the env with the gate on the approach boundary must fail a test.
2. **Approach demand timing.** Recompute every far-end departure from the committed schedule,
   `W` and `T_A`. Recompute `T_A` from the calibration run's raw crossing times. Re-run the arrival
   and burst checks on seeds 1 and 2. Confirm that the controlled sets equal the gated ones and
   that `.rou.xml` and `_schedule.json` hold the same vehicles.
3. **Approach super-segments are in every metric.** `jam` = `jam_network` + `jam_approach` in jam
   time on every record; recompute onset, prevented and drain from a trace on all 36 columns; a
   trace edited so that only an approach jams changes `jam` and not `jam_network`.
4. **The lookahead and the corrections act on approach vehicles.** From a per-write log on one B9
   episode of each of the lookahead, rule, take-the-slower, learned and oracle arms: approach
   vehicles receive writes; every written fraction is at most `a_k[plan_row]` (the lookahead's
   own value for the lookahead); the lookahead's approach fractions recompute from the version's
   36-row snapshot and `downstream`; corrected arms step down approach vehicles; and the sensing
   egos include approach edges, with candidates across the approach-to-entry junction.
5. **`pending` is recorded and is zero.** It is in every trace sample and record, and no episode
   is flagged. Recompute `queue_m` from a trace sample's vehicle positions.
6. **Seeds.** Blocks M to K hold only seeds 16-30; block Q and the calibration only 1 and 2. The
   selection file is the committed one (`5bce02db...`), and no G, T or S record exists under
   `results/jam_approach/`.
7. **Collisions.** Zero colliding vehicles and zero collision teleports in every episode of every
   block, no control included. The gated block P had one collision (the plan at penetration 0.75,
   B9, seed 23), so the mixed fleet is where to look.
8. **The clock.** Every constant in the table of section 2.4 is shifted by `W` and by nothing
   else. The feed stamps of a record equal the gated record's stamps plus `W`, version by version,
   for the same seed. The decision instants start at `300 + W`.
9. **The original network and the models are unchanged.** Run the numeric net comparison, the
   sha256 of `mainz.net.xml`, `mainz_segments.json`, the camera model, the predictor, the
   checkpoint and the selection, and the 12 x 5 assertion on the Q-network input.
10. **Regression.** The gated-boundary records of step S5 are reproduced exactly.
11. **Where results go.** Nothing is written under `results/jam/`; the approach results and
    summaries are under `results/jam_approach/` and the traces under `outputs/jam_approach/`.
12. **Provenance.** Headers carry the commit, `dirty: false`, `W`, `T_A` and every approach sha256,
    and the footers say the tree did not change during the run.
13. **Storage measures.** Recompute the approach excess and system excess vehicle-minutes of
    three records from their traces and the schedule.
14. **The one-lane follower test** exists, runs SUMO, and fails when the leader is not capped.
15. **Timing.** Per-episode walls are within a factor of 2 of section 7.2.

## 10. Risks

| risk | what it would do to the comparison with the gated results | what this plan does |
|---|---|---|
| The approach is too short for some arm, so a queue reaches the insertion point | The backlog returns off the map in that arm, and its approach jam time is understated against the others | Queue and `pending` recorded every 10 s; flag; block Q on the worst arms before block M; rebuild at 5 km on a Q failure; stop after a block with a flag |
| Travel-time spread smooths arrivals, and the one-lane approach delays the B9 burst by about 40 s | Burst-driven onsets on the approach boundary start a little later than on the gated boundary, so onset differences between the boundaries under about 40 s are not interpretable | Mean shift per entry; arrival and burst checks with stated tolerances; per-entry lags recorded; open item 3 |
| Controlled arms store vehicles on the approaches by slowing them | Jam time can improve against the gated results while delay does not; a reader who sees only jam time would credit the controller with what is storage | Approach excess and system excess vehicle-minutes in every record and paired in the comparison (open item 2) |
| The thresholds were selected on the gated boundary | The approach results use thresholds that may not be the best ones on this boundary, so a worse result here does not show that the method is worse | Stated in every summary; re-selection out of scope |
| The lookahead's self-induced slowing extends 3 km upstream | The lookahead loses more flow on BG than on the gated boundary, which widens its gap to the two-layer controller for a reason unrelated to warning time | Storage measures; BG flow reported per arm |
| The p = 0.5 deficit persists (open item 1) | Block P on the approach boundary would not support the claim that the gate caused it | Reported as measured; no design change |
| A collision in the mixed fleet | Breaks the zero-collision requirement for that arm | Counted in every block; not diagnosed in this plan |
| The approach super-segment length sets the size of approach jam time | Approach jam-minutes are not the same unit as network jam-minutes, so the all-segment total depends on the 1 km choice | Network and approach parts reported separately; network part directly comparable with gated jam time (open item 4) |
| The refactor changes the gated path | Every comparison with the gated records would carry an unknown offset | Exact regression on two committed records (step S5) |
| Compute overrun | Sweeps are cut | Cut order of section 7.2; resume and pause |
| Isolated slow vehicles near the insertion point (open item 9) | A small, undiagnosed contribution to approach jam time in controlled arms | Recorded per approach; excluded from the queue by contiguity |
| Same seeds, different trajectories | Differences between the boundaries paired by seed are descriptive, not a controlled comparison of one mechanism | Stated in the comparison file |

## Appendix: measurements made while planning (2026-10-17)

All read-only on the repository. Scripts and outputs are in
`scratchpad/dsrc_jam/approach_plan/` (`build_trial_net.py`, `build_trial_demand.py`, `pilot.py`,
`pilot_sense.py`, `arrivals.py`, `compare_nets.py`, `sanity_follow.py`). Environment: `.venv` with
SUMO and libsumo 1.27.1. Every pilot used training seeds 1 or 2.

**A1. The gated runs (committed traces under `outputs/jam/run_c03cca5/traces/`).** The largest
number held at one instant was 207 (K, rule `b0.8`, B9), 187 in block M (plan and take-the-slower,
B9), and 174 in block L (lookahead `m = 0`, B9). The largest SUMO backlog under no control was 107
(M, B9). Mean held vehicle-minutes on B9 in block M: plan 9,035 (maximum 10,530), take-the-slower
8,534, learned 8,694, lookahead 6,900, rule 6,836 (maximum 9,237), oracle 5,668; no control's mean
pending vehicle-minutes 5,208 (maximum 6,374). Mean vehicles in the network after 900 s on B9: no
control 1,039, plan 1,048, lookahead 1,182, rule 980; in block P at penetration 0.5, plan 1,138
and rule 1,177.

**A2. Entry links and travel time.** Every entry link starts at a dead-end node with no incoming
edge. Lanes 2, 3, 2, 1, 3, 3, 2, 2 and lengths 360, 334, 372, 524, 161, 1,685, 84, 182 m for
entries 277, 289, 231, 255, 140, 182, 293, 295, all at 16.67 m/s. Per-entry background rates are
496, 496, 296, 296, 596, 596, 396, 396 veh/h; B9 burst rates are 1,248, 1,248, 744, 744, 1,500,
1,500, 996, 996 veh/h. On a 3 km trial build with a provisional 180 s shift (W = 240 s), BG under
no control on seed 1 gave mean travel times of 189.90, 189.58, 188.70, 194.67, 190.32, 189.42,
190.28 and 189.44 s for entries 140, 182, 231, 255, 277, 289, 293 and 295 (detected at 1 s
resolution). After the demand was rebuilt with these values, seed 2 gave mean lags of -1.46 to
+1.21 s, a standard deviation of 12.4 to 14.3 s, lags from -14.2 to +89.4 s, and at most 5
vehicles of difference in any 300 s window; all 6,848 due vehicles reached their entry links. On
the 5 km build the standard deviation was 20 to 24 s.

**A3. Pilots on the 3 km build.** These used a pilot environment in the scratchpad, not the planned
code. Approach vehicles received the entry segment's plan fraction. In the pilots marked
"no sensing" the camera did not run. In the lookahead pilots, approach vehicles used the entry
segment's lookahead chain instead of the approach chain of section 4.3. "Queue" is the contiguous
queue of section 5.4. "Network jam" is jammed segment-minutes on the original 12 over
`[1,140, cap]`.

| scenario | arm | seed | penetration | wall s | longest queue (approach) | network jam | mean vehicles in the network after 1,140 s | approach excess veh-min |
|---|---|---|---|---|---|---|---|---|
| B9 | no control (no sensing) | 1 | 1.0 | 27.6 | 227 m (295) | 221.8 | 1,115 | +18 |
| B9 | no control, passive camera | 1 | 1.0 | 37.0 | 227 m (295) | 221.8 | 1,115 | |
| B9 | plan (no sensing) | 1 | 1.0 | 34.7 | 709 m (182) | 68.8 | 1,028 | +11,252 |
| B9 | plan (no sensing) | 2 | 1.0 | 34.5 | 448 m (295) | 112.2 | 1,053 | |
| B9 | plan (no sensing) | 1 | 0.5 | 31.8 | 257 m (182) | 151.2 | 1,071 | +5,098 |
| B9 | plan (no sensing) | 2 | 0.5 | 31.2 | 174 m (295) | 159.7 | 1,079 | |
| B9 | lookahead J2 (no sensing) | 1 | 1.0 | 38.8 | 210 m (295) | 9.0 | 1,077 | +15,736 |
| B9 | rule m0, camera | 1 | 1.0 | 42.8 | 799 m (182) | 1.3 | 959 | +10,579 |
| B9 | rule m0, camera | 1 | 0.5 | 43.0 | 200 m (295) | 253.5 | 1,226 | |
| B9 | oracle 60 m | 1 | 1.0 | 37.8 | 799 m (182) | 2.5 | 928 | |
| I_exit | no control (no sensing) | 1 | 1.0 | 23.2 | 0 | 151.0 | 913 | -132 |
| I_exit | plan (no sensing) | 1 | 1.0 | 25.0 | 767 m (140) | 13.0 | 756 | +5,546 |
| I_exit | plan (no sensing) | 2 | 1.0 | 25.8 | 740 m (140) | 13.8 | 782 | |
| I_exit | plan (no sensing) | 1 | 0.5 | 27.8 | 0 | 89.8 | 936 | |
| I_exit | lookahead J2 (no sensing) | 1 | 1.0 | 36.5 | 109 m (277) | 24.0 | 1,001 | |
| I_exit | rule m0, camera | 1 | 1.0 | 35.7 | 767 m (140) | 8.0 | 822 | +6,607 |
| S45 | no control (no sensing) | 1 | 1.0 | 7.6 | 0 | 17.8 | 942 | |
| S45 | plan (no sensing) | 1 | 1.0 | 9.6 | 116 m (295) | 3.3 | 943 | |
| S45 | plan (no sensing) | 1 | 0.5 | 9.0 | 0 | 10.3 | 968 | |
| S45 | lookahead J2 (no sensing) | 1 | 1.0 | 10.9 | 112 m (255) | 0.0 | 932 | |
| S45 | rule m0, camera | 1 | 1.0 | 13.2 | 142 m (293) | 0.0 | 998 | |

Every one of these episodes, and both BG episodes, had zero colliding vehicles, zero collision
teleports, an insertion backlog of 0 at every sample, and (where counted) zero writes above the
plan. These are pilots on two training seeds and are not results. Block M's evaluation seeds were
not used.

**A4. The original network on the trial build.** 414 of 414 original edges and 767 of 767
original connections unchanged; only the eight entry start nodes changed (dead end to priority);
one formatting difference, "-0.00" against "0.00", in a lane shape of edge 289. Approach edges
measured 999.99 to 1,000.01 m.

**A5. The one-lane follower.** On `a255_0`, an `av` leader inserted at 0 s at maximum speed and
capped at 8.335 m/s at 5 s fell to 7.67 m/s and then held 8.34 m/s. A `human` follower inserted at
6 s at 16.67 m/s was at 11.25 m/s by 16.5 s and between 8.19 and 8.60 m/s from 20.5 s to 116.5 s,
with its front about 15 m behind the leader's, and did not pass it.

**A6. Isolated slow vehicles near the insertion point.** Under the plan on I_exit, seed 1, the
first 30 logged vehicles below 19 km/h within 200 m of a far end were all controlled, capped at
8.34 m/s, aged 2.5 to 16.5 s since insertion, and driving at 4.17 to 4.79 m/s; about half had no
leader within 100 m. A lone capped vehicle in the follower test fell only to 7.67 m/s, so the cap
transition alone does not explain them. Not diagnosed (open item 9).

## Orchestrator record (2026-10-17)

- **Block Q passed** (worktree at `5cea9d3`, 60 episodes on seeds 1 and 2): no approach-short flag (longest
  approach queue 832.4 m, `Q/oracle/tau10_m0_R60/B9/1` on approach 182, against the 2,000 m limit;
  corrected after validation round 2, R2-2: an earlier draft of this record said 767 m, which came from sorting the
  per-approach queue lists by their first element), `pending` 0, zero collisions, teleports, writes above the plan
  and read-back mismatches.
- **Validation round 1** (findings in `plans/2026-10-17-approach-roads-validation.md`): no defect in
  production code; every quantity checked against the real block Q records by independent recomputation;
  the gated path reproduces every field on seed 3 at both commits. Six SHOULD FIX findings are missing tests
  (13 surviving mutants); the orchestrator re-ran mutant JE1 and it survives `test_approach_engines.py`.
- **Decision:** because the round 1 findings are test-only, blocks M, L, P and K start at once from the same
  clean worktree at `5cea9d3` that produced block Q, while the implementer adds the killing tests in a
  test-only batch (`src/` and `scripts/` unchanged since `5cea9d3`, checked in round 2). If a new test shows a
  production defect, the blocks it affects are re-run.

- **Validation round 2** (6fc1639, tests only; `src/`, `scripts/` and `data/` unchanged since `5cea9d3`): no
  production finding; all 13 round 1 survivors killed by the new tests; the double flow-window marking opens the
  window late only under `--smoke` (window at exactly 900 s in all 1,695 committed gated records and at 1,140 s
  in the 75 approach records then written); R2-1 (leads clock-start and cap shift not pinned by a test; no effect
  on 22 real traces) not fixed; R2-2 corrected above. The validation cap of two rounds is reached.
