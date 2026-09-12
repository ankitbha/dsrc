# Results

What the paper cites, kept small enough to live in the repository. Everything here is
reproducible from `scripts/`; it is committed so a number in the paper can be traced to
the run that produced it without re-running anything.

## `checkpoints/`

The two selected policies and the runs that produced them. Seeds 1-10 train, 11-15
select, 16-30 evaluate, 80 episodes, Mainz with the EIDM fleet and a three-into-two lane
drop at the exit, sustained 4,500 veh/h.

- `mainz_here_best.pt` — the traffic-API observation: speed, free flow, jam factor, lane
  count and length per super-segment.
- `mainz_src_best.pt` — SRC's original six features, four of which no vehicle can obtain.
- `*_result.json` — validation trajectory, the held-out test read, and the no-control
  baseline on the same seeds.
- `*_training.log` — the episode-by-episode record, including the degradation after the
  selected checkpoint that selecting on validation seeds exists to catch.

Paired on seed over seeds 16-30: the traffic-API arm gains 230 +/- 44 veh/h and SRC's
six features 224 +/- 61, and the two differ by 0.2%.

## `evaluation/`

The per-seed table those two figures are computed from, which did not exist until
2026-09-12: `mainz_paired_seeds.jsonl` is one line per episode over seeds 16-30 and four
arms, `mainz_paired_seeds.json` the summary, both from
`scripts/evaluate_mainz_checkpoints.py`. Measured: here 3,764.89 (+229.93 +/- 44.36), src
3,759.26 (+224.30 +/- 60.95), no control 3,534.96.

The fourth arm is the point of the file. `zeroed_head` loads the same checkpoint as `here`
with its output layer zeroed, and reads 2,846.67 veh/h -- 15.74 times no control's own
two-standard-error bar below it. **A result from an instrument never seen to fail is not
evidence**, so the falsification arm runs from the same command line and lands in the same
artefact as the arms it exists to bound.

## `flow_density/`

- `exit_supersegment.{png,json}` — the exit super-segment's flow-density curve under a
  metered exit. Peak 826 veh/h/lane at 17.0 veh/km/lane, falling 28%, with the loading
  and unloading halves of the rush agreeing to 1% where they overlap.
- `straight_road_w99.png`, `straight_road_krauss.png` — the same measurement on a plain
  two-lane road under both fleets. **The pair is the point**: Krauss produces the hill
  too, at 15%, and Krauss has no capacity drop. A link's `q = k v` hill is not evidence
  of one.

## `gates/`

The measurement that has to pass before a training run is worth making: served flow
falling as offered demand rises, by more than the seed spread.

- `mainz_w99_no_capacity_drop.log` — the paper's W99 calibration. Served flow is flat
  within 1.4% across a 3.6x density range at every exit setting. FAILS.
- `mainz_eidm_free_exit.log` — EIDM lifts capacity from 2,865 to 5,907 veh/h and flow is
  still flat, because link 218 ends into free outflow and a capacity drop needs a merge.
  FAILS.
- `mainz_eidm_lane_drop.log` — EIDM with a three-into-two drop at the exit. Peak 3,887
  veh/h falling to 3,449, a fall of 437 against a combined bar of 106. PASSES, and it is
  the only configuration in the leg that does.

## Not here

The USB latency campaign's raw logs are 82 MB and stay out; `plans/results_task42_43_47_usb_campaign.md`
carries its numbers. The generated SUMO network and route files are rebuilt by one
command from `data/mainz/Mainz20base.inpx`, so only the input is committed.
