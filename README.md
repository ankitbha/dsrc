# DSRC — deploying self-regulating cars

A phone-plus-Jetson advisory rig that runs a centrally trained traffic-control policy on
a vehicle, and the simulation that establishes what that policy is worth at scale.

`plans/paper_deploying_self_regulating_cars.md` is the argument and where every result
sits, including a map of which file to read for each section of the paper.
`plans/implementation_records.md` is what was built and what each step found.

`plans/detours.md` is everything the project did that the paper does not rest on, kept
rather than deleted. Read its warning before reading anything in it: it describes an
earlier formulation, and it is the easiest way to write the wrong paper.

## The deployment

- `phone/` — the Android app. Camera, GPS, IMU and the HERE query, each at the rate the
  Jetson commands, plus the driver display. The phone stays dumb and forwards raw data.
- `deployment/jetson/` — perception, fusion, policy inference, advisory decode and the
  sampling controller. All interpretation and control lives here.
- `specs/` — the wire protocol, the observation and action schemas, the controller
  contract. `transport_protocol.md` is the eight-channel link the two devices speak over.
- `src/safety/` — the safety and etiquette filters that bound the advisory.
- `scripts/run_*.py`, `with_device.py`, `watch_drive.py` — the drive and bench harnesses.

## The simulation

- `src/sumo/mainz.py` — the SRC network with the observation restricted to what a traffic
  API returns.
- `src/rl/src_q.py` — SRC's own Q-learning, ported.
- `scripts/build_mainz_scenario.py` — builds the scenario from the published `.inpx`.
- `scripts/train_mainz_src.py` — train, select on validation seeds, read the test seeds
  once.
- `scripts/measure_mainz_fundamental_diagram.py` — the gate that must pass before any
  training run: served flow has to fall as offered demand rises, by more than the seed
  spread, or there is no capacity drop for a controller to recover.
- `scripts/measure_fd_straight.py`, `measure_fd_open_road.py`, `measure_fd_exit.py` — the
  road-level flow-density measurements the fleet and the threshold were chosen from.

## What is not here

The highway-env topology ladder, the non-learning baselines, the MAPPO stack and the
`inverted_tree` networks were the project's earlier formulation. None of it is touched by
the paper. The code was removed on 2026-09-12 and is in the git history; the records of it
are in `plans/detours.md`.
