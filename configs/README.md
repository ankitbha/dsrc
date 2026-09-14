# Configs

What remains is the fleet. Every other config directory -- topology, demand, experiments,
training -- belonged to the highway-env topology ladder and the MAPPO formulation, and is
in the git history rather than here.

## `human_models/`

`eidm_reaction.yaml` is the fleet every simulation result uses. Its header carries the
measurements that chose it: the SRC paper's Wiedemann-99 calibration has no capacity drop
on SUMO, and without one served flow is a constant and no controller can recover
anything.

`w99_calibrated.yaml` is that calibration, kept because it is the paper's own and because
the comparison between the two is a result.

The network, the demand and the exit constraint are not configs. They are arguments to
`scripts/build_mainz_scenario.py`, which writes the scenario from the published `.inpx`.
