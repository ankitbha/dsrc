"""Is the branch imbalance structural or emergent? Same demand, different seeds.

Measured at 2400 veh/h with no control: leaves a1-a3 carry a queue (density ratio
0.16 to 0.25) while a4-a6 run nearly empty (0.031), although the route writer splits
the demand evenly and the network is structurally symmetric -- b1 and b2 both zipper
into c with identical lane connections.

If the same branch loses on every seed it is structural and a defect. If which branch
loses varies, it is merge bistability: whichever side falls behind first is slower,
and being slower makes it lose more of the zipper.
"""
import sys, statistics
sys.path.insert(0, "/Users/ankit_nash/Desktop/ankit_summer_2026/dsrc")
from src.config.loaders import load_named_config
from src.sumo.env import SumoTopologyEnv

print(f"{'seed':>5} {'b1':>7} {'b2':>7} {'a1-a3':>8} {'a4-a6':>8} {'trunk':>7} {'served':>7}")
for seed in (7, 17, 27, 37, 47):
    config = {
        "topology": load_named_config("topology", "inverted_tree"),
        "demand": load_named_config("demand", "sumo_capacity_drop"),
        "human_model": load_named_config("human_model", "w99_calibrated"),
        "duration_steps": 6000, "dt": 0.1, "warmup_steps": 3000,
        "work_dir": f"/tmp/dsrc_asym_{seed}",
    }
    env = SumoTopologyEnv("inverted_tree", config)
    env.reset(seed=seed)
    acc = {}
    try:
        for _ in range(6000):
            _, _, _, _, info = env.step({})
            for name, ratio in info["density_ratios"].items():
                acc.setdefault(name, []).append(ratio)
        mean = {k: statistics.fmean(v) for k, v in acc.items()}
        left = statistics.fmean(mean[f"tree_leaf_a{i}"] for i in (1, 2, 3))
        right = statistics.fmean(mean[f"tree_leaf_a{i}"] for i in (4, 5, 6))
        print(f"{seed:>5} {mean['tree_middle_b1']:>7.3f} {mean['tree_middle_b2']:>7.3f} "
              f"{left:>8.3f} {right:>8.3f} {mean['tree_trunk_c']:>7.3f} "
              f"{env.arrived_total:>7}")
    finally:
        env.close()
