"""Would rescaling the speed bins produce a gradient signal? Measured, not argued.

The recommendation to rescale rests on a mechanism: the three values are 20, 27 and
30 m/s while the traffic runs at 7.3, so `setSpeed` -- an upper bound SUMO's
car-following dominates -- makes them equivalent on 97% of decisions. That mechanism
predicts that a rescaling which binds would produce a gradient above the noise floor.
This tests the prediction on four arms.

NOTHING IN THE REPOSITORY CHANGES. `_apply_speed` is replaced on the instance for the
duration of each arm, so the deployed contract is untouched and the arms differ only
in what the three action names command.

Every arm carries the same two controls as `scripts/measure_gradient_signal.py`: the
advantages shuffled across the batch is the floor, and an advantage built to
correlate with the action is what the instrument reads when signal is present.
"""
import sys, dataclasses, statistics
sys.path.insert(0, "/Users/ankit_nash/Desktop/ankit_summer_2026/dsrc")
sys.argv = ["x", "--training", "mappo_sumo"]
import torch
from scripts.train_policy import parse_args, load_training_bundle
from src.rl.trainers import TrainingConfig, make_trainer, seed_everything
from src.rl.ppo import PPOConfig
import src.sumo.env as env_module

cfg = load_training_bundle(parse_args())
base_t = TrainingConfig.from_mapping(cfg)
base_p = PPOConfig.from_mapping(cfg.get("training", cfg))
_sumo = env_module._sumo

OFFSETS = {"slow": -10.0, "nominal": -3.0, "fast": 0.0}
FACTORS_A = {"slow": 0.6, "nominal": 0.85, "fast": 1.0}
FACTORS_B = {"slow": 0.5, "nominal": 0.75, "fast": 1.0}
bind_events = {"binding": 0, "total": 0}


def make_apply(kind):
    def apply_speed(self, agent_id, bin_name):
        if bin_name is None:
            return
        name = str(bin_name)
        current = float(_sumo.vehicle.getSpeed(agent_id))
        allowed = float(_sumo.vehicle.getAllowedSpeed(agent_id))
        if kind == "shipped":
            target = max(12.0, allowed + OFFSETS[name])
        elif kind == "prevailing":
            # Option 1: the context is the speed traffic is actually holding, and
            # the 12 m/s floor is gone. `free_flow_speed_mps` is what the parameter
            # is called; the shipped env passes the lane limit instead.
            segment = self.get_segment_metrics().get(
                self.network.segment_for_edge(_sumo.vehicle.getRoadID(agent_id)) or "", {})
            prevailing = float(segment.get("mean_speed", current) or current)
            target = max(0.0, prevailing + OFFSETS[name])
        elif kind == "factor_a":
            target = max(0.0, current * FACTORS_A[name])
        else:
            target = max(0.0, current * FACTORS_B[name])
        bind_events["total"] += 1
        # The command binds when it is below the speed the vehicle is holding: only
        # then can the choice change anything.
        bind_events["binding"] += target < current - 1e-9
        self.command_speed(agent_id, target)
    return apply_speed


def gradient_norm(trainer, batch, advantages, clip):
    log_probs, _ = trainer.actor.evaluate_actions(batch.observations, batch.actions)
    ratio = torch.exp(log_probs - batch.old_log_probs)
    loss = -torch.min(ratio * advantages,
                      torch.clamp(ratio, 1 - clip, 1 + clip) * advantages).mean()
    trainer.actor.zero_grad(set_to_none=True)
    loss.backward()
    return float(torch.sqrt(sum((q.grad.detach() ** 2).sum()
                                for q in trainer.actor.parameters()
                                if q.grad is not None)))


original = env_module.SumoTopologyEnv._apply_speed
print(f"{'arm':>12} {'measured/floor':>15} {'per seed':>28} {'oracle/floor':>13} {'binds':>7}")
for kind in ("shipped", "prevailing", "factor_a", "factor_b"):
    env_module.SumoTopologyEnv._apply_speed = make_apply(kind)
    bind_events["binding"] = bind_events["total"] = 0
    ratios, ceilings = [], []
    try:
        for seed in (7, 17, 27):
            seed_everything(0)
            t = dataclasses.replace(base_t, rollout_steps=150,
                                    work_dir=f"/tmp/dsrc_bin_{kind}_{seed}")
            trainer = make_trainer(t, base_p, device="cpu")
            buf, _ = trainer.collect_rollout(seed=seed)
            batch = buf.compute_returns_and_advantages(
                gamma=base_p.gamma, gae_lambda=base_p.gae_lambda, group_by_agent=True)
            n = batch.advantages.shape[0]
            measured = gradient_norm(trainer, batch, batch.advantages, base_p.clip_coef)
            permutation = torch.randperm(n, generator=torch.Generator().manual_seed(seed))
            shuffled = gradient_norm(trainer, batch, batch.advantages[permutation],
                                     base_p.clip_coef)
            oracle = (batch.actions[:, 0] == 0).float() * 2.0 - 1.0
            oracle = (oracle - oracle.mean()) / (oracle.std() + 1e-8)
            ceilings.append(gradient_norm(trainer, batch, oracle, base_p.clip_coef)
                            / max(shuffled, 1e-12))
            ratios.append(measured / max(shuffled, 1e-12))
    finally:
        env_module.SumoTopologyEnv._apply_speed = original
    share = bind_events["binding"] / max(bind_events["total"], 1)
    print(f"{kind:>12} {statistics.fmean(ratios):>15.3f} "
          f"{str([round(r, 3) for r in ratios]):>28} "
          f"{statistics.fmean(ceilings):>13.1f} {share:>7.1%}", flush=True)
