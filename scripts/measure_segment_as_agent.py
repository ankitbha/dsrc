"""If the agent were the LINK rather than the vehicle, would the gradient appear?

The paper's controller is attached to the road: one action per super-segment per
decision, applied to every vehicle on it. This project's is attached to a vehicle, so
co-located AVs act independently and out of phase, and at a mean travel time of 250
to 300 s a one-minute lever leaves each agent four or five decisions in its life.

This measures the paper's formulation directly, WITHOUT changing the trainer. One
action is sampled per segment per decision from a representative agent's observation
and applied to every AV on that segment, and one transition is recorded per
(segment, decision). The effective agent is the link, realised by whichever vehicles
happen to be on it -- and nothing reads a global state, so execution stays
decentralized.

Same instrument and same two controls as tasks 105 to 110: the floor is the measured
advantages permuted, estimated from 60 permutations; the ceiling is a synthetic
advantage built to correlate with the action, which shows the statistic can move.
"""
import sys, dataclasses, statistics, collections
sys.path.insert(0, "/Users/ankit_nash/Desktop/ankit_summer_2026/dsrc")
import torch
from src.config.loaders import load_named_config
from src.rl.trainers import TrainingConfig, make_trainer, seed_everything
from src.rl.encoders import encode_local_batch
from src.rl.ppo import PPOConfig
from src.rl.rewards import build_threshold_reward
from src.rl.rollout_buffer import RolloutBuffer

PERMUTATIONS = 60
DECISIONS = 20


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


cfg = load_named_config("training", "mappo_src")
env_keys = ("topology", "demand", "human_model", "duration_steps", "dt")
bundle = {"training": cfg, "env": {k: cfg[k] for k in env_keys if k in cfg}, "seed": 7}
base_t = TrainingConfig.from_mapping(bundle)
base_p = PPOConfig.from_mapping(cfg)

print(f"{'seed':>5} {'segment-decisions':>18} {'measured':>10} {'floor':>9} {'sd':>8} "
      f"{'z':>7} {'ceiling':>9}")
zs = []
for seed in (7, 17, 27):
    seed_everything(0)
    t = dataclasses.replace(base_t, rollout_steps=DECISIONS,
                            work_dir=f"/tmp/dsrc_segagent_{seed}")
    trainer = make_trainer(t, base_p, device="cpu")
    env = trainer.build_env()
    repeat = trainer.action_repeat()
    buffer = RolloutBuffer()
    try:
        observations, _ = env.reset(seed=seed)
        for _ in range(DECISIONS):
            agent_ids, obs_tensor = encode_local_batch(observations)
            if not agent_ids:
                observations, _, _, _, _ = trainer.hold_action(env, {}, repeat)
                continue
            segment_of = {s.vehicle_id: s.segment_id for s in env.vehicle_snapshots()}
            # One REPRESENTATIVE agent per segment: the first in sorted order, which
            # is a rule that does not depend on the action about to be taken.
            representative = {}
            for index, agent_id in enumerate(agent_ids):
                segment = segment_of.get(agent_id)
                if segment and segment not in representative:
                    representative[segment] = index
            if not representative:
                observations, _, _, _, _ = trainer.hold_action(env, {}, repeat)
                continue
            rows = sorted(representative.values())
            with torch.no_grad():
                actions, indices, log_probs, _ = trainer.actor.sample(obs_tensor[rows])
                value_obs = trainer.value_observation_tensor(
                    env.get_global_state(), obs_tensor[rows], len(rows),
                    trainer.privileged_features(env, [agent_ids[r] for r in rows]))
                values = trainer.critic(value_obs)
            by_segment = {seg: k for k, seg in enumerate(
                s for s, _ in sorted(representative.items(), key=lambda kv: kv[1]))}
            # EVERY AV on a segment gets that segment's one action.
            action_map = {}
            for agent_id in agent_ids:
                segment = segment_of.get(agent_id)
                if segment in by_segment:
                    action_map[agent_id] = actions[by_segment[segment]]
            nxt, terminated, _, infos, _ = trainer.hold_action(env, action_map, repeat)
            reward = base_p.reward_scale * sum(
                build_threshold_reward(i.get("segment_metrics", {}),
                                       i.get("density_ratios", {}),
                                       **dict(base_t.threshold_reward or {}))
                for i in infos)
            reward = max(-base_p.reward_clip, min(base_p.reward_clip, reward))
            for segment, k in by_segment.items():
                buffer.add(observation=obs_tensor[rows[k]], action=indices[k],
                           log_prob=log_probs[k], reward=reward, value=values[k],
                           done=False, value_observation=value_obs[k],
                           agent_id=segment)
            observations = nxt
    finally:
        env.close()

    batch = buffer.compute_returns_and_advantages(
        gamma=base_p.gamma, gae_lambda=base_p.gae_lambda, group_by_agent=True)
    n = batch.advantages.shape[0]
    measured = gradient_norm(trainer, batch, batch.advantages, base_p.clip_coef)
    generator = torch.Generator().manual_seed(seed)
    floor = [gradient_norm(trainer, batch,
                           batch.advantages[torch.randperm(n, generator=generator)],
                           base_p.clip_coef)
             for _ in range(PERMUTATIONS)]
    mean, sd = statistics.fmean(floor), statistics.pstdev(floor)
    ceiling_adv = (batch.actions[:, 0] == 0).float() * 2.0 - 1.0
    ceiling_adv = (ceiling_adv - ceiling_adv.mean()) / (ceiling_adv.std() + 1e-8)
    ceiling = gradient_norm(trainer, batch, ceiling_adv, base_p.clip_coef) / max(mean, 1e-12)
    z = (measured - mean) / max(sd, 1e-12)
    zs.append(z)
    print(f"{seed:>5} {n:>18} {measured:>10.5f} {mean:>9.5f} {sd:>8.5f} {z:>+7.2f} "
          f"{ceiling:>9.1f}", flush=True)

se = statistics.stdev(zs) / len(zs) ** 0.5
print(f"\nsegment-as-agent, {len(zs)} seeds: mean z {statistics.fmean(zs):+.2f} +/- {se:.2f}")
print(f"vehicle-as-agent, same config (task 115): -1.82, +0.21, third seed pending")
