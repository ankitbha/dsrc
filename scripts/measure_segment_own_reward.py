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

THE NULL IS THE RESAMPLED-ACTION ONE, not the permutation floor this script first
used. Permuting advantages across the batch destroys the action-advantage pairing AND
each agent's temporal profile, and a segment agent holds about nineteen consecutive
decisions whose advantages are smooth -- shuffling makes the sequence rough, a rough
sequence cancels less against `grad log pi`, and the floor comes out high. Measured
that way these arms read z of -1.18 and -1.75, a value systematically BELOW its own
null, which is the signature of a null that is not exchangeable with the data. Those
readings are void. See `scripts/measure_action_alignment.py` and its tests.
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

sys.path.insert(0, "/Users/ankit_nash/Desktop/ankit_summer_2026/dsrc/scripts")
from measure_action_alignment import alignment_z, gradient_norm  # noqa: E402

RESAMPLES = 60
DECISIONS = 20


cfg = load_named_config("training", "mappo_src")
env_keys = ("topology", "demand", "human_model", "duration_steps", "dt")
bundle = {"training": cfg, "env": {k: cfg[k] for k in env_keys if k in cfg}, "seed": 7}
base_t = TrainingConfig.from_mapping(bundle)
base_p = PPOConfig.from_mapping(cfg)

print(f"{'seed':>5} {'segment-decisions':>18} {'measured':>10} {'null':>9} {'sd':>8} "
      f"{'z':>7} {'ceiling':>9}")
zs = []
for seed in (7, 17, 27):
    # THE NETWORK INITIALISATION VARIES WITH THE SEED, not just the traffic.
    # With `seed_everything(0)` the actor and critic weights were bit-identical
    # across every seed -- verified by comparing parameters -- so a statistic that
    # depends on the initialisation had ONE draw and its cross-seed standard error
    # understated the uncertainty. It is how the segment arm came to read
    # -1.32 +/- 0.16: a systematic actor-critic relationship, repeated three times.
    seed_everything(seed)
    t = dataclasses.replace(base_t, rollout_steps=DECISIONS,
                            work_dir=f"/tmp/dsrc_segown_{seed}")
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
            # EACH SEGMENT IS PAID ITS OWN TERM, not the network sum. The paper's
            # reward is a sum of per-segment terms and its agent is centralized, so
            # the sum is the right credit there. For a per-SEGMENT agent it is not: a
            # shared network reward is moved by one segment's action by about a
            # ninth, which is the same dilution that defeated the per-vehicle
            # formulation, one level up. Verified a genuine decomposition: a clear
            # segment reads +1.000 and a jammed one -0.850, summing to the network's
            # +0.150.
            weights = dict(base_t.threshold_reward or {})
            for segment, k in by_segment.items():
                own = base_p.reward_scale * sum(
                    build_threshold_reward(
                        {segment: i.get("segment_metrics", {}).get(segment, {})},
                        {segment: i.get("density_ratios", {}).get(segment, 0.0)},
                        **weights)
                    for i in infos)
                own = max(-base_p.reward_clip, min(base_p.reward_clip, own))
                buffer.add(observation=obs_tensor[rows[k]], action=indices[k],
                           log_prob=log_probs[k], reward=own, value=values[k],
                           done=False, value_observation=value_obs[k],
                           agent_id=segment)
            observations = nxt
    finally:
        env.close()

    batch = buffer.compute_returns_and_advantages(
        gamma=base_p.gamma, gae_lambda=base_p.gae_lambda, group_by_agent=True)
    n = batch.advantages.shape[0]
    z, measured, mean, sd = alignment_z(
        trainer, batch.observations, batch.actions, batch.old_log_probs,
        batch.advantages, clip=base_p.clip_coef, resamples=RESAMPLES, seed=seed)
    ceiling_adv = (batch.actions[:, 0] == 0).float() * 2.0 - 1.0
    ceiling_adv = (ceiling_adv - ceiling_adv.mean()) / (ceiling_adv.std() + 1e-8)
    ceiling = gradient_norm(trainer, batch.observations, batch.actions,
                            batch.old_log_probs, ceiling_adv,
                            base_p.clip_coef) / max(mean, 1e-12)
    zs.append(z)
    print(f"{seed:>5} {n:>18} {measured:>10.5f} {mean:>9.5f} {sd:>8.5f} {z:>+7.2f} "
          f"{ceiling:>9.1f}", flush=True)

se = statistics.stdev(zs) / len(zs) ** 0.5
print(f"\nsegment-as-agent with its OWN reward term, {len(zs)} seeds: mean z {statistics.fmean(zs):+.2f} +/- {se:.2f}")
print("vehicle-as-agent under the same valid null: see scripts/measure_action_alignment.py")
