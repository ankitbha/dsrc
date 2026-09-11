#!/usr/bin/env python3
"""Train SRC's controller on the Mainz network with a HERE-shaped observation.

    .venv/bin/python scripts/train_mainz_src.py --episodes 30

Train on seeds 1-10, evaluate every checkpoint on seeds 11-15, keep the best by
validation return, and report the held-out seeds 16-20 once at the end. The test seeds
are read on the final line and nowhere else.

`--features src` trains the same model on SRC's original six features instead, which
are not obtainable from a traffic API. The difference between the two on the same test
seeds is what restricting the controller to deployable observations costs.
"""
from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.rl.src_q import (  # noqa: E402
    DISCOUNT, SrcQNetwork, epsilon_actions, greedy_actions, td_loss,
)
from src.sumo.mainz import HERE_FEATURES, SRC_FEATURES, MainzEnv  # noqa: E402

TRAIN_SEEDS = tuple(range(1, 11))
VALIDATION_SEEDS = tuple(range(11, 16))
TEST_SEEDS = tuple(range(16, 21))


def run_episode(model: SrcQNetwork, seed: int, features: tuple[str, ...],
                duration_s: float, epsilon: float,
                generator: torch.Generator | None, step_length: float = 1.0,
                window_start_s: float = 0.0, gate_entries: bool = True) -> dict:
    """One episode. Returns the trajectory and the outcome metrics.

    The controller acts from t=0 so it can prevent a jam rather than inherit one, but
    throughput is counted only from `window_start_s`, after the network has filled.
    """
    env = MainzEnv(seed=seed, duration_s=duration_s, features=features,
                   step_length_s=step_length, gate_entries=gate_entries)
    states, actions, rewards = [], [], []
    marked = window_start_s <= 0.0
    try:
        state = torch.tensor(env.reset(), dtype=torch.float)
        while True:
            action = (greedy_actions(model, state) if generator is None
                      else epsilon_actions(model, state, epsilon, generator))
            raw, reward, done = env.step(action.tolist())
            if not marked and env.metrics()["time_s"] >= window_start_s:
                env.mark_window()
                marked = True
            states.append(state)
            actions.append(action)
            rewards.append(torch.tensor(reward, dtype=torch.float))
            state = torch.tensor(raw, dtype=torch.float)
            if done:
                break
        metrics = env.metrics()
    finally:
        env.close()
    return {
        "states": torch.stack(states),
        "actions": torch.stack(actions),
        "rewards": torch.stack(rewards),
        "return": float(torch.stack(rewards).mean(dim=1).sum()),
        "arrived": metrics["arrived"],
        "flow": metrics["flow_veh_per_h"],
        "mean_speed_kmh": metrics["mean_speed_kmh"],
        "held_at_entry": metrics["held_at_entry"],
    }


def evaluate(model: SrcQNetwork, seeds, features, duration_s, step_length=1.0,
             window_start_s=0.0, gate_entries=True) -> dict:
    runs = [run_episode(model, s, features, duration_s, 0.0, None, step_length,
                        window_start_s, gate_entries) for s in seeds]
    return {
        "return": statistics.fmean(r["return"] for r in runs),
        "arrived": statistics.fmean(r["arrived"] for r in runs),
        "flow": statistics.fmean(r["flow"] for r in runs),
        "mean_speed_kmh": statistics.fmean(r["mean_speed_kmh"] for r in runs),
        "held_at_entry": statistics.fmean(r["held_at_entry"] for r in runs),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--episodes", type=int, default=30)
    parser.add_argument("--duration-s", type=float, default=2500.0)
    parser.add_argument("--features", choices=("here", "src"), default="here")
    parser.add_argument("--learning-rate", type=float, default=0.05)
    parser.add_argument("--epsilon", type=float, default=0.3)
    parser.add_argument("--epsilon-final", type=float, default=0.02)
    parser.add_argument("--window-start-s", type=float, default=0.0,
                        help="open the throughput window here, so the ramp-up is "
                             "excluded and the score is a steady-state flow rate "
                             "rather than a count dominated by how far the fill got")
    parser.add_argument("--step-length", type=float, default=1.0,
                        help="0.1 to resolve the capacity drop; at 1.0 served flow "
                             "RISES with density and there is nothing to recover")
    parser.add_argument("--no-gate-entries", action="store_true",
                        help="drop the entry gate from the TRAINED arm, leaving SUMO "
                             "to insert on its own terms. The gate stands in for the "
                             "policy acting on the link above each entry link, which "
                             "this scenario does not simulate, so it belongs to the "
                             "policy and never to the no-control baseline")
    parser.add_argument("--validate-every", type=int, default=5)
    parser.add_argument("--out", default="outputs/mainz_src")
    args = parser.parse_args()

    features = HERE_FEATURES if args.features == "here" else SRC_FEATURES
    out = REPO_ROOT / args.out / args.features
    out.mkdir(parents=True, exist_ok=True)

    gate = not args.no_gate_entries
    probe = MainzEnv(seed=1, duration_s=1.0, warmup_s=0.0, features=features,
                     gate_entries=gate)
    num_segments = len(probe.segments)
    model = SrcQNetwork(num_segments, len(features), 3)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    generator = torch.Generator().manual_seed(0)
    torch.manual_seed(0)

    print(f"features {args.features}: {features}")
    print(f"{num_segments} super-segments, {sum(p.numel() for p in model.parameters())} parameters")
    print(f"entry gate {'on' if gate else 'off'} for the trained arm, "
          f"never for no control")
    print(f"train {TRAIN_SEEDS}  validate {VALIDATION_SEEDS}  test {TEST_SEEDS}\n")
    print(f"{'ep':>4} {'seed':>5} {'eps':>5} {'loss':>10} {'return':>10} "
          f"{'flow':>8} {'speed':>7}   validation")

    history, best = [], {"return": float("-inf"), "episode": -1}
    for episode in range(args.episodes):
        fraction = episode / max(args.episodes - 1, 1)
        epsilon = args.epsilon + (args.epsilon_final - args.epsilon) * fraction
        seed = TRAIN_SEEDS[episode % len(TRAIN_SEEDS)]
        started = time.time()
        run = run_episode(model, seed, features, args.duration_s, epsilon,
                          generator, args.step_length, args.window_start_s, gate)
        loss = td_loss(model, run["states"], run["actions"], run["rewards"], DISCOUNT)
        total = loss.sum(dim=1).mean()
        optimizer.zero_grad()
        total.backward()
        optimizer.step()

        line = (f"{episode:>4} {seed:>5} {epsilon:>5.2f} {float(total.detach()):>10.3f} "
                f"{run['return']:>10.2f} {run['flow']:>8.0f} "
                f"{run['mean_speed_kmh']:>7.2f}")
        note = ""
        if (episode + 1) % args.validate_every == 0 or episode == args.episodes - 1:
            scores = evaluate(model, VALIDATION_SEEDS, features, args.duration_s,
                              args.step_length, args.window_start_s, gate)
            note = (f"   return {scores['return']:.1f}  flow {scores['flow']:.0f}"
                    f"  speed {scores['mean_speed_kmh']:.2f}")
            if scores["return"] > best["return"]:
                best = {"return": scores["return"], "episode": episode, **scores}
                torch.save(model.state_dict(), out / "best.pt")
                note += "  <- best"
            history.append({"episode": episode, **scores})
        print(line + note + f"   [{time.time()-started:.0f}s]", flush=True)

    print(f"\nbest checkpoint: episode {best['episode']}, "
          f"validation return {best['return']:.2f}")
    model.load_state_dict(torch.load(out / "best.pt", weights_only=True))

    print("\nreading the held-out test seeds, once")
    test = evaluate(model, TEST_SEEDS, features, args.duration_s, args.step_length,
                    args.window_start_s, gate)
    baseline = evaluate_no_control(TEST_SEEDS, features, args.duration_s,
                                   args.step_length, args.window_start_s)
    print(f"  trained    flow {test['flow']:>7.0f} veh/h  return {test['return']:>8.1f}"
          f"  speed {test['mean_speed_kmh']:>6.2f} km/h  arrived {test['arrived']:>7.1f}"
          f"  held at entry {test['held_at_entry']:>7.1f}")
    print(f"  no control flow {baseline['flow']:>7.0f} veh/h  return {baseline['return']:>8.1f}"
          f"  speed {baseline['mean_speed_kmh']:>6.2f} km/h  arrived {baseline['arrived']:>7.1f}"
          f"  held at entry {baseline['held_at_entry']:>7.1f}")
    print(f"  difference {100*(test['flow']-baseline['flow'])/baseline['flow']:+.1f}% "
          f"in steady-state flow")

    (out / "result.json").write_text(json.dumps(
        {"features": args.features, "best": best, "test": test,
         "no_control": baseline, "history": history}, indent=1))
    return 0


def evaluate_no_control(seeds, features, duration_s, step_length=1.0,
                        window_start_s=0.0) -> dict:
    """Every vehicle left to the car-following model, on the same seeds.

    NO CONTROL RUNS WITHOUT THE ENTRY GATE, and that is deliberate rather than an
    omission. The gate is not a piece of the network. It stands in for the policy
    acting on the link above the entry link, which is a link this scenario does not
    simulate: inside the network a link is protected by slowing the link above it, and
    at the boundary that link is outside the map, so the gate supplies its effect. It
    is a boundary condition on the controlled system.

    A run with no policy has nothing acting on that upstream link either, so gating its
    entries would credit the baseline with a control action it is not taking, and the
    comparison would no longer be against an uncontrolled network. The exit meter is
    the opposite case and applies to every arm, because a signal on a road is part of
    the network whoever is driving.
    """
    from src.sumo.mainz import DECISION_INTERVAL_S

    runs = []
    for seed in seeds:
        env = MainzEnv(seed=seed, duration_s=duration_s, features=features,
                       step_length_s=step_length, gate_entries=False)
        total = 0.0
        marked = window_start_s <= 0.0
        try:
            env.reset()
            while True:
                env.release()
                env._advance(DECISION_INTERVAL_S)
                total += float(env.reward().mean())  # matches run_episode
                if not marked and env.metrics()["time_s"] >= window_start_s:
                    env.mark_window(); marked = True
                if env.metrics()["time_s"] >= duration_s:
                    break
            metrics = env.metrics()
        finally:
            env.close()
        runs.append({"return": total, **metrics})
    return {
        "return": statistics.fmean(r["return"] for r in runs),
        "arrived": statistics.fmean(r["arrived"] for r in runs),
        "flow": statistics.fmean(r["flow_veh_per_h"] for r in runs),
        "mean_speed_kmh": statistics.fmean(r["mean_speed_kmh"] for r in runs),
        "held_at_entry": statistics.fmean(r["held_at_entry"] for r in runs),
    }


if __name__ == "__main__":
    raise SystemExit(main())
