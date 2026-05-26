#!/usr/bin/env python3
import sys
from importlib import metadata

import numpy as np


REQUIRED_MODULES = (
    ("torch", None),
    ("highway_env", "highway-env"),
    ("gymnasium", None),
    ("numpy", None),
    ("pygame", None),
)


class CheckFailure(RuntimeError):
    pass


def assert_true(condition: bool, message: str) -> None:
    if not condition:
        raise CheckFailure(message)


def import_package(module_name: str, dist_name: str | None = None) -> str:
    __import__(module_name)
    try:
        return metadata.version(dist_name or module_name)
    except metadata.PackageNotFoundError:
        return "unknown"


def check_python_runtime(results: list[tuple[str, str]]) -> None:
    assert_true(sys.version_info[:2] == (3, 11), f"expected Python 3.11, got {sys.version.split()[0]}")
    assert_true(
        sys.executable.startswith("/ext3/miniforge3/"),
        f"expected /ext3/miniforge3 Python, got {sys.executable}",
    )
    results.append(("python_runtime", f"{sys.executable} ({sys.version.split()[0]})"))


def check_imports(results: list[tuple[str, str]]) -> None:
    for module_name, dist_name in REQUIRED_MODULES:
        version = import_package(module_name, dist_name)
        results.append((f"import:{module_name}", version))


def validate_frame(frame: np.ndarray, env_name: str) -> str:
    assert_true(isinstance(frame, np.ndarray), f"{env_name} render did not return a numpy array")
    assert_true(frame.ndim == 3, f"{env_name} frame should be rank-3, got shape {frame.shape}")
    assert_true(frame.shape[0] > 0 and frame.shape[1] > 0, f"{env_name} frame has invalid shape {frame.shape}")
    assert_true(frame.shape[2] in (3, 4), f"{env_name} frame should have 3 or 4 channels, got {frame.shape}")
    assert_true(frame.size > 0, f"{env_name} frame is empty")
    return str(frame.shape)


def run_single_agent_smoke(env_name: str, steps: int, results: list[tuple[str, str]]) -> None:
    import gymnasium as gym
    import highway_env  # noqa: F401

    env = gym.make(env_name, render_mode="rgb_array")
    try:
        obs, info = env.reset(seed=7)
        assert_true(obs is not None, f"{env_name} reset returned no observation")
        assert_true(isinstance(info, dict), f"{env_name} reset info should be a dict")
        first_frame = env.render()
        first_shape = validate_frame(first_frame, env_name)
        results.append((f"{env_name}:reset_frame", first_shape))

        for _ in range(steps):
            action = env.action_space.sample()
            obs, reward, terminated, truncated, info = env.step(action)
            assert_true(obs is not None, f"{env_name} step returned no observation")
            assert_true(np.isfinite(reward), f"{env_name} reward is not finite: {reward}")
            assert_true(isinstance(terminated, bool), f"{env_name} terminated flag is not bool")
            assert_true(isinstance(truncated, bool), f"{env_name} truncated flag is not bool")
            assert_true(isinstance(info, dict), f"{env_name} step info should be a dict")
            if terminated or truncated:
                obs, info = env.reset(seed=7)
                assert_true(obs is not None, f"{env_name} reset after termination returned no observation")

        final_frame = env.render()
        final_shape = validate_frame(final_frame, env_name)
        results.append((f"{env_name}:final_frame", final_shape))
        results.append((f"{env_name}:steps", str(steps)))
    finally:
        env.close()


def run_multi_agent_smoke(results: list[tuple[str, str]]) -> None:
    import gymnasium as gym
    import highway_env  # noqa: F401

    env = gym.make(
        "highway-v0",
        render_mode="rgb_array",
        config={
            "controlled_vehicles": 2,
            "vehicles_count": 6,
            "observation": {
                "type": "MultiAgentObservation",
                "observation_config": {"type": "Kinematics"},
            },
            "action": {
                "type": "MultiAgentAction",
                "action_config": {"type": "DiscreteMetaAction"},
            },
        },
    )
    try:
        obs, info = env.reset(seed=11)
        assert_true(isinstance(obs, tuple), f"multi-agent reset should return a tuple, got {type(obs).__name__}")
        assert_true(len(obs) == 2, f"multi-agent reset should return 2 observations, got {len(obs)}")
        assert_true(isinstance(info, dict), f"multi-agent reset info should be a dict, got {type(info).__name__}")

        sampled_action = env.action_space.sample()
        assert_true(isinstance(sampled_action, tuple), "multi-agent action space should sample a tuple")
        assert_true(len(sampled_action) == 2, f"multi-agent action tuple should have 2 items, got {len(sampled_action)}")

        next_obs, reward, terminated, truncated, info = env.step(sampled_action)
        assert_true(isinstance(next_obs, tuple), "multi-agent step should return tuple observations")
        assert_true(len(next_obs) == 2, f"multi-agent step should return 2 observations, got {len(next_obs)}")
        assert_true(np.isfinite(reward), f"multi-agent reward is not finite: {reward}")
        assert_true(isinstance(terminated, bool), "multi-agent terminated flag is not bool")
        assert_true(isinstance(truncated, bool), "multi-agent truncated flag is not bool")
        assert_true(isinstance(info, dict), f"multi-agent step info should be a dict, got {type(info).__name__}")

        frame = env.render()
        shape = validate_frame(frame, "highway-v0 multi-agent")
        results.append(("highway-v0:multi_agent_obs", str(len(obs))))
        results.append(("highway-v0:multi_agent_action", str(len(sampled_action))))
        results.append(("highway-v0:multi_agent_frame", shape))
    finally:
        env.close()


def main() -> int:
    results: list[tuple[str, str]] = []
    failures: list[str] = []

    checks = (
        ("python runtime", lambda: check_python_runtime(results)),
        ("imports", lambda: check_imports(results)),
        ("highway-fast-v0 smoke", lambda: run_single_agent_smoke("highway-fast-v0", 5, results)),
        ("merge-v0 smoke", lambda: run_single_agent_smoke("merge-v0", 5, results)),
        ("highway-v0 multi-agent smoke", lambda: run_multi_agent_smoke(results)),
    )

    print("Running highway_env integration checks")
    for label, check in checks:
        try:
            check()
            print(f"[PASS] {label}")
        except Exception as exc:  # noqa: BLE001
            failures.append(f"{label}: {exc}")
            print(f"[FAIL] {label}: {exc}")

    print("\nCheck summary")
    for key, value in results:
        print(f"- {key}: {value}")

    if failures:
        print("\nFAILURES")
        for failure in failures:
            print(f"- {failure}")
        return 1

    print("\nAll integration checks passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
