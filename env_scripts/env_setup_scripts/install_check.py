#!/usr/bin/env python3
import os
import sys
from importlib import metadata


def import_package(module_name: str, dist_name: str | None = None) -> None:
    __import__(module_name)
    try:
        version = metadata.version(dist_name or module_name)
    except metadata.PackageNotFoundError:
        version = "unknown"
    print(f"import ok: {module_name} ({version})")


def main() -> None:
    require_cuda = os.environ.get("REQUIRE_CUDA", "1") != "0"

    print("python:", sys.executable)
    print("python version:", sys.version.split()[0])

    if sys.version_info[:2] != (3, 11):
        raise RuntimeError(f"expected Python 3.11, got {sys.version.split()[0]}")

    if not sys.executable.startswith("/ext3/miniforge3/"):
        raise RuntimeError(f"expected /ext3/miniforge3 Python, got {sys.executable}")

    modules = (
        ("torch", None),
        ("torchvision", None),
        ("torchaudio", None),
        ("pytest", None),
        ("highway_env", "highway-env"),
        ("gymnasium", None),
        ("stable_baselines3", "stable-baselines3"),
        ("sb3_contrib", "sb3-contrib"),
        ("pettingzoo", None),
        ("supersuit", None),
        ("numpy", None),
        ("pandas", None),
        ("scipy", None),
        ("matplotlib", None),
        ("yaml", "PyYAML"),
        ("tqdm", None),
        ("networkx", None),
        ("numba", None),
        ("tensorboard", None),
        ("wandb", None),
        ("rich", None),
        ("h5py", None),
        ("pyarrow", None),
        ("pygame", None),
        ("imageio", None),
        ("moviepy", None),
        ("cv2", "opencv-python-headless"),
    )
    for module_name, dist_name in modules:
        import_package(module_name, dist_name)

    import gymnasium as gym
    import torch

    env = gym.make("highway-fast-v0", render_mode="rgb_array")
    obs, info = env.reset()
    for _ in range(5):
        obs, reward, terminated, truncated, info = env.step(env.action_space.sample())
        if terminated or truncated:
            obs, info = env.reset()
    env.close()
    print("highway-env OK")

    print("torch cuda available:", torch.cuda.is_available())
    if require_cuda and not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available to PyTorch")
    if torch.cuda.is_available():
        print("torch cuda device:", torch.cuda.get_device_name(torch.cuda.current_device()))
    elif not require_cuda:
        print("torch cuda check skipped")

    print("all core packages imported")


if __name__ == "__main__":
    main()
