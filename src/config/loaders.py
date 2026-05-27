from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any, Mapping

import yaml


CONFIG_ROOT = Path(__file__).resolve().parents[2] / "configs"
FAMILY_DIRS = {
    "topology": "topology",
    "demand": "demand",
    "human_model": "human_models",
    "experiments": "experiments",
    "training": "training",
}


def deep_merge(base: Mapping[str, Any], overrides: Mapping[str, Any]) -> dict[str, Any]:
    """Recursively merge dictionaries, returning a new dict."""
    merged = deepcopy(dict(base))
    for key, value in overrides.items():
        if isinstance(value, Mapping) and isinstance(merged.get(key), Mapping):
            merged[key] = deep_merge(merged[key], value)
        else:
            merged[key] = deepcopy(value)
    return merged


def load_yaml_file(path: str | Path) -> dict[str, Any]:
    """Load a YAML file into a plain dict."""
    data = yaml.safe_load(Path(path).read_text())
    if not isinstance(data, dict):
        raise ValueError(f"expected mapping YAML in {path}")
    return data


def load_named_config(
    family: str,
    name: str,
    config_root: str | Path = CONFIG_ROOT,
) -> dict[str, Any]:
    """Load one named config family entry by file stem."""
    if family not in FAMILY_DIRS:
        raise KeyError(f"unknown config family '{family}'")

    path = Path(config_root) / FAMILY_DIRS[family] / f"{name}.yaml"
    if not path.exists():
        raise FileNotFoundError(f"config file not found: {path}")
    return load_yaml_file(path)


def compose_experiment_config(
    experiment_name: str,
    config_root: str | Path = CONFIG_ROOT,
) -> dict[str, Any]:
    """Resolve one experiment config into the canonical composed bundle."""
    experiment_cfg = load_named_config("experiments", experiment_name, config_root)
    refs = experiment_cfg.get("refs", {})
    required_refs = ("topology", "demand", "human_model", "training")
    missing_refs = [key for key in required_refs if key not in refs]
    if missing_refs:
        raise ValueError(f"experiment '{experiment_name}' is missing refs: {missing_refs}")

    bundle = {
        "experiment": deepcopy(experiment_cfg.get("experiment", {"id": experiment_name})),
        "topology": load_named_config("topology", refs["topology"], config_root),
        "demand": load_named_config("demand", refs["demand"], config_root),
        "human_model": load_named_config("human_model", refs["human_model"], config_root),
        "training": load_named_config("training", refs["training"], config_root),
        "controller": deepcopy(experiment_cfg.get("controller", {})),
        "metrics": deepcopy(experiment_cfg.get("metrics", {})),
        "outputs": deepcopy(experiment_cfg.get("outputs", {})),
        "resolved_refs": deepcopy(refs),
    }

    for section, overrides in experiment_cfg.get("overrides", {}).items():
        if section not in bundle:
            raise ValueError(f"unsupported override section '{section}'")
        current_section = bundle[section]
        if not isinstance(current_section, Mapping):
            raise ValueError(f"override section '{section}' is not mergeable")
        if not isinstance(overrides, Mapping):
            raise ValueError(f"override section '{section}' must be a mapping")
        bundle[section] = deep_merge(current_section, overrides)

    return bundle

