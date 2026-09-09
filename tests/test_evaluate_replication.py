"""Task 69's gate: evaluate under the block the actor was trained on.

`src/envs/topology_env.py` builds every agent observation through `SensingConfig`,
so the sensing block IS the actor's input distribution. Scoring a checkpoint under
a different block measures the mismatch rather than the controller, and nothing in
the output would say so. These tests pin the refusals that prevent it, because a
silent fallback to library defaults is precisely the failure mode.
"""
from __future__ import annotations

import pytest
import yaml

from scripts.evaluate_replication import sensing_from_checkpoint


def test_the_recorded_block_is_returned(tmp_path):
    (tmp_path / "config_resolved.yaml").write_text(yaml.safe_dump({
        "training": {"algorithm": "mappo",
                     "sensing": {"range_m": 100.0, "latency_s": 0.0,
                                 "position_noise_std": 1.5, "speed_noise_std": 0.15,
                                 "queue_speed_mps": 5.0}},
    }))
    assert sensing_from_checkpoint(tmp_path)["range_m"] == 100.0


def test_a_missing_config_is_refused(tmp_path):
    with pytest.raises(SystemExit) as excinfo:
        sensing_from_checkpoint(tmp_path)
    assert "cannot tell what sensing" in str(excinfo.value)


def test_a_config_without_a_sensing_block_is_refused(tmp_path):
    # The dangerous case: the file exists and parses, so a lenient reader would
    # fall through to defaults and produce a number that looks fine.
    (tmp_path / "config_resolved.yaml").write_text(yaml.safe_dump({
        "training": {"algorithm": "mappo"}, "ppo": {},
    }))
    with pytest.raises(SystemExit) as excinfo:
        sensing_from_checkpoint(tmp_path)
    assert "records no sensing block" in str(excinfo.value)


def test_an_empty_sensing_block_is_refused_too(tmp_path):
    (tmp_path / "config_resolved.yaml").write_text(yaml.safe_dump({
        "training": {"algorithm": "mappo", "sensing": {}},
    }))
    with pytest.raises(SystemExit):
        sensing_from_checkpoint(tmp_path)
