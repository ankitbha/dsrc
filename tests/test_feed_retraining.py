"""Amendment 1, A2: the training script and the evaluator read and write the `_cap`
checkpoint and evaluation names, never the committed files.

The evaluator tests stub the episodes. The last test runs one 600 s training episode
under the cap (plus the script's own 600 s validation and test reads) into a temporary
directory; run this file in its own process.
"""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "scripts"))
import evaluate_mainz_checkpoints as emc  # noqa: E402
import train_mainz_src as tms  # noqa: E402

CHECKPOINTS = REPO_ROOT / "results" / "checkpoints"
COMMITTED = [CHECKPOINTS / "mainz_here_best.pt", CHECKPOINTS / "mainz_src_best.pt",
             REPO_ROOT / "results" / "evaluation" / "mainz_paired_seeds.json",
             REPO_ROOT / "results" / "evaluation" / "mainz_paired_seeds.jsonl"]


def _digests():
    return {p: hashlib.sha256(p.read_bytes()).hexdigest() for p in COMMITTED}


class TestNames:
    def test_the_cap_suffix_names_every_arm_s_file(self, tmp_path):
        assert emc.checkpoint_path_for_arm("here", tmp_path, "_cap") == \
            tmp_path / "mainz_here_cap_best.pt"
        assert emc.checkpoint_path_for_arm("src", tmp_path, "_cap") == \
            tmp_path / "mainz_src_cap_best.pt"
        assert emc.checkpoint_path_for_arm("zeroed_head", tmp_path, "_cap") == \
            tmp_path / "mainz_here_cap_best.pt"
        assert emc.checkpoint_path_for_arm("here", tmp_path) == tmp_path / "mainz_here_best.pt"

    def test_save_as_writes_prefixed_files_and_refuses_existing_ones(self, tmp_path):
        best, result = tms.output_paths("ignored", "here", str(tmp_path / "mainz_here_cap"))
        assert best == tmp_path / "mainz_here_cap_best.pt"
        assert result == tmp_path / "mainz_here_cap_result.json"
        best.write_bytes(b"x")
        with pytest.raises(SystemExit, match="refusing to overwrite"):
            tms.output_paths("ignored", "here", str(tmp_path / "mainz_here_cap"))

    def test_save_as_refuses_the_committed_checkpoint(self):
        with pytest.raises(SystemExit):
            tms.output_paths("ignored", "here", str(CHECKPOINTS / "mainz_here"))


def _stubbed_run(tmp_path, monkeypatch, **kwargs):
    calls = {"load": [], "episode": [], "no_control": []}

    def fake_load(name, checkpoint_dir, num_segments, suffix=""):
        calls["load"].append((name, suffix))
        path = emc.checkpoint_path_for_arm(name, checkpoint_dir, suffix)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(name.encode())
        return emc.LoadedArm(name=name, model=("model", name),
                             features=emc.CHECKPOINT_FEATURES[emc._checkpoint_name_for_arm(name)],
                             checkpoint_path=path,
                             sha256=hashlib.sha256(name.encode()).hexdigest())

    def fake_episode(model, seed, features, duration_s, epsilon, generator, step_length,
                     window_start_s, gate_entries, paths, **extra):
        calls["episode"].append(extra)
        return {"return": 1.0, "arrived": 10, "flow": 3000.0 + seed, "mean_speed_kmh": 1.0,
                "space_mean_speed_kmh": 1.0, "mean_vehicles": 1.0, "held_at_entry": 3,
                "colliding_vehicles": 0, "collision_teleports": 0}

    def fake_no_control(seeds, features, duration_s, step_length, window_start_s, paths,
                        **extra):
        calls["no_control"].append(extra)
        return {"return": 1.0, "arrived": 10, "flow": 2900.0 + seeds[0], "mean_speed_kmh": 1.0,
                "space_mean_speed_kmh": 1.0, "mean_vehicles": 1.0, "held_at_entry": 0,
                "colliding_vehicles": 0, "collision_teleports": 0}

    monkeypatch.setattr(emc, "load_arm_model", fake_load)
    monkeypatch.setattr(emc, "run_episode", fake_episode)
    monkeypatch.setattr(emc, "evaluate_no_control", fake_no_control)
    summary = emc.run(arms=("no_control", "here", "src", "zeroed_head"), seeds=(16, 17),
                      checkpoint_dir=tmp_path / "checkpoints", out_dir=tmp_path / "evaluation",
                      **kwargs)
    return summary, calls


class TestTheCapEvaluation:
    def test_it_reads_the_cap_checkpoints_and_writes_the_cap_files(self, tmp_path, monkeypatch):
        summary, calls = _stubbed_run(tmp_path, monkeypatch, checkpoint_suffix="_cap",
                                      out_name="mainz_paired_seeds_cap")
        assert calls["load"] == [("here", "_cap"), ("src", "_cap"), ("zeroed_head", "_cap")]
        written = sorted(p.name for p in (tmp_path / "evaluation").iterdir())
        assert written == ["mainz_paired_seeds_cap.json", "mainz_paired_seeds_cap.jsonl"]
        assert summary["config"]["checkpoint_suffix"] == "_cap"
        assert summary["config"]["actuation"] == "max_speed"
        assert summary["config"]["checkpoints"]["here"]["path"].endswith("mainz_here_cap_best.pt")
        record = json.loads((tmp_path / "evaluation" / "mainz_paired_seeds_cap.jsonl")
                            .read_text().splitlines()[1])
        assert record["colliding_vehicles"] == 0 and record["collision_teleports"] == 0

    def test_the_default_actuation_passes_nothing_and_the_legacy_one_is_passed(
            self, tmp_path, monkeypatch):
        _, calls = _stubbed_run(tmp_path / "a", monkeypatch)
        assert all(extra == {} for extra in calls["episode"] + calls["no_control"])
        _, calls = _stubbed_run(tmp_path / "b", monkeypatch, actuation="set_speed")
        assert all(extra == {"actuation": "set_speed"}
                   for extra in calls["episode"] + calls["no_control"])

    def test_main_never_replaces_an_existing_evaluation(self, tmp_path):
        (tmp_path / "mainz_paired_seeds_cap.json").write_text("{}")
        with pytest.raises(SystemExit, match="refusing to replace"):
            emc.main(["--out", str(tmp_path), "--out-name", "mainz_paired_seeds_cap",
                      "--checkpoint-suffix", "_cap"])

    def test_main_refuses_the_committed_evaluation_by_default(self):
        before = _digests()
        with pytest.raises(SystemExit, match="refusing to replace"):
            emc.main([])
        assert _digests() == before


def test_one_600_s_training_episode_under_the_cap(tmp_path):
    before = _digests()
    prefix = tmp_path / "mainz_here_cap"
    assert tms.main(["--features", "here", "--episodes", "1", "--duration-s", "600",
                     "--step-length", "0.5", "--window-start-s", "900",
                     "--save-as", str(prefix)]) == 0
    result = json.loads((tmp_path / "mainz_here_cap_result.json").read_text())
    assert (tmp_path / "mainz_here_cap_best.pt").exists()
    assert result["actuation"] == "max_speed"
    assert result["config"]["episodes"] == 1 and result["config"]["step_length_s"] == 0.5
    assert result["test"]["colliding_vehicles"] == 0
    assert result["test"]["collision_teleports"] == 0
    assert result["test"]["arrived"] > 0
    # The trained file loads into the evaluator under the cap suffix.
    loaded = emc.load_arm_model("here", tmp_path, 12, suffix="_cap")
    assert loaded.checkpoint_path == tmp_path / "mainz_here_cap_best.pt"
    assert _digests() == before
