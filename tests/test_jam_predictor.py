"""The supervised onset predictor (plan section 5.5, step S11).

Unit tests use hand traces and synthetic features. The real smoke at the bottom runs two
600 s block T episodes through the runner's episode function and fits on them; run this
file in its own process.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from src.rl import onset_predictor as op  # noqa: E402


def _trace(jammed: dict[int, tuple[float, float]], end: float = 1500.0) -> list[dict]:
    """10 s samples on [900, end]; `jammed[segment] = (first, last)` at 10 km/h."""
    out = []
    t = 900.0
    while t <= end:
        speed = [50.0] * 12
        for segment, (a, b) in jammed.items():
            if a <= t <= b:
                speed[segment] = 10.0
        out.append({"t": t, "speed_kmh": speed, "count": [5] * 12})
        t += 10.0
    return out


def _sample(t, segment):
    return {"vehicle": "v", "t": t, "segment": segment, "features": [0.0] * len(op.FEATURES)}


class TestLabels:
    def test_a_spell_starting_60_s_ahead_on_the_next_segment_is_positive(self):
        trace = _trace({6: (1060.0, 1250.0)})
        _, y, kept = op.label_samples([_sample(1000.0, 5)], trace, v_jam=19.0)
        assert kept.tolist() == [True] and y.tolist() == [1]

    def test_a_spell_starting_130_s_ahead_is_negative(self):
        trace = _trace({6: (1130.0, 1300.0)})
        _, y, kept = op.label_samples([_sample(1000.0, 5)], trace, v_jam=19.0)
        assert y.tolist() == [0]

    def test_a_spell_two_segments_ahead_or_too_short_is_negative(self):
        assert op.label_samples([_sample(1000.0, 5)], _trace({7: (1060.0, 1250.0)}),
                                19.0)[1].tolist() == [0]
        assert op.label_samples([_sample(1000.0, 5)], _trace({6: (1060.0, 1160.0)}),
                                19.0)[1].tolist() == [0]

    def test_a_spell_on_the_own_segment_counts(self):
        _, y, _ = op.label_samples([_sample(1000.0, 5)], _trace({5: (1050.0, 1250.0)}), 19.0)
        assert y.tolist() == [1]

    def test_a_sample_whose_own_segment_is_jammed_is_dropped(self):
        trace = _trace({5: (980.0, 1250.0)})
        X, y, kept = op.label_samples([_sample(1005.0, 5), _sample(1005.0, 4)], trace, 19.0)
        assert kept.tolist() == [False, True] and len(y) == 1 and len(X) == 1

    def test_the_last_segment_looks_at_itself_only(self):
        _, y, _ = op.label_samples([_sample(1000.0, 11)], _trace({11: (1060.0, 1250.0)}), 19.0)
        assert y.tolist() == [1]


class TestFeatures:
    def test_no_feature_is_ego_speed_or_true_density(self):
        names = [n.lower() for n in op.FEATURES]
        for token in op.DENIED:
            assert not any(token in name for name in names), token
        assert len(names) == len(set(names))

    def test_the_card_lists_the_features(self, tmp_path):
        model = op.fit(np.zeros((20, len(op.FEATURES))), np.zeros(20, dtype=int),
                       np.zeros((10, len(op.FEATURES))), np.zeros(10, dtype=int))
        card = op.save(model, tmp_path / "card.json", extra={"seeds": {"train": [1]}})
        assert card["features"] == list(op.FEATURES)
        assert card["positives"]["train"] == 0


class TestFit:
    def test_a_separable_set_reaches_validation_log_loss_below_0_1(self):
        # Separable with a margin: no sample within 0.3 of the boundary.
        rng = np.random.default_rng(0)
        X = rng.normal(size=(3000, len(op.FEATURES)))
        score = X[:, 0] + 0.5 * X[:, 3] - 0.8
        X, score = X[np.abs(score) > 0.3][:2000], score[np.abs(score) > 0.3][:2000]
        y = (score > 0).astype(int)
        model = op.fit(X[:1500], y[:1500], X[1500:], y[1500:])
        assert model.validation_log_loss < 0.1
        assert set(model.candidates) == {"logistic", "mlp"}
        assert model.kind == min(model.candidates, key=lambda k: model.candidates[k])

    def test_save_then_load_predicts_identically(self, tmp_path):
        rng = np.random.default_rng(1)
        X = rng.normal(size=(400, len(op.FEATURES)))
        y = (X[:, 1] > 0).astype(int)
        model = op.fit(X[:300], y[:300], X[300:], y[300:])
        card = op.save(model, tmp_path / "onset_predictor.json")
        loaded = op.load(tmp_path / "onset_predictor.json")
        np.testing.assert_array_equal(model.predict(X), loaded.predict(X))
        assert loaded.card["weights_sha256"] == card["weights_sha256"]
        assert loaded.card["kind"] == model.kind

    def test_a_changed_weights_file_is_refused(self, tmp_path):
        model = op.fit(np.zeros((20, len(op.FEATURES))), np.zeros(20, dtype=int),
                       np.zeros((10, len(op.FEATURES))), np.zeros(10, dtype=int))
        op.save(model, tmp_path / "p.json")
        (tmp_path / "p.pt").write_bytes(b"not the weights")
        with pytest.raises(SystemExit):
            op.load(tmp_path / "p.json")


class TestTrainingSeeds:
    def test_seeds_outside_1_10_are_refused(self):
        import train_onset_predictor as top

        with pytest.raises(SystemExit):
            top.split_seeds([1, 2, 11])
        assert top.split_seeds(list(range(1, 11))) == (list(range(1, 9)), [9, 10])


# ----------------------------------------------------------------- real smoke


def test_two_600_s_block_t_episodes_fit_and_write_a_card(tmp_path):
    from dataclasses import replace

    import run_jam_experiments as rje
    import train_onset_predictor as top

    camera = REPO_ROOT / "outputs/jam/smoke/camera_error_model.json"
    if not camera.exists():
        pytest.skip("the S8 smoke artefact has not been written")
    context = rje.Context(camera_model=camera, window_start_s=300.0, smoke=True,
                          training_dir=tmp_path / "training")
    from src.sumo.jam_episode import FeedSetting

    context.feed = FeedSetting(context.feed_artefact, context.feed_session).parameters()
    specs = [replace(s, duration_s=600.0) for s in rje.block_specs("T", seeds=[1, 9])
             if s.scenario == "B9"]
    import evaluate_mainz_checkpoints as emc

    model = emc._build_and_load(rje.DEFAULT_CHECKPOINT, emc.CHECKPOINT_FEATURES["here"], 12)
    results = tmp_path / "results"
    record = rje.header("T", context, specs, argv=[])
    rje.run_block("T", specs, rje.episode_runner(model, context), results_dir=results,
                  traces_dir=tmp_path / "traces", header_record=record,
                  training_dir=context.training_dir)
    out = tmp_path / "onset_predictor_round1.json"
    assert top.main(["--round", "1", "--results-dir", str(results), "--out", str(out),
                     "--smoke"]) == 0
    card = json.loads(out.read_text())
    assert card["round"] == 1 and card["seeds"]["train"] == [1] and card["seeds"]["validation"] == [9]
    assert card["samples"]["train"] > 0 and card["features"] == list(op.FEATURES)
    assert card["feed_artefact_sha256"] == context.feed["artefact_sha256"]
    predictor = op.load(out)
    assert predictor.predict(np.zeros((3, len(op.FEATURES)))).shape == (3,)
