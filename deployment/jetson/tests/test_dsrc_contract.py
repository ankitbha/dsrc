"""Guard the vendored DSRC network against drift from the simulation source.

The shape/key comparison against `src.rl.src_q.SrcQNetwork` runs only where
that module is importable (guarded, per-test, the way
`test_sim_contract.py`'s action-constant tests are) -- it is not a module-
level `importorskip`, because the golden-action tests below it need no sim
import at all and must still run on the device, where only this vendored
copy is trusted.
"""

from __future__ import annotations

import pytest
import torch

from policy import dsrc_contract


def test_speed_action_fractions_match_mainz():
    from src.sumo import mainz

    assert dsrc_contract.SPEED_ACTION_FRACTIONS == mainz.SPEED_ACTION_FRACTIONS


def test_decision_interval_matches_mainz():
    from src.sumo import mainz

    assert dsrc_contract.DECISION_INTERVAL_S == mainz.DECISION_INTERVAL_S


def test_here_features_match_mainz():
    from src.sumo import mainz

    assert dsrc_contract.HERE_FEATURES == mainz.HERE_FEATURES


def test_vendored_network_matches_sim_state_dict_keys_and_shapes():
    sim_src_q = pytest.importorskip("src.rl.src_q", reason="sim repo not importable")

    vendored = dsrc_contract.SrcQNetwork(12, 5, 3)
    original = sim_src_q.SrcQNetwork(12, 5, 3)

    vendored_state = vendored.state_dict()
    original_state = original.state_dict()
    assert set(vendored_state.keys()) == set(original_state.keys())
    for key in vendored_state:
        assert vendored_state[key].shape == original_state[key].shape, key


def test_vendored_forward_matches_sim_forward_given_the_same_weights():
    """Not just the same shapes -- the same function, weight for weight."""
    sim_src_q = pytest.importorskip("src.rl.src_q", reason="sim repo not importable")

    torch.manual_seed(0)
    vendored = dsrc_contract.SrcQNetwork(12, 5, 3)
    original = sim_src_q.SrcQNetwork(12, 5, 3)
    original.load_state_dict(vendored.state_dict())

    state = torch.randn(12, 5)
    with torch.no_grad():
        assert torch.equal(vendored(state), original(state))


def test_vendored_network_can_load_the_committed_checkpoint_shapes():
    """The two committed checkpoints imply (12, 5, 3) and (12, 6, 3)."""
    twelve_five = dsrc_contract.SrcQNetwork(12, 5, 3)
    assert twelve_five.stack[0].weight.shape == (120, 60)
    assert twelve_five.stack[2].weight.shape == (36, 120)

    twelve_six = dsrc_contract.SrcQNetwork(12, 6, 3)
    assert twelve_six.stack[0].weight.shape == (144, 72)
    assert twelve_six.stack[2].weight.shape == (36, 144)


class TestNetworkFingerprint:
    """Mirrors `TestTheBundleGuardSeesMoreThanTheDimension` in
    `test_sim_contract.py`: the fingerprint must move on a change the shapes
    cannot see, and must be stable otherwise.
    """

    BASE = dict(
        network_id="mainz",
        feature_names=("speed", "free_flow", "jam_factor", "lanes", "length_km"),
        segment_ids=(("a", "b"), ("c",)),
        segment_speed_limits_kmh=(60.0, 60.0),
    )

    def test_stable_across_calls(self):
        assert (
            dsrc_contract.network_fingerprint(**self.BASE)
            == dsrc_contract.network_fingerprint(**self.BASE)
        )

    def test_moves_when_two_segments_are_reordered(self):
        before = dsrc_contract.network_fingerprint(**self.BASE)
        reordered = dict(self.BASE, segment_ids=(("c",), ("a", "b")))
        assert dsrc_contract.network_fingerprint(**reordered) != before

    def test_moves_when_two_edges_within_a_segment_are_reordered(self):
        before = dsrc_contract.network_fingerprint(**self.BASE)
        reordered = dict(self.BASE, segment_ids=(("b", "a"), ("c",)))
        assert dsrc_contract.network_fingerprint(**reordered) != before

    def test_moves_when_a_speed_limit_changes(self):
        before = dsrc_contract.network_fingerprint(**self.BASE)
        changed = dict(self.BASE, segment_speed_limits_kmh=(60.0, 45.0))
        assert dsrc_contract.network_fingerprint(**changed) != before

    def test_moves_when_feature_names_are_reordered(self):
        before = dsrc_contract.network_fingerprint(**self.BASE)
        reordered = dict(
            self.BASE,
            feature_names=("free_flow", "speed", "jam_factor", "lanes", "length_km"),
        )
        assert dsrc_contract.network_fingerprint(**reordered) != before

    def test_moves_when_network_id_changes(self):
        before = dsrc_contract.network_fingerprint(**self.BASE)
        changed = dict(self.BASE, network_id="inverted_tree")
        assert dsrc_contract.network_fingerprint(**changed) != before

    def test_unchanged_when_nothing_changed(self):
        before = dsrc_contract.network_fingerprint(**self.BASE)
        assert dsrc_contract.network_fingerprint(**self.BASE) == before
