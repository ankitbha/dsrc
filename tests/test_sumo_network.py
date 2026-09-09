"""Generating a SUMO network from the topology spec this project already has.

The network is generated rather than hand-written so there is one definition of
`inverted_tree`. A second, hand-maintained `.net.xml` would drift from the spec, and
the drift would show up as every gap the sensing model computes being slightly
wrong, which is close to undiagnosable.

SUMO is used because it cannot produce a collision. Measured on the 3-into-1 merge
shape that broke highway_env, at 6000 veh/h into a two-lane exit -- three times past
capacity -- for 1800 steps with 134 vehicles present: zero collisions.
"""
from __future__ import annotations

import pytest

from src.config.loaders import load_named_config
from src.sumo.network import SumoNetwork


@pytest.fixture(scope="module")
def network(tmp_path_factory):
    return SumoNetwork.build(
        "inverted_tree",
        load_named_config("topology", "inverted_tree"),
        tmp_path_factory.mktemp("net"),
    )


class TestTheNetworkMatchesTheSpec:

    def test_netconvert_accepts_it(self, network):
        assert network.net_file.exists(), "netconvert produced no network file"
        assert network.net_file.stat().st_size > 0

    def test_every_entry_segment_has_an_edge(self, network):
        # inverted_tree has six leaf entries in the spec; each must be an edge a
        # vehicle can be inserted on, or the demand has nowhere to start.
        entries = network.entry_edges()
        assert len(entries) == 6, f"expected 6 entry edges, got {sorted(entries)}"

    def test_the_lane_counts_come_from_the_spec(self, network):
        # leaves: 1, middle: 2, trunk: 2. A wrong lane count changes capacity, which
        # is the quantity the whole replication turns on.
        counts = network.lane_counts()
        for edge in network.entry_edges():
            assert counts[edge] == 1, f"{edge} has {counts[edge]} lanes, spec says 1"
        assert all(counts[e] == 2 for e in network.middle_edges()), counts

    def test_edges_map_back_to_segment_ids(self, network):
        # The sensing model keys on our segment ids, not SUMO edge ids, so the
        # mapping has to be total in the direction the snapshots need.
        for edge in network.entry_edges() | network.middle_edges():
            assert network.segment_for_edge(edge) is not None, f"{edge} maps to no segment"

    def test_a_route_exists_from_every_entry_to_the_exit(self, network):
        # If an entry cannot reach the exit, SUMO discards its vehicles silently and
        # the demand is quietly lower than configured.
        for edge in network.entry_edges():
            assert network.route_to_exit(edge), f"no route from {edge} to the exit"


class TestItIsReproducible:

    @staticmethod
    def _network_body(net_file):
        """The network itself, without netconvert's provenance header.

        The header records a generation timestamp and the absolute input and output
        paths, so two builds in different directories differ on 4 of 134 lines
        while the geometry is identical. Comparing whole files would make this test
        fail on facts that do not affect a single vehicle.
        """
        text = net_file.read_text()
        return text[text.index("<net "):]

    def test_building_twice_gives_the_same_network(self, tmp_path):
        cfg = load_named_config("topology", "inverted_tree")
        first = SumoNetwork.build("inverted_tree", cfg, tmp_path / "a")
        second = SumoNetwork.build("inverted_tree", cfg, tmp_path / "b")
        assert self._network_body(first.net_file) == self._network_body(second.net_file)

    def test_the_comparison_would_notice_a_real_change(self, tmp_path):
        # The control. Without it the test above would pass on any two networks
        # whose bodies happened to be sliced away.
        cfg = load_named_config("topology", "inverted_tree")
        baseline = SumoNetwork.build("inverted_tree", cfg, tmp_path / "base")
        narrowed = dict(cfg)
        road = dict(narrowed["road"])
        road["lane_counts"] = {**road["lane_counts"], "trunk": 1}
        narrowed["road"] = road
        changed = SumoNetwork.build("inverted_tree", narrowed, tmp_path / "changed")
        assert self._network_body(baseline.net_file) != self._network_body(changed.net_file)
        assert changed.lane_counts()["tree_trunk_c"] == 1


class TestTheBottleneckVariantIsBuilt:
    """`SumoNetwork.build` ignored `topology_id` for structure.

    It always emitted six leaves, two middles and one trunk with the exit at
    `tree_trunk_c`, whatever it was asked for. `inverted_tree_bottleneck` declares a
    tenth segment `tree_bottleneck_d`, `lane_counts.bottleneck: 1` and
    `exit_segments: [tree_bottleneck_d]`, so constructing it produced no error and
    simulated the non-bottleneck topology — and any comparison between the two on
    SUMO would have been a comparison of one topology with itself.
    """

    @pytest.fixture(scope="class")
    def bottleneck(self, tmp_path_factory):
        return SumoNetwork.build(
            "inverted_tree_bottleneck",
            load_named_config("topology", "inverted_tree_bottleneck"),
            tmp_path_factory.mktemp("bottleneck"),
        )

    def test_the_bottleneck_segment_exists(self, bottleneck):
        assert "tree_bottleneck_d" in bottleneck.lane_counts()

    def test_the_bottleneck_drops_to_one_lane(self, bottleneck):
        counts = bottleneck.lane_counts()
        assert counts["tree_bottleneck_d"] == 1, counts
        assert counts["tree_trunk_c"] == 2, counts

    def test_the_exit_is_the_bottleneck(self, bottleneck):
        assert bottleneck.exit_edge == "tree_bottleneck_d"

    def test_every_entry_still_reaches_the_exit(self, bottleneck):
        for edge in bottleneck.entry_edges():
            route = bottleneck.route_to_exit(edge)
            assert route, f"no route from {edge}"
            assert route[-1] == "tree_bottleneck_d"

    def test_it_differs_from_the_plain_topology(self, bottleneck, network):
        # The control. Without it every assertion above would pass on a generator
        # that ignored the name, since the plain network would also be returned.
        assert set(bottleneck.lane_counts()) != set(network.lane_counts())
        assert bottleneck.exit_edge != network.exit_edge

    def test_an_unsupported_layout_is_refused_by_name(self, tmp_path):
        config = load_named_config("topology", "inverted_tree")
        road = dict(config["road"])
        road["layout"] = "cloverleaf"
        with pytest.raises(ValueError, match="cloverleaf"):
            SumoNetwork.build("cloverleaf", {**config, "road": road}, tmp_path)
