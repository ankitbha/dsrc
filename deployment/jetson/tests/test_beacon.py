"""The V2V beacon, which no test in this directory imported.

Its only production caller is `run_demo.py` under `config["v2v"]["enabled"]`, which
defaults to false -- so removing the transmit gate entirely left the whole suite
green. Two of the three findings here were named by earlier rounds and not pursued.
"""

from __future__ import annotations

import time

from sensors.gps_reader import GpsFix
from v2v.beacon import BeaconTransceiver, _Peer


def fix(*, age_s: float = 0.0, valid: bool = True, lat: float = 51.49,
        lon: float = -0.20) -> GpsFix:
    return GpsFix(valid=valid, lat=lat, lon=lon, speed_mps=20.0, heading_deg=90.0,
                  fix_quality=1, num_sats=9, hdop=0.9,
                  t_mono=time.monotonic() - age_s, t_wall=0.0)


class TestTheBeaconWillNotBroadcastAStaleFix:
    """`GpsFix.valid` never decays; `is_stale` is a separate method this never called."""

    def test_a_stale_fix_is_not_broadcast(self):
        # The observation builder calls the very same reading stale and falls
        # `ego_speed` back to neutral, while this broadcast it as current -- stamped
        # `t_wall = time.time()` on every send, so a receiver stamps it as heard now
        # and its own TTL never expires it. A car that has not moved in five minutes
        # looks like one reporting live.
        v2v = BeaconTransceiver(port=0)
        assert v2v._fix_is_stale(fix(age_s=300.0)) is True
        assert v2v._fix_is_stale(fix(age_s=0.2)) is False

    def test_a_fix_from_this_clocks_future_is_stale_too(self):
        # The rule `PhoneGpsReader.is_stale` states and every other freshness
        # predicate in the tree follows.
        v2v = BeaconTransceiver(port=0)
        assert v2v._fix_is_stale(fix(age_s=-60.0)) is True

    def test_peers_are_not_ranged_against_a_stale_ego_fix(self):
        # A distance measured from a position the car left minutes ago.
        v2v = BeaconTransceiver(port=0)
        with v2v._lock:
            v2v._peers["other"] = _Peer(
                peer_id="other", lat=51.4901, lon=-0.20, speed_mps=18.0,
                heading_deg=90.0, lane_id=None, t_mono_heard=time.monotonic())
        assert v2v.peers(fix(age_s=0.1)), "a fresh fix should still see the peer"
        assert v2v.peers(fix(age_s=300.0)) == []


class TestPeersExpireEvenWhenOurOwnFixIsGone:

    def test_the_ttl_sweep_runs_before_the_ego_gate(self):
        # The sweep is the only eviction path there is, and it sat past an early
        # return -- so a drive whose own fix went invalid stopped evicting entirely.
        # Peer ids come off the wire, so the key space is not bounded by the fleet.
        v2v = BeaconTransceiver(port=0, peer_ttl_s=2.0)
        stale_heard = time.monotonic() - 10_000.0
        with v2v._lock:
            for i in range(50):
                v2v._peers[f"peer-{i}"] = _Peer(
                    peer_id=f"peer-{i}", lat=51.49, lon=-0.20, speed_mps=0.0,
                    heading_deg=0.0, lane_id=None, t_mono_heard=stale_heard)

        assert v2v.peers(GpsFix()) == []
        assert len(v2v._peers) == 0, (
            f"{len(v2v._peers)} expired peers survived because our own fix was invalid"
        )
