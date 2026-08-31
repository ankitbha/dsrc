"""Recording which network path a run used.

`link_ms` says how long the link segment took; it cannot say whether that was a
direct connection or a relay, and the two differ by tens of milliseconds. A previous
measurement in this project -- a 40 per cent cadence change -- cannot be attributed
to a cause because nothing recorded the path.
"""

from __future__ import annotations

import json
import subprocess

import tailnet


def _fake_run(payload, returncode=0):
    def run(*args, **kwargs):
        return subprocess.CompletedProcess(args, returncode, stdout=payload, stderr="")
    return run


class TestThePathIsNamedNotInferred:

    def test_a_direct_connection_is_reported_as_direct(self, monkeypatch):
        monkeypatch.setattr(subprocess, "run", _fake_run(json.dumps({
            "Self": {"HostName": "laptop"},
            "Peer": {"k": {"HostName": "jetson-orin", "Online": True,
                           "TailscaleIPs": ["100.90.108.88"],
                           "CurAddr": "216.165.95.173:41641", "Relay": "nyc"}},
        })))
        peers = tailnet.peer_paths()["peers"]
        assert peers["jetson-orin"]["path"] == "direct"

    def test_a_relayed_connection_is_reported_as_relayed(self, monkeypatch):
        # `Relay` is set on a direct connection too -- it names the region that would
        # be used -- so the path must key on `CurAddr`, not on `Relay` being present.
        monkeypatch.setattr(subprocess, "run", _fake_run(json.dumps({
            "Self": {"HostName": "laptop"},
            "Peer": {"k": {"HostName": "jetson-orin", "Online": True,
                           "TailscaleIPs": ["100.90.108.88"],
                           "CurAddr": "", "Relay": "nyc"}},
        })))
        peer = tailnet.peer_paths()["peers"]["jetson-orin"]
        assert peer["path"] == "relay"
        assert peer["relay"] == "nyc"

    def test_neither_set_is_its_own_state(self, monkeypatch):
        # An online peer with no connection yet is a third case, and collapsing it into
        # either of the other two would report a connection that does not exist.
        monkeypatch.setattr(subprocess, "run", _fake_run(json.dumps({
            "Self": {"HostName": "laptop"},
            "Peer": {"k": {"HostName": "jetson-orin", "Online": True,
                           "TailscaleIPs": [], "CurAddr": "", "Relay": ""}},
        })))
        assert tailnet.peer_paths()["peers"]["jetson-orin"]["path"] == "unconnected"

    def test_offline_peers_are_omitted(self, monkeypatch):
        monkeypatch.setattr(subprocess, "run", _fake_run(json.dumps({
            "Self": {"HostName": "laptop"},
            "Peer": {"k": {"HostName": "gone", "Online": False, "CurAddr": ""}},
        })))
        assert tailnet.peer_paths()["peers"] == {}


class TestAMachineWithoutTailscale:

    def test_a_missing_binary_is_recorded_rather_than_raised(self, monkeypatch):
        def raise_oserror(*args, **kwargs):
            raise FileNotFoundError("tailscale")
        monkeypatch.setattr(subprocess, "run", raise_oserror)
        result = tailnet.peer_paths()
        assert result["available"] is False
        assert "FileNotFoundError" in result["reason"]

    def test_unparseable_status_is_recorded(self, monkeypatch):
        monkeypatch.setattr(subprocess, "run", _fake_run("not json"))
        result = tailnet.peer_paths()
        assert result["available"] is False
        assert "unparseable" in result["reason"]

    def test_a_nonzero_exit_is_recorded(self, monkeypatch):
        monkeypatch.setattr(subprocess, "run", _fake_run("", returncode=1))
        assert tailnet.peer_paths()["available"] is False


class TestTheSessionsOwnPathIsNotReachability:
    """`peer_paths` says how this machine WOULD reach each peer. That is the same
    whether or not a given session used that route, so a run carried over USB wrote
    the same network record as one carried over the tailnet."""

    def _status(self):
        return {
            "available": True,
            "self": "laptop",
            "peers": {
                "moto g power": {"addresses": ["100.75.142.126"], "direct_addr": "",
                                 "relay": "nyc", "path": "relay"},
            },
            "hostname_collisions": [],
        }

    def test_a_loopback_peer_is_named_as_not_over_the_tailnet(self):
        # This is what `adb reverse` looks like: the phone dials 127.0.0.1 and the data
        # crosses USB. It is the run task 32 exists not to be.
        result = tailnet.path_for_address(self._status(), "127.0.0.1:47811")
        assert result["over_tailnet"] is False
        assert result["path"] == "not_tailnet"

    def test_a_tailnet_peer_resolves_to_its_path(self):
        result = tailnet.path_for_address(self._status(), "100.75.142.126:41234")
        assert result["over_tailnet"] is True
        assert result["path"] == "relay"
        assert result["relay"] == "nyc"
        assert result["peer"] == "moto g power"

    def test_the_two_cases_do_not_write_the_same_record(self):
        # The property the whole change is for.
        over_usb = tailnet.path_for_address(self._status(), "127.0.0.1:47811")
        over_tailnet = tailnet.path_for_address(self._status(), "100.75.142.126:41234")
        assert over_usb != over_tailnet

    def test_an_empty_address_is_named_rather_than_guessed(self):
        result = tailnet.path_for_address(self._status(), "")
        assert result["path"] == "unknown"
        assert "no peer address" in result["detail"]

    def test_an_unavailable_status_does_not_claim_a_path(self):
        result = tailnet.path_for_address({"available": False, "reason": "exit 1"},
                                          "100.75.142.126:41234")
        assert result["over_tailnet"] is False
        assert result["path"] == "unknown"


class TestTwoPeersWithOneHostname:

    def test_a_collision_keeps_both_and_records_it(self, monkeypatch):
        # Tailscale enforces uniqueness on `DNSName`, not `HostName`, and this fleet is
        # two handsets of the same model. Keying on the name alone dropped one, and
        # which one survived was dictionary order.
        monkeypatch.setattr(subprocess, "run", _fake_run(json.dumps({
            "Self": {"HostName": "laptop"},
            "Peer": {
                "a": {"HostName": "moto g power", "Online": True,
                      "TailscaleIPs": ["100.75.142.126"], "CurAddr": "1.2.3.4:1", "Relay": "nyc"},
                "b": {"HostName": "moto g power", "Online": True,
                      "TailscaleIPs": ["100.86.4.9"], "CurAddr": "", "Relay": "nyc"},
            },
        })))
        result = tailnet.peer_paths()
        assert len(result["peers"]) == 2, result["peers"]
        assert result["hostname_collisions"] == ["moto g power"]
        # And both remain addressable, which is what the session lookup needs.
        assert tailnet.path_for_address(result, "100.75.142.126:1")["path"] == "direct"
        assert tailnet.path_for_address(result, "100.86.4.9:1")["path"] == "relay"
