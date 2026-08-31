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
