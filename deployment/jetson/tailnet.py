"""Which network path a tailnet peer is reached over, direct or relayed.

A previous measurement in this project -- a 40 per cent change in cadence -- cannot
be attributed to a cause, because nothing recorded whether Tailscale used a direct
connection or a relay. The two differ by tens of milliseconds on the segment the
link measurement is about, so a run that does not record the path cannot separate a
slow link from a slow relay afterwards.

`Tick` already reports `link_ms` apart from `jetson_ms`. This supplies the other half:
what produced that segment.

Reading only. Nothing here changes the tailnet, and a machine with no `tailscale` on
its path records that fact rather than failing the run.
"""

from __future__ import annotations

import json
import subprocess
from typing import Any


def peer_paths(timeout_s: float = 10.0) -> dict[str, Any]:
    """Every online peer, and whether this machine reaches it directly or by relay.

    `CurAddr` is set when the connection is direct; `Relay` names the DERP region when
    it is not. Both are reported rather than one derived from the other, because an
    empty `CurAddr` and a named relay are two observations and collapsing them would
    lose the case where neither is set -- a peer that is online and not yet connected
    to at all.
    """
    try:
        out = subprocess.run(["tailscale", "status", "--json"],
                             capture_output=True, text=True, timeout=timeout_s)
    except (OSError, subprocess.SubprocessError) as exc:
        return {"available": False, "reason": f"{type(exc).__name__}: {exc}"}
    if out.returncode != 0:
        return {"available": False, "reason": f"exit {out.returncode}"}
    try:
        status = json.loads(out.stdout)
    except ValueError as exc:
        return {"available": False, "reason": f"unparseable status: {exc}"}

    peers = {}
    for peer in (status.get("Peer") or {}).values():
        if not peer.get("Online"):
            continue
        cur_addr = peer.get("CurAddr") or ""
        relay = peer.get("Relay") or ""
        peers[peer.get("HostName", "?")] = {
            "addresses": peer.get("TailscaleIPs", []),
            "direct_addr": cur_addr,
            "relay": relay,
            # Named rather than inferred at read time, so a reader of the artifact does
            # not have to know the CurAddr/Relay convention to know what happened.
            "path": "direct" if cur_addr else ("relay" if relay else "unconnected"),
            "rx_bytes": peer.get("RxBytes"),
            "tx_bytes": peer.get("TxBytes"),
        }
    return {
        "available": True,
        "self": (status.get("Self") or {}).get("HostName"),
        "peers": peers,
    }
