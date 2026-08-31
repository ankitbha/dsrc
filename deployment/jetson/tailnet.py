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
    collisions: list[str] = []
    for peer in (status.get("Peer") or {}).values():
        if not peer.get("Online"):
            continue
        cur_addr = peer.get("CurAddr") or ""
        relay = peer.get("Relay") or ""
        name = peer.get("HostName", "?")
        if name in peers:
            # Tailscale enforces uniqueness on `DNSName`, not on `HostName`, and this
            # project's fleet is two handsets of the same model. Keying on the name
            # alone dropped one of them and which one survived was dictionary order.
            # Recorded rather than overwritten: a lost peer that nothing mentions is
            # the failure this whole module exists to avoid one level up.
            if name not in collisions:
                collisions.append(name)
            # Unique by construction. Suffixing the first address alone was not: an
            # online peer with an empty `TailscaleIPs` -- a shape this suite already
            # constructs -- produced the constant suffix `[?]`, so a third peer on the
            # same name overwrote the second and one was still lost. Counting the
            # survivors then disagreed with the collision list.
            addresses = peer.get("TailscaleIPs") or []
            suffix = addresses[0] if addresses else f"#{len(peers)}"
            candidate = f"{name} [{suffix}]"
            while candidate in peers:
                candidate = f"{name} [{suffix} #{len(peers)}]"
            name = candidate
        peers[name] = {
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
        # What this machine's tailnet REACHABILITY looks like. Not the path any
        # particular session took -- a peer is listed here whether or not a session
        # crossed the tailnet to reach it. `path_for_address` answers that question.
        "peers": peers,
        "hostname_collisions": collisions,
    }


def path_for_address(status: dict[str, Any], address: str) -> dict[str, Any]:
    """The tailnet path to the peer holding `address`, or a statement that it is not one.

    `address` is the session's own remote address, `host:port`, taken from the accepted
    socket. That is the ground truth of where the bytes went: `127.0.0.1` means the
    phone dialled loopback and the data crossed USB via `adb reverse`; a 100.x address
    means it crossed the tailnet.

    Without this, a record could only say how this machine WOULD reach each online
    peer, which is the same whether or not the session used that route -- so a USB run
    and a tailnet run wrote the same network record.
    """
    host = address.rsplit(":", 1)[0].strip("[]") if address else ""
    if not host:
        return {"session_peer": address, "over_tailnet": False, "path": "unknown",
                "detail": "the session recorded no peer address"}
    if not status.get("available"):
        return {"session_peer": address, "over_tailnet": False, "path": "unknown",
                "detail": f"tailscale status unavailable: {status.get('reason')}"}
    for name, peer in (status.get("peers") or {}).items():
        if host in (peer.get("addresses") or []):
            path = peer.get("path")
            return {
                "session_peer": address,
                "over_tailnet": True,
                "peer": name,
                "path": path,
                # Only on a relayed path. Tailscale sets `Relay` on a direct connection
                # too -- it names the region that WOULD be used -- which is why `path`
                # keys on `CurAddr`. Returning it unconditionally made `run_demo` print
                # "direct via nyc", attributing a relay hop of tens of milliseconds to a
                # connection that made none, which is the one thing this module exists
                # to get right.
                "relay": peer.get("relay") if path == "relay" else "",
            }
    # Belonging to the tailnet and having an online peer are different questions, and
    # only the first is a property of the address. `peer_paths` lists online peers
    # only, so a run that ended BECAUSE the phone left the tailnet -- which is the case
    # the redial timeout exists for -- looked up its own peer after it went offline and
    # recorded `not_tailnet`: the same two field values an `adb reverse` run writes.
    if _in_tailnet_range(host):
        return {
            "session_peer": address,
            "over_tailnet": True,
            "path": "peer_offline",
            "detail": f"{host} is a tailnet address and no online peer holds it now",
        }
    return {
        "session_peer": address,
        "over_tailnet": False,
        "path": "not_tailnet",
        # Named, because this is the case that matters: it is what an `adb reverse`
        # run looks like, and the whole point of task 32 is not to be that run.
        "detail": f"{host} is not a tailnet address of any online peer",
    }


def _in_tailnet_range(host: str) -> bool:
    """Whether an address is in the range Tailscale assigns from.

    100.64.0.0/10, the CGNAT block, and the `fd7a:115c:a1e0::/48` ULA prefix. Address
    membership does not depend on any peer being online, which is the point: it
    separates "this was not the tailnet" from "this was, and the peer has since gone".
    """
    import ipaddress

    try:
        parsed = ipaddress.ip_address(host)
    except ValueError:
        return False
    if parsed.version == 4:
        return parsed in ipaddress.ip_network("100.64.0.0/10")
    return parsed in ipaddress.ip_network("fd7a:115c:a1e0::/48")
