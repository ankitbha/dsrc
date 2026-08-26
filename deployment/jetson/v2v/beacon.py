"""Optional UDP-broadcast cooperation beacons between deployment units.

The sim's cooperation contract (specs/observation_schema.md) only allows
aggregate traffic-state sharing - density, mean speed, queue/congestion
estimates - not per-vehicle coordination. This channel honors that: each
unit broadcasts its own ego state at a low rate, and each receiver
aggregates whatever it heard into the nearby_av_* observation fields.
With zero peers the observation builder falls back to the spec's neutral
values, which is exactly the single-car deployment story.

Transport: JSON over UDP broadcast on the local subnet (e.g. a WiFi
hotspot in the test car, or two units on one AP). This is a stand-in for
DSRC/C-V2X radios; the paper should describe it as a cooperation-ready
interface, not a V2X implementation.

Disabled by default (v2v.enabled: false).
"""

from __future__ import annotations

import json
import math
import socket
import threading
import time
import uuid
from dataclasses import dataclass

from perception.observation_builder import PeerState
from sensors.gps_reader import GpsFix

# Re-exported: this module's callers have imported it from here since task 11, and
# it moved to `geo` only to break an import cycle -- see that module.
from geo import EARTH_RADIUS_M, haversine_m  # noqa: E402,F401


@dataclass
class _Peer:
    peer_id: str
    lat: float
    lon: float
    speed_mps: float
    heading_deg: float
    lane_id: int | None
    t_mono_heard: float


class BeaconTransceiver:
    def __init__(
        self,
        port: int = 47808,
        beacon_hz: float = 2.0,
        peer_ttl_s: float = 2.0,
        range_m: float = 150.0,
        unit_id: str | None = None,
    ) -> None:
        self.port = port
        self.beacon_hz = beacon_hz
        self.peer_ttl_s = peer_ttl_s
        self.range_m = range_m
        self.unit_id = unit_id or f"{socket.gethostname()}-{uuid.uuid4().hex[:6]}"

        self._tx = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self._tx.setsockopt(socket.SOL_SOCKET, socket.SO_BROADCAST, 1)
        self._rx = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self._rx.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self._rx.bind(("", port))
        self._rx.settimeout(0.5)

        self._lock = threading.Lock()
        self._peers: dict[str, _Peer] = {}
        self._latest_fix: GpsFix | None = None
        self._ego_lane: int | None = None
        self._running = False
        self._threads: list[threading.Thread] = []

    def start(self) -> "BeaconTransceiver":
        self._running = True
        for target, name in ((self._tx_loop, "v2v-tx"), (self._rx_loop, "v2v-rx")):
            t = threading.Thread(target=target, name=name, daemon=True)
            t.start()
            self._threads.append(t)
        return self

    def stop(self) -> None:
        self._running = False
        for t in self._threads:
            t.join(timeout=2.0)
        self._tx.close()
        self._rx.close()

    def update_ego(self, fix: GpsFix, lane_id: int | None = None) -> None:
        self._latest_fix = fix
        self._ego_lane = lane_id

    # ------------------------------------------------------------------

    def _tx_loop(self) -> None:
        period = 1.0 / max(self.beacon_hz, 0.1)
        while self._running:
            fix = self._latest_fix
            if fix is not None and fix.valid:
                msg = {
                    "id": self.unit_id,
                    "lat": fix.lat,
                    "lon": fix.lon,
                    "speed_mps": fix.speed_mps,
                    "heading_deg": fix.heading_deg,
                    "lane": getattr(self, "_ego_lane", None),
                    "t_wall": time.time(),
                }
                try:
                    self._tx.sendto(json.dumps(msg).encode(), ("255.255.255.255", self.port))
                except OSError:
                    pass
            time.sleep(period)

    def _rx_loop(self) -> None:
        while self._running:
            try:
                payload, _ = self._rx.recvfrom(4096)
            except socket.timeout:
                continue
            except OSError:
                return
            try:
                msg = json.loads(payload.decode())
            except (ValueError, UnicodeDecodeError):
                continue
            if not isinstance(msg, dict) or msg.get("id") in (None, self.unit_id):
                continue
            try:
                peer = _Peer(
                    peer_id=str(msg["id"]),
                    lat=float(msg["lat"]),
                    lon=float(msg["lon"]),
                    speed_mps=float(msg["speed_mps"]),
                    heading_deg=float(msg.get("heading_deg", 0.0)),
                    lane_id=int(msg["lane"]) if msg.get("lane") is not None else None,
                    t_mono_heard=time.monotonic(),
                )
            except (KeyError, TypeError, ValueError):
                continue
            with self._lock:
                self._peers[peer.peer_id] = peer

    # ------------------------------------------------------------------

    def peers(self, ego: GpsFix) -> list[PeerState]:
        """Cooperating AVs heard recently and within sim sensing range."""
        if ego is None or not ego.valid:
            return []
        now = time.monotonic()
        result = []
        with self._lock:
            for peer in list(self._peers.values()):
                if now - peer.t_mono_heard > self.peer_ttl_s:
                    del self._peers[peer.peer_id]
                    continue
                distance = haversine_m(ego.lat, ego.lon, peer.lat, peer.lon)
                if distance <= self.range_m:
                    result.append(
                        PeerState(
                            peer_id=peer.peer_id,
                            distance_m=distance,
                            speed_mps=peer.speed_mps,
                            lane_id=peer.lane_id,
                        )
                    )
        return result
