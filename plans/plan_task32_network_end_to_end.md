# Task 32 — End-to-end over the network backend, phone and Jetson apart

## The short version

The loop has never crossed a real network. Task 31's drive (`scripts/run_phone_drive.py`)
runs both ends in one process over `LoopbackAcceptor`, and every device run so far has
gone over `adb reverse`, which carries the data over USB. Task 32 is the step before any
USB work, so the data path must be the tailnet.

Hardware checked 2026-08-31: phone `ZY227VV4XC` attached over USB; `jetson-orin`
(100.90.108.88) online, ssh open, answering from this laptop via a relay at 57 ms. The
phone answers this laptop directly over IPv6 at 208 ms.

**One code gap blocks the whole task.** `SensingService.kt:329` constructs `LinkConfig()`
with defaults and the default host is `127.0.0.1`. Nothing can point the app at the
Jetson. `adb reverse` works only because loopback on the handset *is* the Jetson,
forwarded over USB.

## Scope boundary

In: a way to set the link address without a rebuild; deploying the Jetson side to the
Jetson; the drive harness over `TcpAcceptor`; one measured run; and recording which
network path the run actually used.

Out: USB as a data path (task 39); per-stage instrumentation (task 33); whether the
rates are correct (task 35). Also out: changing which side dials. The phone dials and the
Jetson listens because the Jetson's Tegra kernel has no `CONFIG_NF_CONNTRACK_MARK`, so
Tailscale cannot install its connmark rules and the Jetson cannot originate to tailnet
peers. The relay path observed from this laptop is consistent with that.

## Decisions taken

**The address comes from a pushed file.** A build-config value makes an address change a
rebuild, and the address is a property of how the two machines are connected on a given
day. An intent extra requires exporting a service that is deliberately
`exported="false"`, which task 26 worked around rather than relax. A file in the app's
external files directory is `adb push`-able, read once at service start, and falls back
to the current defaults when absent. It is read once, not re-read: an address change
during a session is a reconnect, not a configuration edit.

**One `LinkConfig` instance.** `SensingService.kt:515` builds a second one only to display
the address. Both are defaults today, so they agree. Once one is configurable the status
line will show the default while the link uses the file.

**The run records which path it used, direct or relayed.** A previous measurement in this
project — a 40 per cent cadence change — cannot be attributed to a cause because nothing
recorded whether Tailscale used a direct connection or a relay. The Jetson answered this
laptop by relay, not directly, so the phone may also relay, and a relayed hop adds tens of
milliseconds to the link segment. `Tick` already separates `link_ms` from `jetson_ms`, so
the segment is measured; what is missing is the path that produced it. The run captures
`tailscale status --json` at start and at end and stores both in the run record.

**The harness keeps its structure.** `run_phone_drive.py` already runs the real pipeline,
controller, mode holder and return path, paced at 30 Hz, and survives a redial. Two things
in it are loopback-specific: the acceptor, and the `Peer` class standing in for the app.
Task 32 replaces the first with `TcpAcceptor` and deletes the second, because the app is
the peer. Everything else is unchanged, so a difference between the loopback drive and
this one is a property of the network.

## The work

1. `LinkConfig.load(context)`: read the pushed file, validate through the existing `init`
   requirements, fall back to the defaults, and record which of the two it used.
2. One `LinkConfig` instance in `SensingService`, read by the link and by the status line.
3. `run_phone_drive.py --tcp <host>:<port>`: `TcpAcceptor`, no synthetic peer.
4. Deploy the Jetson side to `jetson-orin`: repository, Python environment, exported
   policy bundle. Check what is already present before assuming any of it is.
5. One run: phone on the tailnet, Jetson listening, no USB in the data path.
6. Record the network path, and report the link segment separately from the Jetson
   segment.

## Tests

- A pushed file with a host and port is used, and the status line shows the same one.
- A missing file falls back to the defaults, and the record says `default` rather than
  the address, so a run that silently used loopback does not read as a configured one.
- A malformed file is refused with a named reason and does not fall back silently: a
  mistyped address that quietly becomes `127.0.0.1` connects to nothing and presents as
  a link failure.
- Every `LinkConfig` validation already in `init` still fires through the new path:
  blank host, port outside 1..65535, backoff ordering.
- The harness over `TcpAcceptor` accepts a real connection, runs the loop, and survives
  the peer hanging up — the same assertions as task 31's drive.

## What the run establishes, and what it does not

It establishes that the loop runs with the two ends on separate machines over a real
network: that a decision reaches the phone, that every advisory the phone holds is about
a frame the Jetson processed, that the cadence rule holds against a channel that can
refuse, and that a redial is survived.

It does not establish latency under load, behaviour on a cellular link, or behaviour in a
moving vehicle. The Jetson is stationary on a desk and the phone is beside it. The link
segment measured here is a tailnet path between two machines in one building, and the
record will say which path that was.

## Open items carried in

- Task 27's HERE parse has never met a real response body.
- Task 28's observation vector cannot be feed-informed without a simulator-side change.
- `MAX_QUERY_RADIUS_M` (10 km) has not been checked against HERE v7's accepted range.
- Four-stamp time sync needs the phone to carry `t4` on its next ping, before task 33.
- From round 6: `achieved["camera_hz"]` derives from the enqueue count and overstates the
  rate the phone sustains when the channel evicts frames. Fixing it requires
  `Session.send` to return the displacement it already computes.
