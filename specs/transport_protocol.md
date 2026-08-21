# Transport Protocol

This file defines the wire format carried between the phone and the Jetson. It
is a cross-language contract: the Python implementation lives in
`deployment/jetson/transport/`, the Kotlin implementation in the phone app, and
both are checked against the frozen vectors in `specs/transport_golden_frames.json`.

The transport is **opaque**. It moves `(channel, header, payload)` and assigns
no meaning to any payload. Message semantics are defined separately.

`PROTOCOL_VERSION = 1`

## Connection Direction

**The phone always opens the connection. The Jetson always listens.**

This is not a preference. The Jetson's Tegra kernel is built with
`CONFIG_NF_CONNTRACK_MARK` unset, so Tailscale cannot install its connmark
rules and the Jetson cannot originate ordinary IP traffic to tailnet peers.
Inbound works, and both directions work on a phone-initiated connection. The
USB (`adb forward`) path tunnels over TCP and is unaffected, but it keeps the
same asymmetry so the two backends are interchangeable.

## Frame Layout

Every frame is:

```text
offset  size  field
0       4     payload_len   uint32, big-endian
4       2     header_len    uint16, big-endian
6       H     header        UTF-8 JSON object, H = header_len
6+H     P     payload       opaque bytes, P = payload_len
```

Total frame size is `6 + header_len + payload_len`.

All multi-byte integers on the wire are **big-endian** (network order). The
header is a JSON *object*; a JSON array, string or number at the top level is
a protocol error.

### Limits

```text
MAX_PAYLOAD_BYTES   4194304   (4 MiB)
MAX_HEADER_BYTES    8192
```

Both limits MUST be checked against the length fields **before** any buffer is
allocated or any payload byte is read. A receiver that allocates on an
unvalidated length is a denial-of-service against itself on the first
corrupted prefix.

`MAX_HEADER_BYTES` is far below the 65535 the field could express; the extra
room exists so a future header addition never needs a version bump for size.

## Header Fields

Required in every frame:

| key          | type   | meaning                                          |
|--------------|--------|--------------------------------------------------|
| `ch`         | string | channel id, from the channel table below         |
| `seq`        | int    | per-channel sequence number, starts at 0         |
| `t_mono_ns`  | int    | sender's monotonic clock at enqueue, nanoseconds |
| `t_wall_ns`  | int    | sender's wall clock at enqueue, nanoseconds      |
| `n`          | int    | payload length; MUST equal `payload_len`         |

`n` is deliberately redundant with the binary prefix. A disagreement means the
two sides have desynchronized and is a protocol error, not something to
reconcile.

Any other key is a header **extension** and carries message-level meaning the
transport ignores. Extensions are additive: a receiver MUST accept and preserve
keys it does not recognize, which is why the header is JSON rather than a
struct.

Two extension keys are reserved for the transport itself on every channel:
`hello` and `heartbeat`, both described below. A sender MUST NOT put either on
a caller's message. A message carrying one would be read as transport traffic
by the peer and consumed instead of delivered -- lost with no drop counted and
no sequence gap to show it, so invisible in the session summary as well.

### Clocks

Two clocks, following the discipline in `deployment/jetson/sensors/time_sync.py`:

- `t_mono_ns` is monotonic and meaningful **only on the device that produced
  it**. Never compare a phone monotonic value with a Jetson monotonic value.
  Cross-device comparison requires the offset estimate built on top of the
  handshake samples, which this protocol does not itself compute.
- `t_wall_ns` is UTC epoch nanoseconds, for log correlation. It can step.

### Sequence numbers

`seq` counts per `(session, channel)` and starts at 0. It is assigned at
enqueue, before any overflow decision, so a **gap in received `seq` is how a
receiver observes that the sender dropped something**. Sequence numbers do not
survive a reconnect: a new session restarts every channel at 0.

One exception, and both sides must implement it: the hello spends `control`
seq 0, so a session's own `control` traffic continues from **1**. A peer that
restarted control at 0 would duplicate the hello's seq, and the gap rule above
detects nothing -- it only fires on a seq greater than expected -- so the
divergence would be silent and would offset every control-channel gap
statistic permanently.

## Channels

One connection carries all traffic, multiplexed by `ch`.

| channel     | direction | priority | overflow    | depth | note                        |
|-------------|-----------|----------|-------------|-------|-----------------------------|
| `control`   | both      | high     | reliable    | 8     | hello, heartbeat            |
| `rate_cmd`  | down      | high     | reliable    | 16    | each command is distinct    |
| `advisory`  | down      | high     | latest_wins | 1     | a stale advisory is useless |
| `gps`       | up        | normal   | reliable    | 64    | each fix is unique          |
| `imu`       | up        | normal   | reliable    | 256   | 50-100 Hz, small            |
| `here`      | up        | normal   | reliable    | 16    | low rate, large-ish JSON    |
| `telemetry` | up        | normal   | reliable    | 32    | thermal, phone-side stats   |
| `camera`    | up        | bulk     | latest_wins | 1     | newest frame only           |

`direction` is documentation, not enforcement: the transport does not reject a
frame for arriving on an unexpected channel. Every channel in this table MUST
have a policy; there is no default policy for an unknown channel, and a frame
naming an unknown channel is a protocol error.

### Priority

The sender drains `high` before `normal` before `bulk`, and round-robins among
channels at the same priority so no channel starves a peer of equal priority.

Priority applies **between** frames, never within one. A frame already being
written is written to completion. The worst-case delay for a high-priority
message is therefore the transmit time of one frame — which is bounded only by
the frame size, so large payloads on a slow link are a rate-control problem,
not a transport problem.

Strict priority means a saturated `high` or `normal` tier can starve `bulk`
indefinitely. This is accepted: in this system the high and normal tiers carry
heartbeats, commands and small sensor records, and cannot saturate a link that
is carrying camera frames at all.

### Overflow

When a channel's outbound queue is full:

- `reliable` — **drop the oldest** queued message and enqueue the new one.
  Recency wins because for every reliable channel here a newer message is
  worth more than an older one. The drop is counted.
- `latest_wins` — the queue holds one message; a new message replaces any
  unsent one. The replaced message is counted as dropped.

Inbound queues use the same policies and depths. Nothing is dropped silently:
every drop increments a per-channel counter, and the receiver additionally
sees the resulting `seq` gap.

## Handshake

Immediately after the connection is established, **both sides send a hello and
then read the peer's hello**. Both send before either reads; the hello is small
enough that no send can block on a socket buffer, so this cannot deadlock.

A hello is an ordinary frame on `control` with `seq` 0, an empty payload, and a
reserved `hello` header extension:

```json
{
  "ch": "control",
  "seq": 0,
  "t_mono_ns": 123456789,
  "t_wall_ns": 1755648000000000000,
  "n": 0,
  "hello": {
    "protocol_version": 1,
    "device_id": "moto-g-power",
    "role": "phone"
  }
}
```

`role` is `phone` or `jetson`. `hello` is reserved: no message-level extension
may use that key on any channel.

Rules:

- The first frame in each direction MUST be a hello. A non-hello first frame is
  a protocol error and the connection is closed.
- If `protocol_version` differs from the local version, the connection is
  closed with both versions logged. No data frame is read from a mismatched
  peer.
- The four timestamps the handshake produces — local monotonic at send, the
  peer's `t_mono_ns` and `t_wall_ns`, local monotonic at receipt — are the
  first sample for clock-offset estimation. This protocol records them and
  computes nothing from them.

## Sessions

A **session** is one accepted connection. The listener accepts one at a time.

- Each session has a monotonically increasing integer id, distinct per listener
  process, and its own per-channel sequence counters starting at 0.
- **Displacement.** A new connection arriving while a session is live displaces
  it: the live session ends with reason `displaced` and the new one starts. The
  alternative, refusing the newcomer, would let one half-open drop that the
  Jetson has not yet noticed lock the phone out for the rest of a drive.
- **Stall.** Each side sends a keepalive every **1.0 s**. A session that has
  made no read progress for **5.0 s** ends with reason `stalled`. Without this,
  a half-open TCP connection looks healthy indefinitely.

  A keepalive is a `control` frame with an empty payload and the reserved
  `heartbeat` header extension set to `true`:

  ```json
  {"ch": "control", "seq": 7, "t_mono_ns": 123, "t_wall_ns": 456,
   "n": 0, "heartbeat": true}
  ```

  It consumes a `control` sequence number like any other frame. A receiver
  MUST consume it and MUST NOT deliver it to the application -- the transport
  generates keepalives, so it also absorbs them. The reserved key is honoured
  on `control` only: the same key arriving on a data channel is a caller's
  message and MUST be delivered.

  "No read progress" means a read completing, where the reader never asks for
  more than **8192 bytes** at a time. Measuring per completed *frame* instead
  would end any session whose frame takes longer than the timeout to arrive --
  at a 4 MiB limit and a 5 s timeout, any link under about 839 KB/s -- and the
  session would then reconnect and re-send, so the link would never recover.

  The floor this actually sets is low, and lower than a chunk-sized figure
  suggests. A read asks for `min(8192, bytes still outstanding)`, and a frame
  begins with a 6-byte prefix read and a header read, so a peer keeps a
  session alive by completing **any one read per timeout** -- a single small
  frame per two timeouts, on the order of tens of bytes per second at the
  shipped values, not kilobytes.

  That is a deliberate trade and not a strong liveness guarantee: a peer
  dribbling a few bytes per second holds a session open and is
  indistinguishable from a healthy one. What limits the damage is
  displacement, not the timer -- a healthy phone reconnecting takes the slot
  back. For a system whose only peer runs our own app, that is the right
  balance, but an implementation should not read this timeout as a bandwidth
  floor.

Session-end reasons: `closed_local`, `peer_closed`, `displaced`, `stalled`,
`framing_error`, `transport_error`.

A disconnect part-way through a frame ends the session. A partial frame is
never delivered to a caller.

## Errors

| condition                                   | outcome                     |
|---------------------------------------------|-----------------------------|
| `payload_len` > `MAX_PAYLOAD_BYTES`         | framing error, session ends |
| `header_len` > `MAX_HEADER_BYTES`           | framing error, session ends |
| `header_len` == 0                           | framing error, session ends |
| header is not valid UTF-8                   | framing error, session ends |
| header is not a JSON object                 | framing error, session ends |
| a required header key is missing            | framing error, session ends |
| `n` != `payload_len`                        | framing error, session ends |
| `ch` not in the channel table               | framing error, session ends |
| connection closes mid-frame                 | session ends, no delivery   |
| first frame is not a hello                  | session ends                |
| `protocol_version` mismatch                 | session ends                |

Framing errors are not recoverable by resynchronization. The stream carries no
frame delimiter to hunt for, so a desynchronized reader cannot find its place
again; ending the session and letting the phone reconnect is the only correct
response.

## Messages

Everything above moves bytes and assigns them no meaning. This section is the
meaning: one message type per channel, which is why no message carries a `kind`
field -- `ch` already says what it is.

Two rules decide where a message's data goes:

- **Small structured records ride in the header** as extension keys. That is
  what an additive JSON header is for, and encoding a hundred-byte GPS fix twice
  would be waste.
- **Blobs ride in the payload**, untouched. A camera JPEG obviously; and HERE's
  response body, which alone would exceed `MAX_HEADER_BYTES`.

### Timestamps

Every message carries `t_capture_mono_ns`: when its content came into being on
the sending device -- shutter time for a frame, fix time for a GPS record,
decision time for an advisory. It is the **same monotonic clock** as the
header's `t_mono_ns`, which is enqueue time on that same device, so

```text
queueing latency = t_mono_ns - t_capture_mono_ns
```

is a valid subtraction. It is the only latency this protocol lets you compute
without the offset estimate, and it is per-device: comparing a phone capture
stamp with a Jetson one remains forbidden.

### Unavailable values are `null`

A field whose value is unknown is **present and `null`**. Never absent, and
never a sentinel.

Absent would conflate "the sensor said nothing" with "the sender is an older
build that never had this field", and the phone app and Jetson runtime are
deployed separately. A sentinel that is a legitimate value in some other unit is
a silent-corruption source.

**NaN and Infinity must never reach the wire.** `MAX_HEADER_BYTES`-level framing
refuses them (see Frame Layout), because Python writes a bare `NaN` token that a
strict parser elsewhere rejects. An encoder converts a non-finite value to
`null` before framing; a decoder may map `null` back to NaN for an in-process
type that uses it, and the round trip must be lossless in both directions.

### The message set

`t_capture_mono_ns` is required on all of them and omitted below.

| channel | header fields | payload |
|---|---|---|
| `camera` | `frame_id`, `width`, `height`, `format`, `quality` | JPEG bytes |
| `gps` | `valid`, `lat`, `lon`, `speed_mps`, `heading_deg`, `fix_quality`, `num_sats`, `hdop`, `altitude_m`, `utc_epoch_ns` | empty |
| `imu` | `ax`, `ay`, `az`, `gx`, `gy`, `gz`, `accuracy` | empty |
| `here` | `request_url`, `status`, `content_type`, `query_lat`, `query_lon`, `query_radius_m`, `t_request_mono_ns`, `t_response_mono_ns` | response body bytes |
| `advisory` | `rec_speed_mps`, `rec_speed_display`, `current_speed_display`, `units`, `headway_target_s`, `lane_text`, `merge_text`, `traffic_text`, `confidence`, `confidence_label`, `action` | empty |
| `rate_cmd` | `rates`, `trigger`, `shadow` | empty |
| `telemetry` | `thermal_status`, `thermal_headroom`, `achieved`, `dropped`, `here_calls`, `here_errors` | empty |
| `control` | owned by the transport: `hello`, `heartbeat` | empty |

Nullable fields: every numeric field of `gps` except `valid`, `fix_quality` and
`num_sats`; `quality` on `camera`; `accuracy` on `imu`; `thermal_headroom` on
`telemetry`; `content_type` on `here`.

`action` is an object with exactly the four v2 action heads, and their allowed
values are the ones in `specs/action_schema.md`:

```json
{"desired_speed_bin": "nominal", "desired_headway_bin": "normal",
 "lane_preference": "keep", "merge_mode": "normal"}
```

`rates` and `achieved` are objects keyed `camera_hz`, `gps_hz`, `imu_hz`,
`here_hz`, with values in `(0, 1000]` Hz for `rates`. `dropped` is keyed
`camera`, `gps`, `imu`, `here`, and its values are **integers** -- they are
counts, and a fractional one is a bug in the sender rather than something to
truncate. `units` is one of `mph`, `kmh`, `mps`. `shadow` is a boolean: whether
the command was gated for real or only recorded.

**These three nested objects are additive.** Every listed key must be present,
and an unrecognised key is **ignored, not refused**. The sensor set will grow,
and refusing an unknown key would break a rolling deploy in both directions at
once -- a new sender's every command dropped by an old receiver, and an old
sender's by a new one. Ignoring unknown keys is safe precisely because the known
ones are required, so a misspelled key still surfaces as a missing key.

`action` is the exception and is **strict**: exactly the four heads, no more and
no fewer, because they are a closed set defined by `specs/action_schema.md`
rather than an extensible list.

### A malformed message is not a malformed stream

| condition | outcome |
|---|---|
| a required field absent | message dropped, counted |
| a field of the wrong JSON type | message dropped, counted |
| an action value outside the schema | message dropped, counted |
| `units` not one of the three | message dropped, counted |
| a rate outside `(0, 1000]` Hz | message dropped, counted |
| `lat`/`lon` out of range while `valid` | message dropped, counted |
| a payload on a channel whose message carries none | message dropped, counted |
| a missing key in a nested object | message dropped, counted |
| an extra head in `action` | message dropped, counted |
| a non-integer value in `dropped` | message dropped, counted |

The session stays open, and the drop is counted per channel and per reason. The
reasons are a closed vocabulary: `missing_field`, `wrong_type`,
`null_not_allowed`, `non_finite`, `out_of_range`, `unknown_value`,
`unexpected_payload`, `reserved_key`, `no_typed_message`. One number cannot
answer whether four thousand drops were one bad field or four, which is the
point of counting them at all.

This is deliberately unlike a framing error, which ends the session. The
difference is recoverability: a framing error means the byte stream has
desynchronised and there is no delimiter to hunt for, so the reader can never
find its place again. A message that framed correctly proves the stream is
fine -- one bad record costs one record, and reconnecting the phone over a
single malformed IMU sample at 50 Hz would be far worse than dropping it.

## Golden Vectors

`specs/transport_golden_frames.json` holds the frozen encoding of a set of
cases. Every implementation MUST encode each case to exactly the recorded bytes
and decode those bytes back to the recorded fields.

The file is frozen. Changing any recorded byte is a protocol change and
requires a `PROTOCOL_VERSION` bump, because an existing peer already agrees
with the old bytes. **Adding a case is not** -- it constrains nothing that was
already agreed. Since the generator rewrites the whole file, a regeneration must
be checked to have left every pre-existing case byte-identical, and a test does
that rather than a habit.

Cases cover: an empty payload; a typical JPEG-sized payload; a payload at
`MAX_PAYLOAD_BYTES`; non-ASCII text in a header extension; an integer at the
`2**53` boundary, where JSON number handling diverges between languages; and a
hello frame.

Payloads in the file are described by a deterministic generator rather than
stored literally, so a 4 MiB case costs nothing to commit.
