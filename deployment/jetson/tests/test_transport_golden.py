"""The frozen cross-language encodings.

specs/transport_golden_frames.json is the artifact that keeps the Python and
Kotlin codecs honest without either device present. Every case must encode to
exactly the recorded bytes and decode back to the recorded fields; a mismatch
names the guilty implementation instead of surfacing later as a mysterious
desync during the first interop attempt.

These tests read the file. They never regenerate it -- a test that rewrote its
own expectations would agree with any bug. Regeneration is a deliberate act via
scripts/generate_transport_golden_frames.py, and it is a protocol change.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from transport.channels import Channel
from transport.frames import (
    MAX_PAYLOAD_BYTES,
    PROTOCOL_VERSION,
    Frame,
    decode,
    encode,
    encode_header,
)

REPO_ROOT = Path(__file__).resolve().parents[3]
GOLDEN = REPO_ROOT / "specs" / "transport_golden_frames.json"
DOCUMENT = json.loads(GOLDEN.read_text())
CASES = DOCUMENT["cases"]
IDS = [case["name"] for case in CASES]


def pattern_payload(length: int) -> bytes:
    """The generator the file documents. Period 256, so it tiles."""
    block = bytes((index * 37 + 11) % 256 for index in range(256))
    whole, remainder = divmod(length, 256)
    return block * whole + block[:remainder]


def frame_for(case) -> Frame:
    fields = case["frame"]
    return Frame(
        channel=Channel(fields["ch"]),
        seq=fields["seq"],
        t_mono_ns=fields["t_mono_ns"],
        t_wall_ns=fields["t_wall_ns"],
        payload=pattern_payload(case["payload"]["length"]),
        extensions=fields["extensions"],
    )


# -- the file itself ---------------------------------------------------------


def test_the_file_is_for_this_protocol_version():
    assert DOCUMENT["protocol_version"] == PROTOCOL_VERSION


def test_the_file_declares_itself_frozen():
    assert DOCUMENT["frozen"] is True
    assert DOCUMENT["byte_order"] == "big-endian"


def test_case_names_are_unique():
    assert len(IDS) == len(set(IDS))


def test_the_payload_generator_is_reproducible_from_its_description():
    assert DOCUMENT["payload_generator"] == "payload[i] = (i * 37 + 11) % 256"
    naive = bytes((index * 37 + 11) % 256 for index in range(1000))
    assert pattern_payload(1000) == naive


def test_the_required_categories_are_covered():
    """The cases exist to catch specific cross-language failures. Losing one
    silently would leave that failure unguarded."""
    names = set(IDS)
    assert {"empty_payload", "max_payload", "non_ascii_extension", "large_ints", "hello"} <= names


# -- encoding ---------------------------------------------------------------


@pytest.mark.parametrize("case", CASES, ids=IDS)
def test_encodes_to_exactly_the_recorded_bytes(case):
    encoded = encode(frame_for(case))
    assert encoded[:6].hex() == case["prefix_hex"]
    assert len(encoded) == case["frame_len"]
    assert hashlib.sha256(encoded).hexdigest() == case["frame_sha256"]


@pytest.mark.parametrize("case", CASES, ids=IDS)
def test_header_bytes_match_exactly(case):
    header_bytes = encode_header(frame_for(case).header())
    assert header_bytes.hex() == case["header_hex"]
    assert len(header_bytes) == case["header_len"]


@pytest.mark.parametrize("case", CASES, ids=IDS)
def test_decodes_back_to_the_recorded_fields(case):
    frame = frame_for(case)
    decoded = decode(encode(frame))
    fields = case["frame"]
    assert decoded.channel.value == fields["ch"]
    assert decoded.seq == fields["seq"]
    assert decoded.t_mono_ns == fields["t_mono_ns"]
    assert decoded.t_wall_ns == fields["t_wall_ns"]
    assert decoded.extensions == fields["extensions"]
    assert len(decoded.payload) == case["payload"]["length"]
    assert decoded.payload == frame.payload


@pytest.mark.parametrize("case", CASES, ids=IDS)
def test_the_recorded_header_bytes_are_themselves_decodable(case):
    """Decode from the file's bytes, not from what this implementation just
    produced -- otherwise a symmetric bug passes both directions."""
    header = json.loads(bytes.fromhex(case["header_hex"]).decode("utf-8"))
    fields = case["frame"]
    assert header["ch"] == fields["ch"]
    assert header["seq"] == fields["seq"]
    assert header["n"] == case["payload"]["length"]
    for key, value in fields["extensions"].items():
        assert header[key] == value


# -- the properties the cases exist to pin ----------------------------------


def case_named(name):
    return next(case for case in CASES if case["name"] == name)


def test_the_max_payload_case_really_sits_on_the_limit():
    assert case_named("max_payload")["payload"]["length"] == MAX_PAYLOAD_BYTES


def test_large_integers_are_recorded_without_precision_loss():
    """A wall clock in nanoseconds is past 2**53; an implementation that puts
    these through a float loses the low digits."""
    header = json.loads(bytes.fromhex(case_named("large_ints")["header_hex"]).decode())
    assert header["seq"] == 9_007_199_254_740_993
    assert header["t_wall_ns"] == 1_755_648_000_987_654_321
    assert header["big"] == 9_223_372_036_854_775_807
    assert header["neg"] == -9_007_199_254_740_993


def test_non_ascii_is_recorded_as_raw_utf8():
    raw = bytes.fromhex(case_named("non_ascii_extension")["header_hex"])
    assert "señal".encode("utf-8") in raw
    assert b"\\u" not in raw


def test_recorded_headers_have_sorted_keys_at_every_level():
    for case in CASES:
        text = bytes.fromhex(case["header_hex"]).decode("utf-8")
        _assert_sorted(json.loads(text, object_pairs_hook=list), case["name"], text)


def _assert_sorted(node, name, text):
    if isinstance(node, list) and node and isinstance(node[0], tuple):
        keys = [key for key, _ in node]
        assert keys == sorted(keys), f"{name}: unsorted keys {keys} in {text}"
        for _, value in node:
            _assert_sorted(value, name, text)
    elif isinstance(node, list):
        for item in node:
            _assert_sorted(item, name, text)


def test_the_hello_case_names_this_protocol_version():
    header = json.loads(bytes.fromhex(case_named("hello")["header_hex"]).decode())
    assert header["hello"]["protocol_version"] == PROTOCOL_VERSION
    assert header["hello"]["role"] == "phone"


# -- the message layer, pinned in the same file -------------------------------

MESSAGE_CASES = [case for case in CASES if case["name"].startswith("message_")]
MESSAGE_IDS = [case["name"] for case in MESSAGE_CASES]


def test_the_file_carries_message_cases_as_well_as_frame_cases():
    assert len(MESSAGE_CASES) >= 7
    frame_cases = [case for case in CASES if not case["name"].startswith("message_")]
    assert len(frame_cases) >= 8


def test_every_channel_with_a_typed_message_has_a_case():
    from transport.messages import MESSAGE_FOR_CHANNEL

    covered = {case["frame"]["ch"] for case in MESSAGE_CASES}
    assert covered == {channel.value for channel in MESSAGE_FOR_CHANNEL}


@pytest.mark.parametrize("case", MESSAGE_CASES, ids=MESSAGE_IDS)
def test_a_message_case_decodes_to_a_valid_typed_message(case):
    """Decoded from the file's own recorded bytes, not from what this
    implementation just produced -- otherwise a symmetric bug passes."""
    from transport.channels import Channel
    from transport.messages import decode_message

    header = json.loads(bytes.fromhex(case["header_hex"]).decode("utf-8"))
    extensions = {k: v for k, v in header.items()
                  if k not in {"ch", "seq", "t_mono_ns", "t_wall_ns", "n"}}
    payload = pattern_payload(case["payload"]["length"])
    message = decode_message(Channel(header["ch"]), extensions, payload)
    assert message.t_capture_mono_ns == extensions["t_capture_mono_ns"]


def test_the_all_null_gps_case_really_is_all_null():
    """The null convention is a cross-language agreement, so it is pinned by
    bytes rather than by a round trip."""
    case = next(c for c in MESSAGE_CASES if c["name"] == "message_gps_all_null")
    header = json.loads(bytes.fromhex(case["header_hex"]).decode("utf-8"))
    for field in ("lat", "lon", "speed_mps", "heading_deg", "hdop", "altitude_m",
                  "utc_epoch_ns"):
        assert field in header, f"{field} must be present, not absent"
        assert header[field] is None, f"{field} should be null"
    assert header["valid"] is False
    assert "NaN" not in bytes.fromhex(case["header_hex"]).decode("utf-8")


def test_no_message_case_contains_a_nan_token():
    """Python would write a bare NaN that a strict parser elsewhere rejects; the
    null convention exists to prevent it, and this checks the bytes."""
    for case in MESSAGE_CASES:
        text = bytes.fromhex(case["header_hex"]).decode("utf-8")
        assert "NaN" not in text and "Infinity" not in text, case["name"]


def test_the_advisory_case_carries_both_the_display_text_and_the_action():
    case = next(c for c in MESSAGE_CASES if c["name"] == "message_advisory")
    header = json.loads(bytes.fromhex(case["header_hex"]).decode("utf-8"))
    assert header["lane_text"] == "Keep lane"
    assert header["units"] == "mph"
    assert set(header["action"]) == {
        "desired_speed_bin", "desired_headway_bin", "lane_preference", "merge_mode",
    }


def test_the_note_says_adding_a_case_is_not_a_protocol_change():
    """Because regeneration rewrites the whole file, and a reader has to know
    which kind of change they are looking at."""
    assert "ADDING a case is not" in DOCUMENT["note"]


def test_regeneration_leaves_every_recorded_case_byte_identical():
    """The guard the note promises. The generator rewrites the whole file, so
    the only thing standing between "added a message case" and "silently moved
    a frame case" is this check -- and it has to be a test, not a habit.
    """
    import subprocess
    import sys
    import tempfile

    script = REPO_ROOT / "scripts" / "generate_transport_golden_frames.py"
    with tempfile.TemporaryDirectory() as tmp:
        out = Path(tmp) / "regenerated.json"
        result = subprocess.run(
            [sys.executable, str(script), "--out", str(out), "--force"],
            capture_output=True, text=True, timeout=300,
        )
        assert result.returncode == 0, result.stderr
        fresh = json.loads(out.read_text())
    recorded = {case["name"]: case for case in DOCUMENT["cases"]}
    regenerated = {case["name"]: case for case in fresh["cases"]}
    assert set(recorded) == set(regenerated), "the case set moved"
    for name, case in recorded.items():
        assert regenerated[name]["frame_sha256"] == case["frame_sha256"], name
        assert regenerated[name]["header_hex"] == case["header_hex"], name
        assert regenerated[name]["prefix_hex"] == case["prefix_hex"], name
