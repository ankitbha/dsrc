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

GOLDEN = Path(__file__).resolve().parents[3] / "specs" / "transport_golden_frames.json"
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
