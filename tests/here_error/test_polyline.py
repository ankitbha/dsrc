import pytest

from here_error.polyline import _ALPHABET, decode


def test_decodes_the_published_example():
    # The example in the README of github.com/heremaps/flexible-polyline.
    pts = decode("BFoz5xJ67i1B1B7PzIhaxL7Y")
    assert pts == [(50.10228, 8.69821), (50.10201, 8.69567), (50.10063, 8.69150), (50.09878, 8.68752)]


def _encode_varint(n: int) -> str:
    out = ""
    while True:
        chunk = n & 0x1F
        n >>= 5
        if n:
            out += _ALPHABET[chunk | 0x20]
        else:
            return out + _ALPHABET[chunk]


def _zigzag(x: int) -> int:
    return ~(x << 1) if x < 0 else x << 1


def encode(points, precision=5, third=False):
    header = precision | ((2 if third else 0) << 4)
    s = _encode_varint(1) + _encode_varint(header)
    last = [0, 0, 0]
    for p in points:
        for i, v in enumerate(p):
            q = round(v * 10 ** precision)
            s += _encode_varint(_zigzag(q - last[i]))
            last[i] = q
    return s


def test_round_trip_with_negative_steps_and_a_third_dimension():
    pts2 = [(40.39253, -74.56386), (40.39100, -74.56001), (40.40000, -74.57000)]
    assert decode(encode(pts2)) == pytest.approx(pts2)
    pts3 = [(40.39253, -74.56386, 20.0), (40.39100, -74.56001, 25.0)]
    assert decode(encode(pts3, third=True)) == pytest.approx([p[:2] for p in pts3])


def test_rejects_other_versions_and_truncated_numbers():
    with pytest.raises(ValueError):
        decode("CFoz5xJ")
    with pytest.raises(ValueError):
        decode("BFoz5xJ67i1B1B7PzIhaxL7")  # last number cut off
