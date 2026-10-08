"""Decoder for HERE's flexible polyline encoding."""
from __future__ import annotations

_ALPHABET = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789-_"
_DECODE = {c: i for i, c in enumerate(_ALPHABET)}


def _varints(text: str) -> list[int]:
    out, value, shift = [], 0, 0
    for ch in text:
        chunk = _DECODE[ch]
        value |= (chunk & 0x1F) << shift
        if chunk & 0x20:
            shift += 5
        else:
            out.append(value)
            value, shift = 0, 0
    if shift:
        raise ValueError("flexible polyline ends inside a number")
    return out


def _signed(x: int) -> int:
    return ~(x >> 1) if x & 1 else x >> 1


def decode(encoded: str) -> list[tuple[float, float]]:
    """Latitude and longitude pairs. A third dimension, when present, is read and discarded."""
    nums = _varints(encoded)
    if len(nums) < 2:
        raise ValueError("flexible polyline has no header")
    if nums[0] != 1:
        raise ValueError(f"unsupported flexible polyline version {nums[0]}")
    header = nums[1]
    precision = header & 15
    third = (header >> 4) & 7
    dims = 2 if third == 0 else 3
    body = nums[2:]
    if len(body) % dims:
        raise ValueError("flexible polyline body is not a whole number of points")
    scale = 10.0 ** precision
    lat = lng = 0
    out = []
    for i in range(0, len(body), dims):
        lat += _signed(body[i])
        lng += _signed(body[i + 1])
        out.append((lat / scale, lng / scale))
    return out
