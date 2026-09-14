"""Guard src/safety/ against drift from the golden safety contract.

Counterpart to deployment/jetson/tests/test_safety_contract.py, which checks
the vendored deployment/jetson/policy/safety_gate.py against the same file.
If src/safety/ is ever deleted this test goes away with it -- the one that
matters is the vendored one, which carries no import of this package at all.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
from pathlib import Path

from src.safety.constraints import SafetyConstraints
from src.safety.safety_layer import SafetyContext

GOLDEN_PATH = Path(__file__).resolve().parents[1] / "specs" / "safety_contract_golden.json"


def _load_golden() -> dict:
    return json.loads(GOLDEN_PATH.read_text())


def _constraints_as_golden(cls) -> dict:
    return {f.name: f.default for f in dataclasses.fields(cls)}


def _context_fields_as_golden(cls) -> list[dict]:
    out = []
    for f in dataclasses.fields(cls):
        if f.default is dataclasses.MISSING:
            out.append({"name": f.name, "has_default": False, "default": None})
        else:
            out.append({"name": f.name, "has_default": True, "default": f.default})
    return out


def _recomputed_hash(golden: dict) -> str:
    payload = {
        "safety_constraints": golden["safety_constraints"],
        "safety_context_fields": golden["safety_context_fields"],
    }
    canonical = json.dumps(payload, sort_keys=False, allow_nan=True)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:16]


def test_golden_hash_is_self_consistent() -> None:
    golden = _load_golden()
    assert golden["hash"] == _recomputed_hash(golden)


def test_safety_constraints_matches_golden() -> None:
    golden = _load_golden()
    assert _constraints_as_golden(SafetyConstraints) == golden["safety_constraints"]


def test_safety_context_fields_match_golden() -> None:
    golden = _load_golden()
    assert _context_fields_as_golden(SafetyContext) == golden["safety_context_fields"]
