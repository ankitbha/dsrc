"""Guard the vendored safety contract against drift from src/safety/.

Unlike test_sim_contract.py, this carries no ``importorskip``: the reference
this checks against is a committed file (``specs/safety_contract_golden.json``),
not a sibling package that can be deleted out from under it. It must run
everywhere and pass or fail on its own -- task 143's finding is that a check
written as "compare the vendored copy against the original" goes vacuous the
moment the original disappears; a golden file cannot go vacuous that way.

The counterpart on the other side is tests/test_safety_contract_matches_golden.py,
which checks src/safety/ against the same file. Neither test imports the other
side's module: a change to either SafetyConstraints or SafetyContext must edit
this JSON file, which then shows up in a diff.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
from pathlib import Path

GOLDEN_PATH = Path(__file__).resolve().parents[3] / "specs" / "safety_contract_golden.json"


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


def test_golden_file_exists() -> None:
    assert GOLDEN_PATH.is_file(), GOLDEN_PATH


def test_golden_hash_is_self_consistent() -> None:
    """The one thing that would let the golden file itself drift silently:
    an edit to safety_constraints or safety_context_fields with no matching
    edit to hash. Runs against the file alone, no vendored or src import.
    """
    golden = _load_golden()
    assert golden["hash"] == _recomputed_hash(golden)


def test_vendored_safety_constraints_matches_golden() -> None:
    from policy.safety_gate import SafetyConstraints

    golden = _load_golden()
    assert _constraints_as_golden(SafetyConstraints) == golden["safety_constraints"]


def test_vendored_safety_context_fields_match_golden() -> None:
    from policy.safety_gate import SafetyContext

    golden = _load_golden()
    assert _context_fields_as_golden(SafetyContext) == golden["safety_context_fields"]
