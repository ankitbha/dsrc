"""No function may read a name that does not exist.

Task 26 split `_build_rest` out of `build_components` and left the new function
referring to the old one's local `cam_cfg`. That is a `NameError` on every call,
and it took out live mode, `bench_latency` and `replay_demo` at once. It shipped
because nothing checks: `scripts/check.sh` runs Android lint, the JVM suites, this
pytest run and the instrumented suite -- there is no Python linter anywhere in the
gate, and `--selfcheck`, the one path a smoke test exercises, never calls
`build_components`.

pyflakes would say this in one line and is not installed. Rather than add a
dependency the Jetson would also need, the check lives here, where the gate
already runs. It is deliberately narrow: a name loaded in a function, not bound
anywhere that function can see and not a module global or a builtin, is an error
with no false-positive story. It does not attempt type checking or anything else a
real linter would do.
"""

from __future__ import annotations

import builtins
import symtable
from pathlib import Path

import pytest

JETSON = Path(__file__).resolve().parents[1]
REPO = JETSON.parents[1]

#: Every module that is an entry point or is imported by one. A NameError in any
#: of these is a defect that reaches a run.
#:
#: The whole repository, not `deployment/jetson/`. Scoped to the jetson tree, this
#: covered 46 of 121 modules -- and the very next extract-a-function commit broke
#: `scripts/run_loopback_pipeline.py`, which is one directory outside that glob.
#: The guard passed, the full suite passed, and the defect it exists to prevent
#: sat in the blind spot. A check is only worth what it looks at.
MODULES = sorted(
    p for p in REPO.rglob("*.py")
    if "__pycache__" not in p.parts
    and "tests" not in p.parts
    and ".venv" not in p.parts
    and "build" not in p.parts
)


def scopes(table: symtable.SymbolTable):
    """Every scope in the module, depth first."""
    yield table
    for child in table.get_children():
        yield from scopes(child)


def module_names(table: symtable.SymbolTable) -> set[str]:
    """What the module binds at its own level.

    `symtable` is used rather than an AST walk because scope is exactly what this
    test is about, and a hand-rolled walk got it wrong in the direction that
    matters: collecting binders from the whole tree counted `cam_cfg` as a module
    name because another function assigns it, so the first version of this test
    passed against the very defect it was written for.
    """
    return {s.get_name() for s in table.get_symbols() if s.is_assigned() or s.is_imported()}


@pytest.mark.parametrize("path", MODULES, ids=lambda p: str(p.relative_to(REPO)))
def test_every_function_reads_only_names_that_exist(path: Path):
    table = symtable.symtable(path.read_text(encoding="utf-8"), str(path), "exec")
    visible = module_names(table) | set(dir(builtins)) | {"__file__", "__name__", "__doc__"}

    offenders: list[str] = []
    for scope in scopes(table):
        if scope.get_type() != "function":
            continue
        for symbol in scope.get_symbols():
            # `is_global()` here means the compiler resolved the name to module or
            # builtin scope -- it found no binding in this function or any
            # enclosing one. If the module does not bind it either, the read is a
            # NameError the moment it executes.
            if not symbol.is_global() or symbol.is_assigned():
                continue
            name = symbol.get_name()
            if name not in visible:
                offenders.append(f"{scope.get_name()}() reads undefined {name!r}")

    assert not offenders, f"{path.relative_to(REPO)}: " + "; ".join(sorted(set(offenders)))
