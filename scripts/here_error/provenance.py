"""Provenance recorded beside every output: git commit, dirty state, input hashes and parameters."""
from __future__ import annotations

import hashlib
import json
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Iterable

from here_error import params
from here_error.models import InputFile, Provenance

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "scripts"))

from record_deployed_commit import _git_commit, _git_is_dirty  # noqa: E402


class ProvenanceError(RuntimeError):
    pass


def sha256_file(path: Path | str) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def make_provenance(inputs: Iterable[Path | str], *, require_clean: bool, repo_root: Path = REPO_ROOT) -> Provenance:
    """Refuses when the commit cannot be read, and, for a final run, when the tree is dirty or its state unknown."""
    commit = _git_commit(repo_root)
    if not commit:
        raise ProvenanceError(f"cannot read the git commit of {repo_root}; refusing to produce unattributed output")
    dirty = _git_is_dirty(repo_root)
    if require_clean and dirty is not False:
        raise ProvenanceError("a final run needs a clean working tree"
                              + ("" if dirty else "; the tree state could not be read"))
    files = tuple(InputFile(str(p), sha256_file(p)) for p in sorted({str(Path(p)) for p in inputs}))
    return Provenance(git_commit=commit, dirty=dirty, inputs=files, params=params.all_params())


def to_json(prov: Provenance) -> str:
    return json.dumps(asdict(prov), indent=2, sort_keys=True, default=list)


def write_sidecar(output: Path, prov: Provenance) -> Path:
    side = output.with_name(output.name + ".provenance.json")
    side.write_text(to_json(prov))
    return side
