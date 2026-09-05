#!/usr/bin/env python3
"""Record the commit that a deploy step is about to rsync out.

B16 (validation round 3): `run_demo.py`'s `_build_provenance` reads
`commit` from `git rev-parse HEAD`, which returns `commit: null` on the
only machine a run actually happens on -- `~/dsrc-task40` (this task's own
jetson tree) is an rsync copy, not a git checkout, so `git rev-parse` exits
128 there every time. The other three provenance fields (`policy_bundle`,
`detector_engine_sha256`, `apk_sha256`) populate fine; only the commit,
arguably the most basic one, was always absent where it mattered most.

B21 (validation round 4): the commit alone has no staleness signal, on the
ONLY machine that reads it. `models/` is gitignored and the deploy rsync
has no `--delete`, so a re-deploy that forgets to run this script leaves
deploy A's sidecar sitting beside deploy B's code -- and `_build_
provenance` always takes the sidecar path on the jetson, since `git rev-
parse` always fails there. It would then report commit A with full
confidence about a tree that is actually B's. `recorded_at` alone would
not catch this: `rsync -a` preserves source mtimes, so a freshly deployed
file can be OLDER than a stale sidecar. This script instead hashes the
deployed source tree itself (`deployment/jetson/`, minus generated
artifacts) immediately before the rsync, and `_build_provenance`
recomputes the identical hash at run time -- a mismatch means "this tree
is not the one this commit names", whatever the actual cause.

Run this on the SOURCE machine -- the one with the real git checkout --
right before the deploy step's rsync, so the commit and hash it names are
the ones about to be copied:

    python3 scripts/record_deployed_commit.py
    rsync -a --exclude .git ./ jetson:~/dsrc-task40/

The written file, `deployment/jetson/models/deployed_commit.json`, is an
ordinary file (not `.git`), so it rides along in the same rsync and lands
on the jetson tree where `run_demo.py`'s `_build_provenance` falls back to
reading it once `git rev-parse` itself has failed. Untracked, same as
every other model artifact (`models/.gitignore` is `*`).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DEPLOY_ROOT = REPO_ROOT / "deployment" / "jetson"
DEFAULT_OUT = DEFAULT_DEPLOY_ROOT / "models" / "deployed_commit.json"

#: Directories that exist under `deployment/jetson/` on BOTH machines but
#: describe something other than the deployed source: `models/` is this
#: task's own generated artifacts, including the sidecar THIS script
#: writes -- hashing it would be self-referential -- and the other two are
#: Python's own bytecode/test caches, which differ between the two
#: machines' Python versions and run histories for reasons that have
#: nothing to do with a real code change. `run_demo.py`'s own copy of this
#: constant must name exactly the same set, or an unchanged tree hashes
#: differently on each side and every run reports a false mismatch.
SOURCE_HASH_EXCLUDE_DIRS = frozenset({"models", "__pycache__", ".pytest_cache", ".git"})
SOURCE_HASH_EXCLUDE_SUFFIXES = frozenset({".pyc", ".pyo"})


def _git_commit(repo_root: Path) -> str | None:
    """`git rev-parse HEAD` in `repo_root`, or `None` when it is not a git
    checkout, git is unreachable, or the tree is genuinely a checkout with
    no commits."""
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=str(repo_root),
            capture_output=True, text=True, timeout=5.0,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    return result.stdout.strip() if result.returncode == 0 else None


def _git_is_dirty(repo_root: Path) -> bool | None:
    """Whether the working tree has uncommitted changes, or `None` when
    that could not be determined -- named alongside `commit` because a
    commit hash with a dirty tree is not the same provenance claim as a
    clean one, and the deploy step is exactly the moment this is still
    knowable (a run on the jetson has no `.git` to ask at all)."""
    try:
        result = subprocess.run(
            ["git", "status", "--porcelain"], cwd=str(repo_root),
            capture_output=True, text=True, timeout=5.0,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    return bool(result.stdout.strip()) if result.returncode == 0 else None


def source_tree_sha256(deploy_root: Path) -> str | None:
    """One sha256 over every source file under `deploy_root`, keyed by its
    path relative to `deploy_root` as well as its bytes (B21, validation
    round 4) -- `run_demo.py`'s `_live_source_tree_sha256` recomputes this
    identically on the jetson side.

    Files are visited in a fixed order (sorted by their relative path's
    own parts, not by however the filesystem happens to enumerate them) and
    each one's relative path goes into the hash alongside its content, so a
    file that MOVED without changing (a rename) also changes the result --
    hashing content alone would miss that, and an unordered walk would make
    two genuinely-identical trees hash differently by accident.

    `None` when `deploy_root` does not exist at all -- distinct from an
    empty tree (which hashes to a real, reproducible digest of nothing) and
    from any other failure this function does not otherwise mask (a
    permission error reading a real file is left to raise, the same as
    `record_installed_apk.py` and `_build_provenance` leave a hash's own
    read errors unguarded elsewhere in this task).
    """
    if not deploy_root.is_dir():
        return None
    paths = [p for p in deploy_root.rglob("*") if p.is_file()]
    paths.sort(key=lambda p: p.relative_to(deploy_root).parts)
    hasher = hashlib.sha256()
    for path in paths:
        rel_parts = path.relative_to(deploy_root).parts
        if any(part in SOURCE_HASH_EXCLUDE_DIRS for part in rel_parts):
            continue
        if path.suffix in SOURCE_HASH_EXCLUDE_SUFFIXES:
            continue
        hasher.update("/".join(rel_parts).encode("utf-8"))
        hasher.update(b"\0")
        hasher.update(path.read_bytes())
        hasher.update(b"\0")
    return hasher.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--repo-root", type=Path, default=REPO_ROOT,
        help="the git checkout to read HEAD from (default: this script's own repo)",
    )
    parser.add_argument(
        "--deploy-root", type=Path, default=DEFAULT_DEPLOY_ROOT,
        help="the tree to hash for the staleness check (default: deployment/jetson/, "
             "the code that actually runs on the jetson)",
    )
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()

    commit = _git_commit(args.repo_root)
    if commit is None:
        print(f"{args.repo_root} is not a git checkout (or git is unreachable) -- "
              f"nothing to record", file=sys.stderr)
        return 1
    dirty = _git_is_dirty(args.repo_root)
    tree_sha256 = source_tree_sha256(args.deploy_root)
    if tree_sha256 is None:
        print(f"warning: {args.deploy_root} does not exist -- source_tree_sha256 will "
              f"be null, and _build_provenance will have nothing to check the deployed "
              f"tree against", file=sys.stderr)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps({
        "commit": commit, "dirty": dirty, "source_tree_sha256": tree_sha256,
    }, indent=2))
    print(f"wrote {args.out}")
    print(f"commit={commit} dirty={dirty} source_tree_sha256={tree_sha256}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
