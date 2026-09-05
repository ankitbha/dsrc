#!/usr/bin/env python3
"""Record the commit that a deploy step is about to rsync out.

B16 (validation round 3): `run_demo.py`'s `_build_provenance` reads
`commit` from `git rev-parse HEAD`, which returns `commit: null` on the
only machine a run actually happens on -- `~/dsrc-task40` (this task's own
jetson tree) is an rsync copy, not a git checkout, so `git rev-parse` exits
128 there every time. The other three provenance fields (`policy_bundle`,
`detector_engine_sha256`, `apk_sha256`) populate fine; only the commit,
arguably the most basic one, was always absent where it mattered most.

Run this on the SOURCE machine -- the one with the real git checkout --
right before the deploy step's rsync, so the commit it names is the one
about to be copied:

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
import json
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUT = REPO_ROOT / "deployment" / "jetson" / "models" / "deployed_commit.json"


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


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--repo-root", type=Path, default=REPO_ROOT,
        help="the git checkout to read HEAD from (default: this script's own repo)",
    )
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()

    commit = _git_commit(args.repo_root)
    if commit is None:
        print(f"{args.repo_root} is not a git checkout (or git is unreachable) -- "
              f"nothing to record", file=sys.stderr)
        return 1
    dirty = _git_is_dirty(args.repo_root)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps({"commit": commit, "dirty": dirty}, indent=2))
    print(f"wrote {args.out}")
    print(f"commit={commit} dirty={dirty}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
