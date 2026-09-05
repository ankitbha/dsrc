"""scripts/record_deployed_commit.py: the sidecar B16 (validation round 3)
made `run_demo.py`'s `_build_provenance` fall back to when `git rev-parse`
itself returns `commit: null` (the jetson's `~/dsrc-task40` is an rsync
copy, not a git checkout).

Imported the same way `test_record_installed_apk.py` imports
`scripts/record_installed_apk.py`.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import record_deployed_commit as rdc  # noqa: E402


def _init_git_repo(repo_root: Path) -> None:
    subprocess.run(["git", "init", "-q"], cwd=str(repo_root), check=True)
    subprocess.run(
        ["git", "-c", "user.email=t@example.com", "-c", "user.name=t", "commit", "-q",
         "--allow-empty", "-m", "initial"],
        cwd=str(repo_root), check=True,
    )


def test_git_commit_reads_head_of_a_real_checkout(tmp_path):
    _init_git_repo(tmp_path)
    expected = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=str(tmp_path), capture_output=True, text=True,
    ).stdout.strip()
    assert rdc._git_commit(tmp_path) == expected
    assert len(expected) == 40


def test_git_commit_is_none_when_the_tree_is_not_a_checkout(tmp_path):
    """The exact case B16 is filed for: an rsync copy with no `.git` at
    all -- `git rev-parse` exits 128, and this must not raise or invent a
    hash."""
    assert rdc._git_commit(tmp_path) is None


def test_git_is_dirty_is_false_on_a_clean_checkout(tmp_path):
    _init_git_repo(tmp_path)
    assert rdc._git_is_dirty(tmp_path) is False


def test_git_is_dirty_is_true_with_an_uncommitted_change(tmp_path):
    _init_git_repo(tmp_path)
    (tmp_path / "untracked.txt").write_text("new file")
    assert rdc._git_is_dirty(tmp_path) is True


def test_git_is_dirty_is_none_when_the_tree_is_not_a_checkout(tmp_path):
    assert rdc._git_is_dirty(tmp_path) is None


def test_main_writes_the_real_checkouts_commit(tmp_path, monkeypatch):
    _init_git_repo(tmp_path)
    expected = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=str(tmp_path), capture_output=True, text=True,
    ).stdout.strip()
    out = tmp_path / "deployed_commit.json"
    monkeypatch.setattr(
        sys, "argv",
        ["record_deployed_commit.py", "--repo-root", str(tmp_path), "--out", str(out)],
    )
    rc = rdc.main()
    assert rc == 0
    record = json.loads(out.read_text())
    assert record["commit"] == expected
    assert record["dirty"] is False


def test_main_refuses_a_tree_that_is_not_a_git_checkout(tmp_path, monkeypatch, capsys):
    """The whole point: this script is meant to run on the SOURCE machine,
    which must actually be a git checkout -- refusing rather than writing
    a null commit keeps a misconfigured deploy step from silently
    shipping a sidecar that is no better than not having one."""
    out = tmp_path / "deployed_commit.json"
    monkeypatch.setattr(
        sys, "argv",
        ["record_deployed_commit.py", "--repo-root", str(tmp_path), "--out", str(out)],
    )
    rc = rdc.main()
    assert rc == 1
    assert not out.exists()
    assert "not a git checkout" in capsys.readouterr().err


def test_main_reads_this_real_repos_own_commit(monkeypatch, tmp_path):
    """Against the real dsrc checkout this task runs in -- not a synthetic
    `git init` fixture -- the same repo `run_demo.py`'s own
    `_build_provenance` calls `git rev-parse HEAD` against."""
    real_repo_root = Path(__file__).resolve().parents[1]
    expected = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=str(real_repo_root), capture_output=True, text=True,
    )
    out = tmp_path / "deployed_commit.json"
    monkeypatch.setattr(
        sys, "argv",
        ["record_deployed_commit.py", "--repo-root", str(real_repo_root), "--out", str(out)],
    )
    rc = rdc.main()
    if expected.returncode != 0:
        # This checkout itself is not a git repo in whatever environment
        # collected this test run -- the refusal path, already covered
        # above, is the correct outcome and there is nothing further to
        # assert against a real commit that does not exist here.
        assert rc == 1
        return
    assert rc == 0
    record = json.loads(out.read_text())
    assert record["commit"] == expected.stdout.strip()
    assert len(record["commit"]) == 40
