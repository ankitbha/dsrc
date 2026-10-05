import json
import sys

import pytest
import requests

from here_error import cli, pipeline, provenance
from test_report import make_asm, SPECS


def test_commit_none_is_refused(monkeypatch, tmp_path):
    monkeypatch.setattr(provenance, "_git_commit", lambda root: None)
    with pytest.raises(provenance.ProvenanceError, match="git commit"):
        provenance.make_provenance([], require_clean=False)


def test_dirty_tree_refused_for_a_final_run_only(monkeypatch, tmp_path):
    monkeypatch.setattr(provenance, "_git_commit", lambda root: "deadbeef")
    monkeypatch.setattr(provenance, "_git_is_dirty", lambda root: True)
    f = tmp_path / "in.txt"
    f.write_text("hello")
    prov = provenance.make_provenance([f], require_clean=False)
    assert prov.dirty is True and prov.git_commit == "deadbeef"
    assert prov.inputs[0].sha256 == "2cf24dba5fb0a30e26e83b2ac5b9e29e1b161e5c1fa7425e73043362938b9824"
    assert "MATCH_M" in prov.params
    with pytest.raises(provenance.ProvenanceError, match="clean"):
        provenance.make_provenance([f], require_clean=True)
    monkeypatch.setattr(provenance, "_git_is_dirty", lambda root: None)
    with pytest.raises(provenance.ProvenanceError):
        provenance.make_provenance([f], require_clean=True)


def test_sidecar_is_written_beside_the_output(monkeypatch, tmp_path):
    monkeypatch.setattr(provenance, "_git_commit", lambda root: "deadbeef")
    monkeypatch.setattr(provenance, "_git_is_dirty", lambda root: False)
    out = tmp_path / "x.csv"
    out.write_text("a")
    side = provenance.write_sidecar(out, provenance.make_provenance([out], require_clean=True))
    assert side.name == "x.csv.provenance.json"
    assert json.loads(side.read_text())["git_commit"] == "deadbeef"


def test_routing_dry_run_makes_no_network_call(monkeypatch, tmp_path, capsys):
    asm = make_asm(SPECS)
    # The synthetic passes carry a fix list; route_requests needs only those.
    monkeypatch.setattr(pipeline, "assemble", lambda d: asm)

    def boom(*a, **k):
        raise AssertionError("network call during a dry run")

    monkeypatch.setattr(requests, "get", boom)
    monkeypatch.delenv("HERE_API_KEY", raising=False)
    rc = cli.main(["--out-dir", str(tmp_path), "routing", "--dry-run"])
    out = capsys.readouterr().out
    assert rc == 0 and "calls made: 0" in out
    n = len(pipeline.route_requests(asm))
    assert f"dry run: {n} requests" in out
    assert not list(tmp_path.glob("routing*"))     # nothing written, not even a cache directory


def test_routing_without_key_refuses(monkeypatch, tmp_path):
    asm = make_asm(SPECS)
    monkeypatch.setattr(pipeline, "assemble", lambda d: asm)
    monkeypatch.setattr(pipeline.Assembly, "input_files", lambda self: [])
    monkeypatch.setattr(provenance, "_git_commit", lambda root: "deadbeef")
    monkeypatch.delenv("HERE_API_KEY", raising=False)
    with pytest.raises(Exception, match="HERE_API_KEY"):
        cli.main(["--out-dir", str(tmp_path), "routing"])


def test_report_exits_nonzero_and_says_incomplete(monkeypatch, tmp_path, capsys):
    asm = make_asm(SPECS)
    monkeypatch.setattr(pipeline, "assemble", lambda d: asm)
    monkeypatch.setattr(pipeline.Assembly, "input_files", lambda self: [])
    monkeypatch.setattr(provenance, "_git_commit", lambda root: "deadbeef")
    monkeypatch.setattr(provenance, "_git_is_dirty", lambda root: False)
    rc = cli.main(["--out-dir", str(tmp_path), "report"])
    assert rc == 3
    assert capsys.readouterr().out.startswith("STATUS: INCOMPLETE")
    text = (tmp_path / "summary.md").read_text()
    assert "**STATUS: INCOMPLETE.**" in text
