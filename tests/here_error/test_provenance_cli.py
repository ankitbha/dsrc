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


def _patch_report(monkeypatch):
    asm = make_asm(SPECS)
    monkeypatch.setattr(pipeline, "assemble", lambda d: asm)
    monkeypatch.setattr(pipeline.Assembly, "input_files", lambda self: [])
    monkeypatch.setattr(provenance, "_git_commit", lambda root: "deadbeef")
    monkeypatch.setattr(provenance, "_git_is_dirty", lambda root: False)


@pytest.mark.parametrize("reason", ["", "   "])
def test_empty_waiver_reason_is_refused(monkeypatch, tmp_path, reason):
    _patch_report(monkeypatch)
    with pytest.raises(SystemExit) as e:
        cli.main(["--out-dir", str(tmp_path), "report", "--labels-waived", reason])
    assert e.value.code != 0
    assert not (tmp_path / "summary.md").exists()


def test_waiver_with_existing_label_scores_is_refused(monkeypatch, tmp_path):
    _patch_report(monkeypatch)
    (tmp_path / "label_scores.json").write_text("[]")
    with pytest.raises(SystemExit, match="contradicts"):
        cli.main(["--out-dir", str(tmp_path), "report", "--labels-waived", "no labelling"])
    assert not (tmp_path / "summary.md").exists()


def test_report_with_waiver_drops_only_the_labels_reason(monkeypatch, tmp_path, capsys):
    _patch_report(monkeypatch)
    rc = cli.main(["--out-dir", str(tmp_path), "report", "--labels-waived", "no labelling"])
    out = capsys.readouterr().out
    assert rc == 3                                   # routing and detections are still missing
    assert "gate V6" not in out.split("# HERE")[0]
    assert "routing results are missing" in out
    assert "- V6: not run (no labelling)." in (tmp_path / "summary.md").read_text()


def test_waiver_reason_is_stored_without_surrounding_whitespace(monkeypatch, tmp_path):
    _patch_report(monkeypatch)
    cli.main(["--out-dir", str(tmp_path), "report", "--labels-waived", "  no labelling \n"])
    assert "- V6: not run (no labelling)." in (tmp_path / "summary.md").read_text()
