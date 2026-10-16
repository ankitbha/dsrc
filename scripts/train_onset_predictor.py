#!/usr/bin/env python3
"""Train the learned correction's onset predictor on block T (jam plan section 5.5).

    .venv/bin/python scripts/train_onset_predictor.py --round 1
    .venv/bin/python scripts/run_jam_experiments.py --block T --dagger \\
        --predictor results/jam/onset_predictor_round1.json
    .venv/bin/python scripts/train_onset_predictor.py --round 2

Round 1 reads block T's plan-alone episodes (`results/jam/T.jsonl`) and writes
`results/jam/onset_predictor_round1.json` with its `.pt`. Round 2 adds the DAgger
episodes run with the round-1 model active at threshold 0.5 (`results/jam/T_dagger.jsonl`)
and refits on the union into `results/jam/onset_predictor.json`. Each sample is labelled
from its episode's 10 s trace at the feed program's jam threshold. Training seeds are
1 to 8 and validation seeds 9 and 10; a record on any other seed is refused. The card
records the feed artefact and camera model the episodes ran on.

`--smoke` trains on whatever seeds a smoke run has, the first as training and the rest as
validation, and is for proving the entry point only.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
for _path in (REPO_ROOT, REPO_ROOT / "scripts"):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

from src.analysis.jam_metrics import load_v_jam  # noqa: E402
from src.rl import onset_predictor as op  # noqa: E402

TRAIN_SEEDS = tuple(range(1, 9))
VALIDATION_SEEDS = (9, 10)


def split_seeds(seeds: list[int]) -> tuple[list[int], list[int]]:
    outside = sorted(set(seeds) - set(TRAIN_SEEDS) - set(VALIDATION_SEEDS))
    if outside:
        raise SystemExit(f"the predictor trains on seeds 1-10 only; refusing {outside}")
    return (sorted(s for s in set(seeds) if s in TRAIN_SEEDS),
            sorted(s for s in set(seeds) if s in VALIDATION_SEEDS))


def _read(jsonl: Path) -> tuple[list[dict], list[dict]]:
    headers, episodes = [], {}
    for line in jsonl.read_text().splitlines():
        try:
            record = json.loads(line)
        except ValueError:
            continue
        if record.get("type") == "header":
            headers.append(record)
        elif record.get("type") == "episode":
            episodes[record["key"]] = record
    return headers, list(episodes.values())


def labelled(episodes: list[dict], v_jam: float) -> dict[int, tuple[np.ndarray, np.ndarray]]:
    """Per seed, the stacked `(X, y)` of its episodes' kept samples."""
    by_seed: dict[int, list[tuple[np.ndarray, np.ndarray]]] = {}
    for record in episodes:
        if not record.get("samples_file"):
            continue
        samples = json.loads(Path(record["samples_file"]).read_text())["samples"]
        trace = json.loads(Path(record["trace_file"]).read_text())["trace"]
        X, y, _ = op.label_samples(samples, trace, v_jam)
        by_seed.setdefault(int(record["seed"]), []).append((X, y))
    return {seed: (np.concatenate([x for x, _ in parts]), np.concatenate([y for _, y in parts]))
            for seed, parts in by_seed.items()}


def _stack(data, seeds):
    if not seeds:
        return np.zeros((0, len(op.FEATURES)), np.float32), np.zeros(0, np.int64)
    return (np.concatenate([data[s][0] for s in seeds]),
            np.concatenate([data[s][1] for s in seeds]))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--round", type=int, choices=(1, 2), required=True)
    parser.add_argument("--results-dir", default="results/jam")
    parser.add_argument("--out", default=None)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args(argv)
    results = Path(args.results_dir)
    results = results if results.is_absolute() else REPO_ROOT / results
    files = [results / "T.jsonl"] + ([results / "T_dagger.jsonl"] if args.round == 2 else [])
    headers, episodes = [], []
    for path in files:
        if not path.exists():
            raise SystemExit(f"{path} does not exist")
        h, e = _read(path)
        headers += h
        episodes += e
    feeds = {(h.get("feed_artefact") or {}).get("artefact_sha256") for h in headers}
    cameras = {(h.get("camera_model") or {}).get("sha256") for h in headers}
    if len(feeds) != 1 or len(cameras) != 1:
        raise SystemExit(f"block T mixes feed artefacts {feeds} or camera models {cameras}")
    v_jam, v_sha = load_v_jam()
    data = labelled(episodes, v_jam)
    seeds = sorted(data)
    if args.smoke:
        train, validation = seeds[:1], seeds[1:]
    else:
        train, validation = split_seeds(seeds)
        if not train or not validation:
            raise SystemExit(f"need training seeds 1-8 and validation seeds 9-10; have {seeds}")
    X, y = _stack(data, train)
    Xv, yv = _stack(data, validation)
    model = op.fit(X, y, Xv, yv)
    out = Path(args.out) if args.out else results / (
        "onset_predictor_round1.json" if args.round == 1 else "onset_predictor.json")
    out = out if out.is_absolute() else REPO_ROOT / out
    card = op.save(model, out, extra={
        "round": args.round, "seeds": {"train": train, "validation": validation},
        "episodes": len(episodes), "sources": [str(p) for p in files],
        "feed_artefact_sha256": feeds.pop(), "camera_model_sha256": cameras.pop(),
        "v_jam": v_jam, "v_jam_sha256": v_sha, "smoke": args.smoke})
    print(f"wrote {out}: {card['kind']} (validation log-loss {card['validation_log_loss']}), "
          f"{card['samples']['train']} training samples with {card['positives']['train']} "
          f"positive, {card['samples']['validation']} validation with "
          f"{card['positives']['validation']} positive")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
