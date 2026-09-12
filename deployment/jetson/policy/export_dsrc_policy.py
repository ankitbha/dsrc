#!/usr/bin/env python3
"""Export a trained DSRC checkpoint to a Jetson-loadable bundle.

Produces two files, the same shape `policy/export_policy.py` uses for the
39-field actor:
  <out>.ts     TorchScript module: state (segments, features) -> Q-values
  <out>.json   manifest: dims, network identity, network_fingerprint,
               provenance, trained flag

The checkpoints this project trains (`scripts/train_mainz_src.py`) are bare
`torch.save(model.state_dict(), ...)` calls with no metadata at all --
plan_task145 section 1.2 measured this directly against both committed
checkpoints. This script is where the network's identity is attached, in
the manifest, read from the SAME network definition file
(`specs/dsrc_network_mainz.json` by default) the device loads at run time --
`policy.dsrc_runtime.DsrcRuntime` refuses to run a bundle whose
`network_fingerprint` does not match its own network definition, with no
grandfather clause.

No trained checkpoint for a network? --random creates a correctly-shaped,
randomly initialised network so the bundle format, the manifest checks and
the latency measurement can all be exercised before training finishes
(weights do not affect shape or latency). The manifest marks it
trained=false.

Usage:
  python3 policy/export_dsrc_policy.py \\
      --checkpoint ../../results/checkpoints/mainz_here_best.pt \\
      --out models/dsrc_policy
  python3 policy/export_dsrc_policy.py --random --out models/dsrc_policy
"""

from __future__ import annotations

import argparse
import datetime
import hashlib
import json
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from perception.segment_state import DEFAULT_NETWORK_DEFINITION_PATH  # noqa: E402
from policy.dsrc_contract import (  # noqa: E402
    SPEED_ACTION_FRACTIONS,
    SrcQNetwork,
    network_fingerprint,
)


def load_network_definition(path: str | Path) -> dict:
    return json.loads(Path(path).read_text())


def _dims(definition: dict) -> tuple[int, int, int]:
    return (
        len(definition["segments"]),
        len(definition["feature_names"]),
        len(SPEED_ACTION_FRACTIONS),
    )


def build_from_checkpoint(checkpoint_path: str, definition: dict) -> tuple[SrcQNetwork, dict]:
    num_segments, num_features, num_actions = _dims(definition)
    model = SrcQNetwork(num_segments, num_features, num_actions)
    # Our own training artifact (scripts/train_mainz_src.py): a bare
    # state_dict, hence weights_only=True is safe and preferred.
    state_dict = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    model.load_state_dict(state_dict)
    info = {
        "num_segments": num_segments,
        "num_features": num_features,
        "num_actions": num_actions,
        "trained": True,
        "source": str(Path(checkpoint_path).resolve()),
    }
    return model, info


def build_random(definition: dict, seed: int = 0) -> tuple[SrcQNetwork, dict]:
    torch.manual_seed(seed)
    num_segments, num_features, num_actions = _dims(definition)
    model = SrcQNetwork(num_segments, num_features, num_actions)
    info = {
        "num_segments": num_segments,
        "num_features": num_features,
        "num_actions": num_actions,
        "trained": False,
        "source": f"random_init(seed={seed})",
    }
    return model, info


def export(
    model: SrcQNetwork,
    info: dict,
    definition: dict,
    out_prefix: str,
    *,
    checkpoint_path: str | None = None,
) -> None:
    model.eval()
    example = torch.zeros(info["num_segments"], info["num_features"])
    scripted = torch.jit.script(model)
    scripted(example)  # sanity forward pass
    out = Path(out_prefix)
    out.parent.mkdir(parents=True, exist_ok=True)
    scripted.save(str(out) + ".ts")

    segment_ids = [list(segment["edge_ids"]) for segment in definition["segments"]]
    speed_limits = [float(segment["speed_limit_kmh"]) for segment in definition["segments"]]
    lanes = [float(segment["lanes"]) for segment in definition["segments"]]
    length_kms = [float(segment["length_km"]) for segment in definition["segments"]]
    polylines = [
        [(float(lat), float(lon)) for edge in segment["edges"] for lat, lon in edge["polyline"]]
        for segment in definition["segments"]
    ]
    feature_names = list(definition["feature_names"])
    network_id = definition["network_id"]

    checkpoint_sha256 = None
    if checkpoint_path is not None:
        checkpoint_sha256 = hashlib.sha256(Path(checkpoint_path).read_bytes()).hexdigest()

    manifest = {
        **info,
        "network_id": network_id,
        "feature_names": feature_names,
        "segment_ids": segment_ids,
        "segment_speed_limits_kmh": speed_limits,
        "segment_lanes": lanes,
        "segment_length_km": length_kms,
        "action_fractions": list(SPEED_ACTION_FRACTIONS),
        # Everything above this line, plus a hash of the polylines (not
        # carried here directly -- segment 5 alone is 490 vertices), in one
        # hash. DsrcRuntime recomputes this from its OWN network definition
        # and refuses on any difference -- the individual fields above are
        # carried for audit, not re-checked field by field at load time.
        "network_fingerprint": network_fingerprint(
            network_id=network_id,
            feature_names=feature_names,
            segment_ids=segment_ids,
            segment_speed_limits_kmh=speed_limits,
            segment_lanes=lanes,
            segment_length_km=length_kms,
            segment_polylines=polylines,
        ),
        # Risk 1 in plan_task145: nothing in best.pt records which network it
        # was trained on. This does not prove the pairing, only makes it
        # auditable after the fact.
        "checkpoint_sha256": checkpoint_sha256,
        "created_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
    }
    with open(str(out) + ".json", "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"wrote {out}.ts and {out}.json (trained={info['trained']})")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--checkpoint", help="path to a mainz_src checkpoint (.pt state_dict)")
    group.add_argument("--random", action="store_true", help="random-init network (bring-up)")
    parser.add_argument("--out", default="models/dsrc_policy", help="output prefix")
    parser.add_argument(
        "--network-definition", default=str(DEFAULT_NETWORK_DEFINITION_PATH),
        help="specs/dsrc_network_mainz.json or an equivalent for another network",
    )
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    definition = load_network_definition(args.network_definition)
    if args.checkpoint:
        model, info = build_from_checkpoint(args.checkpoint, definition)
    else:
        model, info = build_random(definition, args.seed)
    export(model, info, definition, args.out, checkpoint_path=args.checkpoint)


if __name__ == "__main__":
    main()
