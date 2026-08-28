"""Export the trained simulation actor to a Jetson-loadable bundle.

Produces two files from a sim checkpoint:
  <out>.ts     TorchScript module: obs (B, 39) -> {head: logits (B, 3)}
  <out>.json   manifest: dims, action profile, provenance, trained flag

The actor architecture is re-declared here (instead of importing
src.rl.models) because models.py drags in the simulation environment
stack via src.rl.actions. The state_dict layout must match
MultiCategoricalActor exactly: backbone.{0,2,4}.* and
heads.<head_name>.* - tests/test_sim_contract.py checks this against
the sim source when it is importable.

No trained checkpoint yet? --random creates a correctly-shaped,
randomly initialized actor so the full pipeline and all latency
numbers can be produced before training finishes (weights do not
affect latency). The manifest marks it trained=false and the dashboard
shows an UNTRAINED banner.

Usage:
  python3 policy/export_policy.py --checkpoint /path/to/actor.pt --out models/actor_policy
  python3 policy/export_policy.py --random --out models/actor_policy
"""

from __future__ import annotations

import argparse
import datetime
import json
import sys
from pathlib import Path
from typing import Dict

import torch
from torch import nn

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from policy import sim_contract


def mlp(input_dim: int, hidden_sizes: tuple[int, ...], output_dim: int) -> nn.Sequential:
    # verbatim from src/rl/models.py - layer indices define state_dict keys
    layers: list[nn.Module] = []
    last_dim = input_dim
    for hidden_dim in hidden_sizes:
        layers.extend([nn.Linear(last_dim, hidden_dim), nn.Tanh()])
        last_dim = hidden_dim
    layers.append(nn.Linear(last_dim, output_dim))
    return nn.Sequential(*layers)


class VendoredActor(nn.Module):
    """State-dict-compatible twin of src.rl.models.MultiCategoricalActor."""

    def __init__(
        self,
        input_dim: int,
        hidden_sizes: tuple[int, ...] = (128, 128),
        action_profile: str = "full",
    ) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.action_profile = action_profile
        self.backbone = mlp(input_dim, hidden_sizes, hidden_sizes[-1] if hidden_sizes else input_dim)
        body_dim = hidden_sizes[-1] if hidden_sizes else input_dim
        self.heads = nn.ModuleDict(
            {
                head: nn.Linear(body_dim, len(sim_contract.ACTION_VALUES[head]))
                for head in sim_contract.active_heads(action_profile)
            }
        )

    def forward(self, obs: torch.Tensor) -> Dict[str, torch.Tensor]:
        body = self.backbone(obs)
        result: Dict[str, torch.Tensor] = {}
        for name, head in self.heads.items():
            result[name] = head(body)
        return result


def build_from_checkpoint(path: str) -> tuple[VendoredActor, dict]:
    # our own training artifact; pickle load is intentional
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    metadata = checkpoint.get("metadata", {}) or {}
    input_dim = int(metadata.get("input_dim", checkpoint.get("input_dim", 0)))
    action_profile = str(metadata.get("action_profile", checkpoint.get("action_profile", "full")))
    hidden_sizes = tuple(int(v) for v in checkpoint.get("hidden_sizes", (128, 128)))
    actor = VendoredActor(input_dim, hidden_sizes, action_profile)
    actor.load_state_dict(checkpoint["state_dict"])
    info = {
        "input_dim": input_dim,
        "action_profile": action_profile,
        "hidden_sizes": list(hidden_sizes),
        "trained": True,
        "source": str(Path(path).resolve()),
    }
    return actor, info


def build_random(seed: int = 0) -> tuple[VendoredActor, dict]:
    torch.manual_seed(seed)
    input_dim = sim_contract.local_obs_dim()
    actor = VendoredActor(input_dim, (128, 128), "full")
    info = {
        "input_dim": input_dim,
        "action_profile": "full",
        "hidden_sizes": [128, 128],
        "trained": False,
        "source": f"random_init(seed={seed})",
    }
    return actor, info


def export(actor: VendoredActor, info: dict, out_prefix: str) -> None:
    if info["input_dim"] != sim_contract.local_obs_dim():
        raise SystemExit(
            f"checkpoint input_dim={info['input_dim']} does not match the vendored "
            f"contract dim={sim_contract.local_obs_dim()}; the sim observation schema "
            "has changed - update policy/sim_contract.py first."
        )
    actor.eval()
    example = torch.zeros(1, info["input_dim"])
    scripted = torch.jit.script(actor)
    scripted(example)  # sanity forward pass
    out = Path(out_prefix)
    out.parent.mkdir(parents=True, exist_ok=True)
    scripted.save(str(out) + ".ts")
    manifest = {
        **info,
        "contract_dim": sim_contract.local_obs_dim(),
        "sim_commit": sim_contract.SIM_COMMIT,
        # What the dimension cannot see: the field order and the scales.
        "contract_fingerprint": sim_contract.contract_fingerprint(),
        "created_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "action_heads": list(sim_contract.ACTION_HEADS),
    }
    with open(str(out) + ".json", "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"wrote {out}.ts and {out}.json (trained={info['trained']})")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--checkpoint", help="path to a sim training checkpoint (.pt)")
    group.add_argument("--random", action="store_true", help="random-init actor (pipeline bring-up)")
    parser.add_argument("--out", default="models/actor_policy", help="output prefix")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    actor, info = (
        build_from_checkpoint(args.checkpoint) if args.checkpoint else build_random(args.seed)
    )
    export(actor, info, args.out)


if __name__ == "__main__":
    main()
