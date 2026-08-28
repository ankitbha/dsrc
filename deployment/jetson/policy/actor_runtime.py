"""Load and run the exported actor bundle on the Jetson CPU.

The actor is a 39 -> 128 -> 128 -> 4x3 MLP. The TorchScript module is
the artifact of record, but per-tick inference uses a numpy mirror of
its weights: at this size TorchScript interpreter dispatch costs ~4 ms
with multi-ms jitter on the Orin's CPU, while two 128-wide matmuls in
numpy run in tens of microseconds. The numpy mirror is verified against
the TorchScript module at load time (1e-5 agreement on random inputs),
so there is no silent-divergence risk; pass use_numpy=False to fall
back to TorchScript execution.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

from policy import sim_contract

torch.set_num_threads(1)


class _NumpyMlpActor:
    """Weights-only mirror of the exported actor (backbone.{0,2,4} + heads)."""

    def __init__(self, module: torch.jit.ScriptModule, head_names: tuple[str, ...]) -> None:
        params = {name: p.detach().numpy().copy() for name, p in module.named_parameters()}
        layer_ids = sorted(
            {int(name.split(".")[1]) for name in params if name.startswith("backbone.")}
        )
        self.backbone = [
            (params[f"backbone.{i}.weight"].T.copy(), params[f"backbone.{i}.bias"].copy())
            for i in layer_ids
        ]
        self.heads = {
            head: (params[f"heads.{head}.weight"].T.copy(), params[f"heads.{head}.bias"].copy())
            for head in head_names
        }

    def __call__(self, obs: np.ndarray) -> dict[str, np.ndarray]:
        x = obs
        last = len(self.backbone) - 1
        for i, (w, b) in enumerate(self.backbone):
            x = x @ w + b
            if i < last:  # mlp() puts Tanh after every Linear except the final one
                x = np.tanh(x)
        return {head: x @ w + b for head, (w, b) in self.heads.items()}


def _softmax(logits: np.ndarray) -> np.ndarray:
    shifted = logits - logits.max()
    e = np.exp(shifted)
    return e / e.sum()


@dataclass
class PolicyOutput:
    action: dict[str, str]              # all 4 heads (defaults fill inactive ones)
    head_probs: dict[str, list[float]]  # softmax per active head
    chosen_prob: dict[str, float]       # prob of the chosen value per head
    confidence: float                   # min over active heads of max prob
    latency_ms: float


class ActorRuntime:
    def __init__(self, bundle_prefix: str, deterministic: bool = True, use_numpy: bool = True) -> None:
        prefix = Path(bundle_prefix)
        manifest_path = prefix.with_suffix(".json")
        module_path = prefix.with_suffix(".ts")
        if not module_path.exists():
            raise FileNotFoundError(
                f"actor bundle not found at {module_path} - create one with\n"
                "  python3 policy/export_policy.py --random --out models/actor_policy\n"
                "or export a trained checkpoint (see policy/export_policy.py)."
            )
        with open(manifest_path) as f:
            self.manifest = json.load(f)
        if int(self.manifest["contract_dim"]) != sim_contract.local_obs_dim():
            raise RuntimeError(
                f"bundle was exported for obs dim {self.manifest['contract_dim']} but the "
                f"vendored contract is {sim_contract.local_obs_dim()}; re-export the policy."
            )
        # The dimension does not move when the contract does. Reordering two fields or
        # changing a scale leaves it at 39, so an old module loaded silently against a
        # new contract and every value landed in the wrong slot -- silent and total.
        #
        # Absent on bundles exported before this existed, and those are accepted: the
        # check is here to catch a contract that MOVED, and refusing every older bundle
        # would be a different and much more disruptive rule than the one intended.
        exported = self.manifest.get("contract_fingerprint")
        current = sim_contract.contract_fingerprint()
        if exported is not None and exported != current:
            raise RuntimeError(
                f"bundle was exported against contract {exported} and the vendored "
                f"contract is {current} -- same dimension, different fields or scales; "
                f"re-export the policy."
            )
        self.module = torch.jit.load(str(module_path), map_location="cpu")
        self.module.eval()
        self.deterministic = deterministic
        self.is_trained = bool(self.manifest.get("trained", False))
        self.active_heads = sim_contract.active_heads(self.manifest.get("action_profile", "full"))
        self._rng = np.random.default_rng(0)
        self._np_actor = _NumpyMlpActor(self.module, self.active_heads) if use_numpy else None
        if self._np_actor is not None:
            self._verify_numpy_mirror()
        # warm the kernels so the first live tick is not an outlier
        self.act(np.zeros(sim_contract.local_obs_dim(), dtype=np.float32))

    def _verify_numpy_mirror(self) -> None:
        rng = np.random.default_rng(7)
        for _ in range(4):
            obs = rng.normal(0, 2, sim_contract.local_obs_dim()).astype(np.float32)
            with torch.inference_mode():
                torch_logits = self.module(torch.from_numpy(obs).unsqueeze(0))
            np_logits = self._np_actor(obs)
            for head in self.active_heads:
                if not np.allclose(torch_logits[head][0].numpy(), np_logits[head], atol=1e-5):
                    raise RuntimeError(
                        f"numpy actor mirror diverges from TorchScript on head '{head}' - "
                        "unexpected architecture; load with use_numpy=False and report."
                    )

    def _logits(self, encoded_obs: np.ndarray) -> dict[str, np.ndarray]:
        if self._np_actor is not None:
            return self._np_actor(encoded_obs)
        obs = torch.from_numpy(encoded_obs).unsqueeze(0)
        with torch.inference_mode():
            torch_logits = self.module(obs)
        return {head: torch_logits[head][0].numpy() for head in self.active_heads}

    def act(self, encoded_obs: np.ndarray) -> PolicyOutput:
        t0 = time.monotonic()
        logits = self._logits(np.ascontiguousarray(encoded_obs, dtype=np.float32))
        indices = sim_contract.default_indices()
        head_probs: dict[str, list[float]] = {}
        chosen_prob: dict[str, float] = {}
        confidence = 1.0
        for head in self.active_heads:
            probs = _softmax(logits[head])
            if self.deterministic:
                idx = int(probs.argmax())
            else:
                idx = int(self._rng.choice(len(probs), p=probs / probs.sum()))
            indices[head] = idx
            head_probs[head] = [round(float(p), 4) for p in probs]
            chosen_prob[head] = float(probs[idx])
            confidence = min(confidence, float(probs.max()))
        return PolicyOutput(
            action=sim_contract.indices_to_action(indices),
            head_probs=head_probs,
            chosen_prob=chosen_prob,
            confidence=confidence,
            latency_ms=(time.monotonic() - t0) * 1000.0,
        )
