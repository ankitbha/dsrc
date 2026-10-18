"""The learned correction's onset predictor (plan 2026-10-16 section 5.5).

A classifier of "a jam spell of at least 120 s starts on my segment or the next downstream
one within the next 120 s", trained on the simulator's own labels. It reads only what a
device has: the camera's local estimates, the applied feed version and the plan, and the
map. Ego GPS speed and every true density are excluded by name (`DENIED`).

Two models are fitted, a logistic regression and a 2 x 32 ReLU network, with
class-weighted binary cross-entropy, Adam and early stopping on the validation seeds; the
lower validation log-loss wins. The model card lists the features, the scaler, the seeds,
the round and the weights' sha256.
"""
from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

FEATURES: tuple[str, ...] = (
    "k_hat", "s_hat", "n_hat_left", "n_hat_own", "n_hat_right", "visible_lanes",
    "g_hat_m", "own_leader_reported_speed_mps", "mean_leader_reported_speed_mps",
    "mean_leader_reported_speed_slope", "mean_confidence", "detections",
    "feed_age_s", "plan_fraction_own", "plan_fraction_next",
    "feed_speed_own_kmh", "feed_jam_factor_own", "feed_speed_next_kmh",
    "feed_jam_factor_next", "segment_length_km", "segment_lanes", "remaining_in_segment_m",
)
#: No feature name may contain one of these (plan attack point 14).
DENIED: tuple[str, ...] = ("ego", "gps", "k_true", "true_", "oracle")
HORIZON_S: float = 120.0
HOLD_S: float = 120.0
G_NONE_M: float = 80.0
LAST_SEGMENT: int = 11
HIDDEN: int = 32
LEARNING_RATE: float = 1e-2
MAX_EPOCHS: int = 300
PATIENCE: int = 15
BATCH: int = 512


# ------------------------------------------------------------------ features


def _remaining_tail(snapshot, segment_of: dict[str, int]) -> dict[str, float]:
    """Per edge, the length of the edges after it on its route inside the same segment."""
    tail: dict[str, float] = {}
    for edge in segment_of:
        total, current, seen = 0.0, edge, {edge}
        while True:
            nxt = snapshot.successor.get(current)
            if nxt is None or nxt in seen or segment_of.get(nxt) != segment_of[edge]:
                break
            total += snapshot.geometry[nxt].length_m if nxt in snapshot.geometry else 0.0
            seen.add(nxt)
            current = nxt
        tail[edge] = total
    return tail


def engine_features(engine, t: float, index: np.ndarray, table, est, s_hat, v_slope,
                    versions: np.ndarray, segments: np.ndarray) -> np.ndarray:
    """`(len(index), len(FEATURES))` for the egos at rows `index` of a sensing pass.

    On the approach boundary (approach plan section 4.4) a vehicle's own super-segment `e`
    may be an approach one: "next" is the first element of its downstream chain (`e` when
    the chain is empty, which is `min(e + 1, 11)` on the original segments), the feed's
    speed and jam factor come from the version's 36-row snapshot, the plan's fractions
    through `plan_row`, and the segment length and lanes from the snapshot row of the
    vehicle's own super-segment, which on an approach is its own edge. An engine without
    these tables reads exactly as before.
    """
    from src.sumo.mainz import SPEED_ACTION_FRACTIONS

    if not hasattr(engine, "_remaining_tail"):
        engine._remaining_tail = _remaining_tail(table.snapshot, engine._segment_of)
    tail = engine._remaining_tail
    geometry = table.snapshot.geometry
    plan_row = getattr(engine, "plan_row", None) or None
    downstream = getattr(engine, "downstream", None) or None
    approach_states = getattr(engine, "approach_states", None) or {}
    extended: dict[int, np.ndarray] = {}
    rows = []
    for i in np.asarray(index, dtype=np.int64):
        k = int(versions[i])
        e = int(segments[i])
        if downstream is None:
            nxt = min(e + 1, LAST_SEGMENT)
        else:
            nxt = downstream[e][0] if downstream[e] else e
        if k in approach_states:
            if k not in extended:
                extended[k] = np.concatenate([engine.states[k], approach_states[k]])
            x = extended[k]
        else:
            x = engine.states[k]
        free_mps = float(x[e, 1]) / 3.6
        vector = engine.vectors[k]
        own_row = plan_row[e] if plan_row is not None else e
        next_row = plan_row[nxt] if plan_row is not None else nxt
        edge = table.ego_edges[i]
        g = est.g_hat[i]
        rows.append([
            est.k_hat[i], s_hat[i], est.n_hat[i, 0], est.n_hat[i, 1], est.n_hat[i, 2],
            float(table.lanes_visible[i].sum()),
            G_NONE_M if np.isnan(g) else g,
            free_mps if np.isnan(est.own_speed[i]) else est.own_speed[i],
            free_mps if np.isnan(est.mean_speed[i]) else est.mean_speed[i],
            v_slope[i],
            0.0 if np.isnan(est.conf_mean[i]) else est.conf_mean[i],
            float(est.n_det[i]),
            t - engine.feed.publish_of(k),
            SPEED_ACTION_FRACTIONS[vector[own_row]], SPEED_ACTION_FRACTIONS[vector[next_row]],
            x[e, 0], x[e, 2], x[nxt, 0], x[nxt, 2], x[e, 4], x[e, 3],
            geometry[edge].length_m - float(table.ego_x[i]) + tail.get(edge, 0.0),
        ])
    return np.asarray(rows, dtype=np.float32).reshape(len(rows), len(FEATURES))


# -------------------------------------------------------------------- labels


def label_samples(samples: Sequence[dict], trace: Sequence[dict], v_jam: float,
                  horizon_s: float = HORIZON_S, hold_s: float = HOLD_S
                  ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """`(X, y, kept)` for one episode's samples from its 10 s trace.

    `y = 1` when a spell of at least `hold_s` starts on the sample's segment or the next
    downstream one within `(t, t + horizon_s]`; a sample whose own segment is jammed at the
    latest trace sample at or before `t` is dropped (`kept` False).
    """
    from src.analysis.jam_metrics import jam_spells
    from src.sumo.drain import jam_indicator

    starts: dict[int, list[float]] = {}
    for spell in jam_spells(trace, v_jam):
        if spell.length_s >= hold_s - 1e-9:
            starts.setdefault(spell.segment, []).append(spell.start)
    times = np.array([s["t"] for s in trace], dtype=float)
    jammed = jam_indicator(trace, v_jam)
    X, y, kept = [], [], []
    for sample in samples:
        t, e = float(sample["t"]), int(sample["segment"])
        position = int(np.searchsorted(times, t, side="right")) - 1
        if position >= 0 and jammed[position, e]:
            kept.append(False)
            continue
        kept.append(True)
        ahead = {e, min(e + 1, LAST_SEGMENT)}
        y.append(int(any(t < s <= t + horizon_s for seg in ahead for s in starts.get(seg, []))))
        X.append(sample["features"])
    return (np.asarray(X, dtype=np.float32).reshape(len(X), len(FEATURES)),
            np.asarray(y, dtype=np.int64), np.asarray(kept, dtype=bool))


# ----------------------------------------------------------------------- fit


def _network(kind: str, n_features: int):
    import torch.nn as nn

    if kind == "logistic":
        return nn.Linear(n_features, 1)
    return nn.Sequential(nn.Linear(n_features, HIDDEN), nn.ReLU(), nn.Linear(HIDDEN, HIDDEN),
                         nn.ReLU(), nn.Linear(HIDDEN, 1))


@dataclass
class Predictor:
    kind: str
    module: Any
    mean: np.ndarray
    scale: np.ndarray
    validation_log_loss: float | None = None
    candidates: dict[str, float] = field(default_factory=dict)
    positives: dict[str, int] = field(default_factory=dict)
    samples: dict[str, int] = field(default_factory=dict)
    card: dict[str, Any] = field(default_factory=dict)

    def predict(self, X: np.ndarray) -> np.ndarray:
        import torch

        X = (np.asarray(X, dtype=np.float32) - self.mean) / self.scale
        with torch.no_grad():
            logits = self.module(torch.from_numpy(X.astype(np.float32))).squeeze(-1)
            return torch.sigmoid(logits).numpy().astype(np.float64)


def _log_loss(p: np.ndarray, y: np.ndarray) -> float:
    p = np.clip(p, 1e-7, 1 - 1e-7)
    return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p))) if len(y) else math.nan


def _train(kind: str, X: np.ndarray, y: np.ndarray, Xv: np.ndarray, yv: np.ndarray):
    import torch

    torch.manual_seed(0)
    module = _network(kind, X.shape[1])
    positives = int(y.sum())
    weight = (len(y) - positives) / positives if positives else 1.0
    loss_fn = torch.nn.BCEWithLogitsLoss(pos_weight=torch.tensor(float(weight)))
    optimiser = torch.optim.Adam(module.parameters(), lr=LEARNING_RATE)
    Xt, yt = torch.from_numpy(X), torch.from_numpy(y.astype(np.float32))
    Xvt = torch.from_numpy(Xv)
    generator = torch.Generator().manual_seed(0)
    best, best_state, stale = math.inf, None, 0
    for _ in range(MAX_EPOCHS):
        order = torch.randperm(len(Xt), generator=generator)
        for start in range(0, len(order), BATCH):
            batch = order[start:start + BATCH]
            optimiser.zero_grad()
            loss_fn(module(Xt[batch]).squeeze(-1), yt[batch]).backward()
            optimiser.step()
        with torch.no_grad():
            val = _log_loss(torch.sigmoid(module(Xvt).squeeze(-1)).numpy(), yv)
        if val < best - 1e-6:
            best, stale = val, 0
            best_state = {k: v.clone() for k, v in module.state_dict().items()}
        else:
            stale += 1
            if stale >= PATIENCE:
                break
    if best_state is not None:
        module.load_state_dict(best_state)
    return module, best, weight


def fit(X: np.ndarray, y: np.ndarray, Xv: np.ndarray, yv: np.ndarray) -> Predictor:
    """Both models on standardised features; the lower validation log-loss wins."""
    X, Xv = np.asarray(X, np.float32), np.asarray(Xv, np.float32)
    y, yv = np.asarray(y, np.int64), np.asarray(yv, np.int64)
    mean = X.mean(axis=0) if len(X) else np.zeros(X.shape[1], np.float32)
    scale = X.std(axis=0) if len(X) else np.ones(X.shape[1], np.float32)
    scale = np.where(scale > 1e-8, scale, 1.0).astype(np.float32)
    Xs, Xvs = (X - mean) / scale, (Xv - mean) / scale
    fitted, losses, weights = {}, {}, {}
    for kind in ("logistic", "mlp"):
        fitted[kind], losses[kind], weights[kind] = _train(kind, Xs.astype(np.float32), y,
                                                           Xvs.astype(np.float32), yv)
    # A nan loss (no validation sample) never wins; with both nan the logistic is kept.
    kind = min(losses, key=lambda k: (not math.isfinite(losses[k]), losses[k]))
    return Predictor(kind, fitted[kind], mean.astype(np.float32), scale, losses[kind], losses,
                     {"train": int(y.sum()), "validation": int(yv.sum())},
                     {"train": int(len(y)), "validation": int(len(yv))},
                     {"pos_weight": weights[kind]})


def _finite(value):
    return float(value) if value is not None and math.isfinite(value) else None


def save(model: Predictor, card_path: Path, extra: dict[str, Any] | None = None) -> dict[str, Any]:
    """Write the weights beside the card (`.pt`) and the card itself; returns the card."""
    import torch

    card_path = Path(card_path)
    weights_path = card_path.with_suffix(".pt")
    torch.save(model.module.state_dict(), weights_path)
    card = {
        "schema": "onset_predictor/1", "kind": model.kind, "features": list(FEATURES),
        "denied_tokens": list(DENIED), "scaler": {"mean": model.mean.tolist(),
                                                  "scale": model.scale.tolist()},
        "label": (f"a jam spell of at least {HOLD_S:g} s starts on the own segment or the "
                  f"next downstream one within (t, t + {HORIZON_S:g} s]; samples whose own "
                  "segment is jammed at t are dropped"),
        # A loss that is not finite (no validation sample) is written as null.
        "validation_log_loss": _finite(model.validation_log_loss),
        "candidates": {k: _finite(v) for k, v in model.candidates.items()},
        "positives": model.positives, "samples": model.samples,
        "pos_weight": model.card.get("pos_weight"), "hidden": HIDDEN,
        "weights_file": weights_path.name,
        "weights_sha256": hashlib.sha256(weights_path.read_bytes()).hexdigest(),
        **(extra or {}),
    }
    card_path.write_text(json.dumps(card, indent=1))
    model.card = card
    return card


def load(card_path: Path) -> Predictor:
    import torch

    card_path = Path(card_path)
    card = json.loads(card_path.read_text())
    if tuple(card["features"]) != FEATURES:
        raise SystemExit(f"{card_path} was trained on other features")
    weights_path = card_path.with_name(card["weights_file"])
    sha = hashlib.sha256(weights_path.read_bytes()).hexdigest()
    if sha != card["weights_sha256"]:
        raise SystemExit(f"{weights_path} has sha256 {sha}; its card records "
                         f"{card['weights_sha256']}")
    module = _network(card["kind"], len(FEATURES))
    module.load_state_dict(torch.load(weights_path, weights_only=True))
    module.eval()
    return Predictor(card["kind"], module, np.asarray(card["scaler"]["mean"], np.float32),
                     np.asarray(card["scaler"]["scale"], np.float32),
                     card.get("validation_log_loss"), card.get("candidates", {}),
                     card.get("positives", {}), card.get("samples", {}), card)
