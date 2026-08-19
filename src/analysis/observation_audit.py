"""Per-field audit of the encoded AV observation vector.

An ablation study that removes a field which never carried information will
report that the field was unnecessary. That is an artifact of the simulator, not
a finding about sensing. This module measures, per field, whether the encoded
observation actually varies, so later studies only ablate fields that had
something to say.

Three independent criteria are reported per field, because they answer different
questions: whether the value ever changes, whether it changes enough to matter,
and whether it ever departs from the neutral fallback the observation schema
defines for unsensed inputs.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np

from src.baselines import make_baseline
from src.config.loaders import deep_merge, load_named_config
from src.envs.topology_env import HighwayTopologyEnv
from src.rl.encoders import (
    COOPERATION_FIELDS,
    FIELD_SCALES,
    LANE_DISTRIBUTION_LANES,
    LOCAL_OBS_FIELDS,
    encode_local_observation,
)
from src.road.topology_factory import build_topology

DEFAULT_NEAR_CONSTANT_VARIANCE = 1e-4
COOPERATION_PREFIX = "cooperation."
LANE_DISTRIBUTION_PREFIX = "nearby_av_lane_distribution."
_DEFAULT_FREE_FLOW_MPS = 30.0
_FALLBACK_TOLERANCE = 1e-9


def encoded_field_names() -> tuple[str, ...]:
    """Names for the encoder's output dimensions, in encoder order.

    The cooperation block repeats three names that also appear at the top level,
    so those are prefixed; without that the audit would report duplicate fields.
    """
    return (
        *LOCAL_OBS_FIELDS,
        *(f"{COOPERATION_PREFIX}{field}" for field in COOPERATION_FIELDS),
        *(f"{LANE_DISTRIBUTION_PREFIX}{lane}" for lane in LANE_DISTRIBUTION_LANES),
    )


@dataclass(frozen=True)
class FieldAudit:
    """Verdict for one encoded dimension.

    `never_left_fallback` is None for fields the observation schema defines no
    neutral fallback for; reporting False there would imply a fallback exists.
    """

    field: str
    n_samples: int
    unique_values: int
    variance: float
    is_strictly_constant: bool
    is_near_constant: bool
    never_left_fallback: bool | None
    constant_value: float | None


@dataclass(frozen=True)
class AuditResult:
    """Audit broken down along both axes that change the verdict.

    Topology matters because geometry forbids some fields from moving. AV count
    matters because the local-aggregate cooperation fields only populate when two
    AVs come within sensing range, so a low-penetration run reports them dead for
    reasons that have nothing to do with the observation design.
    """

    per_field_pooled: tuple[FieldAudit, ...]
    per_field_by_topology: Mapping[str, tuple[FieldAudit, ...]]
    per_field_by_av_count: Mapping[int, tuple[FieldAudit, ...]]
    samples_by_condition: Mapping[str, int]


@dataclass(frozen=True)
class SampleSpec:
    topology: str
    controller: str
    seed: int
    duration_steps: int = 120
    demand: str = "medium"
    human_model: str = "normal"
    controlled_vehicles: int = 2
    initial_human_vehicles: int = 12

    @property
    def condition_id(self) -> str:
        return f"{self.topology}/{self.controller}/av{self.controlled_vehicles}/seed{self.seed}"


def _scale_for(field: str) -> float:
    """Mirror the encoder's scale lookup for a (possibly prefixed) field name."""
    if field.startswith(LANE_DISTRIBUTION_PREFIX):
        return 1.0
    base = field[len(COOPERATION_PREFIX):] if field.startswith(COOPERATION_PREFIX) else field
    return float(FIELD_SCALES.get(base, 1.0))


def _encode_scalar(field: str, value: float) -> float:
    return float(value) / max(_scale_for(field), 1e-9)


def free_flow_speeds_for_topology(topology_id: str) -> tuple[float, ...]:
    """Distinct lane speed limits, which are what the sensing layer reports as
    free-flow speed. Returned as a set of candidates because a topology may mix
    limits across lanes, and a single constant would then be the wrong target.
    """
    topology_cfg = load_named_config("topology", topology_id)
    road_cfg = topology_cfg.get("road", topology_cfg) if isinstance(topology_cfg, Mapping) else None
    spec = build_topology(topology_id, road_cfg)
    limits = {
        float(lane.speed_limit)
        for lane in spec.road_network.lanes_dict().values()
        if getattr(lane, "speed_limit", None)
    }
    return tuple(sorted(limits)) if limits else (_DEFAULT_FREE_FLOW_MPS,)


def encoded_fallback_candidates(free_flow_speeds: Sequence[float]) -> dict[str, tuple[float, ...]]:
    """Encoded neutral-fallback values per field, from the observation schema.

    Speed-valued fallbacks are free-flow speed, which varies by topology, so each
    field maps to every candidate value rather than a single constant.
    """
    speeds = tuple(free_flow_speeds) or (_DEFAULT_FREE_FLOW_MPS,)
    # src/sensing/local.py writes the same float to the top-level key and its
    # cooperation twin, and the same int to active_av_count_local and
    # nearby_av_count. Identical columns must get identical verdicts.
    fixed: dict[str, float] = {
        "nearby_av_count": 0.0,
        "active_av_count_local": 0.0,
        "nearby_av_density": 0.0,
        "merge_pressure": 0.0,
        "downstream_congestion_estimate": 0.0,
        f"{COOPERATION_PREFIX}merge_pressure": 0.0,
        f"{COOPERATION_PREFIX}downstream_congestion_estimate": 0.0,
    }
    for lane in LANE_DISTRIBUTION_LANES:
        fixed[f"{LANE_DISTRIBUTION_PREFIX}{lane}"] = 0.0

    candidates = {field: (_encode_scalar(field, value),) for field, value in fixed.items()}
    for field in (
        "nearby_av_mean_speed",
        "segment_target_speed",
        f"{COOPERATION_PREFIX}segment_target_speed",
    ):
        candidates[field] = tuple(_encode_scalar(field, speed) for speed in speeds)
    return candidates


def audit_fields(
    samples: np.ndarray,
    field_names: Sequence[str],
    *,
    near_constant_variance: float = DEFAULT_NEAR_CONSTANT_VARIANCE,
    fallback_candidates: Mapping[str, Sequence[float]] | None = None,
) -> tuple[FieldAudit, ...]:
    """Audit each column of `samples` (shape [N, len(field_names)]).

    With no samples every flag is False: absence of evidence is not constancy.
    """
    names = tuple(field_names)
    fallbacks = dict(fallback_candidates or {})
    # Validate width before any reshape: reshaping a wrong-width array succeeds
    # silently and misaligns every column against its field name.
    array = np.asarray(samples, dtype=float)
    if array.size == 0:
        matrix = np.empty((0, len(names)), dtype=float)
    elif array.ndim == 1:
        if not names or array.size % len(names):
            raise ValueError(f"flat samples of size {array.size} do not divide into {len(names)} fields")
        matrix = array.reshape(-1, len(names))
    elif array.ndim == 2:
        if array.shape[1] != len(names):
            raise ValueError(f"samples have {array.shape[1]} columns, expected {len(names)}")
        matrix = array
    else:
        raise ValueError(f"samples must be 1- or 2-dimensional, got {array.ndim}")

    audits: list[FieldAudit] = []
    for index, name in enumerate(names):
        column = matrix[:, index]
        count = int(column.size)
        expected = fallbacks.get(name)
        if count == 0:
            audits.append(
                FieldAudit(
                    field=name,
                    n_samples=0,
                    unique_values=0,
                    variance=0.0,
                    is_strictly_constant=False,
                    is_near_constant=False,
                    never_left_fallback=None if expected is None else False,
                    constant_value=None,
                )
            )
            continue
        unique = np.unique(column)
        strictly_constant = unique.size == 1
        if strictly_constant:
            variance = 0.0
        else:
            finite = column[np.isfinite(column)]
            variance = float(np.var(finite)) if finite.size else 0.0
        never_left = None
        if expected is not None:
            never_left = bool(
                np.all(np.isclose(column[:, None], np.asarray(expected, dtype=float)[None, :], atol=_FALLBACK_TOLERANCE).any(axis=1))
            )
        audits.append(
            FieldAudit(
                field=name,
                n_samples=count,
                unique_values=int(unique.size),
                variance=variance,
                is_strictly_constant=strictly_constant,
                is_near_constant=variance < float(near_constant_variance),
                never_left_fallback=never_left,
                constant_value=float(unique[0]) if strictly_constant else None,
            )
        )
    return tuple(audits)


def build_env_config(spec: SampleSpec) -> dict[str, Any]:
    """Environment config for one condition.

    Mirrors `scripts/run_baseline.py`: `no_av` runs zero AVs, and on the ring its
    would-be AVs are replaced by humans so the population is comparable across
    controllers.
    """
    topology_cfg = load_named_config("topology", spec.topology)
    demand_cfg = load_named_config("demand", spec.demand)
    human_cfg = load_named_config("human_model", spec.human_model)

    if spec.controller == "no_av":
        demand_cfg = deep_merge(demand_cfg, {"av_penetration": 0.0})
    controlled = 0 if spec.controller == "no_av" else spec.controlled_vehicles
    if spec.topology == "ring":
        initial_humans = spec.initial_human_vehicles + (spec.controlled_vehicles - controlled)
    else:
        initial_humans = 0
    return {
        "topology": topology_cfg,
        "demand": demand_cfg,
        "human_model": human_cfg,
        "controller": {
            "name": spec.controller,
            "family": "baseline",
            "safety_mode": make_baseline(spec.controller).metadata.safety_mode,
        },
        "duration_steps": spec.duration_steps,
        "dt": 1.0,
        "controlled_vehicles": controlled,
        "initial_human_vehicles": initial_humans,
    }


def collect_encoded_samples(spec: SampleSpec) -> np.ndarray:
    """Run one condition and return every AV's encoded observation per step.

    Shape is [N, 39] where N is the number of (step, active AV) pairs. `no_av`
    yields an empty array because it has no AVs to observe from.
    """
    width = len(encoded_field_names())
    controller = make_baseline(spec.controller)
    controller.reset(env_metadata={"topology_id": spec.topology}, seed=spec.seed)
    env = HighwayTopologyEnv(spec.topology, build_env_config(spec))
    observations, _ = env.reset(seed=spec.seed)

    rows: list[np.ndarray] = []
    terminated = False
    truncated = False
    while not (terminated or truncated):
        for observation in observations.values():
            rows.append(encode_local_observation(observation).numpy())
        actions = controller.act(observations, global_state=None)
        observations, _, terminated, truncated, _ = env.step(actions)
    for observation in observations.values():
        rows.append(encode_local_observation(observation).numpy())

    if not rows:
        return np.empty((0, width), dtype=float)
    return np.asarray(rows, dtype=float)


def run_audit(
    specs: Sequence[SampleSpec],
    *,
    near_constant_variance: float = DEFAULT_NEAR_CONSTANT_VARIANCE,
) -> AuditResult:
    """Audit pooled across all conditions, and per topology.

    Per-topology results matter because inertness is topology-dependent: lane
    fields cannot move where the topology forbids lane changes, and the
    downstream-bottleneck field is constant where no bottleneck exists.
    """
    names = encoded_field_names()
    by_topology: dict[str, list[np.ndarray]] = {}
    by_av_count: dict[int, list[np.ndarray]] = {}
    counts: dict[str, int] = {}
    for spec in specs:
        samples = collect_encoded_samples(spec)
        counts[spec.condition_id] = int(samples.shape[0])
        by_topology.setdefault(spec.topology, []).append(samples)
        by_av_count.setdefault(spec.controlled_vehicles, []).append(samples)

    def stack(chunks: Sequence[np.ndarray]) -> np.ndarray:
        usable = [chunk for chunk in chunks if chunk.size]
        return np.concatenate(usable, axis=0) if usable else np.empty((0, len(names)), dtype=float)

    def audit(chunks: Sequence[np.ndarray], speeds: Sequence[float]) -> tuple[FieldAudit, ...]:
        return audit_fields(
            stack(chunks),
            names,
            near_constant_variance=near_constant_variance,
            fallback_candidates=encoded_fallback_candidates(speeds),
        )

    per_topology = {
        topology: audit(chunks, free_flow_speeds_for_topology(topology))
        for topology, chunks in by_topology.items()
    }
    all_speeds = sorted({speed for topology in by_topology for speed in free_flow_speeds_for_topology(topology)})
    per_av_count = {count: audit(chunks, all_speeds) for count, chunks in by_av_count.items()}
    pooled = audit([chunk for chunks in by_topology.values() for chunk in chunks], all_speeds)
    return AuditResult(
        per_field_pooled=pooled,
        per_field_by_topology=per_topology,
        per_field_by_av_count=per_av_count,
        samples_by_condition=counts,
    )
