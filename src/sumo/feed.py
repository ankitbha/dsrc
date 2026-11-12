"""Per-vehicle feed versions: the feed's update process, the agreement rules and Lemma 1.

A traffic feed publishes versions `k = 1, 2, ...` stamped `T_k`. Version `k` is first
returned by a query at its availability instant `A_k = T_k + L_k`, and a query at `t`
returns `k(t) = max{k : A_k <= t}`. A fleet of vehicles reads the feed at its own query
instants and applies, from some instant, the action vector the policy computes from the
version it fetched. Which version a vehicle applies at `t` is a rule's whole content, and
every rule here is a pure function of the vehicle's query instants and the process, so
it is tested without SUMO.

Versions are numbered from 1; version 0 means "no version yet". All times are seconds of
simulation time. Plan section 1 states the rules and the lemma; section 3.1 this module.
"""
from __future__ import annotations

import hashlib
import json
import math
from collections import Counter
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

#: The query-phase draw is `default_rng(seed + PHASE_SEED_OFFSET)`, independent of
#: SUMO's own seed so the traffic realisation of a seed is the same in every arm.
PHASE_SEED_OFFSET: int = 1000


def draw_phases(seed: int, count: int) -> np.ndarray:
    """One uniform draw in [0, 1) per vehicle, indexed by schedule position.

    A rule maps the draw to its own phase (`u * P_q` for a periodic schedule), so the
    same vehicle carries the same draw in every arm of a seed.
    """
    return np.random.default_rng(seed + PHASE_SEED_OFFSET).random(count)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


@dataclass(frozen=True)
class FeedProcess:
    """Publication instants `T_k` and availability lags `L_k` (k = 1..K)."""

    publish_s: np.ndarray
    lag_s: np.ndarray
    source: str
    meta: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        publish = np.asarray(self.publish_s, dtype=float)
        lag = np.asarray(self.lag_s, dtype=float)
        if publish.shape != lag.shape or publish.ndim != 1 or not len(publish):
            raise ValueError("publish_s and lag_s must be equal-length, non-empty vectors")
        if np.any(np.diff(publish) <= 0.0):
            raise ValueError("publication instants must be strictly increasing")
        if np.any(lag < 0.0):
            raise ValueError("an availability lag cannot be negative")
        object.__setattr__(self, "publish_s", publish)
        object.__setattr__(self, "lag_s", lag)
        available = publish + lag
        # A query returns the highest-numbered version already available, so when a
        # large lag puts A_k after A_{k+1} version k is never returned. The prefix
        # maximum over availability order gives k(t) for any t in one search.
        order = np.argsort(available, kind="stable")
        object.__setattr__(self, "_sorted_available", available[order])
        object.__setattr__(self, "_version_upto", np.maximum.accumulate(order + 1))

    # ------------------------------------------------------------ constructors

    @classmethod
    def aligned(cls, decision_instants: Iterable[float]) -> "FeedProcess":
        """Publication at the env's decision instants with zero lag: the published arm."""
        instants = np.asarray(list(decision_instants), dtype=float)
        return cls(instants, np.zeros_like(instants), "aligned")

    @classmethod
    def synthetic(cls, publish_s: Sequence[float], lag_s: Sequence[float]) -> "FeedProcess":
        """A process given directly. For tests only; no experiment arm uses it."""
        return cls(np.asarray(publish_s, float), np.asarray(lag_s, float), "synthetic")

    @classmethod
    def from_artefact(cls, path: Path, session: str, seed: int, start_s: float,
                      end_s: float, allow_wrap: bool = False) -> "FeedProcess":
        """Replay a session's recorded `(gap, L_up)` pairs from a seeded start index.

        `T_{k+1} = T_k + gap_k`; version `k` carries lag `L_up_k`. `T_1` is `start_s`,
        moved forward to the recorded stamp's second of the minute when the sequence
        carries it (a third element per pair), so simulation time modulo 60 is the
        replayed stamps' wall-clock second and a fixed-phase `psi` read from the artefact
        means the same second in the simulation.

        The replay never wraps (plan Amendment 1, A9): the start index is drawn by
        `default_rng(seed)` from the indices whose remaining pairs cover `end_s`. A
        session too short for any start is refused, unless `allow_wrap` (the drive-log
        placeholder, which is shorter than one episode); then the start is drawn from
        all indices, and at each wrap the gap is lengthened to the next pair's recorded
        second of the minute, so every replayed stamp keeps its own. The artefact's
        sha256, the session, the start index, the number of wraps and the artefact's
        placeholder flag travel in `meta` so every run that reads it can record them.
        """
        path = Path(path)
        artefact = json.loads(path.read_text())
        sessions = artefact["sessions"]
        if session not in sessions:
            raise KeyError(f"{path} has no session {session!r}; it has {sorted(sessions)}")
        sequence = sessions[session]["sequence"]
        pairs = [(float(item[0]), float(item[1])) for item in sequence]
        phases = [(float(item[2]) if len(item) > 2 and item[2] is not None else None)
                  for item in sequence]
        n = len(pairs)
        if not n:
            raise ValueError(f"session {session!r} in {path} has no (gap, lag) pairs")

        def first_stamp(index: int) -> float:
            first = float(start_s)
            if phases[index] is not None:
                first += (phases[index] - first) % 60.0
            return first

        covering = [j for j in range(n)
                    if first_stamp(j) + sum(gap for gap, _ in pairs[j:]) > end_s]
        rng = np.random.default_rng(seed)
        if covering:
            start = covering[int(rng.integers(len(covering)))]
        elif allow_wrap:
            start = int(rng.integers(n))
        else:
            raise ValueError(f"session {session!r} spans {sum(g for g, _ in pairs):g} s of "
                             f"gaps, too short to replay [{start_s:g}, {end_s:g}] s "
                             "without wrapping")
        publish, lag = [first_stamp(start)], []
        step = wraps = 0
        while True:
            index = (start + step) % n
            gap, version_lag = pairs[index]
            lag.append(version_lag)
            following = (index + 1) % n
            next_stamp = publish[-1] + gap
            if following == 0:
                if phases[following] is not None:
                    next_stamp += (phases[following] - next_stamp) % 60.0
            if next_stamp > end_s:
                break
            if following == 0:
                wraps += 1
            publish.append(next_stamp)
            step += 1
        return cls(np.asarray(publish), np.asarray(lag), "artefact", {
            "artefact": str(path), "artefact_sha256": _sha256(path), "session": session,
            "start_index": start, "pairs": n, "wraps": wraps,
            "covering_starts": len(covering),
            "placeholder": bool(artefact.get("placeholder", False)),
            "artefact_source": artefact.get("source"),
        })

    def with_lag_multiplier(self, m: float) -> "FeedProcess":
        """The same stamps with every availability lag multiplied by `m` (jam plan J25:
        `m = 0` is zero lag, `m = 2` doubles each version's own lag)."""
        if m < 0.0:
            raise ValueError(f"a lag multiplier cannot be negative, got {m}")
        return FeedProcess(self.publish_s.copy(), self.lag_s * float(m), self.source,
                           {**self.meta, "lag_multiplier": float(m)})

    # ----------------------------------------------------------------- queries

    @property
    def available_s(self) -> np.ndarray:
        return self.publish_s + self.lag_s

    @property
    def count(self) -> int:
        return len(self.publish_s)

    def publish_of(self, version: int) -> float:
        return float(self.publish_s[version - 1])

    def version_at(self, t: float | np.ndarray) -> int | np.ndarray:
        """The version a query at `t` returns, 0 before the first availability."""
        position = np.searchsorted(self._sorted_available, t, side="right") - 1
        versions = np.where(position >= 0, self._version_upto[np.maximum(position, 0)], 0)
        return int(versions) if np.ndim(versions) == 0 else versions

    def gaps(self) -> np.ndarray:
        return np.diff(self.publish_s)

    def describe(self) -> dict[str, Any]:
        """The lemma's parameters for this process, for a run header or summary."""
        gaps = self.gaps()
        return {
            "source": self.source, "versions": self.count,
            "g_min": float(gaps.min()) if len(gaps) else None,
            "L_bar": float(self.lag_s.max()), "L_low": float(self.lag_s.min()),
            "J": float(self.lag_s.max() - self.lag_s.min()), **self.meta,
        }


# --------------------------------------------------------------------- vehicles


@dataclass
class VehicleFeedState:
    """One controlled vehicle's feed bookkeeping.

    `u` is its uniform draw from `draw_phases`; a rule turns it into a query phase.
    `held` is the newest version it has fetched, `applied` the version whose action it
    applies and `switched_at` the instant `applied` last changed.
    """

    vehicle_id: str
    u: float
    held: int = 0
    applied: int = 0
    switched_at: float = math.nan


def _queries(phase: float, period: float, end_s: float) -> np.ndarray:
    """`phase + m * period` from the last instant at or before 0 through `end_s`.

    Starting one period before 0 means a vehicle has a query in every window of length
    `period`, including the first, as if it had been reading the feed before it was
    inserted.
    """
    first = math.floor((0.0 - phase) / period)
    last = math.ceil((end_s - phase) / period)
    return phase + period * np.arange(first, last + 1, dtype=float)


def _fetches(queries: np.ndarray, feed: FeedProcess) -> tuple[np.ndarray, np.ndarray]:
    """The queries at which the returned version is newer than any fetched before."""
    versions = np.asarray(feed.version_at(queries))
    previous = np.concatenate(([0], np.maximum.accumulate(versions)[:-1]))
    new = versions > previous
    return queries[new], versions[new]


class Rule:
    """A rule turns a vehicle's query instants and the process into a step function.

    `switch_schedule` returns the instants at which the vehicle's applied version
    changes and the version it changes to, both increasing. `applied_version` reads that
    step function at `t`.
    """

    name = "rule"

    def switch_schedule(self, state: VehicleFeedState, feed: FeedProcess,
                        end_s: float) -> tuple[np.ndarray, np.ndarray]:
        raise NotImplementedError

    def applied_version(self, state: VehicleFeedState, feed: FeedProcess, t: float) -> int:
        instants, versions = self.switch_schedule(state, feed, t)
        return version_from_schedule(instants, versions, t)

    def describe(self) -> dict[str, Any]:
        return {"rule": self.name}


def version_from_schedule(instants: np.ndarray, versions: np.ndarray, t: float) -> int:
    position = int(np.searchsorted(instants, t, side="right")) - 1
    return int(versions[position]) if position >= 0 else 0


@dataclass
class NoRule(Rule):
    """Query every `P_q` at phase `u * P_q` and apply the fetched version on receipt."""

    P_q: float
    name = "no_rule"

    def switch_schedule(self, state, feed, end_s):
        return _fetches(_queries(state.u * self.P_q, self.P_q, end_s), feed)

    def describe(self):
        return {"rule": self.name, "P_q": self.P_q}


@dataclass
class FixedPhase(Rule):
    """Query once per `P_q` (60 s) at wall-clock second `psi` and apply on receipt.

    A vehicle's request completes at `psi + u * r_bar` within the minute, `r_bar` being
    the largest request round-trip time, so the fleet's requests land inside
    `[psi, psi + r_bar]` and a version made available inside that window splits it.
    """

    psi: float
    r_bar: float
    P_q: float = 60.0
    name = "fixed_phase"

    def switch_schedule(self, state, feed, end_s):
        return _fetches(_queries(self.psi + state.u * self.r_bar, self.P_q, end_s), feed)

    def describe(self):
        return {"rule": self.name, "psi": self.psi, "r_bar": self.r_bar, "P_q": self.P_q}


@dataclass
class VersionKeyed(Rule):
    """Algorithm 2: apply the action of stamp `T` from `T + D`, or on receipt if later.

    A vehicle queries every `P_q` at phase `u * P_q`. On fetching version `k` at `q` it
    applies `a_k` from `max(T_k + D, q)`. A later version replaces it only at that
    version's own switch instant, which is always later, because both `T_k + D` and the
    fetch instant increase with `k`.
    """

    P_q: float
    D: float
    name = "version_keyed"

    def switch_schedule(self, state, feed, end_s):
        queries, versions = _fetches(_queries(state.u * self.P_q, self.P_q, end_s), feed)
        instants = np.maximum(feed.publish_s[versions - 1] + self.D, queries)
        keep = instants <= end_s
        return instants[keep], versions[keep]

    def describe(self):
        return {"rule": self.name, "P_q": self.P_q, "D": self.D}


@dataclass
class Synchronous(Rule):
    """Every vehicle switches to a version at its availability instant."""

    name = "synchronous"

    def switch_schedule(self, state, feed, end_s):
        order = np.argsort(feed.available_s, kind="stable")
        instants = feed.available_s[order]
        versions = np.maximum.accumulate(order + 1)
        previous = np.concatenate(([0], versions[:-1]))
        keep = (versions > previous) & (instants <= end_s)
        return instants[keep], versions[keep]


# ----------------------------------------------------------------------- lemma


def lemma_bound(D: float, L_bar: float, P_q: float, g: float, tau: float) -> float:
    """Lemma 1 (b): the expected fraction not applying `a_k` at `T_k + D + tau`.

    `1 - (D - L_bar + tau) / P_q`, clipped to [0, 1]. For `D >= L_bar` this is the
    lemma's statement; for `D < L_bar` the same expression clipped at 1 is still an
    upper bound, because a vehicle cannot hold a version before it is available.
    Defined for `0 <= tau < g`, the cycle until the next version's switch.
    """
    if not 0.0 <= tau < g:
        raise ValueError(f"tau must lie in [0, g) = [0, {g}), got {tau}")
    return float(min(1.0, max(0.0, 1.0 - (D - L_bar + tau) / P_q)))


def lemma_cycle_average(D: float, L_bar: float, P_q: float, g: float) -> float:
    """`lemma_bound` averaged over a cycle of length `g`.

    `(P_q - D + L_bar)^2 / (2 P_q g)` for `L_bar <= D < L_bar + P_q`, zero for
    `D >= L_bar + P_q`, and `(L_bar - D + P_q / 2) / g` for `D < L_bar` (capped at the
    cycle), all from integrating the clipped line exactly.
    """
    x = D - L_bar
    full = max(0.0, min(g, -x))
    low, high = max(0.0, -x), min(g, P_q - x)
    linear = 0.0
    if high > low:
        linear = (high - low) - ((x + high) ** 2 - (x + low) ** 2) / (2.0 * P_q)
    return float((full + linear) / g)


def lemma_skip_bound(P_q: float, g: float, J: float) -> float:
    """Lemma 1 (c): the fraction that can skip a version, `1 - (g - J) / P_q` in [0, 1]."""
    return float(min(1.0, max(0.0, 1.0 - (g - J) / P_q)))


# ---------------------------------------------------------------- disagreement


def disagreement(applied_versions: Iterable[int]) -> float:
    """The fraction of vehicles whose applied version is not the plurality version.

    nan for an empty fleet, so a time average skips instants with no vehicle rather than
    counting them as agreement.
    """
    counts = Counter(applied_versions)
    total = sum(counts.values())
    if not total:
        return math.nan
    return 1.0 - max(counts.values()) / total


def command_disagreement(applied: Sequence[int], segments: Sequence[int],
                         action_vectors: Mapping[int, Sequence[float]]) -> float:
    """The fraction of vehicles whose command on their own segment is not that segment's
    plurality command.

    A vehicle on segment `e` applying version `k` commands `action_vectors[k][e]`; a
    vehicle with no version yet (`k` absent from `action_vectors`) commands nothing,
    which is a command value of its own. At most `disagreement(applied)`, and it is the
    quantity the flow responds to.
    """
    if len(applied) != len(segments):
        raise ValueError("one segment per vehicle")
    if not applied:
        return math.nan
    by_segment: dict[int, Counter] = {}
    for version, segment in zip(applied, segments):
        vector = action_vectors.get(version)
        command = None if vector is None else float(vector[segment])
        by_segment.setdefault(segment, Counter())[command] += 1
    agreeing = sum(max(counts.values()) for counts in by_segment.values())
    return 1.0 - agreeing / len(applied)
