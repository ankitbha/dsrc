"""Frozen records passed between the modules of the HERE travel-time error analysis.

Times are seconds since the Unix epoch on the GPS UTC axis unless a name says otherwise.
"""
from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class GpsFix:
    utc_s: float
    wall_s: float
    lat: float
    lon: float
    speed_mps: float | None
    heading_deg: float | None


@dataclass(frozen=True)
class HereSubSegment:
    length_m: float
    speed_mps: float | None
    free_flow_mps: float | None
    jam_factor: float | None


@dataclass(frozen=True)
class HereSegment:
    """One directed road segment of a HERE Traffic Flow response.

    `key` identifies it across responses: HERE gives no stable id in `shape` mode, and the
    direction is carried by the order of the shape points.
    """

    key: str
    description: str
    stated_length_m: float
    lat: tuple[float, ...]
    lon: tuple[float, ...]
    speed_mps: float | None
    free_flow_mps: float | None
    jam_factor: float | None
    confidence: float | None
    functional_class: int | None
    subsegments: tuple[HereSubSegment, ...]


@dataclass(frozen=True)
class HereBody:
    run: str
    seq: int
    source_file: str
    data_time_s: float
    response_utc_s: float
    query_lat: float
    query_lon: float
    segments: tuple[HereSegment, ...]


@dataclass(frozen=True)
class Frame:
    run: str
    pos: int
    frame_id: int
    capture_utc_s: float

    @property
    def key(self) -> str:
        return f"{self.run}:{self.frame_id}"


@dataclass(frozen=True)
class Pass:
    """A run of consecutive one-second fixes lying on one HERE segment."""

    pass_id: str
    run: str
    segment_key: str
    description: str
    functional_class: int | None
    fixes: tuple[GpsFix, ...]
    chain_start_m: float
    chain_end_m: float
    segment_length_flag: bool
    path_positions_m: float
    path_speed_m: float | None
    exclusion: str | None

    @property
    def first_utc_s(self) -> float:
        return self.fixes[0].utc_s

    @property
    def last_utc_s(self) -> float:
        return self.fixes[-1].utc_s

    @property
    def observed_s(self) -> float:
        return self.last_utc_s - self.first_utc_s

    @property
    def matched_m(self) -> float:
        return self.chain_end_m - self.chain_start_m

    @property
    def v3_rel_diff(self) -> float | None:
        if self.path_speed_m is None or self.path_positions_m <= 0:
            return None
        return abs(self.path_positions_m - self.path_speed_m) / self.path_positions_m


@dataclass(frozen=True)
class PassTimes:
    pass_id: str
    reading_seq: int | None
    reading_data_time_s: float | None
    reading_age_s: float | None
    reading_arrived_before_start: bool | None
    observed_s: float
    here_s: float | None
    free_flow_s: float | None
    signed_error: float | None
    free_flow_error: float | None
    jam_factor: float | None
    confidence: float | None
    stopped_s: float
    moving_s: float
    sensitivity_seq: int | None
    sensitivity_signed_error: float | None
    shuffled_seq: int | None
    shuffled_signed_error: float | None
    exclusion: str | None


@dataclass(frozen=True)
class Stretch:
    stretch_id: str
    run: str
    pass_ids: tuple[str, ...]
    start_utc_s: float
    end_utc_s: float
    matched_m: float
    observed_s: float
    here_s: float
    free_flow_s: float
    stopped_s: float

    @property
    def signed_error(self) -> float:
        return (self.here_s - self.observed_s) / self.observed_s

    @property
    def free_flow_error(self) -> float:
        return (self.free_flow_s - self.observed_s) / self.observed_s


@dataclass(frozen=True)
class FrameDetections:
    run: str
    pos: int
    frame_id: int
    utc_s: float
    n_vehicles: int
    leader: bool
    brightness: float

    @property
    def key(self) -> str:
        return f"{self.run}:{self.frame_id}"


@dataclass(frozen=True)
class RouteRequest:
    request_id: str
    kind: str
    run: str
    origin_lat: float
    origin_lon: float
    dest_lat: float
    dest_lon: float
    departure_utc_s: float
    observed_s: float
    matched_m: float
    fix_lat: tuple[float, ...] = field(repr=False)
    fix_lon: tuple[float, ...] = field(repr=False)


@dataclass(frozen=True)
class RouteResult:
    request_id: str
    kind: str
    duration_s: float
    no_traffic_duration_s: float | None
    length_m: float
    observed_s: float
    signed_error: float
    follows_path: bool
    share_within: float
    length_rel_diff: float
    exclusion: str | None


@dataclass(frozen=True)
class InputFile:
    path: str
    sha256: str


@dataclass(frozen=True)
class Provenance:
    git_commit: str
    dirty: bool | None
    inputs: tuple[InputFile, ...]
    params: dict

    def __post_init__(self) -> None:
        if not self.git_commit:
            raise ValueError("provenance needs a git commit; refusing to record None")
