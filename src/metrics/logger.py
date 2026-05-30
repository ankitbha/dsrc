from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any, Mapping, Sequence


class MetricsLogger:
    """Accumulate episode metrics and write JSON/CSV artifacts."""

    def __init__(
        self,
        *,
        experiment_id: str,
        output_root: str | Path = "outputs/metrics",
        output_paths: Mapping[str, str] | None = None,
    ) -> None:
        self.experiment_id = experiment_id
        self.output_root = Path(output_root)
        self.output_paths = dict(output_paths or {})
        self.step_rows: list[dict[str, Any]] = []
        self.segment_rows: list[dict[str, Any]] = []

    def record_step(self, step_metrics: Mapping[str, Any]) -> None:
        self.step_rows.append(_flatten_row(step_metrics))

    def record_segments(
        self,
        *,
        time_s: float,
        segment_metrics: Mapping[str, Mapping[str, Any]],
    ) -> None:
        for segment_id, metrics in segment_metrics.items():
            self.segment_rows.append(_flatten_row({"time": time_s, "segment_id": segment_id, **metrics}))

    def write_episode(
        self,
        episode_summary: Mapping[str, Any],
    ) -> dict[str, str]:
        episode_path = self._path("episode_summary", "episode_summary.json")
        step_path = self._path("step_metrics", "step_metrics.csv")
        segment_path = self._path("segment_metrics", "segment_metrics.csv")
        episode_path.parent.mkdir(parents=True, exist_ok=True)
        step_path.parent.mkdir(parents=True, exist_ok=True)
        segment_path.parent.mkdir(parents=True, exist_ok=True)
        episode_path.write_text(json.dumps(_json_ready(episode_summary), indent=2, sort_keys=True) + "\n")
        _write_csv(step_path, self.step_rows)
        _write_csv(segment_path, self.segment_rows)
        return {
            "episode_summary": str(episode_path),
            "step_metrics": str(step_path),
            "segment_metrics": str(segment_path),
        }

    def _path(self, key: str, filename: str) -> Path:
        configured = self.output_paths.get(key)
        if configured:
            return Path(configured)
        return self.output_root / self.experiment_id / filename


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    fieldnames = sorted({field for row in rows for field in row})
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fieldnames})


def _flatten_row(row: Mapping[str, Any]) -> dict[str, Any]:
    flattened: dict[str, Any] = {}
    for key, value in row.items():
        if isinstance(value, Mapping):
            flattened[key] = json.dumps(_json_ready(value), sort_keys=True)
        elif isinstance(value, (list, tuple)):
            flattened[key] = json.dumps(_json_ready(value))
        else:
            flattened[key] = _json_ready(value)
    return flattened


def _json_ready(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _json_ready(inner) for key, inner in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_ready(inner) for inner in value]
    if value == float("inf"):
        return "inf"
    if value == float("-inf"):
        return "-inf"
    return value
