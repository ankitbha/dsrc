# Action Schema

This file defines the canonical public AV action format.

## Canonical Action Object

Every AV action should use the same dict structure:

```python
{
    "desired_speed": float,
    "desired_lane": str,
}
```

## Field Definitions

`desired_speed`

- units: meters per second
- meaning: target cruising speed requested by the controller
- wrapper responsibility: clamp to environment limits and hand off to the safety layer

`desired_lane`

- uses normalized lane preference labels, not raw simulator lane IDs
- allowed values for v1:
  - `keep`
  - `left`
  - `right`
  - `leftmost`
  - `rightmost`

The public interface should not expose topology-specific lane indices because those change across ring, straight, merge, and tree layouts.

## Action Mapping

Controllers should return one mapping keyed by AV identifier:

```python
{
    "av_0": {"desired_speed": 22.0, "desired_lane": "keep"},
    "av_1": {"desired_speed": 18.5, "desired_lane": "left"},
}
```

## Repo Ownership

The executable contract should live in:

- `src/envs/base_ctde_env.py`
- `src/envs/wrappers.py`
- `src/controllers/base.py`

