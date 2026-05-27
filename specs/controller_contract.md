# Controller Contract

This file defines the controller interface shared by baselines and learned policies.

## Purpose

Controllers should be interchangeable across topology sweeps, demand sweeps, and RL evaluation without changing caller code.

## Public Interface

Every controller should support:

```python
reset(env_metadata=None, seed=None) -> None
act(local_obs, global_state=None) -> dict[str, dict]
```

Optional metadata should be exposed through:

- `name`
- `metadata`

## `reset(...)`

- called at episode start
- may receive topology or experiment metadata
- should clear any controller state

## `act(local_obs, global_state=None)`

- `local_obs` is required and keyed by AV identifier
- `global_state` is optional so the same interface works for decentralized baselines and CTDE controllers
- return value is the canonical AV action mapping

## Metadata

Metadata should stay lightweight and include:

- `name`
- `family`
- `version`
- `requires_global_state`

## Repo Ownership

The executable base contract should live in:

- `src/controllers/base.py`

Baseline and RL implementations should later live under:

- `src/baselines/`
- `src/rl/`

