"""The action contract the safety layer decodes.

What remains is `base_ctde_env` and `wrappers`, which define `AVAction` and the bin
decoders. The topology environment they were written for is gone with the rest of the
highway-env ladder.
"""

from src.envs.base_ctde_env import AVAction

__all__ = ["AVAction"]
