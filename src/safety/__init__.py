"""The bound on a recommended speed: two rules, and the controller that
turns a bounded speed into an acceleration.

`deployment/jetson/policy/safety_gate.py` vendors this module onto the
device, so this is the reference copy rather than a second implementation.
"""

from src.safety.constraints import SafetyConstraints
from src.safety.safety_layer import (
    SafetyContext,
    SafetyDecision,
    apply_safety_layer,
    physical_control_command,
    safety_penalty_terms,
)

__all__ = [
    "SafetyConstraints",
    "SafetyContext",
    "SafetyDecision",
    "apply_safety_layer",
    "physical_control_command",
    "safety_penalty_terms",
]
