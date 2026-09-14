"""SRC's Q-learning controller, ported.

The network, the update and the action set are SRC's `train.py`: one MLP over the
whole network state emitting a Q-value per super-segment per action, a per-segment TD
target against that segment's own reward, argmax execution, discount 0.9.

What changes is the input. SRC's six features include the mean following gap and the
counts of vehicles entering and leaving a super-segment, which are computed in Vissim
from vehicle-level ground truth and which no traffic API returns. A vehicle cannot
evaluate a policy that reads them. The state here is what HERE returns, so the trained
policy is one a vehicle can actually run.

Nothing about the decentralization needs a separate mechanism: the input is the whole
network's state, the policy is a deterministic argmax, so every vehicle holding the
same weights and reading the same state computes the same action. Distributed
execution is identical to centralized by construction rather than by approximation.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

#: SRC's discount, per 60 s decision.
DISCOUNT: float = 0.9


class SrcQNetwork(nn.Module):
    """SRC's network: flatten every segment's features, emit every segment's Q-values.

    The hidden width is twice the input as in SRC, and the two commented-out deeper
    variants in its train.py are left out rather than guessed at.
    """

    def __init__(self, num_segments: int, num_features: int, num_actions: int) -> None:
        super().__init__()
        self.num_segments = num_segments
        self.num_features = num_features
        self.num_actions = num_actions
        self.input_length = num_segments * num_features
        self.output_length = num_segments * num_actions
        self.stack = nn.Sequential(
            nn.Linear(self.input_length, 2 * self.input_length),
            nn.ReLU(),
            nn.Linear(2 * self.input_length, self.output_length),
        )

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """State `(segments, features)` to Q-values `(segments, actions)`."""
        logits = self.stack(torch.flatten(state))
        return logits.view((self.num_segments, self.num_actions))


def greedy_actions(model: SrcQNetwork, state: torch.Tensor) -> torch.Tensor:
    """One action per super-segment.

    SRC applies a softmax before the argmax, which cannot change which index is
    largest; it is kept so the port is the same function, not merely the same choice.
    """
    with torch.no_grad():
        return torch.argmax(F.softmax(model(state), dim=1), dim=1).detach()


def epsilon_actions(model: SrcQNetwork, state: torch.Tensor, epsilon: float,
                    generator: torch.Generator) -> torch.Tensor:
    """Greedy actions with each segment independently randomised with probability eps."""
    actions = greedy_actions(model, state)
    if epsilon <= 0.0:
        return actions
    draw = torch.rand(actions.shape, generator=generator)
    random = torch.randint(0, model.num_actions, actions.shape, generator=generator)
    return torch.where(draw < epsilon, random, actions)


def td_loss(model: SrcQNetwork, states: torch.Tensor, actions: torch.Tensor,
            rewards: torch.Tensor, discount: float = DISCOUNT) -> torch.Tensor:
    """SRC's update, vectorised.

    Its train.py builds `q_sa` and `q_sa_max` with a Python double loop and then
    computes `0.5 * (q_sa[:-1] - (reward[1:] + discount * q_sa_max[1:])) ** 2`. This is
    that expression with the loop replaced by a gather, which is the same arithmetic.

    The target is bootstrapped from the NEXT state's own maximum, per segment, so each
    super-segment is trained against its own reward. That is SRC's credit assignment
    and it is why the shared reward problem of a team-reward formulation does not
    arise here.
    """
    if states.shape[0] < 2:
        # One decision gives an empty TD target, whose mean is nan, and a backward
        # pass over nan updates nothing while the run still looks like it trained.
        raise ValueError(
            f"a TD update needs at least two decisions, got {states.shape[0]}; "
            "the episode is shorter than one decision interval past warm-up")
    values = torch.stack([model(state) for state in states])
    taken = values.gather(2, actions.unsqueeze(-1)).squeeze(-1)
    best = values.max(dim=2).values
    target = rewards[1:] + discount * best[1:]
    return 0.5 * (taken[:-1] - target.detach()) ** 2
