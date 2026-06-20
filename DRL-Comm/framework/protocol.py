from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Optional

import torch


@dataclass
class RolloutBatch:
    """Trajectory data and metadata for one learner update.

    This represents a fixed-length rollout collected by a single actor.
    The first dimension is time (T = args.num_steps); there is no explicit
    vectorized-env dimension in the current setup.
    """

    # Core rollout tensors, shape: (T, *obs_shape) or (T,)
    obs: torch.Tensor        # observations at each step
    actions: torch.Tensor    # actions taken at each step
    logprobs: torch.Tensor   # log π(a_t | s_t) from the actor policy
    rewards: torch.Tensor    # scalar rewards r_t
    dones: torch.Tensor      # done flags for each step (0.0 or 1.0)
    values: torch.Tensor     # value estimates from actor, may be zeros on learner

    # Bootstrap state at the end of the rollout
    next_obs: torch.Tensor   # observation after the final step
    next_done: torch.Tensor  # done flag for the bootstrap state (0.0 or 1.0)

    # Optional actor-side advantages/returns (learner recomputes its own)
    advantages: Optional[torch.Tensor] = None
    returns: Optional[torch.Tensor] = None

    # Comms / scheduling metadata
    actor_id: int = 0               # which actor produced this batch
    learner_step: int = 0           # learner-side logical step when enqueued
    collected_at: float = field(    # monotonic timestamp when batch was created
        default_factory=time.monotonic
    )