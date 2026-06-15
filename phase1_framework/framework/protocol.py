from dataclasses import dataclass, field
import torch, time
from typing import Optional

@dataclass
class RolloutBatch:
    # ── Core rollout tensors ──────────────────────────────────────
    obs:        torch.Tensor   # (T, num_envs, *obs_dim)
    actions:    torch.Tensor   # (T, num_envs, *act_dim)
    logprobs:  torch.Tensor   # (T, num_envs)
    rewards:    torch.Tensor   # (T, num_envs)
    dones:      torch.Tensor   # (T, num_envs)
    values:     torch.Tensor   # (T, num_envs)

    # ── Bootstrap tensors (end-of-rollout state) ──────────────────
    next_obs:   torch.Tensor   # (num_envs, *obs_dim) — for value bootstrap
    next_done:  torch.Tensor   # (num_envs,)          — bootstrap masking

    # advantages and returns: computed actor-side for reference/debugging.
    # The learner recomputes these fresh via compute_advantages() in training.py.
    advantages: Optional[torch.Tensor] = None
    returns:    Optional[torch.Tensor] = None



    # ── Phase 3 comms metadata ────────────────────────────────────
    actor_id:     int   = 0
    learner_step: int   = 0
    collected_at: float = field(default_factory=time.monotonic)