import time

import numpy as np
import torch
import torch.nn as nn

from framework.protocol import RolloutBatch


def compute_advantages(
    batch: RolloutBatch,
    agent,
    args,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Compute learner-side GAE advantages and returns for one rollout batch.

    This function recomputes value estimates with the learner's current critic
    rather than relying on actor-provided `batch.values`. That keeps advantage
    computation consistent with the learner parameters used for the PPO update.

    Args:
        batch: Rollout data for one learner update.
        agent: Learner agent with a critic/value head.
        args: PPO configuration with `num_steps`, `gamma`, and `gae_lambda`.
        device: Target device for intermediate tensors.

    Returns:
        A tuple of `(advantages, returns, learner_values)`.
    """
    with torch.no_grad():
        learner_values = agent.get_value(batch.obs).squeeze()
        next_value = agent.get_value(batch.next_obs).squeeze()

        advantages = torch.zeros_like(batch.rewards, device=device)
        last_gae_lam = 0.0

        for t in reversed(range(args.num_steps)):
            if t == args.num_steps - 1:
                next_non_terminal = 1.0 - batch.next_done
                next_values = next_value
            else:
                next_non_terminal = 1.0 - batch.dones[t + 1]
                next_values = learner_values[t + 1]

            delta = (
                batch.rewards[t]
                + args.gamma * next_values * next_non_terminal
                - learner_values[t]
            )
            last_gae_lam = (
                delta
                + args.gamma * args.gae_lambda * next_non_terminal * last_gae_lam
            )
            advantages[t] = last_gae_lam

        returns = advantages + learner_values

    return advantages, returns, learner_values


def compute_experience_weights(
    batch: RolloutBatch,
    strategy: str = "uniform",
    current_logprobs: torch.Tensor | None = None,
) -> torch.Tensor:
    """Compute non-negative per-sample weights for PPO policy loss scaling.

    Strategies:
        - "uniform": equal weight for every sample.
        - "latency": inverse batch age using `batch.collected_at`.
        - "is": importance-style ratio from current and stored log-probabilities.

    Returns:
        A `(batch_size,)` tensor on the same device as `batch.obs`.
    """
    batch_size = len(batch.obs)
    device = batch.obs.device

    if strategy == "uniform":
        return torch.ones(batch_size, device=device)

    if strategy == "latency":
        batch_age_s = max(time.monotonic() - batch.collected_at, 1e-6)
        return torch.full((batch_size,), 1.0 / batch_age_s, device=device)

    if strategy == "is":
        if current_logprobs is None:
            raise ValueError("strategy='is' requires current_logprobs")
        with torch.no_grad():
            ratio = (current_logprobs - batch.logprobs).exp()
        return ratio.clamp(0.0, 5.0)

    raise ValueError(f"Unknown weighting strategy: {strategy!r}")


def ppo_update(
    agent,
    optimiser,
    batch: RolloutBatch,
    advantages: torch.Tensor,
    returns: torch.Tensor,
    args,
    weights: torch.Tensor | None = None,
    learner_values: torch.Tensor | None = None,
) -> dict[str, float]:
    """Run one PPO update over a rollout batch.

    This implementation uses the actual received batch size rather than
    `args.batch_size` so it can handle partial learner flushes safely.
    """
    batch_size = len(batch.obs)
    minibatch_size = max(1, batch_size // args.num_minibatches)

    clipfracs = []

    for _ in range(args.update_epochs):
        batch_indices = np.random.permutation(batch_size)

        for start in range(0, batch_size, minibatch_size):
            mb_inds = batch_indices[start:start + minibatch_size]

            _, newlogprob, entropy, newvalue = agent.get_action_and_value(
                batch.obs[mb_inds],
                batch.actions[mb_inds],
            )
            logratio = newlogprob - batch.logprobs[mb_inds]
            ratio = logratio.exp()

            with torch.no_grad():
                old_approx_kl = (-logratio).mean()
                approx_kl = ((ratio - 1) - logratio).mean()
                clipfracs.append(
                    ((ratio - 1.0).abs() > args.clip_coef).float().mean().item()
                )

            mb_advantages = advantages[mb_inds]
            if args.norm_adv:
                mb_advantages = (
                    mb_advantages - mb_advantages.mean()
                ) / (mb_advantages.std() + 1e-8)

            pg_loss = torch.max(
                -mb_advantages * ratio,
                -mb_advantages * torch.clamp(
                    ratio,
                    1 - args.clip_coef,
                    1 + args.clip_coef,
                ),
            )

            if weights is not None:
                mb_weights = weights[mb_inds]
                pg_loss = (pg_loss * mb_weights).sum() / mb_weights.sum()
            else:
                pg_loss = pg_loss.mean()

            newvalue = newvalue.squeeze()
            old_values = learner_values if learner_values is not None else batch.values

            if args.clip_vloss:
                v_loss_unclipped = (newvalue - returns[mb_inds]) ** 2
                v_clipped = old_values[mb_inds] + torch.clamp(
                    newvalue - old_values[mb_inds],
                    -args.clip_coef,
                    args.clip_coef,
                )
                v_loss = 0.5 * torch.max(
                    v_loss_unclipped,
                    (v_clipped - returns[mb_inds]) ** 2,
                ).mean()
            else:
                v_loss = 0.5 * ((newvalue - returns[mb_inds]) ** 2).mean()

            entropy_loss = entropy.mean()
            loss = pg_loss - args.ent_coef * entropy_loss + v_loss * args.vf_coef

            optimiser.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(agent.parameters(), args.max_grad_norm)
            optimiser.step()

        if args.target_kl is not None and approx_kl > args.target_kl:
            break

    y_pred = old_values.cpu().numpy()
    y_true = returns.cpu().numpy()
    var_y = np.var(y_true)
    explained_var = np.nan if var_y == 0 else 1 - np.var(y_true - y_pred) / var_y

    return {
        "losses/value_loss": v_loss.item(),
        "losses/policy_loss": pg_loss.item(),
        "losses/entropy": entropy_loss.item(),
        "losses/old_approx_kl": old_approx_kl.item(),
        "losses/approx_kl": approx_kl.item(),
        "losses/clipfrac": np.mean(clipfracs),
        "losses/explained_variance": explained_var,
    }