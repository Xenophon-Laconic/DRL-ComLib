import numpy as np
import torch
import torch.nn as nn
from framework.protocol import RolloutBatch
import time


def compute_advantages(batch: RolloutBatch, agent, args, device) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    with torch.no_grad():
        # Re-evaluate with learner's critic — batch.values from actor are all zeros
        learner_values = agent.get_value(batch.obs).squeeze()     # (num_steps,)
        next_value     = agent.get_value(batch.next_obs).squeeze() # scalar

        advantages  = torch.zeros_like(batch.rewards).to(device)
        lastgaelam  = 0

        for t in reversed(range(args.num_steps)):
            if t == args.num_steps - 1:
                nextnonterminal = 1.0 - batch.next_done
                nextvalues      = next_value
            else:
                nextnonterminal = 1.0 - batch.dones[t + 1]
                nextvalues      = learner_values[t + 1]

            delta         = batch.rewards[t] + args.gamma * nextvalues * nextnonterminal - learner_values[t]
            advantages[t] = lastgaelam = delta + args.gamma * args.gae_lambda * nextnonterminal * lastgaelam

        returns = advantages + learner_values

    return advantages, returns, learner_values



def compute_experience_weights(
    batch,                        # RolloutBatch — has collected_at, logprobs
    strategy: str = "uniform",    # "uniform" | "latency" | "is"
    current_logprobs: torch.Tensor | None = None,  # required for "is"
) -> torch.Tensor:
    """
    Returns a (batch_size,) weight tensor on the same device as batch.obs.
    All strategies output non-negative weights; caller normalises inside ppo_update.
    """
    n = len(batch.obs)
    device = batch.obs.device

    if strategy == "uniform":
        return torch.ones(n, device=device)

    elif strategy == "latency":
        # Δt = time at learner update - time batch was collected
        # batch.collected_at is a scalar float (earliest timestamp in merged batch)
        recv_time = time.monotonic()
        delta_t = max(recv_time - batch.collected_at, 1e-6)  # guard /0
        # All steps in a merged batch share the same collected_at →
        # weight is uniform within the batch but differs across iterations
        w = torch.full((n,), 1.0 / delta_t, device=device)
        return w

    elif strategy == "is":
        # ratio π_θ / π_θ_old = exp(new_logprob - old_logprob)
        # current_logprobs: (n,) from a no_grad forward pass before the update loop
        if current_logprobs is None:
            raise ValueError("strategy='is' requires current_logprobs")
        with torch.no_grad():
            ratio = (current_logprobs - batch.logprobs).exp()
        # Clamp to avoid extreme weights destabilising training
        return ratio.clamp(0.0, 5.0)

    else:
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
):
    actual_batch_size = len(batch.obs)                              # ← Phase 4 fix
    actual_minibatch_size = max(1, actual_batch_size // args.num_minibatches)

    clipfracs = []
    for epoch in range(args.update_epochs):
        b_inds = np.random.permutation(actual_batch_size)          # ← was args.batch_size
        for start in range(0, actual_batch_size, actual_minibatch_size):
            mb_inds = b_inds[start:start + actual_minibatch_size]  # ← was args.minibatch_size

            _, newlogprob, entropy, newvalue = agent.get_action_and_value(
                batch.obs[mb_inds], batch.actions[mb_inds]
            )
            logratio = newlogprob - batch.logprobs[mb_inds]
            ratio = logratio.exp()

            with torch.no_grad():
                old_approx_kl = (-logratio).mean()
                approx_kl    = ((ratio - 1) - logratio).mean()
                clipfracs.append(((ratio - 1.0).abs() > args.clip_coef).float().mean().item())

            mb_advantages = advantages[mb_inds]
            if args.norm_adv:
                mb_advantages = (mb_advantages - mb_advantages.mean()) / (mb_advantages.std() + 1e-8)

            # Policy loss — weighted or plain
            pg_loss = torch.max(
                -mb_advantages * ratio,
                -mb_advantages * torch.clamp(ratio, 1 - args.clip_coef, 1 + args.clip_coef)
            )
            if weights is not None:
                pg_loss = (pg_loss * weights[mb_inds]).sum() / weights[mb_inds].sum()
            else:
                pg_loss = pg_loss.mean()

            # Value loss (unchanged)
            newvalue = newvalue.squeeze()
            old_values = learner_values if learner_values is not None else batch.values
            if args.clip_vloss:
                v_loss_unclipped = (newvalue - returns[mb_inds]) ** 2
                v_clipped = old_values[mb_inds] + torch.clamp(
                    newvalue - old_values[mb_inds], -args.clip_coef, args.clip_coef
                )
                v_loss = 0.5 * torch.max(v_loss_unclipped, (v_clipped - returns[mb_inds]) ** 2).mean()
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
    var_y  = np.var(y_true)
    explained_var = np.nan if var_y == 0 else 1 - np.var(y_true - y_pred) / var_y
    return {
        "losses/value_loss":         v_loss.item(),
        "losses/policy_loss":        pg_loss.item(),
        "losses/entropy":            entropy_loss.item(),
        "losses/old_approx_kl":      old_approx_kl.item(),
        "losses/approx_kl":          approx_kl.item(),
        "losses/clipfrac":           np.mean(clipfracs),
        "losses/explained_variance": explained_var,
    }