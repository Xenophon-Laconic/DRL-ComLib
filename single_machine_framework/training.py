import numpy as np
import torch
import torch.nn as nn
from framework.protocol import RolloutBatch


def compute_advantages(batch: RolloutBatch, agent, args, device) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Compute GAE advantages and returns from a completed rollout batch.
    Returns (advantages, returns), both shape (num_steps,).
    """
    with torch.no_grad():
        next_value = agent.get_value(batch.next_obs).squeeze()
        advantages = torch.zeros_like(batch.rewards).to(device)
        lastgaelam = 0

        for t in reversed(range(args.num_steps)):
            if t == args.num_steps - 1:
                nextnonterminal = 1.0 - batch.next_done
                nextvalues = next_value
            else:
                nextnonterminal = 1.0 - batch.dones[t + 1]
                nextvalues = batch.values[t + 1]

            delta = batch.rewards[t] + args.gamma * nextvalues * nextnonterminal - batch.values[t]
            advantages[t] = lastgaelam = delta + args.gamma * args.gae_lambda * nextnonterminal * lastgaelam

        returns = advantages + batch.values

    return advantages, returns


def ppo_update(
    agent,
    optimiser,
    batch: RolloutBatch,
    advantages: torch.Tensor,
    returns: torch.Tensor,
    args,
    weights: torch.Tensor | None = None,
):
    """
    Run update_epochs passes of minibatch PPO updates over the batch.
    Returns a dict of training metrics for logging.
    """
    clipfracs = []

    for epoch in range(args.update_epochs):
        b_inds = np.random.permutation(args.batch_size)

        for start in range(0, args.batch_size, args.minibatch_size):
            mb_inds = b_inds[start:start + args.minibatch_size]

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

            # Policy loss
            pg_loss = torch.max(
                    -mb_advantages * ratio,
                    -mb_advantages * torch.clamp(ratio, 1 - args.clip_coef, 1 + args.clip_coef)
                )
            if weights is not None:
                pg_loss = (pg_loss * weights[mb_inds]).sum() / weights[mb_inds].sum()
            else:
                pg_loss = pg_loss.mean()

            # Value loss
            newvalue = newvalue.squeeze()
            if args.clip_vloss:
                v_loss_unclipped = (newvalue - returns[mb_inds]) ** 2
                v_clipped = batch.values[mb_inds] + torch.clamp(
                    newvalue - batch.values[mb_inds], -args.clip_coef, args.clip_coef
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

    y_pred = batch.values.cpu().numpy()
    y_true = returns.cpu().numpy()
    var_y  = np.var(y_true)
    explained_var = np.nan if var_y == 0 else 1 - np.var(y_true - y_pred) / var_y

    return {
        "losses/value_loss":        v_loss.item(),
        "losses/policy_loss":       pg_loss.item(),
        "losses/entropy":           entropy_loss.item(),
        "losses/old_approx_kl":     old_approx_kl.item(),
        "losses/approx_kl":         approx_kl.item(),
        "losses/clipfrac":          np.mean(clipfracs),
        "losses/explained_variance": explained_var,
    }