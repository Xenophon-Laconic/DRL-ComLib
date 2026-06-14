import torch
import numpy as np
from dataclasses import dataclass
from typing import Optional


@dataclass
class RolloutBatch:
    obs:       torch.Tensor   # (num_steps, obs_dim)
    actions:   torch.Tensor   # (num_steps,)
    logprobs:  torch.Tensor   # (num_steps,)
    rewards:   torch.Tensor   # (num_steps,)
    dones:     torch.Tensor   # (num_steps,)
    values:    torch.Tensor   # (num_steps,)
    next_obs:  torch.Tensor   # (obs_dim,)  — for bootstrap
    next_done: torch.Tensor   # scalar      — for bootstrap masking


def collect_rollout(env, agent, args, device, writer=None, global_step=0):
    """
    Run the agent in the environment for args.num_steps steps.
    Returns a RolloutBatch and the updated global_step.
    """
    obs      = torch.zeros((args.num_steps,) + env.observation_space.shape).to(device)
    actions  = torch.zeros(args.num_steps, dtype=torch.long).to(device)
    logprobs = torch.zeros(args.num_steps).to(device)
    rewards  = torch.zeros(args.num_steps).to(device)
    dones    = torch.zeros(args.num_steps).to(device)
    values   = torch.zeros(args.num_steps).to(device)

    next_obs, _ = env.reset() if global_step == 0 else (None, None)

    # Carry next_obs/next_done across calls via the returned batch
    # Caller is responsible for passing these in after the first rollout
    raise NotImplementedError(
        "Use collect_rollout_from(env, agent, args, device, next_obs, next_done)"
    )


def collect_rollout_from(env, agent, args, device, next_obs, next_done, writer=None, global_step=0):
    """
    Run the agent for args.num_steps steps starting from (next_obs, next_done).
    Returns (RolloutBatch, next_obs, next_done, global_step).
    next_obs and next_done are the state after the final step, ready for the
    next rollout or bootstrap value computation.
    """
    obs      = torch.zeros((args.num_steps,) + env.observation_space.shape).to(device)
    actions  = torch.zeros(args.num_steps, dtype=torch.long).to(device)
    logprobs = torch.zeros(args.num_steps).to(device)
    rewards  = torch.zeros(args.num_steps).to(device)
    dones    = torch.zeros(args.num_steps).to(device)
    values   = torch.zeros(args.num_steps).to(device)

    for step in range(args.num_steps):
        global_step += 1
        obs[step]   = next_obs
        dones[step] = next_done

        with torch.no_grad():
            action, logprob, _, value = agent.get_action_and_value(next_obs)
            values[step] = value.squeeze()

        actions[step]  = action
        logprobs[step] = logprob

        next_obs_np, reward, termination, truncation, info = env.step(action.item())
        next_obs  = torch.tensor(next_obs_np, dtype=torch.float32).to(device)
        next_done = torch.tensor(float(termination or truncation)).to(device)
        rewards[step] = torch.tensor(reward, dtype=torch.float32).to(device)

        if "episode" in info and writer is not None:
            writer.add_scalar("charts/episodic_return", info["episode"]["r"], global_step)
            writer.add_scalar("charts/episodic_length", info["episode"]["l"], global_step)

        if termination or truncation:
            next_obs_np, _ = env.reset()
            next_obs = torch.tensor(next_obs_np, dtype=torch.float32).to(device)

    batch = RolloutBatch(
        obs=obs,
        actions=actions,
        logprobs=logprobs,
        rewards=rewards,
        dones=dones,
        values=values,
        next_obs=next_obs,
        next_done=next_done,
    )

    return batch, next_obs, next_done, global_step