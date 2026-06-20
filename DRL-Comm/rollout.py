import torch

from framework.protocol import RolloutBatch


def collect_rollout_from(
    env,
    agent,
    args,
    device: torch.device,
    next_obs: torch.Tensor,
    next_done: torch.Tensor,
    global_step: int = 0,
) -> tuple[RolloutBatch, torch.Tensor, torch.Tensor, int, list[dict[str, float]]]:
    """Collect one fixed-length rollout from the current actor policy.

    The rollout starts from the provided `(next_obs, next_done)` carry-over state
    and advances the environment for `args.num_steps` transitions.

    Returns:
        A tuple of:
        - rollout batch containing trajectory tensors,
        - next observation carry-over for the next rollout,
        - next done flag carry-over for the next rollout,
        - updated global environment step count,
        - episodic summaries for any episodes that ended during this rollout.
    """
    obs = torch.zeros((args.num_steps,) + env.observation_space.shape, device=device)
    actions = torch.zeros(args.num_steps, dtype=torch.long, device=device)
    logprobs = torch.zeros(args.num_steps, device=device)
    rewards = torch.zeros(args.num_steps, device=device)
    dones = torch.zeros(args.num_steps, device=device)
    values = torch.zeros(args.num_steps, device=device)

    episode_stats: list[dict[str, float]] = []

    for step in range(args.num_steps):
        global_step += 1
        obs[step] = next_obs
        dones[step] = next_done

        with torch.no_grad():
            action, logprob, _, value = agent.get_action_and_value(next_obs)
            if value is not None:
                values[step] = value.squeeze()

        actions[step] = action
        logprobs[step] = logprob

        next_obs_np, reward, termination, truncation, info = env.step(action.item())
        next_obs = torch.tensor(next_obs_np, dtype=torch.float32, device=device)
        next_done = torch.tensor(float(termination or truncation), device=device)
        rewards[step] = torch.tensor(reward, dtype=torch.float32, device=device)

        if "episode" in info:
            episode_stats.append(
                {
                    "charts/episodic_return": info["episode"]["r"],
                    "charts/episodic_length": info["episode"]["l"],
                }
            )

        if termination or truncation:
            next_obs_np, _ = env.reset()
            next_obs = torch.tensor(next_obs_np, dtype=torch.float32, device=device)

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

    return batch, next_obs, next_done, global_step, episode_stats