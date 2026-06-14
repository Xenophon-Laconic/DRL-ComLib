import os
import random
import time

import torch
import torch.optim as optim
import tyro

from args import Args, compute_runtime_args, make_env
from models import Agent
from rollout import collect_rollout_from
from training import compute_advantages, ppo_update
from logging_utils import (
    make_writer,
    log_hyperparameters,
    log_episode_stats,
    log_training_metrics,
    log_infra_metrics,
)


if __name__ == "__main__":
    args = tyro.cli(Args)
    args.exp_name = os.path.basename(__file__)[: -len(".py")]
    args = compute_runtime_args(args)

    run_name = f"{args.env_id}__{args.exp_name}__{args.seed}__{int(time.time())}"

    if args.track:
        import wandb
        wandb.init(
            project=args.wandb_project_name,
            entity=args.wandb_entity,
            sync_tensorboard=True,
            config=vars(args),
            name=run_name,
            monitor_gym=True,
            save_code=True,
        )

    writer = make_writer(run_name)
    log_hyperparameters(writer, args)

    # Seeding
    random.seed(args.seed)
    import numpy as np
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = args.torch_deterministic

    device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")

    # Environment and agent
    env = make_env(args.env_id, args.capture_video, run_name)()
    assert isinstance(env.action_space, __import__("gymnasium").spaces.Discrete), \
        "only discrete action spaces are supported"

    agent = Agent(env).to(device)
    optimiser = optim.Adam(agent.parameters(), lr=args.learning_rate, eps=1e-5)

    # Initial environment state
    next_obs_np, _ = env.reset(seed=args.seed)
    next_obs  = torch.tensor(next_obs_np, dtype=torch.float32).to(device)
    next_done = torch.tensor(0.0).to(device)

    global_step = 0
    start_time  = time.time()

    for iteration in range(1, args.num_iterations + 1):

        # Learning rate annealing
        if args.anneal_lr:
            frac = 1.0 - (iteration - 1.0) / args.num_iterations
            optimiser.param_groups[0]["lr"] = frac * args.learning_rate

        # Collect rollout
        batch, next_obs, next_done, global_step, episode_stats = collect_rollout_from(
            env, agent, args, device, next_obs, next_done, global_step
        )

        # Compute advantages and returns
        advantages, returns = compute_advantages(batch, agent, args, device)

        # PPO update
        metrics = ppo_update(agent, optimiser, batch, advantages, returns, args)

        # Logging
        sps = int(global_step / (time.time() - start_time))
        print(f"iteration={iteration}, global_step={global_step}, SPS={sps}")

        log_episode_stats(writer, episode_stats, global_step)
        log_training_metrics(writer, metrics, global_step)
        log_infra_metrics(writer, optimiser, global_step, sps)

    env.close()
    writer.close()