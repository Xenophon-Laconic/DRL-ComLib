import os
import random
import time

import torch
import torch.optim as optim
import tyro

from args import Args, compute_runtime_args, make_env
from models import Agent
from training import compute_advantages, ppo_update
from framework.comms import LearnerComms
from logging_utils import (
    make_writer,
    log_hyperparameters,
    log_episode_stats,
    log_training_metrics,
    log_infra_metrics,
    log_batch_meta,
)


if __name__ == "__main__":
    args = tyro.cli(Args)
    if not args.exp_name:
        args.exp_name = os.path.basename(__file__)[: -len(".py")]
    args = compute_runtime_args(args)

    run_name = f"{args.env_id}__{args.exp_name}__{args.seed}__{int(time.time())}"

    # Seeding
    random.seed(args.seed)
    import numpy as np
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = args.torch_deterministic

    device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")

    # Learner needs a dummy env only to instantiate Agent dimensions
    env = make_env(args.env_id, False, run_name)()
    agent     = Agent(env).to(device)
    optimizer = optim.Adam(agent.parameters(), lr=args.learning_rate, eps=1e-5)
    env.close()

    writer = make_writer(run_name)
    log_hyperparameters(writer, args)

    # comms
    comms = LearnerComms(
        pull_addr=args.pull_addr,
        pub_addr=args.pub_addr,
        rep_addr=args.rep_addr,
        device=device,
        buffer_size=args.learner_buffer_size,
        max_batches_per_actor=args.max_batches_per_actor,
        num_actors=args.num_actors,
        staleness_threshold=args.staleness_threshold,
    )

    print(f"[Learner] τ = {args.staleness_threshold}")
    print("Learner ready, waiting for actor handshake...")
    comms.serve_initial_weights(agent.actor.state_dict())
    print("Initial weights sent to actor.")

    global_step = 0
    start_time  = time.time()

    for iteration in range(1, args.num_iterations + 1):

        # Learning rate annealing
        if args.anneal_lr:
            frac = 1.0 - (iteration - 1.0) / args.num_iterations
            optimizer.param_groups[0]["lr"] = frac * args.learning_rate

        # Wait for one batch from any actor
        batch, episode_stats = comms.recv_batch(writer=writer, global_step=global_step)

        global_step += args.num_steps * args.learner_buffer_size
        log_batch_meta(writer, batch, global_step)

        # PPO update
        advantages, returns, learner_values = compute_advantages(batch, agent, args, device)
        metrics = ppo_update(agent, optimizer, batch, advantages, returns, args,
                            learner_values=learner_values)

        # Broadcast updated actor weights
        comms.broadcast_weights(agent, step=iteration)

        # Logging
        sps = int(global_step / (time.time() - start_time))
        print(f"[Learner] iteration={iteration} global_step={global_step} SPS={sps}")

        log_training_metrics(writer, metrics, global_step)
        log_infra_metrics(writer, optimizer, global_step, sps)
        log_episode_stats(writer, episode_stats, global_step)

    writer.close()
    comms.close()