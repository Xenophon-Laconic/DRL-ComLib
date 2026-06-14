import os
import random
import time

import torch
import torch.optim as optim
import zmq
import tyro

from args import Args, compute_runtime_args, make_env
from models import Agent
from rollout import RolloutBatch
from training import compute_advantages, ppo_update
from comms import (
    deserialise_batch,
    serialise_state_dict,
    make_learner_sockets,
)
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

    # ZMQ
    context    = zmq.Context()
    pull, pub, rep = make_learner_sockets(context)
    print("Learner ready, waiting for actor handshake...")

    rep.recv()  # actor sends b"ready"
    rep.send(serialise_state_dict(agent.actor.state_dict()))
    print("Initial weights sent to actor.")

    global_step = 0
    start_time  = time.time()

    for iteration in range(1, args.num_iterations + 1):

        # Learning rate annealing
        if args.anneal_lr:
            frac = 1.0 - (iteration - 1.0) / args.num_iterations
            optimizer.param_groups[0]["lr"] = frac * args.learning_rate

        # Wait for one batch from any actor
        data  = pull.recv()
        batch, episode_stats = deserialise_batch(data)
        batch = RolloutBatch(**{k: v.to(device) for k, v in batch.__dict__.items()
                                if isinstance(v, torch.Tensor)},
                             **{k: v for k, v in batch.__dict__.items()
                                if not isinstance(v, torch.Tensor)})

        global_step += args.num_steps

        # PPO update
        advantages, returns = compute_advantages(batch, agent, args, device)
        metrics = ppo_update(agent, optimizer, batch, advantages, returns, args)

        # Broadcast updated actor weights
        pub.send(serialise_state_dict(agent.actor.state_dict()))

        # Logging
        sps = int(global_step / (time.time() - start_time))
        print(f"[Learner] iteration={iteration} global_step={global_step} SPS={sps}")

        log_training_metrics(writer, metrics, global_step)
        log_infra_metrics(writer, optimizer, global_step, sps)
        log_episode_stats(writer, episode_stats, global_step)

    writer.close()
    context.destroy()