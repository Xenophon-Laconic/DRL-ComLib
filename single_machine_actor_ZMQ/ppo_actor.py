import os
import random
import time

import torch
import zmq
import tyro

from args import Args, compute_runtime_args, make_env
from models import Actor
from rollout import collect_rollout_from
from comms import (
    serialise_batch,
    deserialise_state_dict,
    make_actor_sockets,
)


if __name__ == "__main__":
    args = tyro.cli(Args)
    args.exp_name = os.path.basename(__file__)[: -len(".py")]
    args = compute_runtime_args(args)

    # Seeding
    random.seed(args.seed)
    import numpy as np
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = args.torch_deterministic

    device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")

    # Environment
    run_name = f"{args.env_id}__{args.exp_name}__{args.seed}__{int(time.time())}"
    env = make_env(args.env_id, args.capture_video, run_name)()

    # Actor network only — no critic needed on actor side
    obs_dim    = int(np.array(env.observation_space.shape).prod())
    action_dim = env.action_space.n
    actor      = Actor(obs_dim, action_dim).to(device)

    # ZMQ
    context          = zmq.Context()
    push, sub, req   = make_actor_sockets(context)

    # Initial environment state
    next_obs_np, _ = env.reset(seed=args.seed)
    next_obs  = torch.tensor(next_obs_np, dtype=torch.float32).to(device)
    next_done = torch.tensor(0.0).to(device)
    global_step = 0

    print("Actor ready, requesting initial weights...")
    req.send(b"ready")
    weights_data = req.recv()
    actor.load_state_dict(deserialise_state_dict(weights_data))
    print("Initial weights received, starting rollout.")

    for iteration in range(1, args.num_iterations + 1):

        # Collect rollout using actor as a wrapper around get_action_and_value
        batch, next_obs, next_done, global_step, episode_stats = collect_rollout_from(
            env, actor, args, device, next_obs, next_done, global_step
        )

        # Send batch to learner
        push.send(serialise_batch(batch, episode_stats))

        # Check for updated weights (non-blocking)
        try:
            weights_data = sub.recv(flags=zmq.NOBLOCK)
            actor.load_state_dict(deserialise_state_dict(weights_data))
        except zmq.Again:
            pass  # no update available yet, continue with current weights

        # Print episodic returns locally
        for stat in episode_stats:
            print(f"[Actor] step={global_step} return={float(stat['charts/episodic_return']):.1f}")

    env.close()
    context.destroy()