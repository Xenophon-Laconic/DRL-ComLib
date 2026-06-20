import os
import random
import time

import torch
import tyro

from args import Args, compute_runtime_args, make_env
from models import Actor
from rollout import collect_rollout_from
from framework.comms import ActorComms


if __name__ == "__main__":
    args = tyro.cli(Args)
    if not args.exp_name:
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
    obs_dim = int(np.array(env.observation_space.shape).prod())
    action_dim = env.action_space.n
    actor = Actor(obs_dim, action_dim).to(device)

    comms = ActorComms(
        push_addr=args.push_addr,
        sub_addr=args.sub_addr,
        req_addr=args.req_addr,
        actor_id=args.actor_id,
        cache_size=args.actor_cache_size,
    )

    # Initial environment state
    next_obs_np, _ = env.reset(seed=args.seed)
    next_obs = torch.tensor(next_obs_np, dtype=torch.float32).to(device)
    next_done = torch.tensor(0.0).to(device)
    global_step = 0
    iteration = 0
    should_stop = False

    print(f"[Actor {args.actor_id}] Ready, requesting initial weights...")
    state_dict = comms.request_initial_weights()
    actor.load_state_dict(state_dict)
    print(f"[Actor {args.actor_id}] Initial weights received, starting rollout.")

    try:
        while not should_stop:
            iteration += 1

            batch, next_obs, next_done, global_step, episode_stats = collect_rollout_from(
                env, actor, args, device, next_obs, next_done, global_step
            )

            if args.simulate_outage_at > 0 and iteration == args.simulate_outage_at:
                comms.simulate_outage(args.outage_duration)

            comms.send_batch(batch, episode_stats)

            sync_status = comms.sync_weights(actor)
            if sync_status == "shutdown":
                print(f"[Actor {args.actor_id}] Shutdown signal received, exiting cleanly.")
                should_stop = True

            for stat in episode_stats:
                print(
                    f"[Actor {args.actor_id}] "
                    f"step={global_step} "
                    f"return={float(stat['charts/episodic_return']):.1f}"
                )

    finally:
        env.close()
        comms.close()