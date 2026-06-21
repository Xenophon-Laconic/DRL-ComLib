import os
import random
import time

import numpy as np
import torch
import tyro

from args import Args, compute_runtime_args, make_env
from framework.comms import ActorComms
from models import Actor
from rollout import collect_rollout_from


def handle_sync_status(sync_status: str | None, actor_id: int) -> bool:
    """Handle a learner control message.

    Returns:
        True if the actor should terminate, otherwise False.
    """
    if sync_status == "shutdown":
        print(f"[Actor {actor_id}] Shutdown signal received, exiting cleanly.")
        return True

    if sync_status == "reset":
        print(f"[Actor {actor_id}] Reset applied from learner.")

    return False


def log_episode_summaries(actor_id: int, global_step: int, episode_stats: list[dict]) -> None:
    """Print rollout-local episodic summaries."""
    for stat in episode_stats:
        episodic_return = float(stat["charts/episodic_return"])
        episodic_length = int(stat["charts/episodic_length"])
        print(
            f"[Actor {actor_id}] "
            f"step={global_step} "
            f"return={episodic_return:.1f} "
            f"length={episodic_length}"
        )


if __name__ == "__main__":
    args = tyro.cli(Args)
    if not args.exp_name:
        args.exp_name = os.path.basename(__file__)[:-len(".py")]
    args = compute_runtime_args(args)

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = args.torch_deterministic

    device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")

    run_name = f"{args.env_id}__{args.exp_name}__{args.seed}__{int(time.time())}"
    env = make_env(args, run_name)()

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

    next_obs_np, _ = env.reset(seed=args.seed)
    next_obs = torch.tensor(next_obs_np, dtype=torch.float32, device=device)
    next_done = torch.tensor(0.0, device=device)
    global_step = 0
    iteration = 0

    print(f"[Actor {args.actor_id}] Ready, requesting initial weights...")
    state_dict = comms.request_initial_weights()
    actor.load_state_dict(state_dict)
    print(f"[Actor {args.actor_id}] Initial weights received, starting rollout.")

    try:
        while True:
            iteration += 1

            batch, next_obs, next_done, global_step, episode_stats = collect_rollout_from(
                env=env,
                agent=actor,
                args=args,
                device=device,
                next_obs=next_obs,
                next_done=next_done,
                global_step=global_step,
            )

            if args.simulate_outage_at > 0 and iteration == args.simulate_outage_at:
                comms.simulate_outage(args.outage_duration)

            if handle_sync_status(comms.sync_weights(actor), args.actor_id):
                break

            # Inject synthetic latency before sending the rollout, if enabled.
            if args.simulate_latency and args.latency_ms > 0.0:
                time.sleep(args.latency_ms / 1000.0)

            comms.send_batch(batch, episode_stats)

            if handle_sync_status(comms.sync_weights(actor), args.actor_id):
                break

            if episode_stats:
                log_episode_summaries(args.actor_id, global_step, episode_stats)

    finally:
        env.close()
        comms.close()