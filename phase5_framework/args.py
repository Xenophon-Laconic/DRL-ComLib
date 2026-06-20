import os
from dataclasses import dataclass
from typing import Optional

import gymnasium as gym


@dataclass
class Args:
    exp_name: str = os.path.basename(__file__)[: -len(".py")]
    """the name of this experiment"""
    seed: int = 1
    """seed of the experiment"""
    torch_deterministic: bool = True
    """if toggled, `torch.backends.cudnn.deterministic=False`"""
    cuda: bool = True
    """if toggled, cuda will be enabled by default"""
    track: bool = False
    """if toggled, this experiment will be tracked with Weights and Biases"""
    wandb_project_name: str = "cleanRL"
    """the wandb's project name"""
    wandb_entity: Optional[str] = None
    """the entity (team) of wandb's project"""
    capture_video: bool = False
    """whether to capture videos of the agent performances (check out `videos` folder)"""

    # Algorithm specific arguments
    env_id: str = "CartPole-v1"
    """the id of the environment"""
    total_timesteps: int = 500000
    """total timesteps of the experiments"""
    learning_rate: float = 2.5e-4
    """the learning rate of the optimiser"""
    num_steps: int = 128
    """the number of steps to run per policy rollout"""
    anneal_lr: bool = True
    """Toggle learning rate annealing for policy and value networks"""
    gamma: float = 0.99
    """the discount factor gamma"""
    gae_lambda: float = 0.95
    """the lambda for the general advantage estimation"""
    num_minibatches: int = 4
    """the number of mini-batches"""
    update_epochs: int = 4
    """the K epochs to update the policy"""
    norm_adv: bool = True
    """Toggles advantages normalization"""
    clip_coef: float = 0.2
    """the surrogate clipping coefficient"""
    clip_vloss: bool = True
    """Toggles whether or not to use a clipped loss for the value function, as per the paper."""
    ent_coef: float = 0.01
    """coefficient of the entropy"""
    vf_coef: float = 0.5
    """coefficient of the value function"""
    max_grad_norm: float = 0.5
    """the maximum norm for the gradient clipping"""
    target_kl: Optional[float] = None
    """the target KL divergence threshold"""

    # ── Comms arguments ───────────────────────────────────────────
    push_addr: str = "tcp://localhost:5555"
    """actor pushes batches to this address"""
    pull_addr: str = "tcp://localhost:5555"
    """learner pulls batches from this address"""
    pub_addr: str = "tcp://localhost:5556"
    """learner publishes weights to this address"""
    sub_addr: str = "tcp://localhost:5556"
    """actor subscribes to weights from this address"""
    rep_addr: str = "tcp://localhost:5557"
    """learner REP socket for initial weight handshake"""
    req_addr: str = "tcp://localhost:5557"
    """actor REQ socket for initial weight handshake"""
    actor_id: int = 0
    """unique id for this actor process"""
    weight_timeout_ms: int = 5000
    """how long the actor waits for new weights before continuing with old ones (ms)"""
    num_actors: int = 2
    """number of actor processes (used to compute buffer defaults)"""
    learner_buffer_size: int = 0
    """batches to accumulate before each PPO update. 0 = auto (2 * num_actors)"""
    max_batches_per_actor: int = 0
    """max batches from one actor per buffer fill. 0 = auto (learner_buffer_size // num_actors)"""
    actor_cache_size: int = 16
    """circular cache depth on actor side — how many batches to buffer during outage"""
    staleness_threshold: float = float("inf")
    """maximum age (seconds) a rollout batch may have before the learner discards it; inf = no filtering"""
    partial_flush_timeout_s: float = 5.0
    """seconds to wait for full buffer before proceeding with partial batch"""
    weighting_strategy: str = "uniform"
    """Experience weighting strategy: 'uniform' | 'latency' | 'is'"""

    # to be filled in runtime
    batch_size: int = 0
    """the batch size (computed in runtime)"""
    minibatch_size: int = 0
    """the mini-batch size (computed in runtime)"""
    num_iterations: int = 0
    """the number of iterations (computed in runtime)"""

    # ── Phase 3: outage simulation ────────────────────────────────────────────────
    simulate_outage_at: int = 0
    """iteration at which to trigger a simulated network outage (0 = disabled)"""
    outage_duration: float = 0.0
    """duration of the simulated outage in seconds"""
    

def compute_runtime_args(args: Args) -> Args:
    """Populate fields that are derived from other args at runtime."""

    if args.learner_buffer_size == 0:
        args.learner_buffer_size = 2 * args.num_actors
    if args.max_batches_per_actor == 0:
        args.max_batches_per_actor = max(1, args.learner_buffer_size // args.num_actors)

    args.batch_size        = args.num_steps * args.learner_buffer_size
    args.minibatch_size    = int(args.batch_size // args.num_minibatches)
    args.num_iterations    = args.total_timesteps // args.batch_size  # learner updates

    assert args.learner_buffer_size % args.num_actors == 0, (
    f"learner_buffer_size ({args.learner_buffer_size}) must be divisible by "
    f"num_actors ({args.num_actors}) for even actor load distribution"
    )
    return args


def make_env(env_id: str, capture_video: bool, run_name: str):
    def thunk():
        if capture_video:
            env = gym.make(env_id, render_mode="rgb_array")
            env = gym.wrappers.RecordVideo(env, f"videos/{run_name}")
        else:
            env = gym.make(env_id)
        env = gym.wrappers.RecordEpisodeStatistics(env)
        return env
    return thunk