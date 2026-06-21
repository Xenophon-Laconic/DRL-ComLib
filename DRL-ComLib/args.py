"""Configuration utilities for distributed PPO actor/learner processes."""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Callable, Literal

import gymnasium as gym


WeightingStrategy = Literal["uniform", "latency", "is"]


@dataclass
class AlgorithmArgs:
    """Experiment and PPO hyperparameters."""

    exp_name: str = os.path.basename(__file__)[:-len(".py")]
    """Experiment name used in run naming and TensorBoard logs."""

    seed: int = 42
    """Global random seed."""

    torch_deterministic: bool = True
    """Enable deterministic CuDNN behavior where supported."""

    cuda: bool = False
    """Use CUDA if available."""

    env_id: str = "CartPole-v1"
    """Gymnasium environment ID."""

    total_timesteps: int = 500_000
    """Total environment steps budget."""

    learning_rate: float = 2.5e-4
    """Optimizer learning rate."""

    num_steps: int = 128
    """Rollout horizon collected by each actor per batch."""

    anneal_lr: bool = True
    """Linearly anneal the learning rate over learner iterations."""

    gamma: float = 0.99
    """Discount factor."""

    gae_lambda: float = 0.95
    """GAE lambda parameter."""

    num_minibatches: int = 4
    """Number of minibatches per PPO update."""

    update_epochs: int = 4
    """Number of PPO epochs per learner update."""

    norm_adv: bool = True
    """Normalize advantages before policy loss evaluation."""

    clip_coef: float = 0.2
    """PPO clipping coefficient."""

    clip_vloss: bool = True
    """Use clipped value loss."""

    ent_coef: float = 0.01
    """Entropy bonus coefficient."""

    vf_coef: float = 0.5
    """Value loss coefficient."""

    max_grad_norm: float = 0.5
    """Gradient clipping threshold."""

    target_kl: float | None = None
    """Optional KL threshold for early PPO epoch stopping."""

    video_every_n_episodes: int = 0
    """Actor-side video cadence. 0 disables video recording."""

    batch_size: int = 0
    """Runtime-derived learner batch size."""

    minibatch_size: int = 0
    """Runtime-derived minibatch size."""

    num_iterations: int = 0
    """Runtime-derived number of learner iterations."""


@dataclass
class CommsArgs:
    """Distributed transport, buffering, and actor freshness settings."""

    push_addr: str = "tcp://localhost:5555"
    """Address where actors push rollout batches."""

    pull_addr: str = "tcp://localhost:5555"
    """Address where the learner pulls rollout batches."""

    pub_addr: str = "tcp://localhost:5556"
    """Address where the learner publishes policy weights."""

    sub_addr: str = "tcp://localhost:5556"
    """Address where actors subscribe to learner weights."""

    rep_addr: str = "tcp://localhost:5557"
    """Address for learner-side initial weight handshake."""

    req_addr: str = "tcp://localhost:5557"
    """Address for actor-side initial weight handshake."""

    actor_id: int = 0
    """Unique actor identifier."""

    weight_timeout_ms: int = 5000
    """Maximum wait time for fresh weights before continuing with cached weights."""

    num_actors: int = 2
    """Number of actor processes contributing batches."""

    learner_buffer_size: int = 0
    """Number of rollout batches per learner update; 0 means auto-compute."""

    max_batches_per_actor: int = 0
    """Per-actor cap during learner buffer fill; 0 means auto-compute."""

    actor_cache_size: int = 4
    """Number of cached batches an actor may retain during comms outages."""

    staleness_threshold: float = float("inf")
    """Maximum accepted batch age in seconds; infinity disables filtering."""

    partial_flush_timeout_s: float = 5.0
    """Maximum wait for a full learner buffer before a partial flush."""

    weighting_strategy: WeightingStrategy = "uniform"
    """Experience weighting strategy."""

    enable_policy_reset: bool = True
    """Allow learner-triggered actor resets after repeated staleness."""

    reset_stale_after: int = 5
    """Reset an actor after this many consecutive stale batches."""

    simulate_outage_at: int = 0
    """Trigger a simulated actor outage at this iteration; 0 disables it."""

    outage_duration: float = 0.0
    """Duration of the simulated outage in seconds."""

    simulate_latency: bool = False
    """If True, inject synthetic latency before sending each rollout."""

    latency_ms: float = 0.0
    """Fixed synthetic latency in milliseconds per rollout when simulate_latency is True."""


@dataclass
class Args(AlgorithmArgs, CommsArgs):
    """Full CLI configuration for distributed PPO actor and learner entrypoints."""
    pass


def validate_args(args: Args) -> None:
    """Validate user-provided configuration before deriving runtime fields."""
    if args.num_actors <= 0:
        raise ValueError("num_actors must be positive")
    if args.num_steps <= 0:
        raise ValueError("num_steps must be positive")
    if args.num_minibatches <= 0:
        raise ValueError("num_minibatches must be positive")
    if args.total_timesteps <= 0:
        raise ValueError("total_timesteps must be positive")
    if args.learning_rate <= 0:
        raise ValueError("learning_rate must be positive")
    if args.learner_buffer_size < 0:
        raise ValueError("learner_buffer_size cannot be negative")
    if args.max_batches_per_actor < 0:
        raise ValueError("max_batches_per_actor cannot be negative")
    if args.weighting_strategy not in {"uniform", "latency", "is"}:
        raise ValueError(
            f"Unknown weighting_strategy={args.weighting_strategy!r}. "
            "Expected one of: 'uniform', 'latency', 'is'."
        )
    if args.video_every_n_episodes < 0:
        raise ValueError("video_every_n_episodes cannot be negative")
    if args.latency_ms < 0:
        raise ValueError("latency_ms cannot be negative")


def populate_runtime_args(args: Args) -> Args:
    """Populate fields derived from validated static configuration."""

    # If not set explicitly, scale learner buffer with the number of actors.
    # Using 2 * num_actors gives each actor roughly two pending slots in the
    # learner queue. This is usually enough to absorb rollout bursts from all
    # actors without unbounded queue growth, while keeping the learner close
    # to on-policy (small policy lag)
    if args.learner_buffer_size == 0:
        args.learner_buffer_size = 2 * args.num_actors

    if args.learner_buffer_size % args.num_actors != 0:
        raise ValueError(
            "learner_buffer_size must be divisible by num_actors "
            "for even actor load distribution"
        )

    # cap how many pending batches each actor can enqueue.
    # We divide the total learner buffer across actors so that on average
    # each actor can have at most learner_buffer_size // num_actors in-flight
    # batches (at least one) preventing any actor from being completely blocked
    if args.max_batches_per_actor == 0:
        args.max_batches_per_actor = max(1, args.learner_buffer_size // args.num_actors)

    # Use all collected timesteps in the learner buffer for one PPO update:
    # batch_size = (steps per rollout) * (number of rollouts in learner buffer).
    args.batch_size = args.num_steps * args.learner_buffer_size

    if args.batch_size % args.num_minibatches != 0:
        raise ValueError(
            "batch_size must be divisible by num_minibatches; "
            f"got batch_size={args.batch_size}, "
            f"num_minibatches={args.num_minibatches}"
        )

    # Split each PPO update batch into equal-sized minibatches.
    args.minibatch_size = args.batch_size // args.num_minibatches

    # Run enough update iterations so that num_iterations * batch_size
    # ~= total_timesteps specified for training.
    args.num_iterations = args.total_timesteps // args.batch_size

    if args.num_iterations <= 0:
        raise ValueError(
            "total_timesteps must be at least one full learner batch; "
            f"got total_timesteps={args.total_timesteps}, batch_size={args.batch_size}"
        )

    return args


def compute_runtime_args(args: Args) -> Args:
    """Validate configuration and populate runtime-derived fields."""
    validate_args(args)
    return populate_runtime_args(args)

def make_env(
    args: AlgorithmArgs,
    run_name: str,
    *,
    record_video: bool | None = None,
) -> Callable[[], gym.Env]:
    """Return a thunk that builds a wrapped Gymnasium environment.

    By default, video recording is enabled only when
    `args.video_every_n_episodes > 0`. Callers may override this with
    `record_video=True` or `record_video=False`.
    """
    should_record = (
        args.video_every_n_episodes > 0
        if record_video is None
        else record_video
    )

    def thunk() -> gym.Env:
        if should_record:
            env = gym.make(args.env_id, render_mode="rgb_array")
            env = gym.wrappers.RecordVideo(
                env,
                video_folder=f"videos/{run_name}",
                episode_trigger=lambda episode_id: (
                    episode_id % args.video_every_n_episodes == 0
                ),
            )
        else:
            env = gym.make(args.env_id)

        env = gym.wrappers.RecordEpisodeStatistics(env)
        return env

    return thunk