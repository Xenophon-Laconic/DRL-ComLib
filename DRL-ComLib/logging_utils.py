from __future__ import annotations

import time

import torch
from torch.utils.tensorboard import SummaryWriter

from framework.protocol import RolloutBatch


def make_writer(run_name: str) -> SummaryWriter:
    """Create a TensorBoard SummaryWriter for this run."""
    return SummaryWriter(f"runs/{run_name}")


def log_hyperparameters(writer: SummaryWriter, args) -> None:
    """Log all configuration fields as a Markdown table."""
    rows = [f"|{key}|{value}|" for key, value in vars(args).items()]
    table = "|param|value|\n|-|-|\n" + "\n".join(rows)
    writer.add_text("hyperparameters", table)


def log_episode_stats(
    writer: SummaryWriter,
    episode_stats: list[dict],
    global_step: int,
) -> None:
    """Log episodic returns and lengths collected during a rollout."""
    for stat in episode_stats:
        for key, value in stat.items():
            writer.add_scalar(key, value, global_step)


def log_training_metrics(
    writer: SummaryWriter,
    metrics: dict,
    global_step: int,
) -> None:
    """Write the dict returned by ppo_update() to TensorBoard."""
    for key, value in metrics.items():
        writer.add_scalar(key, value, global_step)


def log_infra_metrics(
    writer: SummaryWriter,
    optimiser,
    global_step: int,
    sps: int,
) -> None:
    """Log learning rate and throughput (steps per second)."""
    lr = optimiser.param_groups[0]["lr"]
    writer.add_scalar("charts/learning_rate", lr, global_step)
    writer.add_scalar("charts/SPS", sps, global_step)


def _log_comms_common(
    writer: SummaryWriter,
    batch: RolloutBatch,
    global_step: int,
) -> None:
    """Log comms latency and learner step gap for a batch."""
    latency = time.monotonic() - batch.collected_at
    writer.add_scalar("comms/transit_latency_s", latency, global_step)
    writer.add_scalar(
        "comms/learner_step_gap",
        global_step - batch.learner_step,
        global_step,
    )


def log_batch_meta(
    writer: SummaryWriter,
    batch: RolloutBatch,
    global_step: int,
) -> None:
    """Log basic comms metadata for a received batch.

    If the batch contains samples merged from multiple actors, actor-specific
    scalar logging is skipped and a mixed-batch indicator is emitted instead.
    """
    _log_comms_common(writer, batch, global_step)

    if batch.actor_id >= 0:
        writer.add_scalar("comms/actor_id", batch.actor_id, global_step)
        writer.add_scalar("comms/mixed_actor_batch", 0, global_step)
    else:
        writer.add_scalar("comms/mixed_actor_batch", 1, global_step)


def log_weight_stats(
    writer: SummaryWriter,
    weights: torch.Tensor,
    strategy: str,
    global_step: int,
) -> None:
    """Diagnostic for the active weighting strategy; call before ppo_update().

    Assumes `weights` is a non-empty tensor of per-sample weights on any device.
    """
    with torch.no_grad():
        writer.add_scalar("weights/mean", weights.mean().item(), global_step)
        writer.add_scalar("weights/std", weights.std().item(), global_step)
        writer.add_scalar("weights/max", weights.max().item(), global_step)
        writer.add_scalar("weights/min", weights.min().item(), global_step)
        writer.add_histogram("weights/dist", weights.cpu(), global_step)
    writer.add_text("weights/strategy", strategy, global_step)


def log_staleness(
    writer: SummaryWriter,
    batch: RolloutBatch,
    global_step: int,
    rejected: bool = False,
) -> None:
    """Log comms timing and whether this batch was rejected as stale."""
    _log_comms_common(writer, batch, global_step)
    writer.add_scalar("staleness/rejected", int(rejected), global_step)


def log_reset_metrics(
    writer: SummaryWriter,
    reset_actor_ids: list[int],
    global_step: int,
) -> None:
    """Log how many and which actors were reset in this learner step."""
    writer.add_scalar("resets/count", len(reset_actor_ids), global_step)
    for actor_id in reset_actor_ids:
        writer.add_scalar(f"resets/actor_{actor_id}", 1, global_step)