from torch.utils.tensorboard import SummaryWriter
import time
from framework.protocol import RolloutBatch
import torch

def make_writer(run_name: str) -> SummaryWriter:
    return SummaryWriter(f"runs/{run_name}")


def log_hyperparameters(writer: SummaryWriter, args) -> None:
    writer.add_text(
        "hyperparameters",
        "|param|value|\n|-|-|\n%s" % ("\n".join([f"|{key}|{value}|" for key, value in vars(args).items()])),
    )


def log_episode_stats(writer: SummaryWriter, episode_stats: list, global_step: int) -> None:
    """Log episodic returns collected during a rollout."""
    for stat in episode_stats:
        for key, value in stat.items():
            writer.add_scalar(key, value, global_step)


def log_training_metrics(writer: SummaryWriter, metrics: dict, global_step: int) -> None:
    """Write the dict returned by ppo_update() to TensorBoard."""
    for key, value in metrics.items():
        writer.add_scalar(key, value, global_step)


def log_infra_metrics(writer: SummaryWriter, optimiser, global_step: int, sps: int) -> None:
    """Log learning rate and throughput."""
    writer.add_scalar("charts/learning_rate", optimiser.param_groups[0]["lr"], global_step)
    writer.add_scalar("charts/SPS", sps, global_step)

def log_batch_meta(writer, batch: RolloutBatch, global_step: int):
    latency = time.monotonic() - batch.collected_at
    writer.add_scalar("comms/transit_latency_s", latency, global_step)
    writer.add_scalar("comms/actor_id", batch.actor_id, global_step)
    writer.add_scalar("comms/learner_step_gap",
                      global_step - batch.learner_step, global_step)
    
def log_weight_stats(writer: SummaryWriter, weights: torch.Tensor, 
                     strategy: str, global_step: int) -> None:
    """Diagnostic for the active weighting strategy — call before ppo_update()."""
    with torch.no_grad():
        writer.add_scalar("weights/mean",     weights.mean().item(),  global_step)
        writer.add_scalar("weights/std",      weights.std().item(),   global_step)
        writer.add_scalar("weights/max",      weights.max().item(),   global_step)
        writer.add_scalar("weights/min",      weights.min().item(),   global_step)
        writer.add_histogram("weights/dist",  weights.cpu(),          global_step)
    writer.add_text("weights/strategy", strategy, global_step)

def log_staleness(writer: SummaryWriter, batch: RolloutBatch,
                  rejected: bool, global_step: int) -> None:
    """Per-iteration staleness record — replaces the raw log_batch_meta call 
    for the latency field, adds a rejection flag for sweep analysis."""
    latency = time.monotonic() - batch.collected_at
    writer.add_scalar("comms/transit_latency_s",   latency,        global_step)
    writer.add_scalar("comms/learner_step_gap",
                      global_step - batch.learner_step,            global_step)
    writer.add_scalar("staleness/rejected",        int(rejected),  global_step)