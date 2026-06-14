from torch.utils.tensorboard import SummaryWriter


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