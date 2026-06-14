import io
import torch
import zmq
from rollout import RolloutBatch


def serialise_batch(batch: RolloutBatch, episode_stats: list) -> bytes:
    buffer = io.BytesIO()
    torch.save({
        "obs":       batch.obs.cpu(),
        "actions":   batch.actions.cpu(),
        "logprobs":  batch.logprobs.cpu(),
        "rewards":   batch.rewards.cpu(),
        "dones":     batch.dones.cpu(),
        "values":    batch.values.cpu(),
        "next_obs":  batch.next_obs.cpu(),
        "next_done": batch.next_done.cpu(),
        "episode_stats": episode_stats,
    }, buffer)
    return buffer.getvalue()


def deserialise_batch(data: bytes):
    buffer = io.BytesIO(data)
    d = torch.load(buffer, weights_only=False)  # weights_only=False to allow plain dicts
    episode_stats = d.pop("episode_stats")
    return RolloutBatch(**d), episode_stats


def serialise_state_dict(state_dict: dict) -> bytes:
    buffer = io.BytesIO()
    torch.save(state_dict, buffer)
    return buffer.getvalue()


def deserialise_state_dict(data: bytes) -> dict:
    buffer = io.BytesIO(data)
    return torch.load(buffer, weights_only=True)


def make_actor_sockets(context: zmq.Context, learner_host: str = "localhost"):
    push = context.socket(zmq.PUSH)
    push.connect(f"tcp://{learner_host}:5555")

    sub = context.socket(zmq.SUB)
    sub.connect(f"tcp://{learner_host}:5556")
    sub.setsockopt_string(zmq.SUBSCRIBE, "")

    # REQ socket to request initial weights
    req = context.socket(zmq.REQ)
    req.connect(f"tcp://{learner_host}:5557")

    return push, sub, req


def make_learner_sockets(context: zmq.Context):
    pull = context.socket(zmq.PULL)
    pull.bind("tcp://*:5555")

    pub = context.socket(zmq.PUB)
    pub.bind("tcp://*:5556")

    # REP socket to serve initial weights on demand
    rep = context.socket(zmq.REP)
    rep.bind("tcp://*:5557")

    return pull, pub, rep