import time
import pickle
import torch
import zmq

from framework.protocol import RolloutBatch


def serialise(obj) -> bytes:
    return pickle.dumps(obj, protocol=5)


def deserialise(raw: bytes):
    return pickle.loads(raw)


class ActorComms:
    def __init__(self, push_addr: str, sub_addr: str, actor_id: int, req_addr: str):
        self._ctx = zmq.Context()

        self._push = self._ctx.socket(zmq.PUSH)
        self._push.connect(push_addr)

        self._sub = self._ctx.socket(zmq.SUB)
        self._sub.connect(sub_addr)
        self._sub.setsockopt_string(zmq.SUBSCRIBE, "weights")

        self._req = self._ctx.socket(zmq.REQ)
        self._req.connect(req_addr)

        self.actor_id = actor_id
        self.last_learner_step: int = 0

    def request_initial_weights(self) -> dict:
        self._req.send(b"ready")
        payload = self._req.recv()
        state_dict, step = deserialise(payload)
        self.last_learner_step = step
        return state_dict

    def send_batch(self, batch: RolloutBatch, episode_stats: list) -> None:
        batch.actor_id     = self.actor_id
        batch.learner_step = self.last_learner_step
        self._push.send(serialise((batch, episode_stats)))

    def recv_weights(self) -> dict | None:
        if self._sub.poll(0):                             # non-blocking
            _, payload = self._sub.recv_multipart()
            state_dict, step = deserialise(payload)
            self.last_learner_step = step
            return state_dict
        return None

    def sync_weights(self, agent) -> bool:
        weights = self.recv_weights()
        if weights is not None:
            agent.load_state_dict(weights)
            return True
        return False

    def close(self) -> None:
        self._push.close()
        self._sub.close()
        self._req.close()
        self._ctx.term()


class LearnerComms:
    def __init__(self, pull_addr: str, pub_addr: str, rep_addr: str,
                 device: torch.device = torch.device("cpu")):
        self.device=device
        self._ctx = zmq.Context()

        self._pull = self._ctx.socket(zmq.PULL)
        self._pull.bind(pull_addr)

        self._pub = self._ctx.socket(zmq.PUB)
        self._pub.bind(pub_addr)

        self._rep = self._ctx.socket(zmq.REP)
        self._rep.bind(rep_addr)

    def serve_initial_weights(self, state_dict: dict, timeout_s: int = 60) -> None:
        deadline = time.monotonic() + timeout_s
        while time.monotonic() < deadline:
            if self._rep.poll(timeout=1000):
                self._rep.recv()
                self._rep.send(serialise((state_dict, 0)))
                return
        raise TimeoutError("No actor connected within 60s")

    def recv_batch(self, timeout_ms: int = 10000) -> tuple[RolloutBatch, list] | None:
        if self._pull.poll(timeout_ms):
            batch, episode_stats = deserialise(self._pull.recv())
            batch = RolloutBatch(
                **{k: v.to(self.device) if isinstance(v, torch.Tensor) else v
                   for k, v in batch.__dict__.items()}
            )
            return batch, episode_stats
        return None

    def broadcast_weights(self, agent, step: int) -> None:
        self._pub.send_multipart([b"weights", serialise((agent.actor.state_dict(), step))])

    def close(self) -> None:
        self._pull.close()
        self._pub.close()
        self._rep.close()
        self._ctx.term()