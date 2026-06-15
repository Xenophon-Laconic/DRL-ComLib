import time
import pickle
import torch
import zmq

from framework.protocol import RolloutBatch


def serialise(obj) -> bytes:
    return pickle.dumps(obj, protocol=5)


def deserialise(raw: bytes):
    return pickle.loads(raw)

def _concat_batches(batches: list[RolloutBatch]) -> RolloutBatch:
    """Merge a list of RolloutBatch objects along the time dimension."""
    return RolloutBatch(
        obs       = torch.cat([b.obs      for b in batches]),
        actions   = torch.cat([b.actions  for b in batches]),
        logprobs  = torch.cat([b.logprobs for b in batches]),
        rewards   = torch.cat([b.rewards  for b in batches]),
        dones     = torch.cat([b.dones    for b in batches]),
        values    = torch.cat([b.values   for b in batches]),
        next_obs  = batches[-1].next_obs,    # bootstrap from most recent batch
        next_done = batches[-1].next_done,
        actor_id     = batches[-1].actor_id,
        learner_step = min(b.learner_step for b in batches),  # most stale
        collected_at = min(b.collected_at for b in batches),  # earliest timestamp
    )

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
        """
        Phase 1: send b"ready", wait for b"ack" from learner.
        Phase 2: wait for initial weights on SUB — arrives when all actors are ready.
        """
        # Phase 1 — register with learner
        self._req.send(b"ready")
        ack = self._req.recv()
        assert ack == b"ack", f"Unexpected handshake response: {ack}"
        print(f"[Actor {self.actor_id}] Registered with learner, waiting for all actors...")

        # Phase 2 — block on SUB until learner broadcasts
        self._sub.poll(timeout=None)   # block indefinitely
        _, payload = self._sub.recv_multipart()
        state_dict, step = deserialise(payload)
        self.last_learner_step = step
        print(f"[Actor {self.actor_id}] All actors ready — starting rollout.")
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
                device: torch.device = torch.device("cpu"),
                buffer_size: int = 1):
        self.device = device
        self._buffer_size = buffer_size
        self._buffer: list[tuple[RolloutBatch, list]] = []
        self.device=device
        self._ctx = zmq.Context()

        self._pull = self._ctx.socket(zmq.PULL)
        self._pull.bind(pull_addr)

        self._pub = self._ctx.socket(zmq.PUB)
        self._pub.bind(pub_addr)

        self._rep = self._ctx.socket(zmq.REP)
        self._rep.bind(rep_addr)
    
    def serve_initial_weights(self, state_dict: dict, timeout_s: int = 60) -> None:
        """
        Phase 1: collect b"ready" from all buffer_size actors, ACK each with b"ack".
        Phase 2: broadcast initial weights via PUB so all actors start simultaneously.
        """
        serialised = serialise((state_dict, 0))
        deadline   = time.monotonic() + timeout_s
        connected  = 0

        # Phase 1 — handshake with each actor individually
        while connected < self._buffer_size:
            if time.monotonic() > deadline:
                raise TimeoutError(
                    f"Only {connected}/{self._buffer_size} actors connected within {timeout_s}s"
                )
            if self._rep.poll(timeout=1000):
                self._rep.recv()           # consume b"ready"
                self._rep.send(b"ack")     # acknowledge — "we see you, keep waiting"
                connected += 1
                print(f"[Learner] {connected}/{self._buffer_size} actors ready.")

        # Phase 2 — all actors connected, broadcast weights simultaneously
        print("[Learner] All actors ready — broadcasting initial weights.")
        time.sleep(0.1)  # brief pause so all SUB sockets are listening before PUB fires
        self._pub.send_multipart([b"weights", serialised])

    def recv_batch(self) -> tuple[RolloutBatch, list]:
        """Block until buffer_size batches have arrived, then return merged batch."""
        while len(self._buffer) < self._buffer_size:
            self._pull.poll(timeout=1000000)   # block until one arrives
            batch, episode_stats = deserialise(self._pull.recv())
            batch = RolloutBatch(
                **{k: v.to(self.device) if isinstance(v, torch.Tensor) else v
                for k, v in batch.__dict__.items()}
            )
            self._buffer.append((batch, episode_stats))

        ready, self._buffer = self._buffer, []
        merged_batch = _concat_batches([b for b, _ in ready])
        merged_stats = [s for _, ss in ready for s in ss]
        return merged_batch, merged_stats

    def broadcast_weights(self, agent, step: int) -> None:
        self._pub.send_multipart([b"weights", serialise((agent.actor.state_dict(), step))])

    def close(self) -> None:
        self._pull.close()
        self._pub.close()
        self._rep.close()
        self._ctx.term()