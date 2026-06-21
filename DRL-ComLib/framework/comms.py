from __future__ import annotations

import pickle
import time
from collections import deque
from typing import Any

import torch
import zmq

from framework.protocol import RolloutBatch


WEIGHTS_TOPIC = b"weights"
SHUTDOWN_TOPIC = b"shutdown"
RESET_TOPIC = b"reset"

MIXED_ACTOR_ID = -1


def serialise(obj: Any) -> bytes:
    """Serialise an object for transport over ZeroMQ."""
    return pickle.dumps(obj, protocol=5)


def deserialise(raw: bytes) -> Any:
    """Deserialise a ZeroMQ payload."""
    return pickle.loads(raw)


def _move_batch_to_device(batch: RolloutBatch, device: torch.device) -> RolloutBatch:
    """Move all tensor fields in a batch onto the learner device."""
    return RolloutBatch(
        **{
            key: value.to(device) if isinstance(value, torch.Tensor) else value
            for key, value in batch.__dict__.items()
        }
    )


def _concat_batches(batches: list[RolloutBatch]) -> RolloutBatch:
    """Merge rollout batches along the leading time dimension.

    The merged batch may contain samples from multiple actors. In that case,
    `actor_id` is set to -1 to indicate mixed provenance.

    Metadata reduction policy:
    - `next_obs` / `next_done`: taken from the most recent constituent batch.
    - `learner_step`: minimum of merged batches, representing the stalest source.
    - `collected_at`: minimum timestamp, representing the oldest source sample.
    """
    if not batches:
        raise ValueError("Cannot concatenate an empty list of batches")

    actor_ids = {batch.actor_id for batch in batches}
    merged_actor_id = next(iter(actor_ids)) if len(actor_ids) == 1 else MIXED_ACTOR_ID

    return RolloutBatch(
        obs=torch.cat([batch.obs for batch in batches]),
        actions=torch.cat([batch.actions for batch in batches]),
        logprobs=torch.cat([batch.logprobs for batch in batches]),
        rewards=torch.cat([batch.rewards for batch in batches]),
        dones=torch.cat([batch.dones for batch in batches]),
        values=torch.cat([batch.values for batch in batches]),
        next_obs=batches[-1].next_obs,
        next_done=batches[-1].next_done,
        advantages=None,
        returns=None,
        actor_id=merged_actor_id,
        learner_step=min(batch.learner_step for batch in batches),
        collected_at=min(batch.collected_at for batch in batches),
    )


class ActorComms:
    """Actor-side ZeroMQ transport for rollout upload and control messages.

    Responsibilities:
    - request the initial policy weights from the learner,
    - upload rollout batches over PUSH,
    - subscribe to learner PUB messages for weights, resets, and shutdown,
    - buffer batches locally during transient outages.
    """

    def __init__(
        self,
        push_addr: str,
        sub_addr: str,
        actor_id: int,
        req_addr: str,
        cache_size: int = 16,
    ) -> None:
        self._push_addr = push_addr
        self._sub_addr = sub_addr
        self._req_addr = req_addr

        self._ctx = zmq.Context()

        self._sub = self._ctx.socket(zmq.SUB)
        self._sub.connect(self._sub_addr)
        self._sub.setsockopt(zmq.SUBSCRIBE, WEIGHTS_TOPIC)
        self._sub.setsockopt(zmq.SUBSCRIBE, SHUTDOWN_TOPIC)
        self._sub.setsockopt(zmq.SUBSCRIBE, RESET_TOPIC)

        self._req = self._ctx.socket(zmq.REQ)
        self._req.connect(self._req_addr)

        self.actor_id = actor_id
        self.last_learner_step = 0

        self._cache: deque[tuple[RolloutBatch, list[dict[str, float]]]] = deque(
            maxlen=cache_size
        )
        self._connected = False
        self._push: zmq.Socket | None = None
        self._outage_ends_at = 0.0

        self._connect_push()

    def _connect_push(self) -> None:
        """Create or recreate the actor PUSH socket."""
        if self._push is not None:
            self._push.close(linger=0)

        self._push = self._ctx.socket(zmq.PUSH)
        self._push.setsockopt(zmq.SNDTIMEO, 100)
        self._push.setsockopt(zmq.LINGER, 0)
        self._push.connect(self._push_addr)
        self._connected = True

    def _disconnect_push(self) -> None:
        """Close the PUSH socket and mark the actor as disconnected."""
        if self._push is not None:
            self._push.close(linger=0)
            self._push = None
        self._connected = False

    def _flush_cache(self) -> int:
        """Attempt to send all cached rollout batches in FIFO order.

        Stops at the first send failure and leaves remaining batches cached.

        Returns:
            Number of cached batches successfully sent.
        """
        sent = 0

        while self._cache:
            batch, episode_stats = self._cache[0]
            try:
                if self._push is None:
                    raise zmq.Again()
                self._push.send(serialise((batch, episode_stats)), zmq.NOBLOCK)
                self._cache.popleft()
                sent += 1
            except zmq.Again:
                self._disconnect_push()
                break

        return sent

    def request_initial_weights(self) -> dict:
        """Block until the learner acknowledges registration and broadcasts weights."""
        self._req.send(b"ready")
        ack = self._req.recv()
        if ack != b"ack":
            raise RuntimeError(
                f"[Actor {self.actor_id}] Unexpected handshake response: {ack!r}"
            )

        print(f"[Actor {self.actor_id}] Registered with learner, waiting for all actors...")

        self._sub.poll(timeout=None)
        topic, payload = self._sub.recv_multipart()
        if topic != WEIGHTS_TOPIC:
            raise RuntimeError(
                f"[Actor {self.actor_id}] Expected initial weights, got topic {topic!r}"
            )

        state_dict, step = deserialise(payload)
        self.last_learner_step = step

        print(f"[Actor {self.actor_id}] All actors ready — starting rollout.")
        return state_dict

    def send_batch(self, batch: RolloutBatch, episode_stats: list[dict[str, float]]) -> bool:
        """Stamp, cache, and attempt to flush a rollout batch.

        The batch is always cached first, then flushed opportunistically.
        If the actor is currently inside a simulated outage window, the batch
        remains cached and this method returns False.

        Returns:
            True if all currently cached batches were sent, otherwise False.
        """
        batch.actor_id = self.actor_id
        batch.learner_step = self.last_learner_step

        self._cache.append((batch, episode_stats))

        if not self._connected:
            if time.monotonic() < self._outage_ends_at:
                print(
                    f"[Actor {self.actor_id}] Outage active — "
                    f"{len(self._cache)} batch(es) cached."
                )
                return False

            print(f"[Actor {self.actor_id}] Reconnecting PUSH socket...")
            self._connect_push()

        self._flush_cache()

        if not self._connected:
            print(
                f"[Actor {self.actor_id}] Outage active — "
                f"{len(self._cache)} batch(es) cached."
            )
            return False

        return True

    def recv_weights(self) -> tuple[str, object] | None:
        """Poll for a learner control-plane message.

        Returns:
            ("weights", state_dict): normal weight update.
            ("reset", state_dict): learner-triggered reset with fresh weights.
            ("shutdown", None): clean learner shutdown.
            None: no message available.
        """
        if not self._sub.poll(0):
            return None

        topic, payload = self._sub.recv_multipart()

        if topic == SHUTDOWN_TOPIC:
            return ("shutdown", None)

        if topic == WEIGHTS_TOPIC:
            state_dict, step = deserialise(payload)
            self.last_learner_step = step
            return ("weights", state_dict)

        if topic == RESET_TOPIC:
            target_actor_id, state_dict, step = deserialise(payload)
            if target_actor_id != self.actor_id:
                return None
            self.last_learner_step = step
            return ("reset", state_dict)

        print(f"[Actor {self.actor_id}] Unknown PUB topic: {topic!r}")
        return None

    def sync_weights(self, agent) -> str:
        """Apply at most one pending learner control message to the local actor.

        Side effects:
        - loads incoming policy weights into `agent`,
        - clears local cached state on reset,
        - does not block if no message is available.

        Returns:
            "updated": a normal weight update was applied.
            "reset": learner-triggered reset was applied.
            "shutdown": learner requested clean termination.
            "none": no message was available.
        """
        msg = self.recv_weights()
        if msg is None:
            return "none"

        msg_type, payload = msg

        if msg_type == "shutdown":
            return "shutdown"

        if msg_type == "reset":
            self.apply_reset()
            agent.load_state_dict(payload)
            return "reset"

        if msg_type == "weights":
            agent.load_state_dict(payload)
            return "updated"

        return "none"

    def simulate_outage(self, duration_s: float) -> None:
        """Simulate a temporary send outage for testing actor-side resilience."""
        print(f"[Actor {self.actor_id}] Simulating outage for {duration_s}s...")
        self._disconnect_push()
        self._outage_ends_at = time.monotonic() + duration_s

    def apply_reset(self) -> None:
        """Clear stale actor-local state after a learner-triggered reset."""
        self._cache.clear()

        if not self._connected:
            print(f"[Actor {self.actor_id}] Reset reconnecting PUSH socket...")
            self._connect_push()

        self._outage_ends_at = 0.0
        print(f"[Actor {self.actor_id}] Local cache cleared due to learner reset.")

    def close(self) -> None:
        """Close all actor-side sockets and terminate the ZeroMQ context."""
        if self._push is not None:
            self._push.close()
        self._sub.close()
        self._req.close()
        self._ctx.term()


class LearnerComms:
    """Learner-side ZeroMQ transport for rollout ingestion and policy broadcasts."""

    def __init__(
        self,
        pull_addr: str,
        pub_addr: str,
        rep_addr: str,
        device: torch.device = torch.device("cpu"),
        buffer_size: int = 1,
        max_batches_per_actor: int = 1,
        num_actors: int = 1,
        staleness_threshold: float = float("inf"),
        enable_policy_reset: bool = True,
        reset_stale_after: int = 5,
    ) -> None:
        self.device = device
        self._buffer_size = buffer_size
        self._max_batches_per_actor = max_batches_per_actor
        self._num_actors = num_actors
        self._staleness_threshold = staleness_threshold
        self._enable_policy_reset = enable_policy_reset
        self._reset_stale_after = reset_stale_after

        self._buffer: list[tuple[RolloutBatch, list[dict[str, float]]]] = []
        self._ctx = zmq.Context()

        self._pull = self._ctx.socket(zmq.PULL)
        self._pull.bind(pull_addr)

        self._pub = self._ctx.socket(zmq.PUB)
        self._pub.bind(pub_addr)

        self._rep = self._ctx.socket(zmq.REP)
        self._rep.bind(rep_addr)

        self._stats = {
            "batches_received": 0,
            "batches_rejected_stale": 0,
        }
        self._actor_registry: dict[int, dict[str, float | int]] = {}
        self._actors_pending_reset: set[int] = set()

    def serve_initial_weights(self, state_dict: dict, timeout_s: int = 60) -> None:
        """Wait for all actors to register, then broadcast initial policy weights."""
        serialised = serialise((state_dict, 0))
        deadline = time.monotonic() + timeout_s
        connected = 0

        while connected < self._num_actors:
            if time.monotonic() > deadline:
                raise TimeoutError(
                    f"Only {connected}/{self._num_actors} actors connected within {timeout_s}s"
                )

            if not self._rep.poll(timeout=1000):
                continue

            ready = self._rep.recv()
            if ready != b"ready":
                raise RuntimeError(f"Unexpected learner handshake payload: {ready!r}")

            self._rep.send(b"ack")
            connected += 1
            print(f"[Learner] {connected}/{self._num_actors} actors ready.")

        print("[Learner] All actors ready — broadcasting initial weights.")
        time.sleep(0.1)
        self._pub.send_multipart([WEIGHTS_TOPIC, serialised])

    def recv_batch(
        self,
        writer=None,
        global_step: int = 0,
        partial_flush_timeout_s: float = 5.0,
    ) -> tuple[RolloutBatch, list[dict[str, float]]]:
        """Receive accepted rollout batches and merge them for one learner update.

        Behavior:
        - waits for up to `buffer_size` accepted batches,
        - if the timeout expires and at least one accepted batch is buffered,
          returns a partial merged batch,
        - filters stale batches and enforces a per-actor buffer cap.
        """
        deadline = time.monotonic() + partial_flush_timeout_s

        while len(self._buffer) < self._buffer_size:
            time_remaining_ms = int((deadline - time.monotonic()) * 1000)

            if time_remaining_ms <= 0:
                if self._buffer:
                    print(
                        f"[Learner] Partial flush: {len(self._buffer)}/{self._buffer_size} "
                        f"batches after {partial_flush_timeout_s:.1f}s timeout."
                    )
                    break

                deadline = time.monotonic() + partial_flush_timeout_s
                continue

            if not self._pull.poll(timeout=min(time_remaining_ms, 1000)):
                continue

            batch, episode_stats = deserialise(self._pull.recv())
            batch = _move_batch_to_device(batch, self.device)

            accepted_before = len(self._buffer)
            self._append_to_buffer(batch, episode_stats)
            batch_was_accepted = len(self._buffer) > accepted_before

            self._maybe_log_recv_metrics(
                writer=writer,
                batch=batch,
                batch_was_accepted=batch_was_accepted,
                global_step=global_step,
            )

        ready, self._buffer = self._buffer, []
        merged_batch = _concat_batches([batch for batch, _ in ready])
        merged_stats = [stat for _, stats in ready for stat in stats]
        return merged_batch, merged_stats

    def _maybe_log_recv_metrics(
        self,
        writer,
        batch: RolloutBatch,
        batch_was_accepted: bool,
        global_step: int,
    ) -> None:
        """Emit periodic learner-side comms telemetry."""
        if self._stats["batches_received"] % 100 == 0 and len(self._buffer) == 0:
            print(
                f"[Learner] Waiting for accepted batches — "
                f"{self._stats['batches_rejected_stale']} rejected so far "
                f"(tau={self._staleness_threshold:.1f}s)"
            )

        if writer is None or self._stats["batches_received"] % 20 != 0:
            return

        rejection_rate = (
            self._stats["batches_rejected_stale"]
            / max(1, self._stats["batches_received"])
        )
        writer.add_scalar("staleness/rejection_rate", rejection_rate, global_step)

        if batch_was_accepted:
            writer.add_scalar(
                "staleness/batch_age_s",
                time.monotonic() - batch.collected_at,
                global_step,
            )

    def _append_to_buffer(
        self,
        batch: RolloutBatch,
        episode_stats: list[dict[str, float]],
    ) -> None:
        """Single entry point for learner buffer writes.

        Enforces:
        - stale-batch rejection,
        - per-actor pending-batch cap,
        - actor reset scheduling after repeated staleness.
        """
        self._stats["batches_received"] += 1

        self._ensure_actor_record(batch.actor_id)
        record = self._actor_registry[batch.actor_id]

        now = time.monotonic()
        age = now - batch.collected_at
        record["last_seen_time"] = now
        record["last_batch_age"] = age
        record["last_learner_step_seen"] = batch.learner_step

        if age > self._staleness_threshold:
            self._stats["batches_rejected_stale"] += 1
            record["consecutive_stale_batches"] += 1

            print(
                f"[Learner] Dropped stale batch from actor {batch.actor_id}: "
                f"age={age:.2f}s > tau={self._staleness_threshold:.2f}s"
            )

            if (
                self._enable_policy_reset
                and record["consecutive_stale_batches"] >= self._reset_stale_after
                and batch.actor_id not in self._actors_pending_reset
            ):
                self._actors_pending_reset.add(batch.actor_id)
                print(
                    f"[Learner] Marking actor {batch.actor_id} for reset after "
                    f"{record['consecutive_stale_batches']} consecutive stale batches."
                )

            return

        record["consecutive_stale_batches"] = 0

        actor_count = sum(
            1 for buffered_batch, _ in self._buffer if buffered_batch.actor_id == batch.actor_id
        )
        if actor_count >= self._max_batches_per_actor:
            for index, (buffered_batch, _) in enumerate(self._buffer):
                if buffered_batch.actor_id == batch.actor_id:
                    self._buffer.pop(index)
                    break

        self._buffer.append((batch, episode_stats))

    def broadcast_weights(self, agent, step: int) -> None:
        """Broadcast the latest actor weights to all subscribed actors."""
        payload = serialise((agent.actor.state_dict(), step))
        self._pub.send_multipart([WEIGHTS_TOPIC, payload])

    def broadcast_shutdown(self) -> None:
        """Broadcast a clean shutdown signal to all subscribed actors."""
        self._pub.send_multipart([SHUTDOWN_TOPIC, b""])
        print("[Learner] Shutdown signal broadcast.")

    def broadcast_reset(self, actor_id: int, state_dict: dict, step: int) -> None:
        """Broadcast a targeted reset command and fresh weights to one actor."""
        payload = serialise((actor_id, state_dict, step))
        self._pub.send_multipart([RESET_TOPIC, payload])

        self._actor_registry[actor_id]["reset_count"] += 1
        self._actor_registry[actor_id]["consecutive_stale_batches"] = 0
        self._actors_pending_reset.discard(actor_id)

        print(f"[Learner] Broadcast reset for actor {actor_id}.")

    def flush_pending_resets(self, agent, step: int) -> list[int]:
        """Broadcast resets for all actors currently marked stale."""
        reset_ids = list(self._actors_pending_reset)
        state_dict = agent.actor.state_dict()

        for actor_id in reset_ids:
            self.broadcast_reset(actor_id, state_dict, step)

        return reset_ids

    def _ensure_actor_record(self, actor_id: int) -> None:
        """Create bookkeeping state for a newly observed actor."""
        if actor_id not in self._actor_registry:
            self._actor_registry[actor_id] = {
                "last_seen_time": 0.0,
                "last_batch_age": 0.0,
                "last_learner_step_seen": 0,
                "consecutive_stale_batches": 0,
                "reset_count": 0,
            }

    def close(self) -> None:
        """Close learner-side sockets and terminate the ZeroMQ context."""
        self._pull.close()
        self._pub.close()
        self._rep.close()
        self._ctx.term()