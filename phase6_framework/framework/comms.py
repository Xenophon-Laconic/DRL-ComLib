import time
import pickle
import torch
import zmq
from collections import deque 

from framework.protocol import RolloutBatch

WEIGHTS_TOPIC = b"weights"
SHUTDOWN_TOPIC = b"shutdown"
RESET_TOPIC = b"reset"

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
    def __init__(self, push_addr: str, sub_addr: str, actor_id: int,
                 req_addr: str, cache_size: int = 16):
        self._push_addr = push_addr
        self._sub_addr = sub_addr
        self._req_addr = req_addr
        self._ctx = zmq.Context()

        self._sub = self._ctx.socket(zmq.SUB)
        self._sub.connect(sub_addr)
        self._sub.setsockopt(zmq.SUBSCRIBE, WEIGHTS_TOPIC)
        self._sub.setsockopt(zmq.SUBSCRIBE, SHUTDOWN_TOPIC)
        self._sub.setsockopt(zmq.SUBSCRIBE, RESET_TOPIC)

        self._req = self._ctx.socket(zmq.REQ)
        self._req.connect(req_addr)

        self.actor_id = actor_id
        self.last_learner_step: int = 0

        # ── Phase 3: circular cache ───────────────────────────────
        self._cache: deque[tuple[RolloutBatch, list]] = deque(maxlen=cache_size)
        self._connected: bool = False
        self._push: zmq.Socket | None = None
        self._connect_push()
        self._outage_ends_at: float = 0.0 
        # ─────────────────────────────────────────────────────────


    def _connect_push(self) -> None:
        """Create (or recreate) the PUSH socket and mark as connected."""
        if self._push is not None:
            self._push.close(linger=0)
        self._push = self._ctx.socket(zmq.PUSH)
        self._push.setsockopt(zmq.SNDTIMEO, 100)   # 100ms send timeout
        self._push.setsockopt(zmq.LINGER, 0)        # don't block on close
        self._push.connect(self._push_addr)
        self._connected = True

    def _flush_cache(self) -> int:
        """
        Attempt to drain the cache over the PUSH socket.
        Stops at the first send failure and marks as disconnected.
        Returns number of batches successfully sent.
        """
        sent = 0
        while self._cache:
            batch, episode_stats = self._cache[0]
            try:
                self._push.send(serialise((batch, episode_stats)), zmq.NOBLOCK)
                self._cache.popleft()
                sent += 1
            except zmq.Again:
                self._connected = False
                if self._push is not None:
                    self._push.close(linger=0)
                    self._push = None
                break
        return sent

    def request_initial_weights(self) -> dict:
        """
        Phase 1: send b"ready", wait for b"ack" from learner.
        Phase 2: wait for initial weights on SUB.
        """
        self._req.send(b"ready")
        ack = self._req.recv()
        assert ack == b"ack", f"Unexpected handshake response: {ack}"
        print(f"[Actor {self.actor_id}] Registered with learner, waiting for all actors...")

        self._sub.poll(timeout=None)
        topic, payload = self._sub.recv_multipart()
        assert topic == WEIGHTS_TOPIC, f"Expected initial weights, got topic {topic!r}"
        state_dict, step = deserialise(payload)
        self.last_learner_step = step
        print(f"[Actor {self.actor_id}] All actors ready — starting rollout.")
        return state_dict

    def send_batch(self, batch: RolloutBatch, episode_stats: list) -> bool:
        """
        Stamp, cache, then attempt to flush.
        Returns True if all cached batches were sent, False if an outage is active.
        collected_at is already set by default_factory at construction time in rollout.py.
        """
        batch.actor_id     = self.actor_id
        batch.learner_step = self.last_learner_step
        # collected_at already stamped at RolloutBatch construction in rollout.py

        self._cache.append((batch, episode_stats))  # always cache first

        if not self._connected:
            if time.monotonic() < self._outage_ends_at:
                # Outage window still active — stay disconnected
                print(f"[Actor {self.actor_id}] Outage active — "
                      f"{len(self._cache)} batch(es) cached.")
                return False
            # Outage window elapsed — reconnect and flush
            print(f"[Actor {self.actor_id}] Reconnecting PUSH socket...")
            self._connect_push()

        sent = self._flush_cache()

        if not self._connected:
            print(f"[Actor {self.actor_id}] Outage active — "
                  f"{len(self._cache)} batch(es) cached.")
            return False

        return True


    def recv_weights(self) -> tuple[str, object] | None:
        """
        Returns:
            ("weights", state_dict) -> normal weight update
            ("reset", state_dict)   -> learner-triggered reset + fresh weights
            ("shutdown", None)      -> learner requested clean termination
            None                    -> no message available right now
        """
        if self._sub.poll(0):
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
        """
        Returns:
            "updated"   -> normal weights applied
            "reset"     -> reset applied and weights reloaded
            "shutdown"  -> learner requested shutdown
            "none"      -> no message available
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
        """
        Test helper: kill the PUSH socket and set a timer.
        The actor loop continues collecting freely; send_batch checks
        the timer before attempting reconnect.
        """
        print(f"[Actor {self.actor_id}] ⚡ Simulating outage for {duration_s}s...")
        if self._push is not None:
            self._push.close(linger=0)
            self._push = None
        self._connected = False
        self._outage_ends_at = time.monotonic() + duration_s  # ← timer not sleep

    def apply_reset(self) -> None:
        """Clear stale local state after a learner-triggered reset."""
        self._cache.clear()
        if not self._connected:
            print(f"[Actor {self.actor_id}] Reset reconnecting PUSH socket...")
            self._connect_push()
        self._outage_ends_at = 0.0
        print(f"[Actor {self.actor_id}] Local cache cleared due to learner reset.")

    def close(self) -> None:
        if self._push is not None:
            self._push.close()
        self._sub.close()
        self._req.close()
        self._ctx.term()


class LearnerComms:
    def __init__(self, pull_addr: str, pub_addr: str, rep_addr: str,
                device: torch.device = torch.device("cpu"),
                buffer_size: int = 1,
                max_batches_per_actor: int = 1,
                num_actors: int = 1,
                staleness_threshold: float = float("inf"),
                enable_policy_reset: bool = True,
                reset_stale_after: int = 5):
        self.device = device
        self._buffer_size = buffer_size
        self._max_batches_per_actor = max_batches_per_actor
        self._num_actors = num_actors
        self._buffer: list[tuple[RolloutBatch, list]] = []
        self._ctx = zmq.Context()

        self._pull = self._ctx.socket(zmq.PULL)
        self._pull.bind(pull_addr)

        self._pub = self._ctx.socket(zmq.PUB)
        self._pub.bind(pub_addr)

        self._rep = self._ctx.socket(zmq.REP)
        self._rep.bind(rep_addr)

        self._stats = {          # for TensorBoard logging
            "batches_received": 0,
            "batches_rejected_stale": 0,
        }
        self._staleness_threshold = staleness_threshold

        self._enable_policy_reset = enable_policy_reset
        self._reset_stale_after = reset_stale_after
        self._actor_registry: dict[int, dict] = {}
        self._actors_pending_reset: set[int] = set()
    
    def serve_initial_weights(self, state_dict: dict, timeout_s: int = 60) -> None:
        """
        Phase 1: collect b"ready" from all buffer_size actors, ACK each with b"ack".
        Phase 2: broadcast initial weights via PUB so all actors start simultaneously.
        """
        serialised = serialise((state_dict, 0))
        deadline   = time.monotonic() + timeout_s
        connected  = 0

        # Phase 1 — handshake with each actor individually
        while connected < self._num_actors:
            if time.monotonic() > deadline:
                raise TimeoutError(
                    f"Only {connected}/{self._num_actors} actors connected within {timeout_s}s"
                )
            if self._rep.poll(timeout=1000):
                self._rep.recv()           # consume b"ready"
                self._rep.send(b"ack")     # acknowledge — "we see you, keep waiting"
                connected += 1
                print(f"[Learner] {connected}/{self._num_actors} actors ready.")

        # Phase 2 — all actors connected, broadcast weights simultaneously
        print("[Learner] All actors ready — broadcasting initial weights.")
        time.sleep(0.1)  # brief pause so all SUB sockets are listening before PUB fires
        self._pub.send_multipart([WEIGHTS_TOPIC, serialised])

    def recv_batch(self, writer=None, global_step: int = 0,
                partial_flush_timeout_s: float = 5.0) -> tuple[RolloutBatch, list]:
        """Block until buffer_size accepted batches arrive, or timeout with partial batch."""
        deadline = time.monotonic() + partial_flush_timeout_s

        while len(self._buffer) < self._buffer_size:
            # Check timeout — proceed with partial buffer if we have at least 1 batch
            time_remaining_ms = int((deadline - time.monotonic()) * 1000)
            if time_remaining_ms <= 0:
                if len(self._buffer) > 0:
                    print(f"[Learner] Partial flush: {len(self._buffer)}/{self._buffer_size} "
                        f"batches after {partial_flush_timeout_s:.1f}s timeout.")
                    break
                else:
                    # Nothing at all — reset deadline and keep waiting
                    deadline = time.monotonic() + partial_flush_timeout_s
                    continue

            if not self._pull.poll(timeout=min(time_remaining_ms, 1_000)):
                continue

            batch, episode_stats = deserialise(self._pull.recv())
            batch = RolloutBatch(
                **{k: v.to(self.device) if isinstance(v, torch.Tensor) else v
                for k, v in batch.__dict__.items()}
            )

            accepted_before = len(self._buffer)
            self._append_to_buffer(batch, episode_stats)
            batch_was_accepted = len(self._buffer) > accepted_before

            # Starvation watchdog
            if self._stats["batches_received"] % 100 == 0 and len(self._buffer) == 0:
                print(f"[Learner] ⚠ Waiting for accepted batches — "
                    f"{self._stats['batches_rejected_stale']} rejected so far "
                    f"(τ={self._staleness_threshold:.1f}s)")

            if writer and self._stats["batches_received"] % 20 == 0:
                rejection_rate = (
                    self._stats["batches_rejected_stale"] /
                    max(1, self._stats["batches_received"])
                )
                writer.add_scalar("staleness/rejection_rate", rejection_rate, global_step)
                if batch_was_accepted:
                    writer.add_scalar(
                        "staleness/batch_age_s",
                        time.monotonic() - batch.collected_at,
                        global_step
                    )

        ready, self._buffer = self._buffer, []
        merged_batch = _concat_batches([b for b, _ in ready])
        merged_stats = [s for _, ss in ready for s in ss]
        return merged_batch, merged_stats
    
    def _append_to_buffer(self, batch: RolloutBatch, episode_stats: list) -> None:
        """Single entry point for all buffer writes. Enforces staleness + per-actor cap."""
        self._stats["batches_received"] += 1

        self._ensure_actor_record(batch.actor_id)
        record = self._actor_registry[batch.actor_id]

        now = time.monotonic()
        age = now - batch.collected_at
        record["last_seen_time"] = now
        record["last_batch_age"] = age
        record["last_learner_step_seen"] = batch.learner_step

        # ── Staleness filter ─────────────────────────────────────────────────
        if age > self._staleness_threshold:
            self._stats["batches_rejected_stale"] += 1
            record["consecutive_stale_batches"] += 1

            print(
                f"[Learner] Dropped stale batch from actor {batch.actor_id}: "
                f"age={age:.2f}s > τ={self._staleness_threshold:.2f}s"
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
        # ─────────────────────────────────────────────────────────────────────

        record["consecutive_stale_batches"] = 0

        # Per-actor cap (unchanged from Phase 3)
        actor_count = sum(1 for b, _ in self._buffer if b.actor_id == batch.actor_id)
        if actor_count >= self._max_batches_per_actor:
            for i, (b, _) in enumerate(self._buffer):
                if b.actor_id == batch.actor_id:
                    self._buffer.pop(i)
                    break

        self._buffer.append((batch, episode_stats))

    def broadcast_weights(self, agent, step: int) -> None:
        self._pub.send_multipart([WEIGHTS_TOPIC, serialise((agent.actor.state_dict(), step))])

    def broadcast_shutdown(self) -> None:
        """Broadcast a clean shutdown signal to all subscribed actors."""
        self._pub.send_multipart([SHUTDOWN_TOPIC, b""])
        print("[Learner] Shutdown signal broadcast.")

    def broadcast_reset(self, actor_id: int, state_dict: dict, step: int) -> None:
        payload = serialise((actor_id, state_dict, step))
        self._pub.send_multipart([RESET_TOPIC, payload])
        self._actor_registry[actor_id]["reset_count"] += 1
        self._actor_registry[actor_id]["consecutive_stale_batches"] = 0
        self._actors_pending_reset.discard(actor_id)
        print(f"[Learner] Broadcast reset for actor {actor_id}.")

    def flush_pending_resets(self, agent, step: int) -> list[int]:
        reset_ids = list(self._actors_pending_reset)
        state_dict = agent.actor.state_dict()
        for actor_id in reset_ids:
            self.broadcast_reset(actor_id, state_dict, step)
        return reset_ids

    def _ensure_actor_record(self, actor_id: int) -> None:
        if actor_id not in self._actor_registry:
            self._actor_registry[actor_id] = {
                "last_seen_time": 0.0,
                "last_batch_age": 0.0,
                "last_learner_step_seen": 0,
                "consecutive_stale_batches": 0,
                "reset_count": 0,
            }

    def close(self) -> None:
        self._pull.close()
        self._pub.close()
        self._rep.close()
        self._ctx.term()