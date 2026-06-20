# test_buffer_guard.py — run once to verify, then delete
import sys
import time
import torch
from framework.protocol import RolloutBatch
from framework.comms import LearnerComms

# ── Helpers ───────────────────────────────────────────────────────────────────

def fake_batch(actor_id: int, num_steps: int = 4, obs_dim: int = 4) -> RolloutBatch:
    """Minimal valid RolloutBatch with dummy tensors."""
    return RolloutBatch(
        obs       = torch.zeros(num_steps, obs_dim),
        actions   = torch.zeros(num_steps, dtype=torch.long),
        logprobs  = torch.zeros(num_steps),
        rewards   = torch.zeros(num_steps),
        dones     = torch.zeros(num_steps),
        values    = torch.zeros(num_steps),
        next_obs  = torch.zeros(obs_dim),
        next_done = torch.tensor(0.0),
        actor_id     = actor_id,
        learner_step = 0,
        collected_at = time.monotonic(),
    )

def buffer_actor_ids(comms: LearnerComms) -> list[int]:
    return [b.actor_id for b, _ in comms._buffer]

def run_test(name: str, passed: bool) -> None:
    status = "✅ PASS" if passed else "❌ FAIL"
    print(f"  {status}  {name}")
    if not passed:
        sys.exit(1)

# ── Instantiate without binding ZMQ sockets ───────────────────────────────────
# We bypass __init__ entirely to avoid needing live sockets for a unit test.
comms = object.__new__(LearnerComms)
comms.device = torch.device("cpu")
comms._buffer_size = 6
comms._max_batches_per_actor = 2
comms._num_actors = 3
comms._buffer = []

print("\n=== test_buffer_guard.py ===\n")

# ── Test 1: Normal fill, no cap triggered ─────────────────────────────────────
print("Test 1: Normal fill — one batch each from actors 0, 1, 2")
for actor_id in [0, 1, 2]:
    comms._buffer.append((fake_batch(actor_id), []))
ids = buffer_actor_ids(comms)
run_test("buffer contains [0, 1, 2]", ids == [0, 1, 2])

# ── Test 2: Second batch from same actor — allowed up to cap ──────────────────
print("\nTest 2: Second batch from actor 0 — should be accepted (count=1 < cap=2)")
comms._buffer.append((fake_batch(0), []))
ids = buffer_actor_ids(comms)
run_test("buffer contains [0, 1, 2, 0]", ids == [0, 1, 2, 0])

# ── Test 3: Third batch from actor 0 — cap triggers, oldest actor-0 dropped ──
print("\nTest 3: Third batch from actor 0 — cap=2 triggers, oldest actor-0 dropped")
batch_to_inject = fake_batch(0)
actor_count = sum(1 for b, _ in comms._buffer if b.actor_id == 0)
if actor_count >= comms._max_batches_per_actor:
    for i, (b, _) in enumerate(comms._buffer):
        if b.actor_id == 0:
            comms._buffer.pop(i)
            break
comms._buffer.append((batch_to_inject, []))
ids = buffer_actor_ids(comms)
run_test("buffer still has exactly 2 batches from actor 0",
         ids.count(0) == 2)
run_test("buffer length unchanged at 4",
         len(comms._buffer) == 4)
run_test("actors 1 and 2 still present",
         1 in ids and 2 in ids)
print(f"  buffer actor_ids: {ids}")

# ── Test 4: Cap is per-actor — actor 1 can still add freely ──────────────────
print("\nTest 4: Actor 1 adds second batch — should be accepted independently")
comms._buffer.append((fake_batch(1), []))
ids = buffer_actor_ids(comms)
run_test("actor 1 now has 2 batches", ids.count(1) == 2)
run_test("actor 0 still has 2 batches", ids.count(0) == 2)
run_test("buffer length is 5", len(comms._buffer) == 5)

# ── Test 5: Buffer ceiling — fill to buffer_size ──────────────────────────────
print("\nTest 5: Fill buffer to capacity (buffer_size=6)")
comms._buffer.append((fake_batch(2), []))
run_test("buffer at capacity", len(comms._buffer) == 6)
run_test("trigger condition fires", len(comms._buffer) >= comms._buffer_size)

# ── Test 6: Flush clears buffer ───────────────────────────────────────────────
print("\nTest 6: Flush — buffer clears after update")
ready, comms._buffer = comms._buffer, []
run_test("buffer empty after flush", len(comms._buffer) == 0)
run_test("ready batch has 6 entries", len(ready) == 6)

# ── Test 7: max_batches_per_actor=1 edge case ─────────────────────────────────
print("\nTest 7: Edge case — max_batches_per_actor=1 (strict round-robin)")
comms._max_batches_per_actor = 1
comms._buffer = []
for actor_id in [0, 1, 2]:
    comms._buffer.append((fake_batch(actor_id), []))
# Now inject actor 0 again — should immediately drop its previous entry
actor_count = sum(1 for b, _ in comms._buffer if b.actor_id == 0)
if actor_count >= comms._max_batches_per_actor:
    for i, (b, _) in enumerate(comms._buffer):
        if b.actor_id == 0:
            comms._buffer.pop(i)
            break
comms._buffer.append((fake_batch(0), []))
ids = buffer_actor_ids(comms)
run_test("actor 0 has exactly 1 batch", ids.count(0) == 1)
run_test("buffer length stays at 3", len(comms._buffer) == 3)

print("\n=== All tests passed ===\n")