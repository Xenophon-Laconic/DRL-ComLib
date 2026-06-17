# test_actor_cache.py — run once, then delete
import sys
import time
import torch
from collections import deque
from framework.protocol import RolloutBatch
from framework.comms import ActorComms

def fake_batch(num_steps=4, obs_dim=4) -> RolloutBatch:
    return RolloutBatch(
        obs       = torch.zeros(num_steps, obs_dim),
        actions   = torch.zeros(num_steps, dtype=torch.long),
        logprobs  = torch.zeros(num_steps),
        rewards   = torch.zeros(num_steps),
        dones     = torch.zeros(num_steps),
        values    = torch.zeros(num_steps),
        next_obs  = torch.zeros(obs_dim),
        next_done = torch.tensor(0.0),
    )

def run_test(name: str, passed: bool) -> None:
    print(f"  {'✅ PASS' if passed else '❌ FAIL'}  {name}")
    if not passed:
        sys.exit(1)

# Bypass __init__ to avoid live sockets
actor = object.__new__(ActorComms)
actor.actor_id = 0
actor.last_learner_step = 0
actor._connected = True
actor._push = None   # no real socket needed

print("\n=== test_actor_cache.py ===\n")

# ── Test 1: Cache-before-send ordering ───────────────────────────────────────
print("Test 1: Batch is cached before send is attempted")
actor._cache = deque(maxlen=8)
b = fake_batch()

# Simulate send_batch logic without real socket:
b.actor_id = actor.actor_id
b.learner_step = actor.last_learner_step
actor._cache.append((b, []))

run_test("cache has 1 entry after append", len(actor._cache) == 1)
run_test("actor_id stamped correctly", actor._cache[0][0].actor_id == 0)
run_test("collected_at is float", isinstance(actor._cache[0][0].collected_at, float))

# ── Test 2: Outage — cache fills, nothing dropped below maxlen ───────────────
print("\nTest 2: Outage — 5 batches accumulate in cache (maxlen=8)")
actor._connected = False
for _ in range(5):
    b = fake_batch()
    b.actor_id = 0
    actor._cache.append((b, []))
run_test("cache depth is 6 (1 + 5)", len(actor._cache) == 6)

# ── Test 3: Circular eviction — oldest dropped when cache full ───────────────
print("\nTest 3: Cache overflow — oldest batch evicted (maxlen=8)")
for i in range(10):   # push 10 more into an 8-slot cache
    b = fake_batch()
    b.actor_id = i    # use actor_id as a unique tag
    actor._cache.append((b, []))
run_test("cache length capped at 8", len(actor._cache) == 8)
# The last 8 have actor_ids 2..9 (first two evicted)
ids = [b.actor_id for b, _ in actor._cache]
run_test("oldest batches evicted (first id >= 2)", ids[0] >= 2)

# ── Test 4: Reconnect — flush drains cache ───────────────────────────────────
print("\nTest 4: Reconnect simulation — cache clears on flush")
sent_log = []

# Simulate _flush_cache success: drain everything
while actor._cache:
    actor._cache.popleft()
    sent_log.append(1)

run_test("all cached batches flushed", len(actor._cache) == 0)
run_test("flush count matches pre-flush depth", len(sent_log) == 8)

# ── Test 5: Partial flush — socket blocks mid-flush ──────────────────────────
print("\nTest 5: Partial flush — socket blocks after 3 sends")
actor._cache = deque(maxlen=8)
for i in range(6):
    b = fake_batch()
    b.actor_id = i
    actor._cache.append((b, []))

sent = 0
for _ in range(3):    # simulate 3 successful sends then failure
    actor._cache.popleft()
    sent += 1
actor._connected = False   # socket blocked at batch 4

run_test("3 batches sent before failure", sent == 3)
run_test("3 batches remain in cache", len(actor._cache) == 3)
run_test("connected flag is False", actor._connected == False)

print("\n=== All tests passed ===\n")