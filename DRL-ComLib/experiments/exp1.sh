#!/usr/bin/env bash
set -euo pipefail

# Number of seeds
SEEDS=(1 2 3 4 5)

# Common hyperparameters
ENV_ID="CartPole-v1"
TOTAL_TIMESTEPS=500000
NUM_STEPS=128
NUM_MINIBATCHES=4

# Addresses
PUSH_ADDR="tcp://localhost:5555"
PULL_ADDR="tcp://localhost:5555"
PUB_ADDR="tcp://localhost:5556"
SUB_ADDR="tcp://localhost:5556"
REP_ADDR="tcp://localhost:5557"
REQ_ADDR="tcp://localhost:5557"

run_sync() {
  local seed=$1

  echo "=== Sync run, seed=${seed} ==="

  # Start learner
  uv run ../ppo_learner.py \
    --env-id "${ENV_ID}" \
    --total-timesteps ${TOTAL_TIMESTEPS} \
    --num-steps ${NUM_STEPS} \
    --num-minibatches ${NUM_MINIBATCHES} \
    --seed ${seed} \
    --num-actors 1 \
    --learner-buffer-size 1 \
    --max-batches-per-actor 1 \
    --actor-cache-size 0 \
    --staleness-threshold inf \
    --weighting-strategy uniform \
    --no-enable-policy-reset \
    --simulate-outage-at 0 \
    --outage-duration 0.0 \
    --push-addr "${PUSH_ADDR}" \
    --pull-addr "${PULL_ADDR}" \
    --pub-addr "${PUB_ADDR}" \
    --sub-addr "${SUB_ADDR}" \
    --rep-addr "${REP_ADDR}" \
    --req-addr "${REQ_ADDR}" \
    --exp-name "exp1_sync_seed${seed}" \
    &

  LEARNER_PID=$!

  # Give learner time to bind sockets
  sleep 1

  # Start actor
  uv run ../ppo_actor.py \
    --env-id "${ENV_ID}" \
    --total-timesteps ${TOTAL_TIMESTEPS} \
    --num-steps ${NUM_STEPS} \
    --num-minibatches ${NUM_MINIBATCHES} \
    --seed ${seed} \
    --num-actors 1 \
    --learner-buffer-size 1 \
    --max-batches-per-actor 1 \
    --actor-cache-size 0 \
    --staleness-threshold inf \
    --weighting-strategy uniform \
    --no-enable-policy-reset \
    --simulate-outage-at 0 \
    --outage-duration 0.0 \
    --push-addr "${PUSH_ADDR}" \
    --pull-addr "${PULL_ADDR}" \
    --pub-addr "${PUB_ADDR}" \
    --sub-addr "${SUB_ADDR}" \
    --rep-addr "${REP_ADDR}" \
    --req-addr "${REQ_ADDR}" \
    --exp-name "exp1_sync_seed${seed}" \
    --actor-id 0

  # Wait for learner to finish
  wait ${LEARNER_PID}
}

run_async() {
  local seed=$1

  echo "=== Async run, seed=${seed} ==="

  uv run ../ppo_learner.py \
    --env-id "${ENV_ID}" \
    --total-timesteps ${TOTAL_TIMESTEPS} \
    --num-steps ${NUM_STEPS} \
    --num-minibatches ${NUM_MINIBATCHES} \
    --seed ${seed} \
    --num-actors 1 \
    --learner-buffer-size 4 \
    --max-batches-per-actor 4 \
    --actor-cache-size 0 \
    --staleness-threshold inf \
    --weighting-strategy uniform \
    --no-enable-policy-reset \
    --simulate-outage-at 0 \
    --outage-duration 0.0 \
    --push-addr "${PUSH_ADDR}" \
    --pull-addr "${PULL_ADDR}" \
    --pub-addr "${PUB_ADDR}" \
    --sub-addr "${SUB_ADDR}" \
    --rep-addr "${REP_ADDR}" \
    --req-addr "${REQ_ADDR}" \
    --exp-name "exp1_async_seed${seed}" \
    &

  LEARNER_PID=$!
  sleep 1

  uv run ../ppo_actor.py \
    --env-id "${ENV_ID}" \
    --total-timesteps ${TOTAL_TIMESTEPS} \
    --num-steps ${NUM_STEPS} \
    --num-minibatches ${NUM_MINIBATCHES} \
    --seed ${seed} \
    --num-actors 1 \
    --learner-buffer-size 4 \
    --max-batches-per-actor 4 \
    --actor-cache-size 0 \
    --staleness-threshold inf \
    --weighting-strategy uniform \
    --no-enable-policy-reset \
    --simulate-outage-at 0 \
    --outage-duration 0.0 \
    --push-addr "${PUSH_ADDR}" \
    --pull-addr "${PULL_ADDR}" \
    --pub-addr "${PUB_ADDR}" \
    --sub-addr "${SUB_ADDR}" \
    --rep-addr "${REP_ADDR}" \
    --req-addr "${REQ_ADDR}" \
    --exp-name "exp1_async_seed${seed}" \
    --actor-id 0

  wait ${LEARNER_PID}
}

# Main loop over seeds
for s in "${SEEDS[@]}"; do
  run_sync "${s}"
  run_async "${s}"
done