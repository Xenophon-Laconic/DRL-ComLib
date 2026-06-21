#!/usr/bin/env bash
set -euo pipefail

ENV_ID="CartPole-v1"
TOTAL_TIMESTEPS=500000
NUM_STEPS=128
NUM_MINIBATCHES=4

# Shared ZMQ addresses (single learner, multiple actors)
PUSH_ADDR="tcp://localhost:5555"
PULL_ADDR="tcp://localhost:5555"
PUB_ADDR="tcp://localhost:5556"
SUB_ADDR="tcp://localhost:5556"
REP_ADDR="tcp://localhost:5557"
REQ_ADDR="tcp://localhost:5557"

kill_existing() {
  echo "Killing existing learner/actor processes (if any)..."
  pkill -f "ppo_learner.py" || true
  pkill -f "ppo_actor.py" || true
  pkill -f "uv run ppo_learner.py" || true
  pkill -f "uv run ppo_actor.py" || true
}

run_one_actor() {
  local seed="$1"
  echo "=== Running 1-actor experiment, seed ${seed} ==="
  kill_existing

  local num_actors=1
  local exp_name="exp1_1actor_seed${seed}"

  # Start learner
  uv run ../ppo_learner.py \
    --env-id "${ENV_ID}" \
    --total-timesteps "${TOTAL_TIMESTEPS}" \
    --num-steps "${NUM_STEPS}" \
    --num-minibatches "${NUM_MINIBATCHES}" \
    --seed "${seed}" \
    --num-actors "${num_actors}" \
    --staleness-threshold inf \
    --weighting-strategy uniform \
    --no-enable-policy-reset \
    --push-addr "${PUSH_ADDR}" \
    --pull-addr "${PULL_ADDR}" \
    --pub-addr "${PUB_ADDR}" \
    --sub-addr "${SUB_ADDR}" \
    --rep-addr "${REP_ADDR}" \
    --req-addr "${REQ_ADDR}" \
    --exp-name "${exp_name}" &
  local learner_pid=$!

  # Start single actor
  uv run ../ppo_actor.py \
    --env-id "${ENV_ID}" \
    --total-timesteps "${TOTAL_TIMESTEPS}" \
    --num-steps "${NUM_STEPS}" \
    --num-minibatches "${NUM_MINIBATCHES}" \
    --seed "${seed}" \
    --num-actors "${num_actors}" \
    --staleness-threshold inf \
    --weighting-strategy uniform \
    --no-enable-policy-reset \
    --push-addr "${PUSH_ADDR}" \
    --pull-addr "${PULL_ADDR}" \
    --pub-addr "${PUB_ADDR}" \
    --sub-addr "${SUB_ADDR}" \
    --rep-addr "${REP_ADDR}" \
    --req-addr "${REQ_ADDR}" \
    --exp-name "${exp_name}" \
    --actor-id 0 &

  wait "${learner_pid}"
}

run_four_actors() {
  local seed="$1"
  echo "=== Running 4-actor experiment, seed ${seed} ==="
  kill_existing

  local num_actors=4
  local exp_name="exp1_4actors_seed${seed}"

  # Start learner
  uv run ../ppo_learner.py \
    --env-id "${ENV_ID}" \
    --total-timesteps "${TOTAL_TIMESTEPS}" \
    --num-steps "${NUM_STEPS}" \
    --num-minibatches "${NUM_MINIBATCHES}" \
    --seed "${seed}" \
    --num-actors "${num_actors}" \
    --staleness-threshold inf \
    --weighting-strategy uniform \
    --no-enable-policy-reset \
    --push-addr "${PUSH_ADDR}" \
    --pull-addr "${PULL_ADDR}" \
    --pub-addr "${PUB_ADDR}" \
    --sub-addr "${SUB_ADDR}" \
    --rep-addr "${REP_ADDR}" \
    --req-addr "${REQ_ADDR}" \
    --exp-name "${exp_name}" &
  local learner_pid=$!

  # Start 4 actors
  for actor_id in 0 1 2 3; do
    uv run ../ppo_actor.py \
      --env-id "${ENV_ID}" \
      --total-timesteps "${TOTAL_TIMESTEPS}" \
      --num-steps "${NUM_STEPS}" \
      --num-minibatches "${NUM_MINIBATCHES}" \
      --seed "${seed}" \
      --num-actors "${num_actors}" \
      --staleness-threshold inf \
      --weighting-strategy uniform \
      --no-enable-policy-reset \
      --push-addr "${PUSH_ADDR}" \
      --pull-addr "${PULL_ADDR}" \
      --pub-addr "${PUB_ADDR}" \
      --sub-addr "${SUB_ADDR}" \
      --rep-addr "${REP_ADDR}" \
      --req-addr "${REQ_ADDR}" \
      --exp-name "${exp_name}" \
      --actor-id "${actor_id}" &
  done

  wait "${learner_pid}"
}

main() {
  # seeds 1..5
  for seed in 1 2 3; do
    run_one_actor "${seed}"
    run_four_actors "${seed}"
  done

  echo "All runs completed."
}

main "$@"