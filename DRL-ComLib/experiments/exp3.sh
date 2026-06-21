#!/usr/bin/env bash
set -euo pipefail

ENV_ID="CartPole-v1"
TOTAL_TIMESTEPS=500000
NUM_STEPS=128
NUM_MINIBATCHES=4
NUM_ACTORS=4


# Fixed stress configuration (from Exp 2)
LATENCY_MS=100.0          # high synthetic latency per rollout
TAU="2.0"                 # moderate staleness threshold (seconds)

# Weighting strategies to compare
WEIGHTING_STRATEGIES=("latency" "is" "uniform")

# Random seeds
SEEDS=(1 2 3)

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

run_config() {
  local weighting="$1"
  local seed="$2"

  echo "=== Running Exp3: weighting=${weighting}, seed=${seed} ==="
  kill_existing

  local exp_name="exp3_weight_${weighting}_lat${LATENCY_MS}ms_tau${TAU}_seed${seed}"

  # Start learner
  uv run ../ppo_learner.py \
    --env-id "${ENV_ID}" \
    --total-timesteps "${TOTAL_TIMESTEPS}" \
    --num-steps "${NUM_STEPS}" \
    --num-minibatches "${NUM_MINIBATCHES}" \
    --seed "${seed}" \
    --num-actors "${NUM_ACTORS}" \
    --staleness-threshold "${TAU}" \
    --weighting-strategy "${weighting}" \
    --no-enable-policy-reset \
    --push-addr "${PUSH_ADDR}" \
    --pull-addr "${PULL_ADDR}" \
    --pub-addr "${PUB_ADDR}" \
    --sub-addr "${SUB_ADDR}" \
    --rep-addr "${REP_ADDR}" \
    --req-addr "${REQ_ADDR}" \
    --exp-name "${exp_name}" &
  local learner_pid=$!

  # Start actors
  for actor_id in $(seq 0 $((NUM_ACTORS - 1))); do
    uv run ../ppo_actor.py \
      --env-id "${ENV_ID}" \
      --total-timesteps "${TOTAL_TIMESTEPS}" \
      --num-steps "${NUM_STEPS}" \
      --num-minibatches "${NUM_MINIBATCHES}" \
      --seed "${seed}" \
      --num-actors "${NUM_ACTORS}" \
      --staleness-threshold "${TAU}" \
      --weighting-strategy "${weighting}" \
      --no-enable-policy-reset \
      --push-addr "${PUSH_ADDR}" \
      --pull-addr "${PULL_ADDR}" \
      --pub-addr "${PUB_ADDR}" \
      --sub-addr "${SUB_ADDR}" \
      --rep-addr "${REP_ADDR}" \
      --req-addr "${REQ_ADDR}" \
      --exp-name "${exp_name}" \
      --actor-id "${actor_id}" \
      --simulate-latency \
      --latency-ms "${LATENCY_MS}" &
  done

  # Wait for learner to finish
  wait "${learner_pid}"
  echo "=== Completed Exp3: weighting=${weighting}, seed=${seed} ==="
}

main() {
  for weighting in "${WEIGHTING_STRATEGIES[@]}"; do
    for seed in "${SEEDS[@]}"; do
      run_config "${weighting}" "${seed}"
    done
  done

  echo "All Exp3 runs completed."
}

main "$@"