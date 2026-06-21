#!/usr/bin/env bash
set -euo pipefail

ENV_ID="CartPole-v1"
TOTAL_TIMESTEPS=500000
NUM_STEPS=128
NUM_MINIBATCHES=4
NUM_ACTORS=4

# Experiment 2 grid
LATENCIES_MS=(10)                # low and high latency regimes
TAUS=("0.1" "2.0")             # staleness thresholds (seconds)
SEEDS=(1 2 3)                        # random seeds

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
  local latency_ms="$1"
  local tau="$2"
  local seed="$3"

  echo "=== Running Exp2: latency=${latency_ms}ms, tau=${tau}, seed=${seed} ==="
  kill_existing

  local exp_name="exp2_latency${latency_ms}ms_tau${tau}_seed${seed}"

  # Start learner
  uv run ../ppo_learner.py \
    --env-id "${ENV_ID}" \
    --total-timesteps "${TOTAL_TIMESTEPS}" \
    --num-steps "${NUM_STEPS}" \
    --num-minibatches "${NUM_MINIBATCHES}" \
    --seed "${seed}" \
    --num-actors "${NUM_ACTORS}" \
    --staleness-threshold "${tau}" \
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

  # Start actors
  for actor_id in $(seq 0 $((NUM_ACTORS - 1))); do
    uv run ../ppo_actor.py \
      --env-id "${ENV_ID}" \
      --total-timesteps "${TOTAL_TIMESTEPS}" \
      --num-steps "${NUM_STEPS}" \
      --num-minibatches "${NUM_MINIBATCHES}" \
      --seed "${seed}" \
      --num-actors "${NUM_ACTORS}" \
      --staleness-threshold "${tau}" \
      --weighting-strategy uniform \
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
      --latency-ms "${latency_ms}" &
  done

  # Wait for learner to finish
  wait "${learner_pid}"
  echo "=== Completed Exp2: latency=${latency_ms}ms, tau=${tau}, seed=${seed} ==="
}

main() {
  for latency_ms in "${LATENCIES_MS[@]}"; do
    for tau in "${TAUS[@]}"; do
      for seed in "${SEEDS[@]}"; do
        run_config "${latency_ms}" "${tau}" "${seed}"
      done
    done
  done

  echo "All Exp2 runs completed."
}

main 