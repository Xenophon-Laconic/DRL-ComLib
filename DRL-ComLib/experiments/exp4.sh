#!/usr/bin/env bash
set -euo pipefail

ENV_ID="CartPole-v1"
TOTAL_TIMESTEPS=500000
NUM_STEPS=128
NUM_MINIBATCHES=4
NUM_ACTORS=4

# Heterogeneous latency: fast vs slow actors
FAST_LATENCY_MS=0.0         # baseline latency
SLOW_LATENCY_MS=100.0       # extra delay per rollout for slow actors

# Staleness threshold and weighting from Exp 2 stress config
TAU="5.0"
WEIGHTING_STRATEGY="uniform"

# Seeds
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
  local reset_mode="$1"   # "reset_off" or "reset_on"
  local seed="$2"

  echo "=== Running Exp4: mode=${reset_mode}, seed=${seed} ==="
  kill_existing

  local exp_name="exp4_${reset_mode}_lathet_tau${TAU}_seed${seed}"

  # Learner: reset policy on or off
  local reset_flag=()
  if [[ "${reset_mode}" == "reset_off" ]]; then
    reset_flag=(--no-enable-policy-reset)
  else
    # enable_policy_reset = True (default), so no flag needed
    reset_flag=()
  fi

  uv run ../ppo_learner.py \
    --env-id "${ENV_ID}" \
    --total-timesteps "${TOTAL_TIMESTEPS}" \
    --num-steps "${NUM_STEPS}" \
    --num-minibatches "${NUM_MINIBATCHES}" \
    --seed "${seed}" \
    --num-actors "${NUM_ACTORS}" \
    --staleness-threshold "${TAU}" \
    --weighting-strategy "${WEIGHTING_STRATEGY}" \
    "${reset_flag[@]}" \
    --push-addr "${PUSH_ADDR}" \
    --pull-addr "${PULL_ADDR}" \
    --pub-addr "${PUB_ADDR}" \
    --sub-addr "${SUB_ADDR}" \
    --rep-addr "${REP_ADDR}" \
    --req-addr "${REQ_ADDR}" \
    --exp-name "${exp_name}" &
  local learner_pid=$!

  # Start heterogeneous actors
  for actor_id in $(seq 0 $((NUM_ACTORS - 1))); do
    # Assign latency by actor id: 0,1 fast; 2,3 slow
    local latency_ms="${FAST_LATENCY_MS}"
    if [[ "${actor_id}" -ge 2 ]]; then
      latency_ms="${SLOW_LATENCY_MS}"
    fi

    uv run ../ppo_actor.py \
      --env-id "${ENV_ID}" \
      --total-timesteps "${TOTAL_TIMESTEPS}" \
      --num-steps "${NUM_STEPS}" \
      --num-minibatches "${NUM_MINIBATCHES}" \
      --seed "${seed}" \
      --num-actors "${NUM_ACTORS}" \
      --staleness-threshold "${TAU}" \
      --weighting-strategy "${WEIGHTING_STRATEGY}" \
      "${reset_flag[@]}" \
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
  echo "=== Completed Exp4: mode=${reset_mode}, seed=${seed} ==="
}

main() {
  for reset_mode in reset_on reset_off; do
    for seed in "${SEEDS[@]}"; do
      run_config "${reset_mode}" "${seed}"
    done
  done

  echo "All Exp4 runs completed."
}

main "$@"