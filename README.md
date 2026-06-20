# Distributed PPO over ZeroMQ

A compact distributed Proximal Policy Optimization (PPO) implementation with separate actor and learner processes, ZeroMQ-based communication, rollout staleness handling, optional experience weighting, and learner-triggered actor resets.[1][2][3][4]

The repository is organized around a simple actor-learner split: actors collect rollouts with an actor-only policy network, the learner owns the full actor-critic model and optimization loop, and `framework.comms` moves weights, batches, resets, and shutdown messages between them.[2][5][3]

## Overview

The actor process builds an environment, requests initial weights from the learner, collects rollouts, reacts to reset or shutdown control messages, and pushes rollout batches back over ZeroMQ.[2] The learner waits for actor handshakes, receives and merges accepted batches, computes advantages and optional experience weights, performs PPO updates, then broadcasts fresh policy weights back to the actors.[3][4]

Rollouts are represented as `RolloutBatch` objects that carry observations, actions, log-probabilities, rewards, dones, bootstrap state, and comms metadata such as `learner_step`, `actor_id`, and `collected_at` for staleness-aware training and diagnostics.[4][6]

## Features

- Actor-learner architecture with separate `ppo_actor.py` and `ppo_learner.py` entrypoints.[2][3]
- ZeroMQ transport using PUSH/PULL for rollout upload, PUB/SUB for weights and control messages, and REQ/REP for initial synchronization.[2][3]
- Configurable learner buffering with `learner_buffer_size`, per-actor caps via `max_batches_per_actor`, and partial flush timeout support.[1][3]
- Staleness-aware training controls, including `staleness_threshold`, rejection accounting, and learner-triggered actor resets after repeated stale batches.[1][3]
- Optional experience weighting strategies: `uniform`, `latency`, and `is` importance-style weighting.[1][4]
- TensorBoard logging for training losses, SPS, rollout stats, comms timing, reset counts, and weight diagnostics.[6][3]
- Outage simulation hooks on the actor side for resilience testing.[1][2]

## Repository layout

```text
.
├── args.py
├── logging_utils.py
├── models.py
├── ppo_actor.py
├── ppo_learner.py
├── rollout.py
├── training.py
└── framework/
    ├── comms.py
    └── protocol.py
```

- `args.py` defines the experiment, PPO, communication, staleness, reset, and outage-simulation configuration, plus runtime-derived values such as `batch_size`, `minibatch_size`, and `num_iterations`.[1]
- `models.py` defines a separate actor network, critic network, and learner-side `Agent` container that owns both modules.[5]
- `rollout.py` collects one rollout window from an environment and packages it as a `RolloutBatch` together with episodic statistics.[7]
- `training.py` computes advantages, derives optional sample weights, and applies PPO updates using the learner-side critic and optimizer.[4]
- `logging_utils.py` centralizes TensorBoard logging for training, infrastructure, comms, resets, and weighting diagnostics.[6]
- `ppo_actor.py` and `ppo_learner.py` are the main process entrypoints.[2][3]

## Process model

### Actor

The actor creates an environment, instantiates an actor-only policy, requests initial weights through the REQ/REP handshake, and begins collecting rollouts once the learner broadcasts the initial policy.[2] During training, it checks for learner control messages before and after sending a batch so it can react promptly to weight updates, targeted resets, or shutdown.[2]

### Learner

The learner instantiates the full actor-critic `Agent`, serves the initial weights after all actors have connected, receives batches through `LearnerComms.recv_batch`, computes advantages with the learner critic, then runs PPO updates and broadcasts new actor weights.[3][4][5] It also tracks staleness and can schedule actor resets after repeated stale submissions when reset handling is enabled.[1][3]

## Training flow

1. Start the learner; it waits for all actors to complete the ready/ack handshake before broadcasting initial weights.[3]
2. Start one or more actors; each actor requests the initial policy, collects `num_steps` environment interactions, and pushes a rollout batch to the learner.[2][1]
3. The learner accumulates accepted batches up to `learner_buffer_size` or flushes partially after `partial_flush_timeout_s`.[1][3]
4. The learner computes advantages and returns using the learner-side critic, then computes optional weighting based on rollout latency or log-probability ratios.[4]
5. PPO optimization runs on the merged batch, after which the learner broadcasts fresh policy weights and any pending targeted resets.[3][4]

## Configuration

The core PPO configuration includes `env_id`, `total_timesteps`, `learning_rate`, `num_steps`, `num_minibatches`, `update_epochs`, `gamma`, `gae_lambda`, clipping settings, entropy/value coefficients, and optional `target_kl` stopping.[1] Distributed-system configuration includes socket addresses, `num_actors`, learner buffering, actor cache depth, staleness threshold, partial flush timeout, reset policy, and outage simulation controls.[1]

Some runtime values are derived automatically. If `learner_buffer_size == 0`, it defaults to `2 * num_actors`; if `max_batches_per_actor == 0`, it defaults to `max(1, learner_buffer_size // num_actors)`; and `batch_size`, `minibatch_size`, and `num_iterations` are computed from the rollout and buffer settings.[1]

## Running the code

### 1. Install dependencies

At minimum, the code depends on PyTorch, Gymnasium, Tyro, NumPy, TensorBoard, and PyZMQ based on the imported modules in the training, rollout, logging, and communication files.[6][1][2][3][4]

A minimal environment might look like this:

```bash
pip install torch gymnasium pyzmq tyro numpy tensorboard
```

### 2. Start the learner

```bash
python ppo_learner.py --env-id CartPole-v1 --num-actors 2
```

The learner creates the optimizer and writer, waits for actor handshake completion, and then starts the distributed training loop.[3]

### 3. Start actors

In separate terminals, launch one actor process per actor id:

```bash
python ppo_actor.py --actor-id 0 --env-id CartPole-v1 --num-actors 2
python ppo_actor.py --actor-id 1 --env-id CartPole-v1 --num-actors 2
```

Each actor connects to the learner, requests initial weights, collects rollouts, and sends batches while continuing to poll for reset, weight, and shutdown broadcasts.[2]

## Monitoring

The learner creates a TensorBoard `SummaryWriter` under `runs/<run_name>` and logs hyperparameters, training losses, episodic statistics, learning rate, SPS, comms latency, learner-step gap, reset counts, and experience-weight diagnostics.[6][3] To inspect logs:

```bash
tensorboard --logdir runs
```

## Staleness and resets

Each batch carries a collection timestamp and learner-step metadata so the learner can measure transit age and training delay.[6][4] If a batch exceeds `staleness_threshold`, the learner can reject it, track repeated staleness per actor, and eventually trigger a targeted actor reset after `reset_stale_after` consecutive stale batches when `enable_policy_reset` is enabled.[1][3]

This setup is useful when experimenting with delayed actors, network partitions, unbalanced actor throughput, or alternative experience-weighting strategies for stale data.[1][2][4]

## Experience weighting

The learner supports three weighting strategies inside `compute_experience_weights()`.[4]

| Strategy | Behavior | Notes |
|---|---|---|
| `uniform` | All samples receive weight 1.0.[4] | Baseline PPO behavior.[4] |
| `latency` | Weight is proportional to `1 / delta_t`, where `delta_t` is learner receive time minus batch collection time.[4] | Downweights older batches uniformly within the merged batch.[4] |
| `is` | Weight is based on `exp(current_logprobs - batch.logprobs)` with clipping.[4] | Requires a learner-side no-grad policy evaluation before PPO update.[3][4] |

## Notes and caveats

The actor-side model is policy-only, while the learner owns both actor and critic; actor-side `values` are therefore placeholders and the learner recomputes values with its critic before advantage estimation.[2][5][4] This keeps actor execution lighter while preserving standard PPO-style value estimation on the learner.[5][4]

The current configuration code still uses an `assert` to validate even actor load distribution in `compute_runtime_args`; replacing that with an explicit exception would make the config contract robust under optimized Python execution as well.[1]

## Suggested next improvements

- Add a `requirements.txt` or `pyproject.toml` so environment setup is reproducible.[6][1][2][3]
- Include a small `framework/protocol.py` section or diagram in the repo once that file is finalized, since `RolloutBatch` is the central interface between collection, comms, and learning.[7][4]
- Add a smoke-test script for the `num_actors=1` and `num_actors=2` cases, especially for outage and reset scenarios.[1][2][3]