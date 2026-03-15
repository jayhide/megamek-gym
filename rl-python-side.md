# MegaMek RL — Python Side Tasks

Companion to the Java RL bridge (`megamek.client.bot.rl`). This doc outlines the work needed to build the Python Gymnasium environment and training pipeline in a separate repo.

## Prerequisites

- Java RL bridge is functional (smoke test passes)
- Known issue: Java-side move enumeration produces very few legal moves (~3) — needs fixing before serious training

## Tasks

### 1. `megamek_env.py` — Gymnasium Environment

Create a `gymnasium.Env` subclass that manages the Java process and communicates over the JSON/TCP bridge.

- **`reset()`**: Start a new Java `RLGameRunner` process (or connect to a running one), receive initial observation
- **`step(action)`**: Send `{"type": "action", "move_index": N}`, receive next observation, return `(obs, reward, terminated, truncated, info)`
- **`close()`**: Kill the Java process

**Observation space**: Flatten the JSON observation into a fixed-size numeric array. Board grid (elevation per hex) + unit stats (position, facing, armor, heat, weapons) as a `Box` space.

**Action space**: `Discrete(N)` where N is the max legal moves across episodes. Pad with no-ops (index 0 = stand still) when fewer moves are available. Use action masking if the RL library supports it.

**Reward computation**: Compute rewards in Python from the observation data, not from the Java-side `RewardCalculator`. The Java side sends a default reward, but Python should override it. Reasoning: reward shaping is the most frequently iterated part of RL training — recompiling Java for each experiment is too slow. The observation already includes full per-location armor/internal data for all units, so Python has everything it needs.

### 2. `reward.py` — Reward Functions

Separate module for reward computation so different strategies can be swapped easily.

- **Damage delta**: (enemy armor lost - own armor lost) per step, normalized
- **Win/loss**: +1 / -1 terminal bonus from unit survival
- **Shaping options to experiment with**:
  - Range control (bonus for staying at optimal weapon range)
  - Facing bonus (keeping front armor toward enemy)
  - Heat management penalty
  - Critical hit penalty

### 3. `train.py` — Training Script

Use stable-baselines3 (or similar) to train an agent.

- PPO is a good starting point (handles discrete actions, stable)
- Wrap env with `Monitor` for logging
- Use `SubprocVecEnv` for parallel training (each subprocess launches its own Java process on a different port)
- Log to TensorBoard / W&B
- Save checkpoints periodically

### 4. `evaluate.py` — Evaluation

- Run trained agent against Princess, report win rate over N games
- Optionally save game replays for analysis
- Compare against random baseline

### 5. Java Process Management

The env needs to reliably start/stop the Java side:

- Launch via `./gradlew :megamek:runRLGameRunner` or direct `java -cp` invocation
- Each env instance uses a unique RL bridge port (for parallel training)
- Handle Java process crashes gracefully (return truncated episode)
- Consider a "stay alive" mode where the Java process handles multiple episodes without restarting (much faster for training)

### 6. Configuration

Expose as env kwargs:
- `rl_unit`: unit name for the RL player (default: "Locust LCT-1V")
- `opponent_unit`: unit name for Princess (default: "Commando COM-2D")
- `board_width`, `board_height`: map dimensions
- `rl_port`: bridge port
- `java_timeout`: max seconds per episode
- `reward_fn`: callable for custom reward computation

## Verification Milestones

1. `gymnasium.utils.env_checker.check_env(MegaMekEnv())` passes
2. Random agent completes 10 episodes without crashes
3. PPO training for 500 episodes shows non-decreasing reward trend
4. Trained agent beats random baseline win rate
