# megamek-gym

Gymnasium-compatible reinforcement learning environment for MegaMek (BattleTech tabletop simulator). This is the Python side of a Java/Python bridge for training RL agents in 1v1 mech combat.

## Quick Start

```bash
poetry install
poetry run pytest                # unit tests
poetry run python smoke_test.py --megamek-dir ../megamek --port 9999  # integration test
```

Always use `poetry run python` instead of bare `python`.

## Architecture

The environment launches a Java subprocess (MegaMek game engine) via Gradle and communicates over a JSON/TCP socket:

```
Python (Gymnasium Env)  ←— JSON/TCP on port 9999 —→  Java (RLBotClient in MegaMek)
        │                                                       │
        ├── Receives observation JSON                           ├── Enumerates legal moves
        ├── Flattens to 382-float vector                        ├── Builds JSON observation
        ├── Computes reward (Python-side)                       ├── Sends obs to Python
        └── Sends action index                                  └── Translates index → MovePath
```

- **Observation space**: `Box(shape=(W*H + 110,), float32)` — board elevations (W*H) + RL unit state (55) + enemy unit state (55). Default board (16x17) gives 382.
- **Action space**: `Discrete(max_legal_moves)` with action masking for legal moves
- **Reward**: computed Python-side via composable `RewardFunction` classes (default: DamageDelta + 10x WinLoss)

## Dependencies on `../megamek` Repo

This project requires a sibling checkout of the [megamek](https://github.com/MegaMek/megamek) repo with RL bridge files added. When the Java implementation is relevant to the issue (e.g., debugging protocol errors, changing observations/actions/rewards), read the corresponding Java-side code in `../megamek/src/megamek/client/bot/rl/`. Start with `../megamek/src/megamek/client/bot/rl/CLAUDE.md` for Java-side architecture and known limitations.

The Java side lives at:

**`megamek/src/megamek/client/bot/rl/`**

| File | Purpose |
|------|---------|
| `RLGameRunner.java` | Headless game lifecycle manager; entry point launched by Gradle |
| `RLBotClient.java` | Bot client that opens a `ServerSocket` bridge to the Python agent |
| `ObservationBuilder.java` | Serializes full game state (board, units, legal moves) to JSON |
| `ActionTranslator.java` | Parses `{"type": "action", "move_index": N}` and returns the corresponding `MovePath` |
| `RewardCalculator.java` | Tracks armor/internal damage between steps; computes per-step and episode rewards |
| `CLAUDE.md` | Architecture docs and known limitations for the Java side |
| `smoke_test_agent.py` | Minimal Python script that connects to the bridge and sends random actions |
| `verbose_smoke_test.py` | Dumps full JSON observations for debugging |

**Gradle task** in `megamek/build.gradle` (~line 596):
```
./gradlew :megamek:runRLGameRunner -PrlArgs="unit1|unit2|board|port|timeout|maxSaves|paranoidSave|rlStartPos|oppStartPos|rlDeployment"
```
Launches `RLGameRunner.main()` with pipe-delimited arguments. All args are optional and positional. See `megamek/src/megamek/client/bot/rl/CLAUDE.md` for the full arg reference table.

**Documentation**: `docs/rl-python-side.md` in the megamek repo contains the original task spec for this Python environment.

## Communication Protocol

Newline-delimited JSON over TCP (default port 9999):

- **Java → Python** (observation): `{"type": "observation", "round": N, "phase": "MOVEMENT", "board": {...}, "units": [...], "legal_moves": [...], "reward": 0.0, "terminated": false, "truncated": false}`
- **Python → Java** (action): `{"type": "action", "move_index": N}`
- Terminal observations have empty board/units/legal_moves with `terminated: true`

## Project Structure

```
megamek_gym/
├── __init__.py          # Gymnasium env registration (MegaMekGym/MegaMek-v0)
├── config.py            # MegaMekConfig dataclass with YAML save/load
├── env.py               # MegaMekEnv — full Gymnasium.Env implementation
├── java_process.py      # JavaProcess — subprocess wrapper for Gradle launcher
├── observation.py       # Flattens variable JSON observations → fixed float array
└── reward.py            # RewardFunction base class + DamageDelta, WinLoss, Composite

configs/
└── default.yaml         # Default configuration (all params documented)

tests/
├── test_config.py       # Config parsing, validation, and YAML roundtrip tests
├── test_observation.py  # Observation flattening correctness tests
└── test_reward.py       # Reward function logic tests

smoke_test.py            # End-to-end integration test with live Java process
perf_test.py             # Multi-env startup timing and diagnostics
train_ppo.py             # CleanRL-style PPO training script
eval.py                  # Evaluation script for trained checkpoints
```

## Configuration

Game parameters are managed via `MegaMekConfig` (a dataclass in `megamek_gym/config.py`). Configs can be saved/loaded as YAML for experiment reproducibility.

```python
from megamek_gym import MegaMekConfig

# Defaults
cfg = MegaMekConfig()

# Custom
cfg = MegaMekConfig(rl_unit="Locust LCT-1V", rl_port=10000)

# Save/load YAML
cfg.save("configs/my_experiment.yaml")
cfg = MegaMekConfig.load("configs/my_experiment.yaml")

# Use with gymnasium
env = gymnasium.make("MegaMekGym/MegaMek-v0", config=cfg)

# Or pass kwargs directly (backward compatible)
env = gymnasium.make("MegaMekGym/MegaMek-v0", rl_unit="Locust LCT-1V")
```

Board dimensions are auto-derived from the board name (e.g., `"16x17"` in `"Map Set 6/16x17 BattleForce 2"`). For boards without parseable dimensions, set `board_width` and `board_height` explicitly.

The `smoke_test.py` accepts `--config path/to/config.yaml` with optional `--megamek-dir` and `--port` overrides.

## Training

CleanRL-style PPO with action masking. Uses `AsyncVectorEnv` for parallel environments (each env spawns its own JVM).

```bash
# Basic training run
poetry run python train_ppo.py --megamek-dir ../megamek

# Fewer envs, shorter run (smoke test)
poetry run python train_ppo.py --megamek-dir ../megamek --num-envs 1 --total-timesteps 500 --num-steps 64

# Resume from checkpoint
poetry run python train_ppo.py --megamek-dir ../megamek --resume runs/megamek-ppo__1__*/checkpoints/latest.pt

# View metrics
tensorboard --logdir runs/
```

**Evaluation:**
```bash
poetry run python eval.py --checkpoint runs/megamek-ppo__1__*/checkpoints/latest.pt --num-episodes 10
poetry run python eval.py --checkpoint path/to/checkpoint.pt --deterministic  # greedy policy
```

**Run directory structure:**
```
runs/{exp_name}__{seed}__{timestamp}/
├── events.out.tfevents.*           # TensorBoard logs
└── checkpoints/
    ├── step_NNNNN.pt               # Periodic checkpoints
    └── latest.pt                   # Most recent checkpoint
```

**Key hyperparameters to tune:**
- `--ent-coef` (default 0.01) — entropy bonus; increase if agent converges to a bad policy too quickly
- `--num-envs` (default 4) — more envs = more data per update, but more JVM processes
- `--num-steps` (default 128) — rollout length; longer = better advantage estimates but more memory
- `--learning-rate` (default 3e-4) — with `--anneal-lr` enabled by default

**Dependencies:** `torch`, `tensorboard` (added to pyproject.toml alongside gymnasium/numpy/pyyaml)

## Known Pitfalls

### JVM shutdown race (terminal observation lost)

When a game ends, Java sends a terminal observation (`"terminated": true`) then exits. If `System.exit(0)` fires before the OS finishes transmitting the TCP data, the terminal observation is lost and Python's `readline()` hangs forever. With `AsyncVectorEnv`, this blocks all envs (they step in lockstep). Fixed on the Java side with `SO_LINGER` + a 200ms shutdown delay — see the Java-side `CLAUDE.md` for details.

**Symptom**: training hangs after a game completes; one env's `readline()` never returns; Java logs show the terminal observation was "sent" but the JVM exited in the same millisecond.

### Gymnasium info dict and AsyncVectorEnv

Gymnasium's `AsyncVectorEnv._add_info()` merges info dicts from all envs into a single vectorized dict. It **cannot handle nested dicts or lists** (like `rl_unit`, `enemy_unit`, `legal_moves`) — it recurses into them and crashes when the structure differs between terminal and non-terminal steps. The `_build_info()` method in `env.py` only returns vector-safe types: scalars, numpy arrays, and `n_legal_moves` (an int count). If you need to add new info fields, keep them flat (no nested dicts/lists).

### diagnose_reset.py

Use `diagnose_reset.py` to reproduce multi-env issues in isolation:
- **Test 1**: Single-process game + reset (no multiprocessing)
- **Test 2**: Same as test 1 but inside a child process
- **Test 3**: Two envs with `AsyncVectorEnv`, continues stepping through multiple auto-resets (reproduces the train_ppo.py pattern)

```bash
poetry run python diagnose_reset.py --megamek-dir ../megamek --test 3
```

## Development Notes

- **V1 scope**: movement phase only, 1v1, single mek per side, BattleForce 2 map
- **Reward shaping** is done in Python (not Java) so you can iterate without recompiling
- Parallel training uses per-environment port offsets: `port = rl_port + env_index`
- The opponent is MegaMek's built-in Princess AI

## Performance Testing

Use `perf_test.py` to diagnose timeout issues when running multiple parallel environments:

```bash
# Baseline: single env, no contention
poetry run python perf_test.py --megamek-dir ../megamek --num-envs 1 --mode sequential

# Reproduce parallel startup (mimics AsyncVectorEnv)
poetry run python perf_test.py --megamek-dir ../megamek --num-envs 4 --mode parallel

# Test staggered startup to reduce contention
poetry run python perf_test.py --megamek-dir ../megamek --num-envs 4 --mode staggered --stagger-delay 10
```

Reports per-env timing breakdown (Java start, socket connect, first observation) and diagnoses bottlenecks. Connection timeout is configurable via `connection_retries` and `connection_retry_delay` in config.

## Debugging Java Errors

Each Java subprocess writes its stderr to a log file at `{megamek_dir}/rl_java_{port}.log`. With the default port (9999) and 4 training envs, the logs are:

```
../megamek/rl_java_9999.log   # env 0
../megamek/rl_java_10000.log  # env 1
../megamek/rl_java_10001.log  # env 2
../megamek/rl_java_10002.log  # env 3
```

When you see `ConnectionError: Java process closed the connection`, the Java-side error (stack trace, OOM, etc.) will be in these log files. Check the log for the port mentioned in the error. The logging is set up in `megamek_gym/java_process.py`.
