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

This project requires a sibling checkout of the [megamek](https://github.com/MegaMek/megamek) repo with RL bridge files added. The Java side lives at:

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

## Development Notes

- **V1 scope**: movement phase only, 1v1, single mek per side, BattleForce 2 map
- **Reward shaping** is done in Python (not Java) so you can iterate without recompiling
- Parallel training uses per-environment port offsets: `port = rl_port + env_index`
- The opponent is MegaMek's built-in Princess AI
