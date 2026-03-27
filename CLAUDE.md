# megamek-gym

Gymnasium-compatible reinforcement learning environment for MegaMek (BattleTech tabletop simulator). This is the Python side of a Java/Python bridge for training RL agents in 1v1 mech combat.

## Quick Start

```bash
poetry install
poetry run pytest                # unit tests
poetry run python smoke_test.py --megamek-dir ../megamek --port 9999  # integration test
```

Always use `poetry run python` instead of bare `python`.

## Smoke Test

**Always run after changes to the RL bridge (Python or Java side):**

```bash
poetry run python smoke_test_all.py --megamek-dir ../megamek
```

Tests: basic episode, truncation signal, termination signal, persistent reset, Python-vs-Java cross-validation. ~2-3 min. Exit 0 = all pass. Use `--verbose` for per-step output.

The older `smoke_test.py` and `smoke_test_truncation.py` are kept for quick manual debugging but are superseded by `smoke_test_all.py`.

## Architecture

The environment launches a Java subprocess (MegaMek game engine) via Gradle and communicates over a JSON/TCP socket:

```
Python (Gymnasium Env)  ←— JSON/TCP on port 9999 —→  Java (RLBotClient in MegaMek)
        │                                                       │
        ├── Receives observation JSON                           ├── Enumerates legal moves
        ├── Flattens to fixed-size float vector                    ├── Builds JSON observation
        ├── Computes reward (Python-side)                       ├── Sends obs to Python
        └── Sends action index                                  └── Translates index → MovePath
```

- **Observation space**: `Box(shape=(W*H + 111 + max_legal_moves * 10,), float32)` — board elevations (W*H) + RL unit state (55) + enemy unit state (55) + global features (1) + move features (max_legal_moves × 10). Default board (16x17) with 400 max moves gives 4383. The global feature is `rl_moves_first` (1.0 if RL moves before opponent, 0.0 if after). The 10 per-move features are: `dest_x/W`, `dest_y/H`, `facing/5`, `mp_used/20`, `dist_to_enemy/max(W,H)`, `range_quality` (RL weapon effectiveness from dest, arc-aware, [-0.5,1.0]), `enemy_range_quality` (enemy weapon effectiveness at this distance, arc-aware, [-0.5,1.0]), `terrain_cover/2` (Light Woods=0.5, Heavy Woods=1.0), `elevation_diff/10` (dest elevation minus enemy elevation), `has_los` (1.0 if a standing Mech at dest has line-of-sight to enemy hex, 0.0 if blocked by terrain/elevation; precomputed once per JVM lifetime via `LosLookupTable`). Tactical features default to 0.0 when the enemy is missing/undeployed. Unused slots (index >= n_legal_moves) are zero-padded.
- **Critic input**: The value network (critic) receives only the 111 state features (2×55 unit features + 1 global), not the per-move/per-destination action-space features. This prevents the critic from having to learn to ignore ~1875 dims of action-space description. **Future experiment**: add current-hex terrain features (cover + elevation at RL unit's hex and enemy's hex) to the critic input — this information is currently only available indirectly through the action-space features, which the critic doesn't see.
- **Action space**: `Discrete(max_legal_moves)` with action masking for legal moves
- **Reward**: computed Python-side via composable `RewardFunction` classes (default: DamageDelta + LocationDestruction + 0.5x RangeAdvantage + 0.05x Cover + 0.5x PronePenalty + 1x WinLoss). DamageDelta weights internal structure damage at 2x armor and normalizes by 20 (so a 20-damage hit = reward 1.0). LocationDestruction gives a bonus/penalty when a location is fully destroyed, weighted by tactical significance (CT/HD=1.0, torsos=0.4, legs=0.3, arms=0.2). RangeAdvantage blends absolute RL range quality with the differential advantage: `absolute_weight * rl_quality + (1 - absolute_weight) * (rl_quality - enemy_quality)` (default absolute_weight=0.3). This ensures the agent is rewarded for closing to firing range even in mirror matchups where the differential cancels out. RangeAdvantage uses the `prev_round_enemy_x/y/facing` fields from the observation (captured by Java at the start of the firing phase, after both units have moved) to evaluate positions against the correct post-movement enemy position regardless of initiative order. Without this, the reward would be wrong ~50% of the time when initiative changes between rounds (the live enemy position in `units[]` may reflect an extra move from the current round). Range quality scores how well a unit's weapons perform at the current hex distance (short=1.0, medium=0.5, long=0.0, out-of-range/below-min=-0.5) using damage-weighted averages. Weapons outside their firing arc (based on unit facing and weapon location) have their range score multiplied by 0.5 — they still contribute for being at favorable distance but at reduced value since they can't fire this turn. Firing arcs are twist-aware: upper-body weapons (HD, CT, RT, LT, RA, LA) check all achievable torso twist positions (±1 hex-side for standard mechs) and count as "in arc" if any twist brings them to bear. Leg weapons (RL, LL) use primary (leg) facing only since they don't rotate with the torso. Arc geometry matches MegaMek's `FacingArc` system: forward arc (±60° from facing, 120° cone) for torso/head/leg weapons, arm arcs extend 60° further on their side (180° cone each). The raw (no-twist) `in_firing_arc` function is cross-validated against Java's `ComputeArc.isInArc` via smoke test; the twist-aware wrapper `in_firing_arc_with_twist` is used by `range_quality`. Cover rewards the RL unit for positioning in terrain with to-hit modifiers (Light Woods=1.0, Heavy Woods=2.0); only RL cover is scored since the agent can't control enemy positioning. PronePenalty applies -1.0 when the RL unit transitions from not-prone to prone (all such transitions are involuntary falls since the Java move enumeration never offers "go prone").

## Dependencies on `../megamek` Repo

This project requires a sibling checkout of the [megamek](https://github.com/MegaMek/megamek) repo with RL bridge files added. When the Java implementation is relevant to the issue (e.g., debugging protocol errors, changing observations/actions/rewards), read the corresponding Java-side code in `../megamek/src/megamek/client/bot/rl/`. Start with `../megamek/src/megamek/client/bot/rl/CLAUDE.md` for Java-side architecture and known limitations.

The Java side lives at:

**`megamek/src/megamek/client/bot/rl/`**

| File | Purpose |
|------|---------|
| `RLGameRunner.java` | Headless game lifecycle manager; entry point launched by Gradle |
| `RLBotClient.java` | Bot client that opens a `ServerSocket` bridge to the Python agent |
| `ObservationBuilder.java` | Serializes full game state (board, units, legal moves) to JSON. Weapon damage for cluster weapons (SRM/LRM) is serialized as effective damage (rackSize × per-missile damage: SRM=2, LRM=1) rather than the raw `getDamage()` sentinel (-2). |
| `ActionTranslator.java` | Parses `{"type": "action", "move_index": N}` and returns the corresponding `MovePath` |
| `RewardCalculator.java` | Tracks armor/internal damage between steps; computes per-step and episode rewards |
| `LosLookupTable.java` | Precomputed LOS for all hex pairs; built once per JVM, cached in `RLBotClient` |
| `CLAUDE.md` | Architecture docs and known limitations for the Java side |
| `smoke_test_agent.py` | Minimal Python script that connects to the bridge and sends random actions |
| `verbose_smoke_test.py` | Dumps full JSON observations for debugging |

**Gradle task** in `megamek/build.gradle` (~line 596):
```
./gradlew :megamek:runRLGameRunner -PrlArgs="unit1|unit2|board|port|timeout|maxSaves|paranoidSave|rlStartPos|oppStartPos|rlDeployment|firingStrategy|maxGameRounds|perfLog|opponentType|forceGC|memLog|autoWakePilot|forceUnconsciousOnTurn|rlFixedX|rlFixedY|oppFixedX|oppFixedY|enableGameReports|validateCaches"
```
Launches `RLGameRunner.main()` with pipe-delimited arguments. All args are optional and positional. See `megamek/src/megamek/client/bot/rl/CLAUDE.md` for the full arg reference table.

**Documentation**: `docs/rl-python-side.md` in the megamek repo contains the original task spec for this Python environment.

## Communication Protocol

Newline-delimited JSON over TCP (default port 9999):

- **Java → Python** (observation): `{"type": "observation", "round": N, "phase": "MOVEMENT", "board": {...}, "units": [...], "legal_moves": [...], "reward": 0.0, "terminated": false, "truncated": false, "prev_round_enemy_x": X, "prev_round_enemy_y": Y, "prev_round_enemy_facing": F}` — `prev_round_enemy_*` fields contain the enemy's position/facing after the previous round's movement phase (captured at the start of firing), used by Python reward functions. Values are -1 when unavailable (e.g., round 1).
- **Python → Java** (action): `{"type": "action", "move_index": N}`
- Terminal observations have empty board/units/legal_moves with `terminated: true` and `game_outcome: "WIN"|"LOSS"|"DRAW"`

## Project Structure

```
megamek_gym/
├── __init__.py          # Gymnasium env registration (MegaMekGym/MegaMek-v0)
├── agent.py             # Agent nn.Module, checkpoint loading, action selection utilities
├── config.py            # MegaMekConfig dataclass with YAML save/load
├── env.py               # MegaMekEnv — full Gymnasium.Env implementation
├── java_process.py      # JavaProcess — subprocess wrapper for Gradle launcher
├── observation.py       # Flattens variable JSON observations → fixed float array
└── reward.py            # RewardFunction base class + DamageDelta, LocationDestruction, WinLoss, Composite

configs/
├── default.yaml         # Default: COM-2D mirror matchup
├── asymmetric.yaml      # COM-2D (RL) vs Flea FLE-15 (short-range opponent)
└── quick_test.yaml      # Fast smoke test settings

smoke_test_all.py        # Consolidated smoke test — run after any RL bridge changes
smoke_test.py            # Quick single-episode integration test with live Java process
smoke_test_truncation.py # Tests max_game_rounds truncation (RL bot stands still)
perf_test.py             # Multi-env startup timing and diagnostics
mem_benchmark.py         # Measure memory usage across N parallel environments
mem_growth.py            # Track JVM memory growth over many games (single JVM)
train_ppo.py             # CleanRL-style PPO training script
tb_summary.py            # TensorBoard run analyzer (text summaries, diagnostics, comparison)
eval.py                  # Evaluation script for trained checkpoints
clean_saves.py           # Delete training artifacts (saves, run dirs, logs, heap dumps)
analyze_full_moves.py    # Compare current (longest-only) vs full (Pareto frontier) legal moves
analyze_move_distributions.py  # Multi-game legal move distribution analysis with matplotlib charts
visualize_hex_map.py     # Interactive hex map viewer: board terrain, reachable hexes, unit positions
game_viewer.py           # Combined hex map + transcript viewer (side-by-side HTML)
bench_sim.py             # Profile sim latency (per-phase breakdown) and memory usage
validate_sim.py          # Cross-validate Python sim against Java MegaMek (tiered tests)

tests/
├── test_config.py            # Config parsing, validation, and YAML roundtrip tests
├── test_cross_validation.py  # Python-vs-Java cross-validation (hex distance, firing arcs)
├── test_observation.py       # Observation flattening correctness tests
├── test_reward.py            # Reward function logic tests
└── sim_validation/           # Sim-vs-Java cross-validation suite
    ├── collector.py          # Run Java game, collect per-step observation traces
    ├── reconstruct.py        # Build sim.Unit from Java observation dict
    ├── test_board.py         # Board terrain/elevation comparison
    ├── test_unit_template.py # Unit armor, weapons, MP comparison
    ├── test_los.py           # LOS comparison
    ├── test_legal_moves.py   # Legal move set comparison (highest priority)
    ├── test_to_hit.py        # To-hit modifier tables + distance cross-validation
    ├── test_damage.py        # Damage consistency via armor deltas
    ├── test_heat.py          # Heat consistency
    └── test_statistical.py   # Game distribution comparison
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

**Fixed deployment coordinates**: Set `rl_fixed_coords: [x, y]` and/or `opponent_fixed_coords: [x, y]` in YAML config (or as tuples in Python) to pin a unit to an exact hex every game. Uses MegaMek's `Board.START_ANY` zone with a 1-hex rectangle. When not set (default `null`), the heuristic picks a hex within the configured zone as before.

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

# Analyze a run (compact text summary for LLM or terminal)
poetry run python tb_summary.py runs/sanity-check__1__*

# Compare multiple runs
poetry run python tb_summary.py runs/run_a runs/run_b

# Quick health check
poetry run python tb_summary.py runs/latest_run --diagnostics-only

# Last 20% of training only
poetry run python tb_summary.py runs/latest_run --tail 20
```

**Evaluation:**
```bash
poetry run python eval.py --checkpoint runs/megamek-ppo__1__*/checkpoints/latest.pt --num-episodes 10
poetry run python eval.py --checkpoint path/to/checkpoint.pt --deterministic  # greedy policy

# Random baseline (uniform over legal moves)
poetry run python eval.py --random --num-episodes 10
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

**Recommended fast-training args:**
```bash
poetry run python train_ppo.py --megamek-dir ../megamek --config configs/default.yaml \
  --num-envs 8 --num-steps 256 --stagger-delay 3
```
- `--config configs/default.yaml` — uses `firing_strategy: naive` (faster Princess AI) and `max_rotating_round_saves: 0` (no disk I/O from saves)
- `--num-envs 8` — amortizes reset blocking (when one env resets, 7 others already contributed recent steps)
- `--num-steps 256` — longer rollouts = fewer rollouts = less reset overhead per update
- `--stagger-delay 3` — faster startup (default 10s is conservative)

**Timing instrumentation:** Each update prints `rollout=Xs train=Ys episodes=N` to identify bottlenecks. Also logged to TensorBoard under `timing/rollout_seconds`, `timing/train_seconds`, `timing/episodes_per_rollout`.

**Dependencies:** `torch`, `tensorboard` (added to pyproject.toml alongside gymnasium/numpy/pyyaml)

## Persistent JVM (Reset Without Restart)

By default, the JVM stays alive between games. When `reset()` is called after a completed game, Python sends `{"type": "reset"}` over the existing socket instead of killing and restarting the JVM. Java tears down the current game (server, clients) and starts a fresh one using the same TCP connection.

**Protocol flow (after terminal observation):**
1. Java sends terminal obs (`terminated: true`), then waits for reset message
2. Python sends `{"type": "reset"}\n`
3. Java creates new Server, clients, game — sends first observation
4. If Python disconnects (EOF) instead of sending reset, Java exits gracefully

**Fallback:** If the persistent reset fails (socket error, Java crash), Python automatically falls back to a full cold restart (kill JVM, start new one, reconnect). This makes the feature backward-compatible with older Java versions that don't support the game loop.

**Performance:** Reset drops from ~3.3s (cold JVM restart) to ~0.5s (persistent), roughly doubling training SPS.

## Known Pitfalls

### JVM shutdown race (terminal observation lost)

When a game ends, Java sends a terminal observation (`"terminated": true`) then exits. If `System.exit(0)` fires before the OS finishes transmitting the TCP data, the terminal observation is lost and Python's `readline()` hangs forever. With `AsyncVectorEnv`, this blocks all envs (they step in lockstep). Fixed on the Java side with `SO_LINGER` + a 200ms shutdown delay — see the Java-side `CLAUDE.md` for details.

**Symptom**: training hangs after a game completes; one env's `readline()` never returns; Java logs show the terminal observation was "sent" but the JVM exited in the same millisecond.

### Java crash recovery during training

When the JVM crashes mid-game (e.g., NPE in Princess AI), `env.py` detects the broken connection within `step_timeout_seconds` (default 30s) and returns a synthetic terminal observation (`phase: "CRASH"`, `info["java_crash"] = 1`) instead of raising an exception. This allows `AsyncVectorEnv` to auto-reset the crashed env without blocking other envs. Crash counts are logged to TensorBoard (`charts/java_crashes`) and printed in training summaries as `C:{count}`.

**Config**: `step_timeout_seconds` in `MegaMekConfig` (default 30). The longer 360s timeout is still used during `reset()` for initial connection.

### Early termination (prone + leg destroyed)

When the RL unit is prone with at least one destroyed leg (LL or RL with `internal == 0`), it cannot stand and is effectively doomed. Rather than waste training time playing out a lost game, `env.py` detects this in `step()` and immediately terminates the episode as a loss. The observation gets `game_outcome: "LOSS"` injected so WinLoss reward fires normally. Early terminations force a cold JVM restart (`_java_crashed = True`) since Java is still mid-game. Counts are logged to TensorBoard (`charts/early_terminations`) and printed as `E:{count}` in training summaries.

### Gymnasium info dict and AsyncVectorEnv

Gymnasium's `AsyncVectorEnv._add_info()` merges info dicts from all envs into a single vectorized dict. It **cannot handle nested dicts or lists** (like `rl_unit`, `enemy_unit`, `legal_moves`) — it recurses into them and crashes when the structure differs between terminal and non-terminal steps. The `_build_info()` method in `env.py` only returns vector-safe types: scalars, numpy arrays, and `n_legal_moves` (an int count). If you need to add new info fields, keep them flat (no nested dicts/lists).

### Unconscious pilot (prone + immobile with full MP)

When a Mech falls, the pilot must pass a consciousness check. On failure, `crew.isUnconscious()` returns true, making `entity.isImmobile()` true even though the entity has full walk/run MP. The Mech gets only 1 legal move (stand still) until the pilot wakes up. The pilot retries each round — may wake up in 1 round or stay unconscious for several.

**Auto-wake (enabled by default)**: The `auto_wake_pilot` config option (default `true`) makes the Java RL bridge automatically clear the unconscious flag on the RL entity before move enumeration. This prevents wasted training steps since RL bots don't have human pilots subject to consciousness checks. The auto-wake only applies to the RL entity; the opponent (Princess) is unaffected.

**Symptom (when auto-wake is disabled)**: `n_legal_moves=1` for multiple consecutive rounds, entity position unchanged. Java log shows `enumerateLegalMoves: entity=... is IMMOBILE (prone=true shutdown=false crew_unconscious=true)`.

### Mid-game reset sends action instead of reset

`_reset_persistent()` sends `{"type": "reset"}` over the socket, but if the game is still in progress (not terminated/truncated), Java is waiting for an action, not a reset. Java's `ActionTranslator` can't parse the reset message (no `move_index` field), falls back to stand-still, and the game continues. Python reads the next observation thinking it's a new game. **This only affects scripts that call `env.reset()` before the game ends.** Training is unaffected because `AsyncVectorEnv` only resets after terminated/truncated.

### Move enumeration: walk/run dedup and safety filters

`enumerateLegalMoves()` in `RLBotClient.java` uses `getAllComputedPathsUnordered()` from `LongestPathFinder` and deduplicates to at most **two paths per (hex, facing)**: a walk-speed path (mpUsed ≤ walkMP) and a run-speed path (mpUsed > walkMP). Within each category, the path that moved through the **most hexes** is kept (maximizes TMM defensive modifier), with ties broken by lowest fall risk (highest PSR success probability). Run paths are only included if they provide more hexes moved than the walk path to the same destination.

Additionally, paths are filtered out if they would:
- **Collapse a building** (unit weight + 10-ton margin exceeds building CF) — ported from Princess `PathRanker.willBuildingCollapse()`
- **Exceed fall tolerance** (cumulative PSR success probability < 0.3, i.e. >70% chance of falling) — ported from Princess `PathRanker.getMovePathSuccessProbability()` using `SharedUtility.getPSRList()`

The walk vs run distinction matters strategically: walking has no attack penalty but lower defensive modifier; running gives +1 TMM but +1 to-hit penalty on own attacks. The `mp_used/20` per-move feature already captures this distinction, so no Python-side observation changes were needed.

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
- **Fire mechanics disabled**: `tacops_start_fire` and `woods_burn_down` are set to false in `RLGameRunner.initializeServer()`. This prevents accidental fire ignition and woods burning down. Empirically negligible: fire only triggers on weapon *misses* in wooded hexes, gated by a 2d6≤3 accident roll (8.3%) then a TN 9 ignition roll (27.8%), yielding ~0.014 expected ignitions per round with Trebuchet-class loadouts (~1 fire per 70 rounds). Safe to leave disabled.
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

## Memory Benchmarking

Use `mem_benchmark.py` to measure actual memory consumption across different numbers of parallel environments:

```bash
# Test 1, 2, 4, and 8 envs (default)
poetry run python mem_benchmark.py --megamek-dir ../megamek --env-counts 1,2,4,8

# Quick test with fewer episodes and shorter games
poetry run python mem_benchmark.py --megamek-dir ../megamek --env-counts 1,2 --episodes-per-env 2 --max-game-rounds 5
```

For each env count, starts N environments, runs a few episodes to stabilize memory, then measures per-JVM RSS (actual resident memory from `/proc`), JVM heap usage (from `[rl-mem]` log lines), and Python process RSS. Reports a summary table.

The `mem_log` config field (default 0) controls Java-side memory logging verbosity: 0=off, 1=basic heap+delta, 2=pool breakdown, 3=class histogram. The benchmark enables level 1 automatically.

## Sim Profiling

Use `bench_sim.py` to profile the pure-Python simulator's per-phase latency and memory usage:

```bash
# Default: 20 games, full report with tracemalloc
poetry run python bench_sim.py

# Quick test with verbose per-game output
poetry run python bench_sim.py --num-games 5 --verbose

# Scaling benchmark (1/2/4/8 in-process envs)
poetry run python bench_sim.py --scaling --env-counts 1,2,4,8
```

Monkey-patches sim internals with timing wrappers (zero sim code changes) to break down per-step latency into phases: move enumeration (RL + opponent), Princess AI, firing, heat, observation flattening, and reward computation. Also reports RSS memory at checkpoints, key object sizes, and tracemalloc top allocations.

## Memory Growth Tracking

Use `mem_growth.py` to track how JVM memory grows over many games within a single JVM:

```bash
# Track memory over 30 games
poetry run python mem_growth.py --megamek-dir ../megamek

# Quick test
poetry run python mem_growth.py --megamek-dir ../megamek --num-games 5 --verbose
```

Runs a single JVM through N games via persistent reset, capturing per-game RSS (from `/proc`) and JVM heap usage (from `[rl-mem]` log lines). Reports a time-series table and summary of memory growth.

## Move Distribution Analysis

Use `analyze_move_distributions.py` to measure legal move set sizes, reachable hex coverage, walk/run overlap, and facing coverage across multiple games:

```bash
# Default: 10 games, saves move_distributions.png
poetry run python analyze_move_distributions.py --megamek-dir ../megamek

# More games for robust statistics
poetry run python analyze_move_distributions.py --megamek-dir ../megamek --games 20 --output moves.png

# Different unit (adjust --walk-mp to match unit's walk MP)
poetry run python analyze_move_distributions.py --megamek-dir ../megamek --walk-mp 8 --config configs/locust.yaml
```

Generates an 8-panel PNG visualization (move count histograms, box plots by round, reachable hex % over time, hex category stacked bars, facing coverage scatter, walk vs run scatter, summary stats) plus detailed text statistics to stdout. Separates mobile steps (>10 moves) from immobile steps (prone/fallen) for cleaner analysis.

## Debugging Java Errors

Each Java subprocess writes its stderr to a log file at `{megamek_dir}/rl_java_{port}.log`. With the default port (9999) and 4 training envs, the logs are:

```
../megamek/rl_java_9999.log   # env 0
../megamek/rl_java_10000.log  # env 1
../megamek/rl_java_10001.log  # env 2
../megamek/rl_java_10002.log  # env 3
```

When you see `ConnectionError: Java process closed the connection`, the Java-side error (stack trace, OOM, etc.) will be in these log files. Check the log for the port mentioned in the error. The logging is set up in `megamek_gym/java_process.py`.
