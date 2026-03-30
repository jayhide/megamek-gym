# megamek-gym

Gymnasium-compatible reinforcement learning environment for MegaMek (BattleTech tabletop simulator). This is the Python side of a Java/Python bridge for training RL agents in 1v1 mech combat.

## Quick Start

```bash
poetry install
poetry run pytest                # unit tests only (fast, no JVM needed)
```

Always use `poetry run python` instead of bare `python`.

## Testing

All tests are unified under pytest with markers. **Run after any non-trivial code change:**

```bash
# Unit tests only (fast, no JVM) — default (~2 min)
poetry run pytest

# Integration smoke tests (require JVM, ~2-3 min)
poetry run pytest -m integration --megamek-dir ../megamek

# Sim-vs-Java cross-validation (require JVM, ~5-7 min)
poetry run pytest -m validation --megamek-dir ../megamek

# Everything (unit + integration + validation)
poetry run pytest -m "" --megamek-dir ../megamek

# Useful options
poetry run pytest -m integration --megamek-dir ../megamek -v          # verbose
poetry run pytest -m integration --megamek-dir ../megamek --port 9999  # custom port
poetry run pytest -m validation --megamek-dir ../megamek --random-actions  # better for damage/heat
poetry run pytest -m validation --megamek-dir ../megamek --trace-workers 11  # max parallelism
```

**Typical timing** (8-core machine):
| Suite | Time | Notes |
|-------|------|-------|
| Unit tests (`poetry run pytest`) | ~2 min | No JVM. Slowest: `test_many_games_with_damage_falls` (20 sim games, parallelized via ProcessPoolExecutor) |
| Validation (`-m validation`) | ~5-7 min | 11 JVM traces collected in parallel (default: all 11 concurrent, `--trace-workers` to tune). Trace collection ~15-33s depending on workers, then `test_statistical` ~45s (50 sim games, parallelized), `test_prone_validity` ~4 min |
| Integration (`-m integration`) | ~2-3 min | Each test launches its own JVM |

**Markers:**
- *(no marker)* — unit tests: config, observation, reward, early termination, heat MP. No JVM needed.
- `integration` — smoke tests: basic episode, truncation, termination, persistent reset, cross-validation, auto-wake, fixed deployment, board consistency, pilot stats, hierarchical actions, feature distributions. Each launches its own JVM.
- `validation` — sim cross-validation: board, unit template, LOS, legal moves (walk patrol + run patrol with dual-bot scripted movement), prone moves (via `inject_state` — RL unit starts prone), distances, to-hit, per-move features (distance + weapon arcs cross-validated against Java's `java_dist_to_enemy` and `java_weapon_arcs[]`), observation features (121-dim base vector end-to-end validated against expected values from raw Java obs), firing (weapon fireability + TNs: measurement test on java_trace + asserting test on walk patrol trace), damage, heat, crits (monotonicity + side effects + MP consistency), statistical. Legal moves and firing walk patrol tests use dual-bot mode (`opponentType=rl`, `firingStrategy=none`) with scripted waypoint paths so both units traverse diverse board positions. Prone moves test uses `inject_state` to deterministically set the RL unit prone at game start (no stochastic falls needed). Injected-state tests use `inject_state` with `firingStrategy=naive` for deterministic testing: `damaged_patrol_trace` (low armor on both units → guaranteed damage/crits/location destructions), `heated_patrol_trace` (RL heat=15 → guaranteed heat dissipation and MP penalty cross-validation), `weapons_damaged_patrol_trace` (RL weapons 0,2 destroyed → verifies unfireable in firing_report), `shutdown_patrol_trace` (RL heat=35 + opponent all weapons destroyed → deterministic heat progression, validates shutdown transitions only at heat >= 14). Heat generation cross-validation uses firing_report weapon data to verify `curr_heat == max(0, prev_heat + weapon_heat + running_heat + engine_heat - sinks)` on heated and damaged traces. `gyro_destroyed_trace` (both units gyro_hits=2 + reduced limb armor → PSR auto-fails on any trigger, validates fall mechanics), `hip_damaged_trace` (RL right hip hit → validates MP halving ceil(5/2)=3). `collect_dual_game_trace()` accepts `firing_strategy` and `initial_state` parameters. The Java-side `applyDeferredInjection` calls `sendUpdateEntity(entity)` to push injected state to the server (without this, MegaMek's server state sync overwrites client-only modifications). Crit slot damage (`crit_state` field) requires persistent re-application before every observation build because it doesn't survive `sendUpdateEntity` server round-trips. Other tests share a single Java game trace (session-scoped fixture).

The standalone scripts `smoke_test_all.py` and `validate_sim.py` are kept for manual debugging but the pytest wrappers (`tests/test_smoke.py`, `tests/test_sim_validation.py`) are the canonical way to run these tests.

**Future work:** Ammo tracking cross-validation — Java sends `ammo[].shots_remaining` and `reconstruct.py` reads it (`extract_ammo_state()`), but no test verifies that the sim correctly tracks ammo consumption across rounds.

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

- **Base features** (OBS_SIZE = 121): 2×60 unit features + 1 global + 6 tactical. `INCLUDE_BOARD_ELEVATION` is `False` so the board elevation block is 0. Per-unit features (60 each): position(2) + facing one-hot(6) + walk/run/jump MP(3) + heat(1) + status flags(4: prone, destroyed, deployed, retreated) + armor per location(32: 8 locations × 4 values) + weapon destroyed flags(7) + terrain_cover at current hex(/2.0, 1 feature) + elevation at current hex(/10.0, 1 feature) + engine_hits(/3.0, 1 feature) + gyro_hits(/2.0, 1 feature) + sensor_hits(/2.0, 1 feature). Global feature: `rl_moves_first` (1.0 if RL moves before opponent, 0.0 if after). Tactical features (6): hex_distance_to_enemy/(W+H), rl_range_quality from current position (arc-aware, _norm_rq), enemy_range_quality from current position, has_los from current position, relative_elevation (rl - enemy)/10, round_number/50. Tactical features default to 0.0 when the enemy is missing/undeployed.
- **Flat observation space** (`action_space_type: flat`): `Box(shape=(121 + max_legal_moves * 10,), float32)` — base features + per-move features. The 10 per-move features are: `dest_x/W`, `dest_y/H`, `facing/5`, `mp_used/20`, `dist_to_enemy/(W+H)`, `range_quality` (RL weapon effectiveness from dest, arc-aware, normalized from [-0.5,1.0] to [0.0,1.0]), `enemy_range_quality` (enemy weapon effectiveness at this distance, arc-aware, normalized to [0.0,1.0]), `terrain_cover/2` (Light Woods=0.5, Heavy Woods=1.0), `elevation_diff/10` (dest elevation minus enemy elevation), `has_los` (1.0 if a standing Mech at dest has line-of-sight to enemy hex, 0.0 if blocked by terrain/elevation; precomputed once per JVM lifetime via `LosLookupTable`). Unused slots (index >= n_legal_moves) are zero-padded. Action space: `Discrete(max_legal_moves)` with action masking.
- **Hierarchical observation space** (`action_space_type: hierarchical`): `Box(shape=(121 + max_destinations * 9 + max_destinations * 6,), float32)` — base features + per-destination features (9 each) + per-destination facing features (6 facings × 1 feature each). Destinations are grouped by (hex, mp_category) with walk preferred over run. Action space: `MultiDiscrete([max_destinations, 6])` with separate destination and facing masks.
- **Spatial/CNN observation space** (`use_cnn: true`): `Box(shape=(121 + 14 * H * W,), float32)` — base features + 14-channel board grid (channel-major). Channels: elevation/3 (0), woods cover (1), self location (2), enemy location (3), self facing with arc (4), enemy facing with arc (5), reachable mask (6), mp_used/20 (7), dist_to_enemy (8), elevation advantage (9), has_los (10), enemy range quality (11), best RL range quality (12), terrain cover (13). Facing channels encode 1.0 at unit hex and 0.5 at front-arc neighbor hexes. Per-destination channels are nonzero only at reachable hexes; walk moves preferred over run for same hex. Action space: `MultiDiscrete([H*W, 6])` with spatial destination and facing masks.
- **Critic input**: Flat/hierarchical critics receive the 121 base features only (not per-move/per-destination features). The spatial/CNN critic receives 32 global-average-pooled CNN features + 121 base features = 153 dims. This prevents the critic from having to learn to ignore action-space description dims while giving the spatial critic board-level spatial awareness.
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
| `NoFiringStrategy.java` | No-fire strategy: sends empty attack vector during firing phase (used for dual-bot validation tests) |
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
Launches `RLGameRunner.main()` with pipe-delimited arguments. All args are optional and positional. Key options: `firingStrategy` accepts `"princess"` (default), `"naive"`, or `"none"` (no firing); `opponentType` accepts `"princess"` (default) or `"rl"` (second RLBotClient sharing the same socket, for dual-bot validation). See `megamek/src/megamek/client/bot/rl/CLAUDE.md` for the full arg reference table.

**Documentation**: `docs/rl-python-side.md` in the megamek repo contains the original task spec for this Python environment.

## Communication Protocol

Newline-delimited JSON over TCP (default port 9999):

- **Java → Python** (observation): `{"type": "observation", "round": N, "phase": "MOVEMENT", "board": {...}, "units": [...], "legal_moves": [...], "reward": 0.0, "terminated": false, "truncated": false, "prev_round_enemy_x": X, "prev_round_enemy_y": Y, "prev_round_enemy_facing": F, "firing_report": {...}}` — `prev_round_enemy_*` fields contain the enemy's position/facing after the previous round's movement phase (captured at the start of firing), used by Python reward functions. Values are -1 when unavailable (e.g., round 1). `firing_report` contains per-weapon fireability and to-hit TNs for both entities at the previous round's firing phase start (post-movement, pre-firing). Absent in the first observation. Contains: `rl_entity`/`opp_entity` (position, facing, delta_distance, mp_used, moved, heat) and `rl_weapons`/`opp_weapons` (per-weapon: weapon_name, weapon_index, location, destroyed, can_fire, to_hit_value, to_hit_desc, impossible).
- **Python → Java** (action): `{"type": "action", "move_index": N}`
- **Python → Java** (inject_state): `{"type": "inject_state", "entities": [{"id": 0, "heat": 15, "prone": true, "armor": {"CT": 5, "LA": 0}, "internal": {"LA": 2}, "weapons_destroyed": [0, 2], "crit_state": {"gyro_hits": 2, "right_leg": {"hip_hits": 1}}}]}` — Test-only mechanism for setting unit damage/heat/prone/crit state at game start. Sent as the response to the first observation (before the first action). Java parses the spec, defers application until each entity's `continueMovementFor()` fires (ensuring server state sync doesn't override), then re-enumerates legal moves and re-sends the updated observation. Only specified fields are applied; omitted fields keep defaults. Location names use BipedMek abbreviations: `HD`, `CT`, `RT`, `LT`, `RA`, `LA`, `RL`, `LL`. Entity IDs come from the observation's `units[i].id`. The `entities` list can target both RL and opponent units. The `crit_state` object supports `gyro_hits` (int) and per-leg actuator damage via `right_leg`/`left_leg` objects with `hip_hits`, `upper_leg_hits`, `lower_leg_hits`, `foot_hits` (all int). Crit slot damage is re-applied before every observation build because it doesn't survive `sendUpdateEntity` server round-trips (the serialized entity broadcast replaces client entities). Used by `dual_collector.py` and `collector.py` for deterministic test state setup.
- Terminal observations have empty board/units/legal_moves with `terminated: true` and `game_outcome: "WIN"|"LOSS"|"DRAW"`

## Project Structure

```
megamek_gym/
├── __init__.py          # Gymnasium env registration (MegaMekGym/MegaMek-v0)
├── agent.py             # Agent/HierarchicalAgent/SpatialHierarchicalAgent nn.Modules, checkpoint loading
├── config.py            # MegaMekConfig dataclass with YAML save/load
├── env.py               # MegaMekEnv — full Gymnasium.Env implementation
├── java_process.py      # JavaProcess — subprocess wrapper for Gradle launcher
├── observation.py       # Flattens variable JSON observations → fixed float array
├── pytest_plugin.py     # pytest11 entry point: registers --megamek-dir, --port CLI options
└── reward.py            # RewardFunction base class + DamageDelta, LocationDestruction, WinLoss, Composite

configs/
├── default.yaml         # Default: COM-2D mirror matchup
├── asymmetric.yaml      # COM-2D (RL) vs Flea FLE-15 (short-range opponent)
├── quick_test.yaml      # Fast smoke test settings
└── sanity_check_cnn.yaml # CNN spatial agent with sim backend

smoke_test_all.py        # Consolidated smoke test — run after any RL bridge changes
smoke_test.py            # Quick single-episode integration test with live Java process
smoke_test_truncation.py # Tests max_game_rounds truncation (RL bot stands still)
perf_test.py             # Multi-env startup timing and diagnostics
mem_benchmark.py         # Measure memory usage across N parallel environments
mem_growth.py            # Track JVM memory growth over many games (single JVM)
train_ppo.py             # CleanRL-style PPO training script
train_queue.py           # Sequential training queue runner (overnight sweeps)
tb_summary.py            # TensorBoard run analyzer (text summaries, diagnostics, comparison)
eval.py                  # Evaluation script for trained checkpoints
clean_saves.py           # Delete training artifacts (saves, run dirs, logs, heap dumps)
analyze_full_moves.py    # Compare current (longest-only) vs full (Pareto frontier) legal moves
analyze_move_distributions.py  # Multi-game legal move distribution analysis with matplotlib charts
visualize_hex_map.py     # Interactive hex map viewer: board terrain, reachable hexes, unit positions
game_viewer.py           # Combined hex map + transcript viewer (side-by-side HTML)
bench_sim.py             # Profile sim latency (per-phase breakdown) and memory usage
validate_sim.py          # Cross-validate Python sim against Java MegaMek (tiered tests)
diagnose_features.py     # Test if critic features can predict returns (sklearn R² diagnostic)

tests/
├── conftest.py               # Shared fixtures: megamek_dir, base_port, java_trace
├── test_config.py            # Config parsing, validation, and YAML roundtrip tests
├── test_cross_validation.py  # Python-vs-Java cross-validation (hex distance, firing arcs)
├── test_observation.py       # Observation flattening correctness tests
├── test_reward.py            # Reward function logic tests
├── test_smoke.py             # @integration: pytest wrappers for smoke_test_all.py (11 tests)
├── test_sim_validation.py    # @validation: pytest wrappers for validate_sim.py (9 tests)
└── sim_validation/           # Sim-vs-Java cross-validation suite
    ├── collector.py          # Run Java game, collect per-step observation traces
    ├── dual_collector.py     # Dual-bot collector: both units RL-controlled with scripted waypoints
    ├── reconstruct.py        # Build sim.Unit from Java observation dict
    ├── test_board.py         # Board terrain/elevation comparison
    ├── test_unit_template.py # Unit armor, weapons, MP comparison
    ├── test_los.py           # LOS comparison
    ├── test_legal_moves.py   # Legal move set comparison (highest priority)
    ├── test_to_hit.py        # To-hit modifier tables + distance cross-validation
    ├── test_firing.py        # Weapon fireability + to-hit TN cross-validation
    ├── test_princess_behavior.py  # Princess movement selection comparison
    ├── test_damage.py        # Damage consistency via armor deltas
    ├── test_crits.py         # Crit state consistency (monotonicity, side effects, MP)
    ├── test_heat.py          # Heat consistency
    ├── test_psr_falls.py     # PSR/fall mechanics (prone transitions, triggers, stand-up, gyro/hip)
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

## Training Queue

Queue multiple training runs to execute sequentially (e.g., overnight hyperparameter sweeps). Each run is a subprocess with full process isolation — crashes in one run don't affect the queue.

```bash
# Run a queue
poetry run python train_queue.py sweep.yaml

# Preview commands without running
poetry run python train_queue.py sweep.yaml --dry-run

# Resume an interrupted queue
poetry run python train_queue.py --resume runs/queue__*/queue_state.yaml
```

**Queue YAML format** — each run references a complete config file (the config is the source of truth):
```yaml
runs:
  - name: "lr-high"              # optional display label
    config: configs/lr_high.yaml
  - name: "lr-low"
    config: configs/lr_low.yaml
  - config: configs/ent_sweep.yaml   # name defaults to config stem
```

**Error recovery:** If a run fails (nonzero exit), the queue logs the failure and continues to the next run. A summary table at the end shows which succeeded/failed.

**Interrupt handling:** First Ctrl+C lets the current run finish gracefully, then stops the queue. Second Ctrl+C force-quits.

**State persistence:** Progress is saved to `runs/queue__{timestamp}/queue_state.yaml` after each run. Use `--resume` to pick up where an interrupted queue left off.

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

## Feature Predictiveness Diagnostic

Use `diagnose_features.py` to test whether the critic's 121 base state features contain enough signal to predict GAE returns. Collects rollout data, fits sklearn models (LinearRegression + RandomForest), and reports R² scores with feature importance rankings.

```bash
# Random policy baseline (sim backend, fast)
poetry run python diagnose_features.py --config configs/sanity_check_sim.yaml --num-steps 4096 --num-envs 30

# With a trained checkpoint
poetry run python diagnose_features.py --config configs/sanity_check_sim.yaml --checkpoint runs/.../latest.pt --num-steps 4096 --num-envs 30

# Re-analyze saved data (no envs needed)
poetry run python diagnose_features.py --load feature_data.npz
```

Saves collected data to `feature_data.npz` for reuse. Interpretation guide: RF CV R² < 30% means features are the bottleneck; 30-60% means both features and capacity matter; 60%+ means the neural net needs more capacity or training. Use `--num-envs 30` with the sim backend for fast collection (~120k samples in under a minute).

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
