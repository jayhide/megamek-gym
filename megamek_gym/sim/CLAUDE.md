# Python MegaMek Simulator

Pure-Python reimplementation of the MegaMek game engine for RL training. Replaces the Java bridge (JVM + TCP/JSON) with a direct Gymnasium env, eliminating all IPC overhead.

## Scope

Only implements the rules exercised during RL training:
- 1v1 Trebuchet TBT-5S mirror match on the 16x17 Woodland board
- Walk/run movement (no jumping)
- Weapon firing with to-hit modifiers, hit location tables, damage cascade, SRM clusters
- Heat tracking and dissipation
- Simplified Princess AI opponent
- No physical attacks, no multi-unit, no deployment choices, no aerospace

## Architecture

```
MegaMekSimEnv (gymnasium.Env)
  └── Game (game loop: initiative → movement → firing → heat → end)
        ├── Board (16x17 hex grid, terrain, elevation)
        ├── Unit × 2 (state: armor, weapons, heat, position)
        ├── Movement (BFS move enumeration, PSR filtering)
        ├── LosTable (precomputed LOS for all 272×272 hex pairs)
        ├── Firing (to-hit calc, hit location, damage, crits)
        ├── Heat (generation, dissipation, shutdown/explosion)
        └── Princess (heuristic opponent: score_move → select best)
```

## Module Reference

| Module | Purpose |
|--------|---------|
| `board.py` | Hex grid with hardcoded Woodland terrain data. Odd-column offset coords. Singleton `BOARD` instance. |
| `unit.py` | `UnitTemplate` (static data) + `Unit` (mutable game state). TBT-5S hardcoded. Hit location tables, cluster hit table, damage transfer map. |
| `movement.py` | BFS move enumeration. Tracks walk vs run paths, PSR probability filtering (< 0.3 = rejected). Pre-computed neighbor table. |
| `los.py` | Hex-line trace LOS algorithm. `LosTable` precomputes all pairs once (~0.7s), cached across resets. |
| `firing.py` | `compute_to_hit()` (gunnery + range + movement + TMM + terrain + heat modifiers), `roll_hit_location()` (front/left/right/rear tables), `apply_damage()` (armor → internal → transfer → crits). Reuses `in_firing_arc_with_twist()` from `reward.py`. |
| `heat.py` | Running = +2 heat, weapon heat per weapon. Dissipate = min(heat, sinks). Shutdown check at 14+ heat, ammo explosion at 19+. |
| `princess.py` | Scores moves by: range_quality (×3), enemy_range_quality (×-1), cover (×0.5), LOS (+1), TMM proxy (×0.5), elevation (×0.3). |
| `game.py` | Round loop: initiative roll → movement (loser first) → simultaneous firing → heat → end checks. Caches RL legal moves between `_build_observation()` and `step()`. |
| `env.py` | `MegaMekSimEnv(gymnasium.Env)` — drop-in replacement for `MegaMekEnv`. Same obs/action spaces, reuses `observation.py` and `reward.py`. |

## Performance

- **64 steps/sec** (vs ~20 SPS with Java bridge — 3x speedup)
- **5ms resets** (vs 500ms persistent / 3300ms cold with Java)
- **0.7s** one-time LOS table precomputation (amortized across all games)
- No JVM, no TCP, no subprocess overhead

## Compatibility

Produces the same observation dict format as Java's `ObservationBuilder`, so:
- `observation.py::flatten_observation_hierarchical()` works unchanged
- `reward.py::CompositeReward` works unchanged
- `train_ppo.py` and `agent.py` work unchanged — just swap env class

## Usage

```python
from megamek_gym.sim.env import MegaMekSimEnv

env = MegaMekSimEnv()
obs, info = env.reset(seed=42)
while True:
    action = env.action_space.sample()
    obs, reward, terminated, truncated, info = env.step(action)
    if terminated or truncated:
        break
```

## Known Simplifications vs Java MegaMek

- **Critical hits**: Random equipment in location rather than numbered slot table. Leg locations can roll hip or non-hip actuator crits (tracked per-leg), reducing walk MP.
- **Princess AI**: Heuristic scoring rather than full Princess path ranking
- **LOS**: Simplified hex-line trace rather than MegaMek's full LOS algorithm
- **PSR**: Only checks running-in-heavy-woods and elevation-change-≥2 triggers
- **Damage PSR**: Not implemented (20+ damage should trigger PSR in real rules)
- **Ammo explosion from crits**: Simplified (full bin explodes at once)
- **Starting facing**: Fixed (RL faces south, opponent faces north) rather than calculated

## Adding New Units

Currently only Trebuchet TBT-5S is defined. To add a new unit:
1. Add a `UnitTemplate` to `unit.py::UNIT_TEMPLATES` with armor, weapons, ammo data
2. Pass the unit name to `MegaMekSimEnv(rl_unit="New Unit Name")`

## Adding New Boards

Currently the Woodland board is hardcoded. To add a new board:
1. Parse the `.board` file and add terrain data to `board.py`
2. The `LosTable` will recompute automatically for any board size

## Cross-Validation Against Java MegaMek

Run `validate_sim.py` to compare the Python sim against a live Java MegaMek game:

```bash
# All tiers (requires Java MegaMek)
poetry run python validate_sim.py --megamek-dir ../megamek

# Tier 1 only (deterministic: board, unit, LOS, legal moves, distances, to-hit)
poetry run python validate_sim.py --megamek-dir ../megamek --tier 1 --verbose

# Tier 2 only (damage/heat consistency via armor deltas)
poetry run python validate_sim.py --megamek-dir ../megamek --tier 2 --random-actions --max-rounds 15

# Single test
poetry run python validate_sim.py --megamek-dir ../megamek --only legal_moves --verbose
```

### Validation tiers

| Tier | Tests | Method |
|------|-------|--------|
| 1 | board, unit_template, los, legal_moves, distances, to_hit_components | Exact match (deterministic) |
| 2 | damage, heat | Infer outcomes from Java armor/heat deltas |
| 3 | statistical | Compare distributions over many games |

### Current status (known divergences)

- **LOS**: ~2% mismatch rate. Remaining mismatches are from hex line tracing differences (cube-coordinate interpolation vs Java's geometric intersection). The accumulated woods threshold (≥3 points), foliage height (hexEl+2), elevation gating (strict `>`), and divided-line handling all match Java.
- **Legal moves**: ~97% hex match rate. Walk/run variant mismatches remain at some destinations where Java has both walk and run paths but the sim finds only one. Root cause: Java's `LongestPathFinder` uses relaxation-based search with a deque of Pareto-optimal paths per `(hex, facing)` state, exploring from ALL non-dominated paths. The sim's BFS keeps at most one path per state per walk/run category, so when intermediate states have multiple routes (short/cheap vs long/high-TMM), only one gets explored. The sim uses multi-objective exploration (re-explore on min-MP or max-hexes improvement) and matches Java's `isBetterPath` output criterion (prefer most hexes moved, tiebreak by PSR), which brings run-speed MP values closer to Java's. However, some walk-speed paths remain undiscoverable because they require exploring intermediate states via routes the BFS has already pruned. A full fix would require porting Java's multi-path deque relaxation, which is a significantly more complex algorithm. Leg actuator damage reduces MP matching Java's `BipedMek.getWalkMP()` (hip = halve MP, each non-hip actuator crit = -1 MP).
- Board, unit template, distances, to-hit components, damage consistency, heat consistency all **pass**.
