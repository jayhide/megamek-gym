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
| `movement.py` | Move enumeration (default: deque-relaxation port of Java's `LongestPathFinder`; also has BFS). Tracks walk vs run paths, PSR probability filtering (< 0.3 = rejected). Pre-computed neighbor table. |
| `los.py` | Geometric hex-line LOS algorithm ported from Java's `IdealHex`/`Coords.intervening()`. `LosTable` precomputes all pairs once (~2.3s), cached across resets. |
| `firing.py` | `compute_to_hit()` (gunnery + range + movement + TMM + terrain + heat modifiers), `roll_hit_location()` (front/left/right/rear tables), `apply_damage()` (armor → internal → transfer → crits). Reuses `in_firing_arc_with_twist()` from `reward.py`. |
| `heat.py` | Running = +2 heat, weapon heat per weapon, engine damage = +5 heat per engine hit per turn. Dissipate = min(heat, sinks). Shutdown check at 14+ heat, ammo explosion at 19+. |
| `princess.py` | Ports Java's `BasicPathRanker.rankPath()` formula for 1v1: `utility = -fallMod + braveryMod - aggressionMod - herdingMod - facingMod`. Uses Java's default behavior weights and quick damage estimate (maxDmgAtRange × 0.42). Herding = distance to own current position × 1.0 (self-herding in 1v1). |
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

- **Critical hits**: Random equipment in location rather than numbered slot table. Crit roll uses TW table (8-9=1 crit, 10-11=2, 12=3 or head blown off). Through-armor crits (TAC) on hit location roll of 2 grant 1 guaranteed crit. Head crits include cockpit (kills pilot), sensors, life support. Heat sink crits in torso locations (CT/RT/LT) for non-integral sinks. Ammo explosion damage uses actual per-round weapon damage (not hard-coded). Leg locations can roll hip or non-hip actuator crits (tracked per-leg), reducing walk MP.
- **Princess AI**: Uses Java's `BasicPathRanker` formula for 1v1 (omits crowding, self-preservation, off-board, movement mod). Includes self-herding (Java's `getFriendEntities()` returns the unit itself in 1v1). Two evaluation paths matching Java: `evaluateMovedEnemy` (terrain LOS, actual positions, 0.42 discount, includes kick if adjacent) and `evaluateUnmovedEnemy` (forward arc check instead of LOS, closest-reachable-hex range approximation via `max(1, dist - run_mp)`, no LOS for enemy damage, 0.25 discount, flank kick if enemy can reach adjacent). Cluster weapons use rackSize (not full effective damage) and `getMaxDamageAtRange` ignores firing arcs, matching Java.
- **LOS**: Ported from Java's `IdealHex`/`Coords.intervening()` geometric intersection algorithm. 0% mismatch vs Java on tested boards.
- **PSR / mid-movement falls**: The sim models mid-movement falls from PSR failures. Each move dict stores the full hex path traversed during enumeration. During move execution in `game.py`, `_resolve_movement_psrs()` walks the path hex-by-hex and rolls an independent PSR (2d6 vs piloting skill + damage modifiers) at each elevation change >= 2 levels. On the first failure: the unit falls at that hex (prone, random facing via 1d6), takes fall damage (`tonnage // 10 * (fall_height + 1)`, where fall_height is the elevation drop for downhill or 0 for uphill/flat), and attempts automated recovery — if enough remaining MP (>= 2), rolls a stand-up PSR, and on success walks greedily toward the original destination with remaining MP (walk only, no running after a fall). This replaces the previous atomic model. Paths with <30% cumulative PSR success are still pre-filtered during enumeration. Note: on the Woodland board, max adjacent elevation diff is 1, so PSR falls from elevation changes do not trigger naturally; they will matter on boards with cliff hexes. Remaining simplifications vs Java: no skid PSR, no terrain-based PSR (rubble, water, ice), fall damage applied to a single random front-hit location rather than potentially multiple locations.
- **Damage-induced falls**: Implemented via a pending PSR queue on Unit. Three triggers: (1) 20+ damage in a phase — PSR with modifier `damage // 20`; (2) gyro crit — first hit PSR +3, second hit (destroyed) automatic fall; (3) hip actuator crit PSR +2, leg actuator crit PSR +1. Leg destruction also queues automatic fall (with fall damage + facing randomization, replacing the old direct `prone = True`). PSRs are resolved after firing via `Game._resolve_pending_psrs()` which handles cascading (fall damage can trigger crits that queue more PSRs). Weight class modifier not used (TAC OPS optional rule).
- **Ammo explosion from crits**: Simplified (full bin explodes at once)
- **Starting facing**: Computed toward opponent using `hex_bearing`, matching Java's `Coords.direction()`

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
| 2 | damage, heat, firing | Infer outcomes from Java armor/heat deltas; compare weapon fireability + TNs |
| 3 | statistical | Compare distributions over many games |

### Current status (known divergences)

- **LOS**: 0% mismatch rate. Ported Java's geometric hex-line tracing algorithm (`IdealHex.isIntersectedBy()` cross-product test + `Coords.intervening()` neighbor walk). Divided-line detection via `degree % 60 == 30`, triplet evaluation matching `losDivided()`, adjacency via actual hex distance. LosTable precomputation ~2.3s (up from ~0.7s with the old cube-coordinate interpolation).
- **Legal moves**: Two algorithms available via `algorithm` parameter in `enumerate_moves()`:
  - **`"deque"`** (default): Faithful port of Java's `LongestPathFinder` with lazy Pareto-deque relaxation. Key features: **lazy relaxation** matching Java's `AbstractPathFinder.run()` (candidates relaxed at pop time, not at generation time); separate forward and backward passes (backward limited to walk MP, matching Java's `BackwardStep.setRunProhibited(true)`); backward movement blocked across elevation changes (matching `MoveStep.isMovementPossible()`); per-pass walk/run dedup matching Java's `addWalkAndRunPaths`; turn constraints matching Java (no opposite turn, max 3 consecutive); **enemy-hex blocking** during pathfinding matches Java's `MovePathLegalityFilter` → `isMovementPossible()` (Mechs cannot enter enemy-occupied hexes, preventing exploration through the enemy); stacking filter at output prevents ending on enemy hex (defense-in-depth). FIFO insertion counter for PQ tie-breaking approximates Java's `PriorityQueue` binary heap ordering for comparator-equal entries. ~2.1ms/call.
  - **`"bfs"`**: Multi-objective BFS, forward-only. Fast (~1.0ms/call), ~97% hex match vs Java.
  - **Deque match rate**: **100% for standing moves, 100% for prone moves on clean entity state**. Mismatches only occur on "partial-move" steps — when Java gives a prone entity a second movement turn in the same round after a mid-movement fall (PSR failure). In that case, the entity has accumulated state (`mpUsed > 0`, `movedBackwards=true`) that restricts its available moves. Python always enumerates from a clean state (`mpUsed=0`) because the sim doesn't model mid-movement falls (see PSR simplification above). These steps are detected by `mp_used > 0` in the observation and skipped during cross-validation. **Zero java-only hexes** (Python never misses a move Java finds). PSR triggers match Java's `SharedUtility.doPSRCheck()`: no false triggers for running-in-heavy-woods or forward-elevation-change (these are server-side enforcement only, not in pre-movement PSR). Standing-still emits only the original facing (matching Java's empty `MovePath`). GET_UP costs 2 MP (matching `GetUpStep.java`, not `ceil(walk/2)`).
  - **Prone (deque)**: Models Java's `hasJustStood` free-turn chain: starts from a single facing (the entity's current facing) and chains through free turns with proper turn constraints (no opposite turn, max 3 consecutive). The `_STEP_JS_TURN_LEFT/RIGHT` step types track the "just stood" state (free turns until first FORWARD). This replaces the previous all-6-facings seed, which missed turn constraints on the opposite facing (3 turns away, at max consecutive) causing sim-only divergence.
  - **`_apply_move_legality_filter()`**: Post-filter matching `MoveStep.compileIllegal()` edge cases for damaged units: gyro destroyed (2 hits) restricts to turn-one-hex-side when prone or stand-still when standing; both arms + leg destroyed prevents standing up. Also, Java's `shutdown` status is now reported in observations, preventing move enumeration for shut-down Mechs.
  - **PSR fall tolerance**: Both algorithms compute cumulative PSR for elevation changes ≥2 levels and filter paths with `success_probability < 0.3`, matching Java's `getMovePathSuccessProbability() < FALL_TOLERANCE`. The GET_UP PSR is excluded (matching Java's skip of "getting up" rolls).
  - **Path storage**: Each move dict includes a `"path"` field — a tuple of `(x, y)` hexes entered via FORWARD steps during BFS/deque enumeration. Used by `game.py` to walk the path hex-by-hex for PSR fall resolution. Paths are short (typically 3-5 hexes for Trebuchet).
  - **`max_moves` cap removed**: `enumerate_moves()` returns all paths; truncation for the RL observation space is handled by the env layer (`config.max_legal_moves`).
  - Leg actuator damage reduces MP matching Java's `BipedMek.getWalkMP()` (hip = halve MP, each non-hip actuator crit = -1 MP).
- **Firing**: `compute_to_hit()` includes base gunnery, range modifier, min range penalty, attacker movement (+1/+2 walk/run), TMM, heat gunnery modifier, terrain (intervening + target woods with elevation gating), attacker prone (+2, leg weapons impossible), target prone (-2 adjacent / +1 at range), target immobile (-4), arm actuator damage (shoulder +4, upper/lower arm +1 each), sensor damage (+2 per sensor hit). `resolve_firing()` blocks all fire when sensors are destroyed (2+ hits) and enforces prone arm restriction (biped can only fire arm weapons from one arm when prone). Validated against Java's `WeaponAttackAction.toHit()` via `firing_report` telemetry. Known gaps: rare arc edge cases.
- Board, unit template, distances, to-hit components, damage consistency, heat consistency all **pass**.

### Java-side per-move debug logging

Enable with `-Drl.debug.moves=true` JVM flag. Logs `[rl-moves-debug]` lines with full per-move details (destination, facing, MP, hexes moved, walk/run, PSR, step sequence) and `[rl-moves-filtered]` lines for paths filtered by legality, building collapse, or PSR with trigger details.

## Princess AI Scoring Debug Logs

Each time the opponent (Princess) selects a move, the Java log emits a `[rl-princess-debug]` line with all scoring components for the **chosen** path. This is invaluable for comparing `princess.py` against Java's `BasicPathRanker.rankPath()`.

```bash
grep "rl-princess-debug" ../megamek/rl_java_9999.log
```

**Example output:**
```
[rl-princess-debug] entity=Trebuchet TBT-5S (Princess) dest=(13,4) facing=1 rank=-5.33
  successProb=1.0000 fallMod=0.00
  myFiring=11.34 myPhysical=0.00 damageExpTotal=11.34 dmgExpPath=0.00
  braveryMod=5.67 closestEnemyDist=2.00 aggressionMod=5.00
  facingDiff=0.00 facingMod=0.00
  herdingMod=6.00 movementMod=0.00
  friendsX=13 friendsY=4 friendsDist=0.0 utility=-5.33
```

**Field reference (maps to Java's `BasicPathRanker.rankPath()` scores):**

| Field | Java source | Formula |
|-------|-------------|---------|
| `successProb` | PSR success probability (1.0 = no fall risk) | Product of all piloting rolls |
| `fallMod` | `(1 - successProb) * fallShame` (500 default) | Subtracted from utility |
| `myFiring` | `calculateMyDamagePotential()` | `maxDamageAtRange × 0.42` (quick estimate) |
| `myPhysical` | `calculateMyKickDamagePotential()` | `kickDmg × hitProb` if adjacent |
| `damageExpTotal` | Sum of all enemy damage estimates | Firing × discount + kick + hazards |
| `dmgExpPath` | `calculateMovePathPSRDamage()` | Fall damage from PSR failures on path |
| `braveryMod` | `successProb × (firing + physical) × 1.5 - damageExpTotal` | Added to utility |
| `closestEnemyDist` | Hex distance to nearest enemy | Used for aggression |
| `aggressionMod` | `closestEnemyDist × 2.5` | Subtracted from utility |
| `facingDiff` | Hexsides off from optimal (after tolerance=1) | 0-2 range |
| `facingMod` | `50 × facingDiff` | Subtracted from utility |
| `herdingMod` | `distToFriends × 1.0` | In 1v1: dist to own position (self-herding) |
| `movementMod` | `TMM × (selfPreservation + favorHigherTMM)` | Usually 0 (enemies visible) |
| `friendsX/Y` | Median friendly position | In 1v1: the unit's own current position |
| `friendsDist` | Distance from own current pos to friendsCoords | -1.0 if no friends (round 1) |
| `utility` | Final score = `-fall + bravery - aggr - herding + movement - facing` | Higher = better |

**Quick estimate mode**: `RLPrincess.useQuickDamageEstimate()` returns `true`, so all damage estimates use `maxDamageAtRange × 0.42` (approximate average hit probability) rather than weapon-by-weapon to-hit computation. `getMaxDamageAtRange` sums weapon damage for all weapons in range, ignoring firing arcs. Cluster weapons (SRM/LRM) contribute `rackSize` (missile count), not `rackSize × perMissileDamage`.

**Comparing Python vs Java**: Run a game, then side-by-side the Java log values with Python's `score_move()` output. The remaining divergences are typically in `closestEnemyDist` (enemy position reconstruction) rather than the scoring formula itself. See `tests/sim_validation/test_princess_behavior.py` for the automated comparison framework.

This logging is defined in `../megamek/src/megamek/client/bot/princess/Princess.java` in `calculateMoveTurn()`, right after `getBestPath()`.
