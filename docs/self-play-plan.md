# Self-Play Training: Implementation Plan

## Context

Currently the RL agent trains against MegaMek's built-in Princess AI. Self-play trains the agent against itself using a shared policy with perspective-flipped observations. The opponent improves as the agent improves, preventing overfitting to Princess's fixed strategy.

**Design decisions:**
- Shared policy (one network plays both sides, observations flipped so agent always sees itself as "rl_unit")
- Two TCP sockets per game (one per player, ports P and P+1)
- Gymnasium-compatible stepping (`step()` alternates which player's transition it returns)
- Reward functions work unchanged (perspective flip means each player sees itself as RL agent)

## Current architecture

```
Python train_ppo.py
  └── AsyncVectorEnv (N envs, each in subprocess)
        └── MegaMekEnv (one TCP socket on port P+i)
              └── JVM (RLBotClient + Princess AI)
```

Each `env.step()` sends one action for the RL bot; Princess acts autonomously in Java. Python sees one observation per step (after both players have moved). Observations already include both units (rl_unit + enemy_unit with owner IDs). Reward is computed Python-side from the RL agent's perspective only.

## Self-play architecture

```
Python train_ppo.py
  └── AsyncVectorEnv (N envs, each in subprocess)
        └── SelfPlayEnv (two TCP sockets on ports P+2i and P+2i+1)
              └── JVM (RLBotClient1 + RLBotClient2)
```

Each `env.step()` handles one player's action. Turns alternate naturally — the JVM sends observations on the active player's socket. Player 2's observations are perspective-flipped so the shared policy always sees itself as "rl_unit".

---

## Step 1: Java — Add `opponentType` Gradle parameter

**File:** `RLGameRunner.java` (arg parsing in `main()`, ~line 100)

Add a new positional arg `opponentType` to the pipe-delimited args (after existing params). Values: `"princess"` (default) or `"rl"`. Store as instance field. No behavioral change yet — Princess is still created regardless.

**Verification:** Run existing smoke test to confirm no regression.

---

## Step 2: Java — Create second RLBotClient when `opponentType=rl`

**File:** `RLGameRunner.java` (lines 259-269 in `run()`, lines 363-372 in persistent loop)

When `opponentType == "rl"`:
- Instead of creating Princess, create a second `RLBotClient("RLBot2", serverIP, serverPort, rlPort + 1)`
- Call `startBridge()` and `waitForAgent()` on RLBot2 (same as RLBot1)
- Connect to game server, set player color/starting position
- Set RLBot2's firing strategy (same as RLBot1, or configurable separately)

When `opponentType == "princess"` (default): no change from current behavior.

Both code paths exist: `run()` (single game) and the persistent game loop (~line 328+). Both need the conditional.

**Key detail:** RLBot2 needs its own `RewardCalculator` instance (already happens in `RLBotClient.initialize()` since it uses `getLocalPlayer().getId()`).

**Verification:** Start JVM with `opponentType=rl`, confirm two bridge sockets open (ports 9999 and 10000). No Python agent needed yet — just verify both sockets accept connections.

---

## Step 3: Java — Persistent reset with two RL bots

**File:** `RLGameRunner.java` (persistent game loop, ~line 328+)

Currently the persistent loop waits for a reset message from the single RL agent's socket. With two RL bots:
- Wait for reset from the primary connection (RLBot1) only — this is the "controller"
- On reset, tear down server + both bot clients
- Recreate server, watcher, both RLBotClients (reusing existing sockets/readers/writers)
- Start new game

**Fallback:** If either bot's socket dies during reset, exit the persistent loop (Python will cold-restart the JVM).

**Verification:** Run two Python scripts that connect, play a game, send reset, play another game.

---

## Step 4: Python — Config changes

**File:** `megamek_gym/config.py`

Add to `MegaMekConfig`:
- `opponent_type: str = "princess"` — values: `"princess"` or `"rl"`

Update Gradle arg construction in `java_process.py` to pass `opponentType` to Java.

**File:** `configs/selfplay.yaml` — new config file for self-play experiments.

**Verification:** `poetry run pytest tests/test_config.py`

---

## Step 5: Python — Observation perspective flipping

**File:** `megamek_gym/observation.py`

Add a `flip_perspective(obs, config)` function that:
- Swaps the two 55-feature unit blocks (positions `[W*H .. W*H+55]` and `[W*H+55 .. W*H+110]`)
- Flips the `rl_moves_first` global feature (index `W*H+110`): `1.0 → 0.0`, `0.0 → 1.0`
- Returns a new observation array

This is a simple slice swap — no structural changes to the observation format.

**Verification:** Unit test — encode an observation, flip it, verify unit features are swapped. `poetry run pytest tests/test_observation.py`

---

## Step 6: Python — SelfPlayEnv

**File:** `megamek_gym/selfplay_env.py` (new file)

A Gymnasium env that wraps two socket connections to one JVM:

```python
class SelfPlayEnv(gymnasium.Env):
    # Two sockets: self._sock1 (port P), self._sock2 (port P+1)
    # Shared observation/action spaces (same as MegaMekEnv)

    def reset():
        # Cold start: launch JVM with opponentType=rl
        # Connect both sockets
        # Read first observation from whichever player moves first
        # Return that observation (perspective-flipped if player 2)

    def step(action):
        # Send action on current player's socket
        # Read response observation from same socket
        # If game not over, read next observation from OTHER player's socket
        #   (that player's turn is next)
        # Perspective-flip if player 2
        # Return (obs, reward, terminated, truncated, info)
        # info["active_player"] indicates whose transition this is
```

**Stepping model:** Each `step()` call handles one player's action and returns one transition. The training loop calls `step()` twice per round (once per player). Both transitions use the shared policy since observations are perspective-normalized.

**Key details:**
- `_read_obs()` blocks on whichever socket is active (turns are sequential, no `select()` needed)
- Player 2's observations get `flip_perspective()` applied before returning
- Player 2's reward is computed from the flipped observation (same reward functions)
- Action mask comes from each player's own legal moves
- Terminal handling: when game ends, both sockets receive terminal observations

**Verification:** Smoke test with two random agents playing a full game via `SelfPlayEnv`.

---

## Step 7: Python — Training loop modifications

**File:** `train_ppo.py`

Changes to the rollout collection loop:
- Use `SelfPlayEnv` instead of `MegaMekEnv` when `config.opponent_type == "rl"`
- Each `env.step()` returns one player's transition (alternating)
- Both players' transitions go into the same rollout buffer (shared policy)
- No changes to PPO loss computation — it sees a stream of (obs, action, reward, value) tuples from both sides

**Optional enhancement — opponent freezing (V1.1):**
- Maintain `opponent_agent = copy.deepcopy(agent)` updated every K policy updates
- Player 1 uses `agent` (learning), player 2 uses `opponent_agent` (frozen)
- Requires `SelfPlayEnv` to expose which player is active so the loop knows which network to query
- Without freezing, both sides use the latest weights (simpler, may work for initial experiments)

**Verification:** Short training run (`--total-timesteps 500 --num-envs 1`) completes without errors. Check TensorBoard for reasonable reward curves.

---

## Step 8: Testing & validation

1. **Unit tests:** perspective flip in `test_observation.py`
2. **Smoke test:** `SelfPlayEnv` with random actions, full game completes
3. **Integration:** Short self-play training run, verify both players' transitions in rollout buffer
4. **Comparison:** Train against Princess vs self-play for same number of timesteps, compare eval win rates

---

## Risks & mitigations

| Risk | Mitigation |
|------|------------|
| Training instability (non-stationary opponent) | Add opponent freezing in Step 7 if needed |
| MegaMek game phase issues with two RL bots | Test thoroughly in Steps 2-3; both bots behave identically to Princess from the game server's perspective |
| Persistent reset with two bots | Fall back to cold restart on any error |
| `AsyncVectorEnv` compatibility | `SelfPlayEnv` exposes standard Gymnasium API; each env subprocess manages its own two sockets |

## Files modified

| File | Change |
|------|--------|
| `RLGameRunner.java` | Steps 1-3: dual-bot mode, persistent reset |
| `megamek_gym/config.py` | Step 4: `opponent_type` field |
| `megamek_gym/java_process.py` | Step 4: pass `opponentType` to Gradle |
| `megamek_gym/observation.py` | Step 5: `flip_perspective()` |
| `megamek_gym/selfplay_env.py` | Step 6: new file |
| `train_ppo.py` | Step 7: self-play rollout collection |
| `configs/selfplay.yaml` | Step 4: new config |
| `tests/test_observation.py` | Step 8: perspective flip tests |
