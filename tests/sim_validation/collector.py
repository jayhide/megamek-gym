"""Run a Java game via MegaMekEnv, collect per-step observation traces."""

from __future__ import annotations

import copy
import json
import logging
import random
from dataclasses import dataclass, field
from typing import Callable

import gymnasium

from megamek_gym.config import MegaMekConfig

logger = logging.getLogger(__name__)


@dataclass
class StepTrace:
    """One step of a Java game."""
    step_idx: int
    raw_obs: dict  # Full Java observation (deep-copied)
    action_taken: int  # Flat move index sent to Java


@dataclass
class GameTrace:
    """Complete trace of a Java game."""
    steps: list[StepTrace] = field(default_factory=list)
    rl_owner_id: int = 0
    config: MegaMekConfig | None = None


def _inject_state_via_env(inner_env, raw_obs: dict, state_spec: dict) -> dict:
    """Send inject_state through the MegaMekEnv's socket and read the updated obs.

    Args:
        inner_env: The unwrapped MegaMekEnv instance (has _sock, _reader).
        raw_obs: Current raw observation (used to resolve entity IDs).
        state_spec: State injection spec with "entities" list (owner-based).

    Returns:
        Updated raw observation dict reflecting the injected state.
    """
    # Resolve owner -> entity ID from observation
    owner_to_id = {}
    for unit in raw_obs.get("units", []):
        owner_to_id[unit["owner"]] = unit["id"]

    entities = []
    for spec in state_spec.get("entities", []):
        owner = spec.get("owner")
        if owner is None:
            continue
        entity_id = owner_to_id.get(owner)
        if entity_id is None:
            logger.warning("inject_state: no entity for owner %d", owner)
            continue
        entry = {k: v for k, v in spec.items() if k != "owner"}
        entry["id"] = entity_id
        entities.append(entry)

    msg = {"type": "inject_state", "entities": entities}
    logger.info("Sending inject_state via env: %d entities", len(entities))
    payload = json.dumps(msg) + "\n"
    inner_env._sock.sendall(payload.encode("utf-8"))

    # Read the updated observation (Java re-enumerates and re-sends)
    line = inner_env._reader.readline()
    if not line:
        logger.warning("No response after inject_state")
        return raw_obs
    updated_obs = json.loads(line)

    # Update the env's cached raw obs so step() stays consistent
    inner_env._last_raw_obs = updated_obs
    logger.info("inject_state applied, updated observation received")
    return updated_obs


def collect_game_trace(
    megamek_dir: str,
    port: int = 9999,
    max_rounds: int = 10,
    action_fn: Callable[[dict, list], int] | None = None,
    rl_unit: str = "Trebuchet TBT-5S",
    opponent_unit: str = "Trebuchet TBT-5S",
    board: str = "Map Set 6/16x17 Woodland",
    rl_fixed_coords: tuple[int, int] = (14, 1),
    opponent_fixed_coords: tuple[int, int] = (1, 15),
    firing_strategy: str = "naive",
    initial_state: dict | None = None,
) -> GameTrace:
    """Play a Java game and return all raw observations.

    Args:
        megamek_dir: Path to MegaMek checkout.
        port: TCP port for the Java bridge.
        max_rounds: Max game rounds before truncation.
        action_fn: Callable(raw_obs, legal_moves) -> move_index.
                   Defaults to always action 0 (stand still).
        rl_unit: RL player's mech.
        opponent_unit: Opponent mech.
        rl_fixed_coords: Fixed starting hex for RL unit.
        opponent_fixed_coords: Fixed starting hex for opponent.
        firing_strategy: "naive" or "princess".
        initial_state: Optional state injection spec. Dict with "entities" list,
            each containing "owner" (int) and optional "heat", "prone", "armor",
            "internal" fields. Entity IDs are resolved from the first observation.

    Returns:
        GameTrace with all step observations.
    """
    config = MegaMekConfig(
        megamek_dir=megamek_dir,
        rl_port=port,
        rl_unit=rl_unit,
        opponent_unit=opponent_unit,
        board=board,
        max_game_rounds=max_rounds,
        rl_fixed_coords=rl_fixed_coords,
        opponent_fixed_coords=opponent_fixed_coords,
        firing_strategy=firing_strategy,
        max_rotating_round_saves=0,
        auto_wake_pilot=True,
    )

    env = gymnasium.make("MegaMekGym/MegaMek-v0", config=config)
    inner = env.unwrapped
    trace = GameTrace(config=config)

    try:
        _obs, _info = env.reset()
        raw = inner._last_raw_obs
        trace.rl_owner_id = inner._rl_owner_id

        # Inject initial state if requested (send via env's socket, read updated obs)
        if initial_state is not None:
            raw = _inject_state_via_env(inner, raw, initial_state)

        step_idx = 0
        trace.steps.append(StepTrace(
            step_idx=step_idx,
            raw_obs=copy.deepcopy(raw),
            action_taken=-1,  # No action for initial obs
        ))

        while True:
            legal_moves = raw.get("legal_moves", [])
            if raw.get("terminated") or raw.get("truncated"):
                break

            if action_fn is not None:
                action = action_fn(raw, legal_moves)
            else:
                action = 0  # Stand still

            action = max(0, min(action, len(legal_moves) - 1)) if legal_moves else 0

            # Step through the Java env (flat action space)
            _obs, _reward, terminated, truncated, _info = env.step(action)
            raw = inner._last_raw_obs
            step_idx += 1

            trace.steps.append(StepTrace(
                step_idx=step_idx,
                raw_obs=copy.deepcopy(raw),
                action_taken=action,
            ))

            if terminated or truncated:
                break

    finally:
        env.close()

    return trace


def random_action_fn(raw_obs: dict, legal_moves: list) -> int:
    """Pick a random legal move."""
    if not legal_moves:
        return 0
    return random.randint(0, len(legal_moves) - 1)
