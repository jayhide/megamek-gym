"""Run a Java game via MegaMekEnv, collect per-step observation traces."""

from __future__ import annotations

import copy
import random
from dataclasses import dataclass, field
from typing import Callable

import gymnasium

from megamek_gym.config import MegaMekConfig


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
