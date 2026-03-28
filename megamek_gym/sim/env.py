"""Gymnasium environment wrapping the Python MegaMek simulator.

Drop-in replacement for MegaMekEnv (Java bridge) — same observation and
action spaces, same reward computation.
"""

from __future__ import annotations

import gymnasium
import numpy as np

from megamek_gym.config import MegaMekConfig
from megamek_gym.observation import (
    UNIT_FEATURES,
    GLOBAL_FEATURES,
    DEST_FEATURES,
    FACING_FEATURES,
    flatten_observation_hierarchical,
    compute_obs_size_hierarchical,
    identify_rl_owner,
    _group_moves_by_destination,
)
from megamek_gym.reward import CompositeReward
from megamek_gym.sim.game import Game


class MegaMekSimEnv(gymnasium.Env):
    """Pure-Python MegaMek simulator environment.

    Same interface as MegaMekEnv but runs entirely in Python —
    no JVM, no TCP, no subprocess overhead.
    """

    metadata = {"render_modes": []}

    def __init__(
        self,
        config: MegaMekConfig | None = None,
        *,
        # Backwards-compatible kwargs for direct usage (tests, bench_sim)
        rl_unit: str = "Trebuchet TBT-5S",
        opponent_unit: str = "Trebuchet TBT-5S",
        rl_fixed_coords: tuple[int, int] = (14, 1),
        opponent_fixed_coords: tuple[int, int] = (1, 15),
        max_game_rounds: int = 40,
        max_destinations: int = 125,
        board_width: int = 16,
        board_height: int = 17,
        **kwargs,
    ) -> None:
        super().__init__()

        if config is not None:
            # Config-based construction (from train_ppo.py)
            self.config = config
            self.max_destinations = config.max_destinations
            self.board_width = config.resolved_board_width
            self.board_height = config.resolved_board_height

            self._game = Game(
                rl_unit_name=config.rl_unit,
                opponent_unit_name=config.opponent_unit,
                rl_start=config.rl_fixed_coords or (14, 1),
                opp_start=config.opponent_fixed_coords or (1, 15),
                max_rounds=config.max_game_rounds,
            )
        else:
            # Backwards-compatible kwargs construction
            self.config = None
            self.max_destinations = max_destinations
            self.board_width = board_width
            self.board_height = board_height

            self._game = Game(
                rl_unit_name=rl_unit,
                opponent_unit_name=opponent_unit,
                rl_start=rl_fixed_coords,
                opp_start=opponent_fixed_coords,
                max_rounds=max_game_rounds,
            )

        # Observation and action spaces (matching Java bridge env)
        obs_size = compute_obs_size_hierarchical(
            self.board_width, self.board_height, self.max_destinations
        )
        self.observation_space = gymnasium.spaces.Box(
            low=-np.inf, high=np.inf, shape=(obs_size,), dtype=np.float32
        )
        # MultiDiscrete: [destination_index (0..max_dest-1), facing (0..5)]
        self.action_space = gymnasium.spaces.MultiDiscrete(
            [self.max_destinations, 6]
        )

        # Reward function
        self._reward_fn = CompositeReward()

        # State
        self._rl_owner: int = -1
        self._prev_obs: dict = {}
        self._curr_obs: dict = {}
        self._last_raw_obs: dict | None = None
        self._n_legal_moves: int = 0
        self._destinations: list = []
        self._dest_lookup: dict = {}

    @property
    def reward_fn(self) -> CompositeReward:
        return self._reward_fn

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        if seed is not None:
            self._game.rng.seed(seed)

        obs_dict = self._game.reset(seed=seed)
        self._last_raw_obs = obs_dict

        self._rl_owner = identify_rl_owner(
            obs_dict, obs_dict["active_entity_id"]
        )
        self._reward_fn.reset()
        self._reward_fn.set_rl_owner(self._rl_owner)

        self._prev_obs = {}
        self._curr_obs = obs_dict

        flat_obs = self._flatten(obs_dict)
        info = self._build_info(obs_dict)

        return flat_obs, info

    def step(self, action):
        # Decode hierarchical action [dest_idx, facing_idx]
        dest_idx, facing_idx = int(action[0]), int(action[1])

        # Map hierarchical action to flat move index using the cached
        # legal moves from the observation the agent saw
        move_idx = self._resolve_action(dest_idx, facing_idx)

        # Execute game step (uses game's cached legal moves internally)
        self._prev_obs = self._curr_obs
        obs_dict = self._game.step(move_idx)
        self._curr_obs = obs_dict
        self._last_raw_obs = obs_dict

        terminated = obs_dict.get("terminated", False)
        truncated = obs_dict.get("truncated", False)

        # Compute reward
        reward = self._reward_fn.compute(self._prev_obs, obs_dict, terminated)

        flat_obs = self._flatten(obs_dict)
        info = self._build_info(obs_dict)

        return flat_obs, reward, terminated, truncated, info

    def _resolve_action(self, dest_idx: int, facing_idx: int) -> int:
        """Map hierarchical (dest, facing) action to a flat move index."""
        if not self._destinations:
            return 0

        # Clamp dest_idx
        dest_idx = min(dest_idx, len(self._destinations) - 1)
        if dest_idx < 0:
            return 0

        # Look up the move index for this (dest, facing)
        flat_idx = self._dest_lookup.get((dest_idx, facing_idx))
        if flat_idx is not None:
            return flat_idx

        # Facing not available for this dest — pick any available facing
        dest = self._destinations[dest_idx]
        available = dest["facing_options"]
        if available:
            return next(iter(available.values()))

        return 0

    def _flatten(self, obs_dict: dict) -> np.ndarray:
        """Flatten observation dict to fixed-size array."""
        legal_moves = obs_dict.get("legal_moves", [])
        self._n_legal_moves = len(legal_moves)

        # Cache destination grouping for _resolve_action and action_masks
        if legal_moves:
            walk_mp = self._game.rl_unit.walk_mp
            self._destinations, self._dest_lookup = _group_moves_by_destination(
                legal_moves, walk_mp
            )
        else:
            self._destinations = []
            self._dest_lookup = {}

        return flatten_observation_hierarchical(
            obs_dict,
            self._rl_owner,
            board_width=self.board_width,
            board_height=self.board_height,
            legal_moves=legal_moves,
            max_destinations=self.max_destinations,
        )

    def action_masks(self) -> dict:
        """Return action masks for hierarchical action space."""
        max_dest = self.max_destinations
        dest_mask = np.zeros(max_dest, dtype=bool)
        facing_mask = np.zeros((max_dest, 6), dtype=bool)
        n = min(len(self._destinations), max_dest)
        for i in range(n):
            dest_mask[i] = True
            for facing in self._destinations[i]["facing_options"]:
                facing_mask[i, facing] = True
        return {"dest_mask": dest_mask, "facing_mask": facing_mask}

    def _build_info(self, obs_dict: dict) -> dict:
        """Build info dict (vector-safe types only for AsyncVectorEnv).

        Matches MegaMekEnv._build_info output shape so train_ppo.py works
        with both backends.
        """
        terminated = obs_dict.get("terminated", False)
        truncated = obs_dict.get("truncated", False)
        game_round = obs_dict.get("round", 0)

        game_outcome = 0
        if terminated or truncated:
            outcome_str = obs_dict.get("game_outcome", "UNKNOWN")
            game_outcome = {"WIN": 1, "LOSS": -1, "DRAW": 0}.get(outcome_str, 0)

        moves_truncated = max(0, len(self._destinations) - self.max_destinations)

        return {
            "action_mask": self.action_masks(),
            "round": game_round,
            "phase": obs_dict.get("phase", "MOVEMENT"),
            "n_legal_moves": self._n_legal_moves,
            "n_destinations": len(self._destinations),
            "moves_truncated": moves_truncated,
            "game_outcome": game_outcome,
            "game_rounds": game_round,
            "java_crash": 0,
            "early_termination": 0,
            "auto_wake_count": 0,
        }
