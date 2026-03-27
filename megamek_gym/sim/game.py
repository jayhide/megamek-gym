"""Game loop: phase sequencing, initiative, turn management."""

from __future__ import annotations

import random

from megamek_gym.sim.board import BOARD, Board
from megamek_gym.sim.firing import d6, resolve_firing
from megamek_gym.sim.heat import apply_heat, check_overheat, dissipate_heat
from megamek_gym.sim.los import LosTable
from megamek_gym.sim.movement import enumerate_moves
from megamek_gym.sim.princess import select_move
from megamek_gym.sim.unit import UNIT_TEMPLATES, Unit


class Game:
    """Manages a single 1v1 mech game."""

    def __init__(
        self,
        rl_unit_name: str = "Trebuchet TBT-5S",
        opponent_unit_name: str = "Trebuchet TBT-5S",
        rl_start: tuple[int, int] = (14, 1),
        opp_start: tuple[int, int] = (1, 15),
        max_rounds: int = 40,
        rng: random.Random | None = None,
    ) -> None:
        self.board: Board = BOARD
        self.max_rounds = max_rounds
        self.rng = rng or random.Random()

        # LOS table (computed once, reused across resets)
        self._los_table: LosTable | None = None

        # Unit setup
        self._rl_start = rl_start
        self._opp_start = opp_start

        # Create units
        rl_tmpl = UNIT_TEMPLATES[rl_unit_name]
        opp_tmpl = UNIT_TEMPLATES[opponent_unit_name]
        self.rl_unit = Unit(template=rl_tmpl, entity_id=1, owner=0)
        self.opp_unit = Unit(template=opp_tmpl, entity_id=2, owner=1)

        # Game state
        self.round: int = 0
        self.rl_moves_first: bool = False
        self.terminated: bool = False
        self.truncated: bool = False
        self.game_outcome: str | None = None

        # Post-movement enemy position tracking
        self.prev_round_enemy_x: int = -1
        self.prev_round_enemy_y: int = -1
        self.prev_round_enemy_facing: int = -1

        # Board observation data (cached, never changes)
        self._board_hexes: list[dict] | None = None

        # Cached RL legal moves for the current observation
        # (avoids re-enumerating in step())
        self._cached_rl_moves: list[dict] = []

    @property
    def los_table(self) -> LosTable:
        if self._los_table is None:
            self._los_table = LosTable(self.board)
        return self._los_table

    @property
    def board_hexes(self) -> list[dict]:
        if self._board_hexes is None:
            self._board_hexes = self.board.to_obs_hexes()
        return self._board_hexes

    def reset(self, seed: int | None = None) -> dict:
        """Reset the game to initial state. Returns the first observation."""
        if seed is not None:
            self.rng = random.Random(seed)

        self.rl_unit.reset_state()
        self.opp_unit.reset_state()

        # Deploy units
        self.rl_unit.deploy(self._rl_start[0], self._rl_start[1], 3)
        self.opp_unit.deploy(self._opp_start[0], self._opp_start[1], 0)

        self.round = 1
        self.terminated = False
        self.truncated = False
        self.game_outcome = None
        self.prev_round_enemy_x = -1
        self.prev_round_enemy_y = -1
        self.prev_round_enemy_facing = -1

        self._roll_initiative()
        return self._build_observation()

    def step(self, action: int) -> dict:
        """Execute one step: RL moves, opponent moves, firing, heat.

        The action indexes into the cached legal moves from the previous
        observation (self._cached_rl_moves).
        """
        if self.terminated or self.truncated:
            return self._build_terminal_observation()

        # Use cached RL moves (from the observation the agent saw)
        rl_moves = self._cached_rl_moves

        if not rl_moves:
            action = -1
        else:
            action = max(0, min(action, len(rl_moves) - 1))

        # Enumerate opponent moves
        opp_moves = enumerate_moves(self.opp_unit, self.board, self.rl_unit)

        # Move order based on initiative
        if self.rl_moves_first:
            self._execute_move(self.rl_unit, rl_moves, action)
            opp_action = self._select_opponent_move(opp_moves)
            self._execute_move(self.opp_unit, opp_moves, opp_action)
        else:
            opp_action = self._select_opponent_move(opp_moves)
            self._execute_move(self.opp_unit, opp_moves, opp_action)
            self._execute_move(self.rl_unit, rl_moves, action)

        # Capture post-movement enemy position
        self.prev_round_enemy_x = self.opp_unit.x
        self.prev_round_enemy_y = self.opp_unit.y
        self.prev_round_enemy_facing = self.opp_unit.facing

        # Firing phase (simultaneous)
        self._resolve_firing()

        if self._check_game_end():
            return self._build_terminal_observation()

        # Heat phase
        self._resolve_heat()

        if self._check_game_end():
            return self._build_terminal_observation()

        # End of round
        self.rl_unit.clear_turn_state()
        self.opp_unit.clear_turn_state()
        self.round += 1

        if self.round > self.max_rounds:
            self.truncated = True
            self.game_outcome = "DRAW"
            return self._build_terminal_observation()

        self._roll_initiative()
        return self._build_observation()

    def _roll_initiative(self) -> None:
        while True:
            rl_roll = d6(1, self.rng)
            opp_roll = d6(1, self.rng)
            if rl_roll != opp_roll:
                self.rl_moves_first = rl_roll < opp_roll
                return

    def _execute_move(self, unit: Unit, moves: list[dict], action: int) -> None:
        if not moves or action < 0:
            return

        move = moves[action]
        old_x, old_y = unit.x, unit.y

        unit.x = move["dest_x"]
        unit.y = move["dest_y"]
        unit.facing = move["facing"]
        unit.mp_used = move["mp_used"]

        from megamek_gym.reward import hex_distance
        unit.moved_hexes = hex_distance(old_x, old_y, unit.x, unit.y)

        walk_mp = unit.walk_mp
        if unit.mp_used == 0:
            unit.movement_type = "none"
        elif unit.mp_used <= walk_mp:
            unit.movement_type = "walk"
        else:
            unit.movement_type = "run"

        if unit.prone and unit.mp_used > 0:
            unit.prone = False

    def _select_opponent_move(self, moves: list[dict]) -> int:
        if not moves:
            return 0
        return select_move(
            moves, self.opp_unit.to_obs_dict(), self.rl_unit.to_obs_dict(),
            self.board, self.board_hexes, self.los_table,
        )

    def _resolve_firing(self) -> None:
        if self.rl_unit.destroyed or self.opp_unit.destroyed:
            return

        rl_result = resolve_firing(
            self.rl_unit, self.opp_unit, self.board, self.los_table, self.rng
        )
        opp_result = resolve_firing(
            self.opp_unit, self.rl_unit, self.board, self.los_table, self.rng
        )

        apply_heat(self.rl_unit, rl_result["heat_generated"])
        apply_heat(self.opp_unit, opp_result["heat_generated"])

    def _resolve_heat(self) -> None:
        for unit in (self.rl_unit, self.opp_unit):
            dissipate_heat(unit)
            if unit.heat >= 14:
                effects = check_overheat(unit, self.rng)
                if effects.get("ammo_explosion"):
                    unit.destroyed = True

    def _check_game_end(self) -> bool:
        rl_dead = self.rl_unit.destroyed
        opp_dead = self.opp_unit.destroyed

        if rl_dead and opp_dead:
            self.terminated = True
            self.game_outcome = "DRAW"
            return True
        elif rl_dead:
            self.terminated = True
            self.game_outcome = "LOSS"
            return True
        elif opp_dead:
            self.terminated = True
            self.game_outcome = "WIN"
            return True

        if self.rl_unit.prone:
            if self.rl_unit.loc_destroyed[6] or self.rl_unit.loc_destroyed[7]:
                self.terminated = True
                self.game_outcome = "LOSS"
                return True

        return False

    def _build_observation(self) -> dict:
        """Build observation dict and cache legal moves."""
        rl_moves = enumerate_moves(self.rl_unit, self.board, self.opp_unit)

        # Add LOS info to moves
        ex, ey = self.opp_unit.x, self.opp_unit.y
        for m in rl_moves:
            m["has_los"] = self.los_table.has_los(m["dest_x"], m["dest_y"], ex, ey)

        # Cache for step()
        self._cached_rl_moves = rl_moves

        return {
            "type": "observation",
            "round": self.round,
            "phase": "MOVEMENT",
            "active_entity_id": self.rl_unit.entity_id,
            "board": {
                "width": self.board.width,
                "height": self.board.height,
                "hexes": self.board_hexes,
            },
            "units": [self.rl_unit.to_obs_dict(), self.opp_unit.to_obs_dict()],
            "legal_moves": rl_moves,
            "rl_moves_first": self.rl_moves_first,
            "prev_round_enemy_x": self.prev_round_enemy_x,
            "prev_round_enemy_y": self.prev_round_enemy_y,
            "prev_round_enemy_facing": self.prev_round_enemy_facing,
            "reward": 0.0,
            "terminated": False,
            "truncated": False,
        }

    def _build_terminal_observation(self) -> dict:
        self._cached_rl_moves = []
        return {
            "type": "observation",
            "round": self.round,
            "phase": "END",
            "active_entity_id": self.rl_unit.entity_id,
            "board": {
                "width": self.board.width,
                "height": self.board.height,
                "hexes": self.board_hexes,
            },
            "units": [self.rl_unit.to_obs_dict(), self.opp_unit.to_obs_dict()],
            "legal_moves": [],
            "rl_moves_first": self.rl_moves_first,
            "prev_round_enemy_x": self.prev_round_enemy_x,
            "prev_round_enemy_y": self.prev_round_enemy_y,
            "prev_round_enemy_facing": self.prev_round_enemy_facing,
            "reward": 0.0,
            "terminated": self.terminated,
            "truncated": self.truncated,
            "game_outcome": self.game_outcome,
        }
