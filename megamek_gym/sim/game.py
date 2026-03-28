"""Game loop: phase sequencing, initiative, turn management."""

from __future__ import annotations

import random

from megamek_gym.reward import hex_bearing
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
        move_algorithm: str = "bfs",
    ) -> None:
        self.board: Board = BOARD
        self.max_rounds = max_rounds
        self.rng = rng or random.Random()
        self.move_algorithm = move_algorithm

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

        # Princess herding: friendsCoords is None in round 1 (Java behavior)
        self._opp_friends_coords: tuple[int, int] | None = None

        # Board observation data (cached, never changes)
        self._board_hexes: list[dict] | None = None

        # Cached RL legal moves for the current observation
        # (avoids re-enumerating in step())
        self._cached_rl_moves: list[dict] = []

        # Round event log for transcript/game viewer
        self.round_log: list[dict] = []
        self._last_initiative: dict = {}

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

        # Deploy units facing each other (matching Java's Coords.direction())
        rl_facing = round(hex_bearing(*self._rl_start, *self._opp_start) / 60) % 6
        opp_facing = round(hex_bearing(*self._opp_start, *self._rl_start) / 60) % 6
        self.rl_unit.deploy(self._rl_start[0], self._rl_start[1], rl_facing)
        self.opp_unit.deploy(self._opp_start[0], self._opp_start[1], opp_facing)

        self.round = 1
        self.terminated = False
        self.truncated = False
        self.game_outcome = None
        self.prev_round_enemy_x = -1
        self.prev_round_enemy_y = -1
        self._opp_friends_coords = None
        self.prev_round_enemy_facing = -1

        self.round_log = []
        self._last_initiative = {}

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
        opp_moves = enumerate_moves(self.opp_unit, self.board, self.rl_unit,
                                    algorithm=self.move_algorithm)

        # Move order based on initiative — capture movement info
        if self.rl_moves_first:
            rl_move_info = self._execute_move(self.rl_unit, rl_moves, action)
            opp_action = self._select_opponent_move(opp_moves)
            opp_move_info = self._execute_move(self.opp_unit, opp_moves, opp_action)
        else:
            opp_action = self._select_opponent_move(opp_moves)
            opp_move_info = self._execute_move(self.opp_unit, opp_moves, opp_action)
            rl_move_info = self._execute_move(self.rl_unit, rl_moves, action)

        # Capture post-movement enemy position
        self.prev_round_enemy_x = self.opp_unit.x
        self.prev_round_enemy_y = self.opp_unit.y
        self.prev_round_enemy_facing = self.opp_unit.facing

        # Firing phase (simultaneous)
        rl_firing, opp_firing = self._resolve_firing()

        if self._check_game_end():
            self._append_round_log(rl_move_info, opp_move_info, rl_firing, opp_firing)
            return self._build_terminal_observation()

        # Heat phase
        self._resolve_heat()

        if self._check_game_end():
            self._append_round_log(rl_move_info, opp_move_info, rl_firing, opp_firing)
            return self._build_terminal_observation()

        # Log this round's events before advancing
        self._append_round_log(rl_move_info, opp_move_info, rl_firing, opp_firing)

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

    def _append_round_log(
        self,
        rl_move_info: dict | None,
        opp_move_info: dict | None,
        rl_firing: dict,
        opp_firing: dict,
    ) -> None:
        self.round_log.append({
            "round": self.round,
            "initiative": dict(self._last_initiative),
            "rl_moves_first": self.rl_moves_first,
            "rl_movement": rl_move_info,
            "opp_movement": opp_move_info,
            "rl_firing": rl_firing,
            "opp_firing": opp_firing,
            "unit_states": [self.rl_unit.to_obs_dict(), self.opp_unit.to_obs_dict()],
        })

    def _roll_initiative(self) -> None:
        while True:
            rl_roll = d6(1, self.rng)
            opp_roll = d6(1, self.rng)
            if rl_roll != opp_roll:
                self.rl_moves_first = rl_roll < opp_roll
                self._last_initiative = {
                    "rl_roll": rl_roll,
                    "opp_roll": opp_roll,
                }
                return

    def _execute_move(self, unit: Unit, moves: list[dict], action: int) -> dict | None:
        if not moves or action < 0:
            return None

        move = moves[action]
        old_x, old_y = unit.x, unit.y
        was_prone = unit.prone

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

        return {
            "entity_id": unit.entity_id,
            "owner": unit.owner,
            "name": f"{unit.template.chassis} {unit.template.model}",
            "from_pos": (old_x, old_y),
            "to_pos": (unit.x, unit.y),
            "facing": unit.facing,
            "movement_type": unit.movement_type,
            "prone": unit.prone,
            "was_prone": was_prone,
        }

    def _select_opponent_move(self, moves: list[dict]) -> int:
        if not moves:
            return 0
        # If RL moved first, the enemy (RL) has already moved from Princess's POV
        enemy_has_moved = self.rl_moves_first

        # Pre-compute enemy reachable hexes for unmoved enemy evaluation
        enemy_reachable = None
        if not enemy_has_moved:
            rl_moves = enumerate_moves(self.rl_unit, self.board, self.opp_unit,
                                       algorithm=self.move_algorithm)
            enemy_reachable = {(m["dest_x"], m["dest_y"]) for m in rl_moves}

        result = select_move(
            moves, self.opp_unit.to_obs_dict(), self.rl_unit.to_obs_dict(),
            self.board, self.board_hexes, self.los_table,
            enemy_has_moved=enemy_has_moved,
            friends_coords=self._opp_friends_coords,
            enemy_reachable_hexes=enemy_reachable,
        )
        # Update friends_coords for next round (Java sets this after each move)
        self._opp_friends_coords = (self.opp_unit.x, self.opp_unit.y)
        return result

    def _resolve_firing(self) -> tuple[dict, dict]:
        if self.rl_unit.destroyed or self.opp_unit.destroyed:
            return {}, {}

        rl_result = resolve_firing(
            self.rl_unit, self.opp_unit, self.board, self.los_table, self.rng
        )
        opp_result = resolve_firing(
            self.opp_unit, self.rl_unit, self.board, self.los_table, self.rng
        )

        apply_heat(self.rl_unit, rl_result["heat_generated"])
        apply_heat(self.opp_unit, opp_result["heat_generated"])

        return rl_result, opp_result

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

        if self.opp_unit.prone:
            if self.opp_unit.loc_destroyed[6] or self.opp_unit.loc_destroyed[7]:
                self.terminated = True
                self.game_outcome = "WIN"
                return True

        return False

    def _build_observation(self) -> dict:
        """Build observation dict and cache legal moves."""
        rl_moves = enumerate_moves(self.rl_unit, self.board, self.opp_unit,
                                   algorithm=self.move_algorithm)

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
