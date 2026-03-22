"""Convert variable-structure JSON observations into fixed-size numpy arrays."""

from __future__ import annotations

import numpy as np

from megamek_gym.reward import cover_value, hex_distance, range_quality

BOARD_WIDTH = 16
BOARD_HEIGHT = 17
BOARD_SIZE = BOARD_WIDTH * BOARD_HEIGHT  # 272
UNIT_FEATURES = 55
GLOBAL_FEATURES = 1  # rl_moves_first
MOVE_FEATURES = 10  # dest_x, dest_y, facing, mp_used, prone, dist_to_enemy, range_quality, enemy_range_quality, terrain_cover, elevation_diff
OBS_SIZE = BOARD_SIZE + 2 * UNIT_FEATURES + GLOBAL_FEATURES  # 383 (without move features)


def compute_obs_size(board_width: int, board_height: int, max_legal_moves: int = 0) -> int:
    """Compute observation vector size for a given board.

    When max_legal_moves > 0, includes a block of move features
    (max_legal_moves * MOVE_FEATURES) appended after the unit features.
    """
    return board_width * board_height + 2 * UNIT_FEATURES + GLOBAL_FEATURES + max_legal_moves * MOVE_FEATURES

MAX_ARMOR_LOCATIONS = 8
MAX_WEAPONS = 7


def flatten_observation(
    obs: dict,
    rl_owner_id: int,
    board_width: int = BOARD_WIDTH,
    board_height: int = BOARD_HEIGHT,
    legal_moves: list | None = None,
    max_legal_moves: int = 0,
) -> np.ndarray:
    """Flatten a JSON observation dict into a fixed-size float32 array.

    When max_legal_moves > 0, appends a block of move features
    (max_legal_moves * MOVE_FEATURES) after the unit features.
    """
    obs_size = compute_obs_size(board_width, board_height, max_legal_moves)
    result = np.zeros(obs_size, dtype=np.float32)

    # Board block: elevation per hex, row-major, normalized by /10
    board = obs.get("board", {})
    hexes = board.get("hexes", [])
    for h in hexes:
        x, y = h["x"], h["y"]
        if 0 <= x < board_width and 0 <= y < board_height:
            idx = y * board_width + x
            result[idx] = h.get("elevation", 0) / 10.0

    # Split units into RL vs enemy
    units = obs.get("units", [])
    rl_unit = None
    enemy_unit = None
    for u in units:
        if u["owner"] == rl_owner_id:
            rl_unit = u
        else:
            enemy_unit = u

    # Encode units
    offset = BOARD_SIZE
    if rl_unit is not None:
        _encode_unit(result, offset, rl_unit, board_width, board_height)
    offset += UNIT_FEATURES
    if enemy_unit is not None:
        _encode_unit(result, offset, enemy_unit, board_width, board_height)
    offset += UNIT_FEATURES

    # Global features
    result[offset] = float(obs.get("rl_moves_first", False))
    offset += GLOBAL_FEATURES

    # Encode move features
    if max_legal_moves > 0 and legal_moves:
        board_hexes = board.get("hexes", [])
        _flatten_move_features(
            result, offset, legal_moves, max_legal_moves, board_width, board_height,
            rl_unit=rl_unit, enemy_unit=enemy_unit, board_hexes=board_hexes,
        )

    return result


def _flatten_move_features(
    buf: np.ndarray,
    offset: int,
    legal_moves: list,
    max_legal_moves: int,
    board_width: int,
    board_height: int,
    rl_unit: dict | None = None,
    enemy_unit: dict | None = None,
    board_hexes: list | None = None,
) -> None:
    """Write normalized move features into buf[offset:offset + max_legal_moves * MOVE_FEATURES].

    Each move gets 10 floats:
      Kinematic: dest_x/W, dest_y/H, facing/5, mp_used/20, prone
      Tactical:  dist_to_enemy, range_quality, enemy_range_quality, terrain_cover, elevation_diff
    Unused slots (index >= len(legal_moves)) stay zero.
    """
    # Pre-compute enemy info and elevation lookup
    has_enemy = (enemy_unit is not None
                 and enemy_unit.get("x", -1) >= 0
                 and enemy_unit.get("y", -1) >= 0)
    if has_enemy:
        ex, ey = enemy_unit["x"], enemy_unit["y"]
        enemy_facing = enemy_unit.get("facing")
    else:
        ex = ey = enemy_facing = None

    max_dim = max(board_width, board_height)
    elev_map: dict[tuple[int, int], float] = {}
    if board_hexes:
        for h in board_hexes:
            elev_map[(h["x"], h["y"])] = h.get("elevation", 0)

    enemy_elev = elev_map.get((ex, ey), 0) if has_enemy else 0

    n = min(len(legal_moves), max_legal_moves)
    for i in range(n):
        m = legal_moves[i]
        base = offset + i * MOVE_FEATURES
        dest_x = m.get("dest_x", 0)
        dest_y = m.get("dest_y", 0)
        facing = m.get("facing", 0)

        # Kinematic features (5)
        buf[base] = dest_x / board_width
        buf[base + 1] = dest_y / board_height
        buf[base + 2] = facing / 5.0
        buf[base + 3] = m.get("mp_used", 0) / 20.0
        buf[base + 4] = float(m.get("prone", False))

        # Tactical features (5)
        if has_enemy:
            dist = hex_distance(dest_x, dest_y, ex, ey)
            buf[base + 5] = dist / max_dim

            # RL weapon effectiveness from this hypothetical position
            buf[base + 6] = range_quality(
                rl_unit, dist,
                target_x=ex, target_y=ey,
                unit_x=dest_x, unit_y=dest_y,
                unit_facing=facing,
            ) if rl_unit else 0.0

            # Enemy weapon effectiveness at this distance
            buf[base + 7] = range_quality(
                enemy_unit, dist,
                target_x=dest_x, target_y=dest_y,
                unit_x=ex, unit_y=ey,
                unit_facing=enemy_facing,
            )

            # Terrain cover at destination
            if board_hexes:
                buf[base + 8] = cover_value(board_hexes, dest_x, dest_y) / 2.0

            # Elevation advantage
            dest_elev = elev_map.get((dest_x, dest_y), 0)
            buf[base + 9] = (dest_elev - enemy_elev) / 10.0


def _encode_unit(
    buf: np.ndarray,
    offset: int,
    unit: dict,
    board_width: int,
    board_height: int,
) -> None:
    i = offset

    # Position normalized 0-1 (use 0 for undeployed units with -1 coords)
    x = unit.get("x", -1)
    y = unit.get("y", -1)
    buf[i] = max(0.0, x) / board_width
    buf[i + 1] = max(0.0, y) / board_height
    i += 2

    # Facing one-hot (6 directions, 0-5)
    facing = unit.get("facing", 0)
    if 0 <= facing < 6:
        buf[i + facing] = 1.0
    i += 6

    # Movement points normalized by /20
    buf[i] = unit.get("mp_walk", 0) / 20.0
    buf[i + 1] = unit.get("mp_run", 0) / 20.0
    buf[i + 2] = unit.get("mp_jump", 0) / 20.0
    i += 3

    # Heat normalized by /30
    buf[i] = unit.get("heat", 0) / 30.0
    i += 1

    # Status flags
    buf[i] = float(unit.get("prone", False))
    buf[i + 1] = float(unit.get("destroyed", False))
    buf[i + 2] = float(unit.get("deployed", False))
    buf[i + 3] = float(unit.get("retreated", False))
    i += 4

    # Armor locations (8 max, 4 values each)
    armor_locs = unit.get("armor", [])
    for loc_idx in range(MAX_ARMOR_LOCATIONS):
        if loc_idx < len(armor_locs):
            loc = armor_locs[loc_idx]
            armor_max = loc.get("armor_max", 1)
            internal_max = loc.get("internal_max", 1)
            rear_max = loc.get("rear_armor_max", 0)

            buf[i] = loc.get("armor", 0) / max(armor_max, 1)
            buf[i + 1] = max(0.0, loc.get("internal", 0)) / max(internal_max, 1)
            buf[i + 2] = loc.get("rear_armor", 0) / max(rear_max, 1) if rear_max > 0 else 0.0
            buf[i + 3] = float(
                loc.get("armor", 0) <= 0
                and loc.get("internal", 0) <= 0
            )
        i += 4

    # Weapon destroyed flags (7 max)
    weapons = unit.get("weapons", [])
    for w_idx in range(MAX_WEAPONS):
        if w_idx < len(weapons):
            buf[i] = float(weapons[w_idx].get("destroyed", False))
        i += 1


def identify_rl_owner(obs: dict, active_entity_id: int) -> int:
    """Determine the RL player's owner ID from the first observation."""
    for unit in obs.get("units", []):
        if unit["id"] == active_entity_id:
            return unit["owner"]
    raise ValueError(
        f"Active entity {active_entity_id} not found in units"
    )
