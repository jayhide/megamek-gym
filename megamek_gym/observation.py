"""Convert variable-structure JSON observations into fixed-size numpy arrays."""

from __future__ import annotations

import numpy as np

from megamek_gym.reward import cover_value, hex_distance, range_quality


def _norm_rq(rq: float) -> float:
    """Normalize range_quality from [-0.5, 1.0] to [0.0, 1.0]."""
    return (rq + 0.5) / 1.5

# Board elevation grid — disabled for MLP (spatial grid is hard for MLPs to use;
# per-dest elevation_diff already captures the decision-relevant signal).
# Re-enable for CNN architecture by setting to True.
INCLUDE_BOARD_ELEVATION = False

BOARD_WIDTH = 16
BOARD_HEIGHT = 17
BOARD_SIZE = BOARD_WIDTH * BOARD_HEIGHT if INCLUDE_BOARD_ELEVATION else 0
UNIT_FEATURES = 60
DEST_FEATURES = 9 # Multi-Discrete Mode
FACING_FEATURES = 1 # Multi-Discrete Mode
GLOBAL_FEATURES = 1  # rl_moves_first
TACTICAL_FEATURES = 6  # hex_distance, rl_range_quality, enemy_range_quality, has_los, relative_elev, round
MOVE_FEATURE_NAMES = [
    "dest_x", "dest_y", "facing", "mp_used",
    "dist_to_enemy", "range_quality", "enemy_range_quality",
    "terrain_cover", "elevation_diff", "has_los",
]
MOVE_FEATURES = len(MOVE_FEATURE_NAMES)
OBS_SIZE = BOARD_SIZE + 2 * UNIT_FEATURES + GLOBAL_FEATURES + TACTICAL_FEATURES  # 121 (without move features)


def _board_block_size(board_width: int, board_height: int) -> int:
    """Size of the board elevation block (0 when INCLUDE_BOARD_ELEVATION is False)."""
    return board_width * board_height if INCLUDE_BOARD_ELEVATION else 0


def compute_obs_size(board_width: int, board_height: int, max_legal_moves: int = 0) -> int:
    """Compute observation vector size for a given board.

    When max_legal_moves > 0, includes a block of move features
    (max_legal_moves * MOVE_FEATURES) appended after the unit features.
    """
    return _board_block_size(board_width, board_height) + 2 * UNIT_FEATURES + GLOBAL_FEATURES + TACTICAL_FEATURES + max_legal_moves * MOVE_FEATURES


def compute_obs_size_hierarchical(board_width: int, board_height: int, max_destinations: int) -> int:
    """Compute observation vector size for hierarchical (dest + facing) action space.

    Layout: [board elevations] + 2 units + global + dest features + facing features.
    """
    base = _board_block_size(board_width, board_height) + 2 * UNIT_FEATURES + GLOBAL_FEATURES + TACTICAL_FEATURES
    return base + max_destinations * DEST_FEATURES + max_destinations * 6 * FACING_FEATURES

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

    board = obs.get("board", {})
    board_hexes = board.get("hexes", [])

    # Build elevation map once for efficient lookup
    elev_map: dict[tuple[int, int], float] = {}
    for h in board_hexes:
        elev_map[(h["x"], h["y"])] = h.get("elevation", 0)

    # Board block: elevation per hex, row-major, normalized by /10
    # (disabled when INCLUDE_BOARD_ELEVATION is False — see module docstring)
    if INCLUDE_BOARD_ELEVATION:
        for h in board_hexes:
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
    offset = _board_block_size(board_width, board_height)
    if rl_unit is not None:
        _encode_unit(result, offset, rl_unit, board_width, board_height,
                     board_hexes=board_hexes, elev_map=elev_map)
    offset += UNIT_FEATURES
    if enemy_unit is not None:
        _encode_unit(result, offset, enemy_unit, board_width, board_height,
                     board_hexes=board_hexes, elev_map=elev_map)
    offset += UNIT_FEATURES

    # Global features
    result[offset] = float(obs.get("rl_moves_first", False))
    offset += GLOBAL_FEATURES

    # Tactical features (relational, computed from both units + board)
    has_los_current = obs.get("has_los_current")
    if has_los_current is None and legal_moves and rl_unit and rl_unit.get("x", -1) >= 0:
        rl_x, rl_y = rl_unit["x"], rl_unit["y"]
        for m in legal_moves:
            if m.get("dest_x") == rl_x and m.get("dest_y") == rl_y:
                has_los_current = m.get("has_los", False)
                break
    _encode_tactical_features(
        result, offset,
        rl_unit=rl_unit, enemy_unit=enemy_unit,
        elev_map=elev_map, board_width=board_width, board_height=board_height,
        game_round=obs.get("round", 0),
        has_los_current=bool(has_los_current) if has_los_current is not None else False,
    )
    offset += TACTICAL_FEATURES

    # Encode move features
    if max_legal_moves > 0 and legal_moves:
        _flatten_move_features(
            result, offset, legal_moves, max_legal_moves, board_width, board_height,
            rl_unit=rl_unit, enemy_unit=enemy_unit, board_hexes=board_hexes,
        )

    return result


def flatten_observation_hierarchical(
    obs: dict,
    rl_owner_id: int,
    board_width: int = BOARD_WIDTH,
    board_height: int = BOARD_HEIGHT,
    legal_moves: list | None = None,
    max_destinations: int = 100,
) -> np.ndarray:
    """Flatten observation for hierarchical (dest + facing) action space.

    Layout: [board elevations] | RL unit | enemy unit | global |
            per-dest features (max_dest * 9) | per-dest facing features (max_dest * 6 * 1)
    """
    obs_size = compute_obs_size_hierarchical(board_width, board_height, max_destinations)
    result = np.zeros(obs_size, dtype=np.float32)

    board = obs.get("board", {})
    board_hexes = board.get("hexes", [])

    # Build elevation map once for efficient lookup
    elev_map: dict[tuple[int, int], float] = {}
    for h in board_hexes:
        elev_map[(h["x"], h["y"])] = h.get("elevation", 0)

    # Board block (same as flat, disabled when INCLUDE_BOARD_ELEVATION is False)
    if INCLUDE_BOARD_ELEVATION:
        for h in board_hexes:
            x, y = h["x"], h["y"]
            if 0 <= x < board_width and 0 <= y < board_height:
                idx = y * board_width + x
                result[idx] = h.get("elevation", 0) / 10.0

    # Split units
    units = obs.get("units", [])
    rl_unit = None
    enemy_unit = None
    for u in units:
        if u["owner"] == rl_owner_id:
            rl_unit = u
        else:
            enemy_unit = u

    # Encode units (same as flat)
    offset = _board_block_size(board_width, board_height)
    if rl_unit is not None:
        _encode_unit(result, offset, rl_unit, board_width, board_height,
                     board_hexes=board_hexes, elev_map=elev_map)
    offset += UNIT_FEATURES
    if enemy_unit is not None:
        _encode_unit(result, offset, enemy_unit, board_width, board_height,
                     board_hexes=board_hexes, elev_map=elev_map)
    offset += UNIT_FEATURES

    # Global features (same as flat)
    result[offset] = float(obs.get("rl_moves_first", False))
    offset += GLOBAL_FEATURES

    # Tactical features (same as flat)
    has_los_current = obs.get("has_los_current")
    if has_los_current is None and legal_moves and rl_unit and rl_unit.get("x", -1) >= 0:
        rl_x, rl_y = rl_unit["x"], rl_unit["y"]
        for m in legal_moves:
            if m.get("dest_x") == rl_x and m.get("dest_y") == rl_y:
                has_los_current = m.get("has_los", False)
                break
    _encode_tactical_features(
        result, offset,
        rl_unit=rl_unit, enemy_unit=enemy_unit,
        elev_map=elev_map, board_width=board_width, board_height=board_height,
        game_round=obs.get("round", 0),
        has_los_current=bool(has_los_current) if has_los_current is not None else False,
    )
    offset += TACTICAL_FEATURES

    # Destination + facing feature blocks
    if legal_moves:
        walk_mp = rl_unit.get("mp_walk", 0) if rl_unit else 0
        _flatten_dest_features(
            result, offset, legal_moves, max_destinations, board_width, board_height,
            walk_mp=walk_mp, rl_unit=rl_unit, enemy_unit=enemy_unit, board_hexes=board_hexes,
        )

    return result


def _flatten_dest_features(
    buf: np.ndarray,
    offset: int,
    legal_moves: list,
    max_destinations: int,
    board_width: int,
    board_height: int,
    walk_mp: int,
    rl_unit: dict | None = None,
    enemy_unit: dict | None = None,
    board_hexes: list | None = None,
) -> None:
    """Write per-destination and per-facing features into the buffer.

    Dest block: buf[offset : offset + max_destinations * DEST_FEATURES]
    Facing block: buf[facing_offset : facing_offset + max_destinations * 6 * FACING_FEATURES]
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

    max_dim = board_width + board_height
    elev_map: dict[tuple[int, int], float] = {}
    if board_hexes:
        for h in board_hexes:
            elev_map[(h["x"], h["y"])] = h.get("elevation", 0)
    enemy_elev = elev_map.get((ex, ey), 0) if has_enemy else 0

    # Group moves into destinations
    destinations, _ = _group_moves_by_destination(legal_moves, walk_mp)
    n_dest = min(len(destinations), max_destinations)

    facing_offset = offset + max_destinations * DEST_FEATURES

    for i in range(n_dest):
        dest = destinations[i]
        dest_x = dest["dest_x"]
        dest_y = dest["dest_y"]

        # --- Per-destination features (9) ---
        base = offset + i * DEST_FEATURES
        buf[base] = dest_x / board_width
        buf[base + 1] = dest_y / board_height
        buf[base + 2] = dest["mp_used"] / 20.0

        best_rl_rq = 0.0

        if has_enemy:
            dist = hex_distance(dest_x, dest_y, ex, ey)
            buf[base + 3] = dist / max_dim

            # Terrain cover at destination
            if board_hexes:
                buf[base + 4] = cover_value(board_hexes, dest_x, dest_y) / 2.0

            # Elevation advantage
            dest_elev = elev_map.get((dest_x, dest_y), 0)
            buf[base + 5] = (dest_elev - enemy_elev) / 10.0

            # has_los — pick from any move in this destination group
            any_move_idx = next(iter(dest["facing_options"].values()))
            buf[base + 6] = float(legal_moves[any_move_idx].get("has_los", False))

            # Enemy range quality (facing-independent — depends on enemy's facing, not ours)
            buf[base + 7] = _norm_rq(range_quality(
                enemy_unit, dist,
                target_x=dest_x, target_y=dest_y,
                unit_x=ex, unit_y=ey,
                unit_facing=enemy_facing,
            ))

        # --- Per-facing features (6 facings × 1: rl_range_quality only) ---
        for facing, move_idx in dest["facing_options"].items():
            f_base = facing_offset + i * 6 * FACING_FEATURES + facing * FACING_FEATURES

            if has_enemy:
                # RL weapon effectiveness at this facing (arc-dependent)
                rl_rq = range_quality(
                    rl_unit, dist,
                    target_x=ex, target_y=ey,
                    unit_x=dest_x, unit_y=dest_y,
                    unit_facing=facing,
                ) if rl_unit else 0.0
                buf[f_base] = _norm_rq(rl_rq)
                if rl_rq > best_rl_rq:
                    best_rl_rq = rl_rq

        # Best RL range quality across all available facings
        buf[base + 8] = _norm_rq(best_rl_rq)


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

    Each move gets MOVE_FEATURES floats:
      Kinematic: dest_x/W, dest_y/H, facing/5, mp_used/20
      Tactical:  dist_to_enemy, range_quality, enemy_range_quality, terrain_cover, elevation_diff, has_los
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

    max_dim = board_width + board_height
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

        # Kinematic features (4)
        buf[base] = dest_x / board_width
        buf[base + 1] = dest_y / board_height
        buf[base + 2] = facing / 5.0
        buf[base + 3] = m.get("mp_used", 0) / 20.0

        # Tactical features (6)
        if has_enemy:
            dist = hex_distance(dest_x, dest_y, ex, ey)
            buf[base + 4] = dist / max_dim

            # RL weapon effectiveness from this hypothetical position
            buf[base + 5] = _norm_rq(range_quality(
                rl_unit, dist,
                target_x=ex, target_y=ey,
                unit_x=dest_x, unit_y=dest_y,
                unit_facing=facing,
            )) if rl_unit else 0.0

            # Enemy weapon effectiveness at this distance
            buf[base + 6] = _norm_rq(range_quality(
                enemy_unit, dist,
                target_x=dest_x, target_y=dest_y,
                unit_x=ex, unit_y=ey,
                unit_facing=enemy_facing,
            ))

            # Terrain cover at destination
            if board_hexes:
                buf[base + 7] = cover_value(board_hexes, dest_x, dest_y) / 2.0

            # Elevation advantage
            dest_elev = elev_map.get((dest_x, dest_y), 0)
            buf[base + 8] = (dest_elev - enemy_elev) / 10.0

            # LOS from destination to enemy (precomputed on Java side)
            buf[base + 9] = float(m.get("has_los", False))


def _encode_tactical_features(
    buf: np.ndarray,
    offset: int,
    rl_unit: dict | None,
    enemy_unit: dict | None,
    elev_map: dict[tuple[int, int], float],
    board_width: int,
    board_height: int,
    game_round: int,
    has_los_current: bool,
) -> None:
    """Write 6 relational/tactical features into buf[offset:offset+TACTICAL_FEATURES].

    Features: hex_distance, rl_range_quality, enemy_range_quality, has_los,
    relative_elevation, round_number.
    """
    i = offset
    max_dim = board_width + board_height

    has_rl = (rl_unit is not None and rl_unit.get("x", -1) >= 0
              and rl_unit.get("y", -1) >= 0)
    has_enemy = (enemy_unit is not None and enemy_unit.get("x", -1) >= 0
                 and enemy_unit.get("y", -1) >= 0)

    if has_rl and has_enemy:
        rl_x, rl_y = rl_unit["x"], rl_unit["y"]
        ex, ey = enemy_unit["x"], enemy_unit["y"]
        rl_facing = rl_unit.get("facing", 0)
        enemy_facing = enemy_unit.get("facing", 0)

        dist = hex_distance(rl_x, rl_y, ex, ey)
        buf[i] = dist / max_dim

        buf[i + 1] = _norm_rq(range_quality(
            rl_unit, dist,
            target_x=ex, target_y=ey,
            unit_x=rl_x, unit_y=rl_y,
            unit_facing=rl_facing,
        ))

        buf[i + 2] = _norm_rq(range_quality(
            enemy_unit, dist,
            target_x=rl_x, target_y=rl_y,
            unit_x=ex, unit_y=ey,
            unit_facing=enemy_facing,
        ))

        buf[i + 3] = float(has_los_current)

        rl_elev = elev_map.get((rl_x, rl_y), 0)
        enemy_elev = elev_map.get((ex, ey), 0)
        buf[i + 4] = (rl_elev - enemy_elev) / 10.0

    # Round number (always available, even without enemy)
    buf[i + 5] = game_round / 50.0


def _encode_unit(
    buf: np.ndarray,
    offset: int,
    unit: dict,
    board_width: int,
    board_height: int,
    board_hexes: list | None = None,
    elev_map: dict[tuple[int, int], float] | None = None,
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

    # Terrain cover at current hex (1 feature)
    x_raw = unit.get("x", -1)
    y_raw = unit.get("y", -1)
    if board_hexes and x_raw >= 0 and y_raw >= 0:
        buf[i] = cover_value(board_hexes, x_raw, y_raw) / 2.0
    i += 1

    # Elevation at current hex (1 feature)
    if elev_map and x_raw >= 0 and y_raw >= 0:
        buf[i] = elev_map.get((x_raw, y_raw), 0) / 10.0
    i += 1

    # System crit hits (3 features)
    crit = unit.get("crit_state", {})
    buf[i] = crit.get("engine_hits", 0) / 3.0
    buf[i + 1] = crit.get("gyro_hits", 0) / 2.0
    buf[i + 2] = crit.get("sensor_hits", 0) / 2.0
    i += 3


def identify_rl_owner(obs: dict, active_entity_id: int) -> int:
    """Determine the RL player's owner ID from the first observation."""
    for unit in obs.get("units", []):
        if unit["id"] == active_entity_id:
            return unit["owner"]
    raise ValueError(
        f"Active entity {active_entity_id} not found in units"
    )


_FACING_NAMES = ["N", "NE", "SE", "S", "SW", "NW"]


def _group_moves_by_destination(legal_moves, walk_mp):
    
    # (x_coord, y_coord, mp_used): {facing: original_index}
    grouping_dict = {}
    for move in legal_moves:
        index = move["index"]
        facing = move["facing"]
        is_run = move["mp_used"] > walk_mp
        dest_idx = (move["dest_x"], move["dest_y"], is_run)
        if dest_idx in grouping_dict:
            grouping_dict[dest_idx]["facing_options"][facing] = index
            grouping_dict[dest_idx]["mp_used"] = min(grouping_dict[dest_idx]["mp_used"], move["mp_used"])
        else:
            features = {k:v for k, v in move.items() if k in ["dest_x", "dest_y", "mp_used"]}
            features["facing_options"] = {facing: index}
            grouping_dict[dest_idx] = features

    grouped_destinations = list(grouping_dict.values())
    lookup = {}
    for i, grouping in enumerate(grouped_destinations):
        for facing, move_idx in grouping["facing_options"].items():
            lookup[(i, facing)] = move_idx
    return grouped_destinations, lookup


def format_observation(obs: dict, rl_owner_id: int | None = None, hierarchical: bool = False) -> str:
    """Format a JSON observation dict as a readable multi-line string.

    Args:
        obs: Raw observation dict from the Java bridge.
        rl_owner_id: RL player's owner ID. If None, inferred from active_entity_id.
        hierarchical: If True, group legal moves by destination (hex + walk/run).

    Returns:
        Human-readable multi-line string.
    """
    lines: list[str] = []
    phase = obs.get("phase", "?")
    rnd = obs.get("round", "?")
    active_id = obs.get("active_entity_id", -1)
    terminated = obs.get("terminated", False)
    truncated = obs.get("truncated", False)

    lines.append(f"=== OBSERVATION: Round {rnd}, Phase {phase} ===")

    reward_str = f"{obs.get('reward', 0.0):.4f}"
    rl_first = obs.get("rl_moves_first", False)
    parts = [f"Active Entity: id={active_id}", f"Reward: {reward_str}",
             f"Terminated: {terminated}", f"Truncated: {truncated}"]
    if "rl_moves_first" in obs:
        parts.append(f"RL Moves First: {rl_first}")
    if "auto_wake_count" in obs:
        parts.append(f"Auto-wake: {obs['auto_wake_count']}")
    lines.append(" | ".join(parts))

    # Terminal observation shortcut
    if terminated or truncated:
        outcome = obs.get("game_outcome")
        if outcome:
            lines.append(f"Game Outcome: {outcome}")
        units = obs.get("units", [])
        if not units:
            return "\n".join(lines)

    # Determine RL owner
    if rl_owner_id is None and active_id >= 0:
        units = obs.get("units", [])
        for u in units:
            if u["id"] == active_id:
                rl_owner_id = u["owner"]
                break

    # Board
    board = obs.get("board", {})
    bw = board.get("width", 0)
    bh = board.get("height", 0)
    lines.append("")
    lines.append(f"--- BOARD ({bw} x {bh}) ---")
    hexes = board.get("hexes", [])
    non_zero = [(h["x"], h["y"], h.get("elevation", 0)) for h in hexes
                if h.get("elevation", 0) != 0]
    if non_zero:
        elev_strs = [f"({x},{y})={e}" for x, y, e in non_zero]
        lines.append("Non-zero elevations: " + "  ".join(elev_strs))
    else:
        lines.append("All hexes at elevation 0")

    # Units
    units = obs.get("units", [])
    rl_units = [u for u in units if rl_owner_id is not None and u.get("owner") == rl_owner_id]
    enemy_units = [u for u in units if rl_owner_id is not None and u.get("owner") != rl_owner_id]

    for label, unit_list in [("RL UNIT", rl_units), ("ENEMY UNIT", enemy_units)]:
        for u in unit_list:
            lines.append("")
            chassis = u.get("chassis", "?")
            model = u.get("model", "")
            uid = u.get("id", "?")
            owner = u.get("owner", "?")
            name = f"{chassis} {model}".strip()
            lines.append(f"--- {label}: {name} (id={uid}, owner={owner}) ---")

            # Position and facing
            x, y = u.get("x", -1), u.get("y", -1)
            facing = u.get("facing", 0)
            facing_name = _FACING_NAMES[facing] if 0 <= facing < 6 else str(facing)
            pos_str = f"({x}, {y})" if x >= 0 else "undeployed"
            mp_w, mp_r, mp_j = u.get("mp_walk", 0), u.get("mp_run", 0), u.get("mp_jump", 0)
            heat = u.get("heat", 0)
            lines.append(f"Position: {pos_str} facing {facing_name} | "
                         f"MP: walk={mp_w} run={mp_r} jump={mp_j} | Heat: {heat}")

            # Status flags
            flags = []
            if u.get("prone"):
                flags.append("PRONE")
            if u.get("destroyed"):
                flags.append("DESTROYED")
            if u.get("retreated"):
                flags.append("RETREATED")
            if u.get("deployed"):
                flags.append("deployed")
            else:
                flags.append("not deployed")
            lines.append(f"Status: {', '.join(flags)}")

            # Armor - build location abbreviation map for weapons
            armor_locs = u.get("armor", [])
            loc_abbrs: dict[int, str] = {}
            armor_parts = []
            for loc_idx, loc in enumerate(armor_locs):
                abbr = loc.get("location", f"L{loc_idx}")
                loc_abbrs[loc_idx] = abbr
                a, a_max = loc.get("armor", 0), loc.get("armor_max", 0)
                i, i_max = loc.get("internal", 0), loc.get("internal_max", 0)
                destroyed = a <= 0 and i <= 0
                part = f"{abbr} {a}/{a_max}"
                if "rear_armor" in loc:
                    ra, ra_max = loc["rear_armor"], loc.get("rear_armor_max", 0)
                    part += f"({ra}/{ra_max}r)"
                if loc.get("internal_max", 0) > 0:
                    part += f" is={i}/{i_max}"
                if destroyed:
                    part += " [DEST]"
                armor_parts.append(part)
            lines.append("Armor: " + "  ".join(armor_parts))

            # Weapons
            weapons = u.get("weapons", [])
            if weapons:
                wpn_strs = []
                for w in weapons:
                    wname = w.get("name", "?")
                    wloc = w.get("location", -1)
                    loc_name = loc_abbrs.get(wloc, f"L{wloc}")
                    status = "DEST" if w.get("destroyed") else "ok"
                    dmg = w.get("damage", 0)
                    sr = w.get("short_range", 0)
                    mr = w.get("medium_range", 0)
                    lr = w.get("long_range", 0)
                    wpn_strs.append(f"{wname} [{loc_name}, {status}, dmg={dmg}, r={sr}/{mr}/{lr}]")
                lines.append("Weapons: " + " | ".join(wpn_strs))
            else:
                lines.append("Weapons: none")

    # Legal moves
    moves = obs.get("legal_moves", [])
    lines.append("")
    if not moves:
        lines.append("--- LEGAL MOVES (0) ---")
    elif hierarchical and moves and "mp_used" in moves[0]:
        _format_moves_hierarchical(lines, moves, rl_owner_id, obs.get("units", []))
    else:
        _format_moves_flat(lines, moves)

    return "\n".join(lines)


def _format_moves_flat(lines: list[str], moves: list) -> None:
    """Format legal moves as a flat list (original format)."""
    is_deployment = "elevation" in moves[0] and "mp_used" not in moves[0]

    if is_deployment:
        lines.append(f"--- DEPLOYMENT MOVES ({len(moves)} total) ---")
        for m in moves[:5]:
            idx = m.get("index", "?")
            dx, dy = m.get("dest_x", "?"), m.get("dest_y", "?")
            f = m.get("facing", 0)
            fn = _FACING_NAMES[f] if 0 <= f < 6 else str(f)
            elev = m.get("elevation", 0)
            lines.append(f"  #{idx}: ({dx},{dy}) facing {fn}, elev={elev}")
        if len(moves) > 5:
            lines.append(f"  ... and {len(moves) - 5} more")
    else:
        mp_vals = [m.get("mp_used", 0) for m in moves]
        xs = [m["dest_x"] for m in moves if m.get("dest_x", -1) >= 0]
        ys = [m["dest_y"] for m in moves if m.get("dest_y", -1) >= 0]
        n_jumping = sum(1 for m in moves if m.get("jumping"))
        n_prone = sum(1 for m in moves if m.get("prone"))

        mp_lo, mp_hi = min(mp_vals), max(mp_vals)
        summary_parts = [f"MP range: {mp_lo}-{mp_hi}"]
        if xs:
            summary_parts.append(f"Dest x=[{min(xs)}..{max(xs)}], y=[{min(ys)}..{max(ys)}]")
        if n_jumping:
            summary_parts.append(f"Jumping: {n_jumping}")
        if n_prone:
            summary_parts.append(f"Prone: {n_prone}")

        lines.append(f"--- LEGAL MOVES ({len(moves)} total) ---")
        lines.append(" | ".join(summary_parts))

        sample_indices: list[int] = [0]
        for si in [1, 2]:
            if si < len(moves) and si not in sample_indices:
                sample_indices.append(si)
        mid = len(moves) // 2
        if mid not in sample_indices and mid < len(moves):
            sample_indices.append(mid)
        closest_idx = None
        closest_dist = float("inf")
        for i, m in enumerate(moves):
            d = m.get("java_dist_to_enemy", -1)
            if d >= 0 and d < closest_dist:
                closest_dist = d
                closest_idx = i
        if closest_idx is not None and closest_idx not in sample_indices:
            sample_indices.append(closest_idx)

        lines.append("Sample moves:")
        for i in sample_indices:
            m = moves[i]
            idx = m.get("index", i)
            dx, dy = m.get("dest_x", -1), m.get("dest_y", -1)
            f = m.get("facing", 0)
            fn = _FACING_NAMES[f] if 0 <= f < 6 else str(f)
            mp = m.get("mp_used", 0)
            tags = []
            if m.get("jumping"):
                tags.append("jumping")
            if m.get("prone"):
                tags.append("prone")
            d = m.get("java_dist_to_enemy", -1)
            if i == closest_idx and d >= 0:
                tags.append(f"closest to enemy, dist={d:.1f}")
            tag_str = f"  [{', '.join(tags)}]" if tags else ""
            lines.append(f"  #{idx}: ({dx},{dy}) facing {fn}, {mp} MP{tag_str}")


def _format_moves_hierarchical(lines: list[str], moves: list, rl_owner_id, units: list) -> None:
    """Format legal moves grouped by destination (hex + walk/run)."""
    # Get walk_mp from RL unit
    walk_mp = 0
    for u in units:
        if u.get("owner") == rl_owner_id:
            walk_mp = u.get("mp_walk", 0)
            break

    destinations, _ = _group_moves_by_destination(moves, walk_mp)

    # Count walk vs run destinations
    n_walk = sum(1 for d in destinations if d["mp_used"] <= walk_mp)
    n_run = len(destinations) - n_walk

    lines.append(f"--- DESTINATIONS ({len(destinations)} from {len(moves)} moves, "
                 f"{n_walk} walk + {n_run} run) ---")

    # Show all destinations (usually 60-80, manageable)
    max_show = 15
    for i, dest in enumerate(destinations[:max_show]):
        x, y = dest["dest_x"], dest["dest_y"]
        mp = dest["mp_used"]
        speed = "run" if mp > walk_mp else "walk"
        facings = dest["facing_options"]
        facing_strs = [_FACING_NAMES[f] for f in sorted(facings.keys())]
        lines.append(
            f"  dest {i:>3d}: ({x:>2d},{y:>2d}) {speed:<4s} {mp:>2d}MP "
            f"facings=[{', '.join(facing_strs)}]"
        )
    if len(destinations) > max_show:
        lines.append(f"  ... and {len(destinations) - max_show} more destinations")
