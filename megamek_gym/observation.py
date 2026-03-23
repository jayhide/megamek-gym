"""Convert variable-structure JSON observations into fixed-size numpy arrays."""

from __future__ import annotations

import numpy as np

from megamek_gym.reward import cover_value, hex_distance, range_quality

BOARD_WIDTH = 16
BOARD_HEIGHT = 17
BOARD_SIZE = BOARD_WIDTH * BOARD_HEIGHT  # 272
UNIT_FEATURES = 55
GLOBAL_FEATURES = 1  # rl_moves_first
MOVE_FEATURES = 11  # dest_x, dest_y, facing, mp_used, prone, dist_to_enemy, range_quality, enemy_range_quality, terrain_cover, elevation_diff, has_los
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

            # LOS from destination to enemy (precomputed on Java side)
            buf[base + 10] = float(m.get("has_los", False))


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


_FACING_NAMES = ["N", "NE", "SE", "S", "SW", "NW"]


def format_observation(obs: dict, rl_owner_id: int | None = None) -> str:
    """Format a JSON observation dict as a readable multi-line string.

    Args:
        obs: Raw observation dict from the Java bridge.
        rl_owner_id: RL player's owner ID. If None, inferred from active_entity_id.

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
    else:
        is_deployment = "elevation" in moves[0] and "mp_used" not in moves[0]

        if is_deployment:
            lines.append(f"--- DEPLOYMENT MOVES ({len(moves)} total) ---")
            # Show first 5
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
            # Movement moves - compute summary stats
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

            # Sample moves: first, a couple short, closest to enemy
            sample_indices: list[int] = [0]  # always show first
            # Add index 1 and 2 if they exist and aren't index 0
            for si in [1, 2]:
                if si < len(moves) and si not in sample_indices:
                    sample_indices.append(si)
            # A mid-range move
            mid = len(moves) // 2
            if mid not in sample_indices and mid < len(moves):
                sample_indices.append(mid)
            # Closest to enemy (by java_dist_to_enemy)
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

    return "\n".join(lines)
