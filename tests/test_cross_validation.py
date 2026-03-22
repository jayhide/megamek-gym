"""Cross-validation of Python tactical features against Java-computed values.

Compares hex_distance and firing arc calculations between Python (reward.py)
and Java (ObservationBuilder.java) using live game observations.

Requires a running MegaMek instance — skipped in normal pytest runs.
Run via smoke_test_all.py (Test 5) or explicitly with:
    poetry run pytest tests/test_cross_validation.py -m integration
"""

from __future__ import annotations

from megamek_gym.reward import hex_bearing, hex_distance, in_firing_arc


def validate_observation(obs: dict, rl_owner_id: int) -> list[str]:
    """Validate Python vs Java calculations for all moves in one observation.

    Returns a list of mismatch descriptions (empty = all match).
    """
    mismatches = []

    units = obs.get("units", [])
    rl_unit = None
    enemy_unit = None
    for u in units:
        if u["owner"] == rl_owner_id:
            rl_unit = u
        else:
            enemy_unit = u

    if rl_unit is None or enemy_unit is None:
        return mismatches

    # Enemy must be deployed (valid position)
    ex, ey = enemy_unit.get("x", -1), enemy_unit.get("y", -1)
    if ex < 0 or ey < 0:
        return mismatches

    weapons = rl_unit.get("weapons", [])
    legal_moves = obs.get("legal_moves", [])

    for move in legal_moves:
        idx = move.get("index", "?")
        java_dist = move.get("java_dist_to_enemy")
        java_arcs = move.get("java_weapon_arcs")

        # Skip moves without cross-validation fields (dest=null or no enemy)
        if java_dist is None or java_dist < 0:
            continue

        dest_x = move["dest_x"]
        dest_y = move["dest_y"]
        facing = move["facing"]

        # Validate hex distance
        py_dist = hex_distance(dest_x, dest_y, ex, ey)
        if py_dist != java_dist:
            mismatches.append(
                f"move[{idx}] distance: py={py_dist} java={java_dist} "
                f"from ({dest_x},{dest_y}) to ({ex},{ey})"
            )

        # Validate weapon arcs
        if java_arcs is not None:
            if len(java_arcs) != len(weapons):
                mismatches.append(
                    f"move[{idx}] weapon count mismatch: "
                    f"py_weapons={len(weapons)} java_arcs={len(java_arcs)}"
                )
                continue

            # Only check arcs if src != dest (bearing is 0 when same hex)
            if dest_x == ex and dest_y == ey:
                continue

            bearing = hex_bearing(dest_x, dest_y, ex, ey)
            for w_idx, weapon in enumerate(weapons):
                weapon_loc = weapon.get("location", 1)
                py_in_arc = in_firing_arc(facing, weapon_loc, bearing)
                java_in_arc = java_arcs[w_idx]
                if py_in_arc != java_in_arc:
                    mismatches.append(
                        f"move[{idx}] weapon[{w_idx}] arc: "
                        f"py={py_in_arc} java={java_in_arc} "
                        f"facing={facing} loc={weapon_loc} bearing={bearing:.1f} "
                        f"from ({dest_x},{dest_y}) to ({ex},{ey})"
                    )

    return mismatches
