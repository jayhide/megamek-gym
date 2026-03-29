"""Per-move feature cross-validation: Python vs Java MegaMek.

Compares per-move distance and weapon arc calculations between Python
(reward.py) and Java (ObservationBuilder.java) using live game traces.

Java sends `java_dist_to_enemy` and `java_weapon_arcs[]` per legal move
as cross-validation ground truth. Python independently computes these
in `_flatten_move_features()` (observation.py).
"""

from __future__ import annotations

from dataclasses import dataclass, field

from megamek_gym.reward import hex_bearing, hex_distance, in_firing_arc
from tests.sim_validation.reconstruct import extract_unit_state


@dataclass
class MoveFeatureResult:
    """Aggregated cross-validation result for one observation step."""
    distance_checks: int = 0
    distance_mismatches: list[str] = field(default_factory=list)
    arc_checks: int = 0
    arc_mismatches: list[str] = field(default_factory=list)

    @property
    def passed(self) -> bool:
        return not self.distance_mismatches and not self.arc_mismatches


def validate_move_features(
    raw_obs: dict,
    rl_owner_id: int,
) -> MoveFeatureResult:
    """Validate Python vs Java per-move features for all legal moves.

    Checks:
    1. hex_distance() vs java_dist_to_enemy
    2. in_firing_arc() vs java_weapon_arcs[] per weapon
    """
    result = MoveFeatureResult()

    # Find RL and enemy units
    rl_unit = extract_unit_state(raw_obs, rl_owner_id)
    enemy_unit = None
    for u in raw_obs.get("units", []):
        if u.get("owner") != rl_owner_id:
            enemy_unit = u
            break

    if rl_unit is None or enemy_unit is None:
        return result

    ex, ey = enemy_unit.get("x", -1), enemy_unit.get("y", -1)
    if ex < 0 or ey < 0:
        return result

    weapons = rl_unit.get("weapons", [])
    legal_moves = raw_obs.get("legal_moves", [])

    for move in legal_moves:
        idx = move.get("index", "?")
        java_dist = move.get("java_dist_to_enemy")
        java_arcs = move.get("java_weapon_arcs")

        # Skip moves without cross-validation fields
        if java_dist is None or java_dist < 0:
            continue

        dest_x = move["dest_x"]
        dest_y = move["dest_y"]
        facing = move["facing"]

        # Distance check
        py_dist = hex_distance(dest_x, dest_y, ex, ey)
        result.distance_checks += 1
        if py_dist != java_dist:
            result.distance_mismatches.append(
                f"move[{idx}] distance: py={py_dist} java={java_dist} "
                f"from ({dest_x},{dest_y}) to ({ex},{ey})"
            )

        # Weapon arc checks
        if java_arcs is None:
            continue

        if len(java_arcs) != len(weapons):
            result.arc_mismatches.append(
                f"move[{idx}] weapon count mismatch: "
                f"py_weapons={len(weapons)} java_arcs={len(java_arcs)}"
            )
            continue

        # Skip arc check when dest == enemy (bearing undefined)
        if dest_x == ex and dest_y == ey:
            continue

        bearing = hex_bearing(dest_x, dest_y, ex, ey)
        for w_idx, weapon in enumerate(weapons):
            weapon_loc = weapon.get("location", 1)
            py_in_arc = in_firing_arc(facing, weapon_loc, bearing)
            java_in_arc = java_arcs[w_idx]
            result.arc_checks += 1
            if py_in_arc != java_in_arc:
                result.arc_mismatches.append(
                    f"move[{idx}] weapon[{w_idx}] arc: "
                    f"py={py_in_arc} java={java_in_arc} "
                    f"facing={facing} loc={weapon_loc} bearing={bearing:.1f} "
                    f"from ({dest_x},{dest_y}) to ({ex},{ey})"
                )

    return result
