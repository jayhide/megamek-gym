"""Tier 1: To-hit number validation.

Verifies the sim's compute_to_hit() components against known BattleTech rules.
Since Java doesn't include to-hit numbers in observations, we validate the
individual modifier tables and formulas rather than doing direct comparison.
Also validates per-move range quality and distance calculations against Java.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from megamek_gym.reward import hex_distance
from megamek_gym.sim.firing import target_movement_modifier, compute_to_hit
from megamek_gym.sim.board import BOARD
from megamek_gym.sim.los import LosTable
from tests.sim_validation.reconstruct import reconstruct_unit, extract_unit_state


@dataclass
class ToHitValidationResult:
    """Results of to-hit validation."""
    component_checks: int = 0
    component_mismatches: list[str] = field(default_factory=list)
    distance_checks: int = 0
    distance_mismatches: list[str] = field(default_factory=list)

    @property
    def passed(self) -> bool:
        return not self.component_mismatches and not self.distance_mismatches


def validate_to_hit_components() -> ToHitValidationResult:
    """Validate to-hit modifier tables against canonical BattleTech rules."""
    result = ToHitValidationResult()

    # TMM table (Total Warfare p.44)
    tmm_expected = {
        0: 0, 1: 0, 2: 0,   # 0-2 hexes
        3: 1, 4: 1,           # 3-4 hexes
        5: 2, 6: 2,           # 5-6 hexes
        7: 3, 8: 3, 9: 3,    # 7-9 hexes
        10: 4, 15: 4, 20: 4, # 10+
    }
    for hexes_moved, expected in tmm_expected.items():
        actual = target_movement_modifier(hexes_moved)
        result.component_checks += 1
        if actual != expected:
            result.component_mismatches.append(
                f"TMM({hexes_moved} hexes): expected={expected} got={actual}"
            )

    # Heat gunnery modifier (Total Warfare p.153)
    # Heat 0-7: +0, 8-12: +1, 13-16: +2, 17-20: +3, 21-24: +4, 25+: +5
    heat_expected = {
        0: 0, 7: 0,
        8: 1, 12: 1,
        13: 2, 16: 2,
        17: 3, 20: 3,
        21: 4, 24: 4,
        25: 5, 30: 5,
    }
    from megamek_gym.sim.unit import UNIT_TEMPLATES, Unit
    tmpl = UNIT_TEMPLATES["Trebuchet TBT-5S"]
    for heat_val, expected in heat_expected.items():
        test_unit = Unit(template=tmpl, entity_id=99, owner=0)
        test_unit.heat = heat_val
        actual = test_unit.gunnery_modifier
        result.component_checks += 1
        if actual != expected:
            result.component_mismatches.append(
                f"heat_modifier(heat={heat_val}): expected={expected} got={actual}"
            )

    return result


def validate_distances(java_obs: dict, rl_owner_id: int) -> ToHitValidationResult:
    """Cross-validate hex distances from Java legal_moves."""
    result = ToHitValidationResult()

    units = java_obs.get("units", [])
    enemy_unit = None
    for u in units:
        if u.get("owner") != rl_owner_id:
            enemy_unit = u
            break

    if enemy_unit is None:
        return result

    ex = enemy_unit.get("x", -1)
    ey = enemy_unit.get("y", -1)
    if ex < 0 or ey < 0:
        return result

    for move in java_obs.get("legal_moves", []):
        java_dist = move.get("java_dist_to_enemy")
        if java_dist is None or java_dist < 0:
            continue

        dx = move["dest_x"]
        dy = move["dest_y"]
        sim_dist = hex_distance(dx, dy, ex, ey)
        result.distance_checks += 1

        if sim_dist != java_dist:
            idx = move.get("index", "?")
            result.distance_mismatches.append(
                f"move[{idx}] ({dx},{dy})→({ex},{ey}): sim={sim_dist} java={java_dist}"
            )

    return result
