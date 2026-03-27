"""Tier 1: LOS comparison between Python sim and Java."""

from __future__ import annotations

from dataclasses import dataclass, field

from megamek_gym.sim.board import BOARD
from megamek_gym.sim.los import LosTable


@dataclass
class LosValidationResult:
    total_checks: int = 0
    mismatches: list[str] = field(default_factory=list)
    sim_sees_java_doesnt: int = 0  # Sim more permissive
    java_sees_sim_doesnt: int = 0  # Sim more restrictive

    @property
    def passed(self) -> bool:
        return not self.mismatches


def validate_los(java_obs: dict, los_table: LosTable,
                 rl_owner_id: int) -> LosValidationResult:
    """Compare has_los for all legal moves in a Java observation."""
    result = LosValidationResult()

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
        java_los = move.get("has_los")
        if java_los is None:
            continue

        dx = move["dest_x"]
        dy = move["dest_y"]

        sim_los = los_table.has_los(dx, dy, ex, ey)
        result.total_checks += 1

        if sim_los != java_los:
            idx = move.get("index", "?")
            result.mismatches.append(
                f"move[{idx}] ({dx},{dy})→({ex},{ey}): "
                f"sim={sim_los} java={java_los}"
            )
            if sim_los and not java_los:
                result.sim_sees_java_doesnt += 1
            else:
                result.java_sees_sim_doesnt += 1

    return result
