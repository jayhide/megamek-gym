"""Tier 1: Board terrain and elevation comparison."""

from __future__ import annotations

from dataclasses import dataclass, field

from megamek_gym.sim.board import BOARD, Terrain


@dataclass
class BoardValidationResult:
    total_hexes: int = 0
    elevation_mismatches: list[str] = field(default_factory=list)
    terrain_mismatches: list[str] = field(default_factory=list)
    dimension_mismatch: str | None = None

    @property
    def passed(self) -> bool:
        return (not self.elevation_mismatches
                and not self.terrain_mismatches
                and self.dimension_mismatch is None)


def _java_terrain_to_sim(terrain_str: str) -> Terrain:
    """Map Java terrain string to sim Terrain enum.

    Java sends terrain as "Level: N  Features: Light Woods; foliage_elev(2, 0); "
    """
    if not terrain_str:
        return Terrain.CLEAR
    ts = terrain_str.lower()
    if "heavy woods" in ts:
        return Terrain.HEAVY_WOODS
    if "light woods" in ts:
        return Terrain.LIGHT_WOODS
    if "water" in ts:
        return Terrain.WATER
    if "rough" in ts:
        return Terrain.ROUGH
    return Terrain.CLEAR


def validate_board(java_obs: dict) -> BoardValidationResult:
    """Compare Python sim board against Java board from first observation."""
    result = BoardValidationResult()

    board_data = java_obs.get("board", {})
    java_width = board_data.get("width", 0)
    java_height = board_data.get("height", 0)

    if java_width != BOARD.width or java_height != BOARD.height:
        result.dimension_mismatch = (
            f"sim=({BOARD.width}x{BOARD.height}) java=({java_width}x{java_height})"
        )
        return result

    java_hexes = board_data.get("hexes", [])
    result.total_hexes = len(java_hexes)

    for jh in java_hexes:
        x = jh.get("x", -1)
        y = jh.get("y", -1)
        if x < 0 or y < 0:
            continue

        # Elevation
        java_elev = jh.get("elevation", 0)
        sim_elev = BOARD.elevation(x, y)
        if java_elev != sim_elev:
            result.elevation_mismatches.append(
                f"({x},{y}): sim={sim_elev} java={java_elev}"
            )

        # Terrain
        java_terrain_str = jh.get("terrain", "")
        java_terrain = _java_terrain_to_sim(java_terrain_str)
        sim_terrain = BOARD.terrain(x, y)
        if java_terrain != sim_terrain:
            result.terrain_mismatches.append(
                f"({x},{y}): sim={sim_terrain.name} java={java_terrain.name} "
                f"(raw='{java_terrain_str}')"
            )

    return result
