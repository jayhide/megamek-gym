"""End-to-end validation of the 127-dim observation feature vector.

Calls flatten_observation() on raw Java obs dicts and compares each feature
against an independently computed expected value derived from the same raw
fields. Catches encoding bugs (wrong divisor, off-by-one index, wrong field).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from megamek_gym.observation import (
    BOARD_HEIGHT,
    BOARD_WIDTH,
    GLOBAL_FEATURES,
    MAX_ARMOR_LOCATIONS,
    MAX_WEAPONS,
    TACTICAL_FEATURES,
    UNIT_FEATURES,
    _norm_rq,
    flatten_observation,
)
from megamek_gym.reward import cover_value, hex_distance, range_quality
from tests.sim_validation.reconstruct import extract_unit_state


# ---------------------------------------------------------------------------
# Feature name map (127 entries) for readable mismatch diagnostics
# ---------------------------------------------------------------------------
def _build_feature_names() -> list[str]:
    names: list[str] = []
    for prefix in ("rl", "enemy"):
        names.append(f"{prefix}_pos_x")
        names.append(f"{prefix}_pos_y")
        for f in range(6):
            names.append(f"{prefix}_facing_{f}")
        names.append(f"{prefix}_mp_walk")
        names.append(f"{prefix}_mp_run")
        names.append(f"{prefix}_mp_jump")
        names.append(f"{prefix}_heat")
        for flag in ("prone", "destroyed", "deployed", "retreated"):
            names.append(f"{prefix}_{flag}")
        loc_names = ["HD", "CT", "RT", "LT", "RA", "LA", "RL", "LL"]
        for loc_idx in range(MAX_ARMOR_LOCATIONS):
            loc_label = loc_names[loc_idx] if loc_idx < len(loc_names) else f"loc{loc_idx}"
            for val in ("armor", "internal", "rear", "destroyed"):
                names.append(f"{prefix}_{loc_label}_{val}")
        for w in range(MAX_WEAPONS):
            names.append(f"{prefix}_weapon{w}_destroyed")
        names.append(f"{prefix}_terrain_cover")
        names.append(f"{prefix}_elevation")
        names.append(f"{prefix}_engine_hits")
        names.append(f"{prefix}_gyro_hits")
        names.append(f"{prefix}_sensor_hits")
    names.append("rl_moves_first")
    for n in ("hex_distance", "rl_range_quality", "enemy_range_quality",
              "has_los", "relative_elevation", "round_number"):
        names.append(n)
    return names


FEATURE_NAMES = _build_feature_names()
assert len(FEATURE_NAMES) == 2 * UNIT_FEATURES + GLOBAL_FEATURES + TACTICAL_FEATURES


# ---------------------------------------------------------------------------
# Result dataclasses
# ---------------------------------------------------------------------------
@dataclass
class FeatureMismatch:
    index: int
    name: str
    expected: float
    actual: float

    def __str__(self) -> str:
        return (f"[{self.index}] {self.name}: expected={self.expected:.6f} "
                f"actual={self.actual:.6f}")


@dataclass
class ObsFeaturesResult:
    total_features: int = 0
    mismatches: list[FeatureMismatch] = field(default_factory=list)
    skipped: bool = False
    skip_reason: str = ""

    @property
    def passed(self) -> bool:
        return not self.mismatches and not self.skipped


# ---------------------------------------------------------------------------
# Independent expected-value builder
# ---------------------------------------------------------------------------
def _encode_unit_expected(
    unit: dict | None,
    board_hexes: list,
    elev_map: dict[tuple[int, int], float],
    board_width: int,
    board_height: int,
) -> list[float]:
    """Build 60 expected feature values for one unit from raw obs dict."""
    feats = [0.0] * UNIT_FEATURES
    if unit is None:
        return feats

    i = 0

    # Position (2)
    x = unit.get("x", -1)
    y = unit.get("y", -1)
    feats[i] = max(0.0, x) / board_width
    feats[i + 1] = max(0.0, y) / board_height
    i += 2

    # Facing one-hot (6)
    facing = unit.get("facing", 0)
    if 0 <= facing < 6:
        feats[i + facing] = 1.0
    i += 6

    # MP (3)
    feats[i] = unit.get("mp_walk", 0) / 20.0
    feats[i + 1] = unit.get("mp_run", 0) / 20.0
    feats[i + 2] = unit.get("mp_jump", 0) / 20.0
    i += 3

    # Heat (1)
    feats[i] = unit.get("heat", 0) / 30.0
    i += 1

    # Status flags (4)
    feats[i] = float(unit.get("prone", False))
    feats[i + 1] = float(unit.get("destroyed", False))
    feats[i + 2] = float(unit.get("deployed", False))
    feats[i + 3] = float(unit.get("retreated", False))
    i += 4

    # Armor locations (8 × 4 = 32)
    armor_locs = unit.get("armor", [])
    for loc_idx in range(MAX_ARMOR_LOCATIONS):
        if loc_idx < len(armor_locs):
            loc = armor_locs[loc_idx]
            armor_max = loc.get("armor_max", 1)
            internal_max = loc.get("internal_max", 1)
            rear_max = loc.get("rear_armor_max", 0)

            feats[i] = loc.get("armor", 0) / max(armor_max, 1)
            feats[i + 1] = max(0.0, loc.get("internal", 0)) / max(internal_max, 1)
            feats[i + 2] = (loc.get("rear_armor", 0) / max(rear_max, 1)
                            if rear_max > 0 else 0.0)
            feats[i + 3] = float(
                loc.get("armor", 0) <= 0 and loc.get("internal", 0) <= 0
            )
        i += 4

    # Weapon destroyed flags (7)
    weapons = unit.get("weapons", [])
    for w_idx in range(MAX_WEAPONS):
        if w_idx < len(weapons):
            feats[i] = float(weapons[w_idx].get("destroyed", False))
        i += 1

    # Terrain cover at current hex (1)
    x_raw = unit.get("x", -1)
    y_raw = unit.get("y", -1)
    if board_hexes and x_raw >= 0 and y_raw >= 0:
        feats[i] = cover_value(board_hexes, x_raw, y_raw) / 2.0
    i += 1

    # Elevation at current hex (1)
    if elev_map and x_raw >= 0 and y_raw >= 0:
        feats[i] = elev_map.get((x_raw, y_raw), 0) / 10.0
    i += 1

    # System crit hits (3)
    crit = unit.get("crit_state", {})
    feats[i] = crit.get("engine_hits", 0) / 3.0
    feats[i + 1] = crit.get("gyro_hits", 0) / 2.0
    feats[i + 2] = crit.get("sensor_hits", 0) / 2.0
    i += 3

    assert i == UNIT_FEATURES
    return feats


def _build_expected(
    raw_obs: dict,
    rl_owner_id: int,
    board_width: int = BOARD_WIDTH,
    board_height: int = BOARD_HEIGHT,
) -> np.ndarray:
    """Build the expected 127-dim vector independently from raw obs fields."""
    n = 2 * UNIT_FEATURES + GLOBAL_FEATURES + TACTICAL_FEATURES
    expected = np.zeros(n, dtype=np.float32)

    board = raw_obs.get("board", {})
    board_hexes = board.get("hexes", [])
    elev_map: dict[tuple[int, int], float] = {}
    for h in board_hexes:
        elev_map[(h["x"], h["y"])] = h.get("elevation", 0)

    # Split units
    rl_unit = None
    enemy_unit = None
    for u in raw_obs.get("units", []):
        if u["owner"] == rl_owner_id:
            rl_unit = u
        else:
            enemy_unit = u

    # RL unit features (0:60)
    rl_feats = _encode_unit_expected(rl_unit, board_hexes, elev_map,
                                     board_width, board_height)
    expected[0:UNIT_FEATURES] = rl_feats

    # Enemy unit features (60:120)
    enemy_feats = _encode_unit_expected(enemy_unit, board_hexes, elev_map,
                                        board_width, board_height)
    expected[UNIT_FEATURES:2 * UNIT_FEATURES] = enemy_feats

    # Global features (120)
    offset = 2 * UNIT_FEATURES
    expected[offset] = float(raw_obs.get("rl_moves_first", False))
    offset += GLOBAL_FEATURES

    # Tactical features (121:127)
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
        expected[offset] = dist / max_dim

        expected[offset + 1] = _norm_rq(range_quality(
            rl_unit, dist,
            target_x=ex, target_y=ey,
            unit_x=rl_x, unit_y=rl_y,
            unit_facing=rl_facing,
        ))

        expected[offset + 2] = _norm_rq(range_quality(
            enemy_unit, dist,
            target_x=rl_x, target_y=rl_y,
            unit_x=ex, unit_y=ey,
            unit_facing=enemy_facing,
        ))

        # has_los: derive from legal_moves (same fallback as flatten_observation)
        has_los_current = raw_obs.get("has_los_current")
        if has_los_current is None:
            legal_moves = raw_obs.get("legal_moves", [])
            for m in legal_moves:
                if m.get("dest_x") == rl_x and m.get("dest_y") == rl_y:
                    has_los_current = m.get("has_los", False)
                    break
        expected[offset + 3] = float(bool(has_los_current)) if has_los_current is not None else 0.0

        rl_elev = elev_map.get((rl_x, rl_y), 0)
        enemy_elev = elev_map.get((ex, ey), 0)
        expected[offset + 4] = (rl_elev - enemy_elev) / 10.0

    expected[offset + 5] = raw_obs.get("round", 0) / 50.0

    return expected


# ---------------------------------------------------------------------------
# Validation entry point
# ---------------------------------------------------------------------------
def validate_obs_features(
    raw_obs: dict,
    rl_owner_id: int,
    board_width: int = BOARD_WIDTH,
    board_height: int = BOARD_HEIGHT,
) -> ObsFeaturesResult:
    """Compare flatten_observation() output against independently computed expected values."""
    result = ObsFeaturesResult()

    # Skip terminal observations
    if raw_obs.get("terminated") or raw_obs.get("truncated"):
        result.skipped = True
        result.skip_reason = "terminal observation"
        return result

    # Get actual vector from flatten_observation
    legal_moves = raw_obs.get("legal_moves", [])
    actual = flatten_observation(
        raw_obs, rl_owner_id,
        board_width=board_width, board_height=board_height,
        legal_moves=legal_moves, max_legal_moves=0,
    )

    # Build expected vector independently
    expected = _build_expected(raw_obs, rl_owner_id, board_width, board_height)

    result.total_features = len(expected)

    # Compare element-wise
    for i in range(len(expected)):
        name = FEATURE_NAMES[i] if i < len(FEATURE_NAMES) else f"feature_{i}"
        # Use looser tolerance for range_quality (float accumulation in
        # damage-weighted averaging)
        atol = 1e-4 if "range_quality" in name else 1e-5
        if not np.isclose(actual[i], expected[i], atol=atol):
            result.mismatches.append(FeatureMismatch(
                index=i, name=name,
                expected=float(expected[i]), actual=float(actual[i]),
            ))

    return result
