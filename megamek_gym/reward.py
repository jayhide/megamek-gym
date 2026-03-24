"""Pluggable reward functions computed from raw JSON observations."""

from __future__ import annotations

import math
from abc import ABC, abstractmethod


class RewardFunction(ABC):
    @abstractmethod
    def compute(self, prev_obs: dict, curr_obs: dict, terminated: bool) -> float:
        ...

    @abstractmethod
    def reset(self) -> None:
        ...


def _weighted_hp(units: list[dict], owner_id: int, internal_multiplier: float = 2.0) -> float:
    """Sum weighted HP across all locations for units belonging to owner.

    Internal structure damage is weighted by *internal_multiplier* (default 2x)
    relative to armor/rear_armor (weight 1x) to reflect its greater tactical
    significance.
    """
    total = 0.0
    for u in units:
        if u["owner"] == owner_id:
            for loc in u.get("armor", []):
                total += loc.get("armor", 0) + loc.get("rear_armor", 0)
                total += max(0, loc.get("internal", 0)) * internal_multiplier
    return total


# Location weights for destruction bonus (BattleTech tactical significance)
LOCATION_WEIGHTS: dict[str, float] = {
    "CT": 1.0,   # Center torso — mech kill
    "HD": 1.0,   # Head — pilot kill
    "LT": 0.4,   # Side torso — loses arm + weapons
    "RT": 0.4,
    "LA": 0.2,   # Arms — weapon loss
    "RA": 0.2,
    "LL": 0.3,   # Legs — mobility kill
    "RL": 0.3,
}


class DamageDeltaReward(RewardFunction):
    """Reward based on net damage dealt minus damage taken.

    Internal structure damage is weighted by *internal_multiplier* (default 2x)
    relative to armor damage, reflecting its greater tactical significance.
    """

    def __init__(self, scale: float = 1.0, normalizer: float = 100.0,
                 internal_multiplier: float = 2.0):
        self.scale = scale
        self.normalizer = normalizer
        self.internal_multiplier = internal_multiplier
        self._prev_own: float | None = None
        self._prev_enemy: float | None = None

    def compute(self, prev_obs: dict, curr_obs: dict, terminated: bool) -> float:
        if terminated and not curr_obs.get("units"):
            # Terminal obs with no units — no delta to compute
            return 0.0

        rl_owner = self._rl_owner
        im = self.internal_multiplier
        own_hp = _weighted_hp(curr_obs.get("units", []), rl_owner, im)
        enemy_hp = 0.0
        enemy_owners = set()
        for u in curr_obs.get("units", []):
            if u["owner"] != rl_owner:
                enemy_owners.add(u["owner"])
        for eid in enemy_owners:
            enemy_hp += _weighted_hp(curr_obs.get("units", []), eid, im)

        if self._prev_own is None:
            self._prev_own = own_hp
            self._prev_enemy = enemy_hp
            return 0.0

        own_lost = self._prev_own - own_hp
        enemy_lost = self._prev_enemy - enemy_hp
        reward = self.scale * (enemy_lost - own_lost) / self.normalizer

        self._prev_own = own_hp
        self._prev_enemy = enemy_hp
        return reward

    def reset(self) -> None:
        self._prev_own = None
        self._prev_enemy = None
        self._rl_owner: int = -1

    def set_rl_owner(self, owner_id: int) -> None:
        self._rl_owner = owner_id


class WinLossReward(RewardFunction):
    """Binary reward at episode end."""

    def __init__(self, scale: float = 1.0):
        self.scale = scale
        self._rl_owner: int = -1

    def compute(self, prev_obs: dict, curr_obs: dict, terminated: bool) -> float:
        if not terminated:
            return 0.0

        # Use explicit outcome from Java if available
        outcome = curr_obs.get("game_outcome")
        if outcome == "WIN":
            return self.scale
        elif outcome == "LOSS":
            return -self.scale
        elif outcome is not None:
            return 0.0

        # Legacy fallback: infer from unit states
        units = curr_obs.get("units", [])
        if not units:
            units = prev_obs.get("units", [])

        own_alive = any(
            not u.get("destroyed", False) and not u.get("retreated", False)
            for u in units if u["owner"] == self._rl_owner
        )
        enemy_alive = any(
            not u.get("destroyed", False) and not u.get("retreated", False)
            for u in units if u["owner"] != self._rl_owner
        )

        if own_alive and not enemy_alive:
            return self.scale
        elif not own_alive and enemy_alive:
            return -self.scale
        return 0.0

    def reset(self) -> None:
        self._rl_owner = -1

    def set_rl_owner(self, owner_id: int) -> None:
        self._rl_owner = owner_id


def _location_internals(units: list[dict], owner_id: int) -> dict[tuple[int, str], float]:
    """Return {(unit_id, location_name): internal} for units belonging to owner."""
    result = {}
    for u in units:
        if u["owner"] == owner_id:
            uid = u["id"]
            for loc in u.get("armor", []):
                loc_name = loc.get("location", "")
                result[(uid, loc_name)] = loc.get("internal", 0)
    return result


class LocationDestructionReward(RewardFunction):
    """Bonus reward when a location's internal structure is fully destroyed.

    Weighted by location tactical significance (CT/HD highest, arms lowest).
    Rewards destroying enemy locations, penalises losing own locations.
    """

    def __init__(self, scale: float = 1.0,
                 location_weights: dict[str, float] | None = None):
        self.scale = scale
        self.location_weights = location_weights or LOCATION_WEIGHTS
        self._prev_own: dict[tuple[int, str], float] | None = None
        self._prev_enemy: dict[tuple[int, str], float] | None = None

    def compute(self, prev_obs: dict, curr_obs: dict, terminated: bool) -> float:
        if terminated and not curr_obs.get("units"):
            return 0.0

        units = curr_obs.get("units", [])
        rl_owner = self._rl_owner

        own_locs = _location_internals(units, rl_owner)
        enemy_locs: dict[tuple[int, str], float] = {}
        for u in units:
            if u["owner"] != rl_owner:
                enemy_locs.update(_location_internals(units, u["owner"]))

        if self._prev_own is None:
            self._prev_own = own_locs
            self._prev_enemy = enemy_locs
            return 0.0

        reward = 0.0
        # Enemy locations destroyed → positive reward
        for key, internal in self._prev_enemy.items():
            if internal > 0 and enemy_locs.get(key, internal) <= 0:
                loc_name = key[1]
                reward += self.location_weights.get(loc_name, 0.2) * self.scale

        # Own locations destroyed → negative reward
        for key, internal in self._prev_own.items():
            if internal > 0 and own_locs.get(key, internal) <= 0:
                loc_name = key[1]
                reward -= self.location_weights.get(loc_name, 0.2) * self.scale

        self._prev_own = own_locs
        self._prev_enemy = enemy_locs
        return reward

    def reset(self) -> None:
        self._prev_own = None
        self._prev_enemy = None
        self._rl_owner: int = -1

    def set_rl_owner(self, owner_id: int) -> None:
        self._rl_owner = owner_id


def _to_cube(x: int, y: int) -> tuple[int, int, int]:
    """Convert odd-q offset coordinates to cube coordinates (MegaMek convention).

    MegaMek uses odd-column offset: odd columns are shifted down by half a hex.
    """
    q = x
    r = y - (x - (x & 1)) // 2
    s = -q - r
    return q, r, s


def hex_distance(x1: int, y1: int, x2: int, y2: int) -> int:
    """Hex distance using offset coordinates (odd-q offset, MegaMek convention)."""
    q1, r1, s1 = _to_cube(x1, y1)
    q2, r2, s2 = _to_cube(x2, y2)
    return (abs(q1 - q2) + abs(r1 - r2) + abs(s1 - s2)) // 2


def hex_bearing(x1: int, y1: int, x2: int, y2: int) -> float:
    """Compass bearing in degrees (0=North, clockwise) from hex (x1,y1) to (x2,y2).

    Uses cube coordinates converted to Cartesian pixel positions for flat-top hexes.
    Returns 0.0 if source and target are the same hex.
    """
    if x1 == x2 and y1 == y2:
        return 0.0

    q1, r1, _ = _to_cube(x1, y1)
    q2, r2, _ = _to_cube(x2, y2)

    # Flat-top hex: pixel_x = 3/2 * q, pixel_y = sqrt(3) * (r + q/2)
    sqrt3 = math.sqrt(3)
    px1 = 1.5 * q1
    py1 = sqrt3 * (r1 + q1 / 2.0)
    px2 = 1.5 * q2
    py2 = sqrt3 * (r2 + q2 / 2.0)

    dx = px2 - px1
    dy = py2 - py1

    # atan2 with North=up: angle from positive-Y axis, clockwise
    # Screen Y increases downward, so negate dy for compass bearing
    angle_rad = math.atan2(dx, -dy)
    angle_deg = math.degrees(angle_rad) % 360
    return angle_deg


def in_firing_arc(facing: int, weapon_location: int, bearing: float) -> bool:
    """Check if a weapon can fire at a target given the unit's facing and bearing.

    Replicates Java's FacingArc + UnitPosition.relativeDotProduct logic exactly:
    1. Round bearing to nearest integer degree (matches Coords.dotProduct)
    2. Compute relative angle = (bearing_int - facing_angle) % 360
    3. Apply FacingArc boundary checks with same >= / <= / > / < operators

    Args:
        facing: Unit facing (0-5), where 0=North, 1=NE, etc.
        weapon_location: MegaMek location index (0=HD, 1=CT, 2=RT, 3=LT, 4=RA, 5=LA, 6=RL, 7=LL)
        bearing: Compass bearing to target in degrees (0=North, clockwise)

    Returns:
        True if the weapon can fire at the target.
    """
    # Java's Coords.dotProduct rounds to int; relativeDotProduct subtracts facing
    bearing_int = round(bearing)
    facing_angle = facing * 60
    target = (bearing_int - facing_angle) % 360

    # FacingArc definitions from FacingArc.java (start, end, condition):
    #   ARC_FORWARD(1):   300, 60  -> target >= 300 || target <= 60
    #   ARC_LEFT_ARM(2):  240, 60  -> target >= 240 || target <= 60
    #   ARC_RIGHT_ARM(3): 300, 120 -> target >= 300 || target <= 120
    if weapon_location == 4:  # RA -> ARC_RIGHT_ARM
        return target >= 300 or target <= 120
    elif weapon_location == 5:  # LA -> ARC_LEFT_ARM
        return target >= 240 or target <= 60
    else:  # HD, CT, RT, LT, RL, LL -> ARC_FORWARD
        return target >= 300 or target <= 60


def range_quality(unit: dict, distance: int,
                   target_x: int | None = None, target_y: int | None = None,
                   unit_x: int | None = None, unit_y: int | None = None,
                   unit_facing: int | None = None) -> float:
    """Score how well a unit's weapons perform at the given distance.

    Returns a damage-weighted average of per-weapon range bracket scores:
    - distance < min_range: -0.5
    - distance <= short_range: 1.0
    - distance <= medium_range: 0.5
    - distance <= long_range: 0.0
    - distance > long_range: -0.5

    When facing info is provided (unit_x, unit_y, unit_facing, target_x, target_y),
    weapons outside their firing arc have their score multiplied by 0.5 — they
    contribute positively for being at favorable distance but at reduced value
    since they can't fire this turn.
    """
    # Compute bearing once if facing info is available
    bearing = None
    if (unit_facing is not None and unit_x is not None and unit_y is not None
            and target_x is not None and target_y is not None):
        bearing = hex_bearing(unit_x, unit_y, target_x, target_y)

    total_score = 0.0
    total_damage = 0.0
    for w in unit.get("weapons", []):
        if w.get("destroyed", False):
            continue
        damage = w.get("damage", 0)
        if damage <= 0:
            continue
        min_range = w.get("min_range", 0)
        short = w.get("short_range", 0)
        medium = w.get("medium_range", 0)
        long = w.get("long_range", 0)

        if distance < min_range:
            score = -0.5
        elif distance <= short:
            score = 1.0
        elif distance <= medium:
            score = 0.5
        elif distance <= long:
            score = 0.0
        else:
            score = -0.5

        # Apply out-of-arc penalty
        if bearing is not None:
            weapon_loc = w.get("location", 1)  # default CT (always in forward arc)
            if not in_firing_arc(unit_facing, weapon_loc, bearing):
                score *= 0.5

        total_score += score * damage
        total_damage += damage

    if total_damage == 0:
        return 0.0
    return total_score / total_damage


class RangeAdvantageReward(RewardFunction):
    """Reward for positioning at favorable weapon ranges relative to opponent.

    Computes the difference in range quality between the RL unit and enemy unit.
    Positive when RL's weapons are more effective at the current distance than
    the enemy's weapons.
    """

    def __init__(self, scale: float = 1.0, absolute_weight: float = 0.3):
        self.scale = scale
        self.absolute_weight = absolute_weight
        self._rl_owner: int = -1

    def compute(self, prev_obs: dict, curr_obs: dict, terminated: bool) -> float:
        units = curr_obs.get("units", [])
        if not units:
            return 0.0

        rl_unit = None
        enemy_unit = None
        for u in units:
            if u.get("destroyed", False):
                continue
            if u["owner"] == self._rl_owner:
                rl_unit = u
            else:
                enemy_unit = u

        if rl_unit is None or enemy_unit is None:
            return 0.0

        # Skip if either unit is undeployed
        if rl_unit.get("x", -1) == -1 or enemy_unit.get("x", -1) == -1:
            return 0.0

        rl_x, rl_y = rl_unit["x"], rl_unit["y"]

        # Use post-movement enemy position if available. This is the enemy's
        # position after the previous round's movement phase completed, captured
        # at the start of the firing phase. This avoids the ~50% reward error
        # that occurs when initiative changes between rounds (the live enemy
        # position in units[] may reflect an extra move from the current round).
        pm_ex = curr_obs.get("prev_round_enemy_x", -1)
        pm_ey = curr_obs.get("prev_round_enemy_y", -1)
        pm_facing = curr_obs.get("prev_round_enemy_facing", -1)

        if pm_ex >= 0 and pm_ey >= 0:
            ex, ey = pm_ex, pm_ey
            enemy_facing = pm_facing if pm_facing >= 0 else enemy_unit.get("facing")
        else:
            ex, ey = enemy_unit["x"], enemy_unit["y"]
            enemy_facing = enemy_unit.get("facing")

        rl_facing = rl_unit.get("facing")

        dist = hex_distance(rl_x, rl_y, ex, ey)

        rl_quality = range_quality(
            rl_unit, dist,
            target_x=ex, target_y=ey,
            unit_x=rl_x, unit_y=rl_y,
            unit_facing=rl_facing,
        )
        enemy_quality = range_quality(
            enemy_unit, dist,
            target_x=rl_x, target_y=rl_y,
            unit_x=ex, unit_y=ey,
            unit_facing=enemy_facing,
        )
        differential = rl_quality - enemy_quality
        aw = self.absolute_weight
        blended = aw * rl_quality + (1 - aw) * differential
        return self.scale * blended

    def reset(self) -> None:
        self._rl_owner = -1

    def set_rl_owner(self, owner_id: int) -> None:
        self._rl_owner = owner_id


def cover_value(board_hexes: list[dict], x: int, y: int) -> float:
    """Return cover value for the hex at (x, y).

    Heavy Woods = 2.0 (+2 to-hit), Light Woods = 1.0 (+1 to-hit), else 0.0.
    """
    for h in board_hexes:
        if h.get("x") == x and h.get("y") == y:
            terrain = h.get("terrain", "")
            if "Heavy Woods" in terrain:
                return 2.0
            if "Light Woods" in terrain:
                return 1.0
            return 0.0
    return 0.0


class CoverReward(RewardFunction):
    """Reward for positioning in terrain that provides defensive cover.

    Returns the cover value of the RL unit's hex only — the agent can't
    control where the enemy stands, so enemy cover is not subtracted.
    """

    def __init__(self, scale: float = 1.0):
        self.scale = scale
        self._rl_owner: int = -1

    def compute(self, prev_obs: dict, curr_obs: dict, terminated: bool) -> float:
        units = curr_obs.get("units", [])
        if not units:
            return 0.0

        rl_unit = None
        for u in units:
            if u.get("destroyed", False):
                continue
            if u["owner"] == self._rl_owner:
                rl_unit = u

        if rl_unit is None:
            return 0.0

        # Skip if undeployed
        if rl_unit.get("x", -1) == -1:
            return 0.0

        board_hexes = curr_obs.get("board", {}).get("hexes", [])
        rl_cover = cover_value(board_hexes, rl_unit["x"], rl_unit["y"])
        return self.scale * rl_cover

    def reset(self) -> None:
        self._rl_owner = -1

    def set_rl_owner(self, owner_id: int) -> None:
        self._rl_owner = owner_id


def _get_prone_status(units: list[dict], owner_id: int) -> bool | None:
    """Return prone status for the first non-destroyed unit belonging to owner.

    Returns None if no matching unit found.
    """
    for u in units:
        if u["owner"] == owner_id and not u.get("destroyed", False):
            return bool(u.get("prone", False))
    return None


class PronePenaltyReward(RewardFunction):
    """Penalty when the RL unit involuntarily falls prone.

    All not-prone → prone transitions are involuntary (failed PSR from damage,
    terrain, etc.) since the Java-side legal move enumeration never includes
    "go prone" as a deliberate action.
    """

    def __init__(self, scale: float = 1.0):
        self.scale = scale
        self._rl_owner: int = -1
        self._prev_prone: bool | None = None

    def compute(self, prev_obs: dict, curr_obs: dict, terminated: bool) -> float:
        units = curr_obs.get("units", [])
        if not units:
            return 0.0

        curr_prone = _get_prone_status(units, self._rl_owner)
        if curr_prone is None:
            return 0.0

        prev_prone = self._prev_prone
        self._prev_prone = curr_prone

        if prev_prone is None:
            # First observation — no transition to judge
            return 0.0

        if not prev_prone and curr_prone:
            # Fell prone — penalty
            return -1.0 * self.scale

        return 0.0

    def reset(self) -> None:
        self._rl_owner = -1
        self._prev_prone = None

    def set_rl_owner(self, owner_id: int) -> None:
        self._rl_owner = owner_id


class CompositeReward(RewardFunction):
    """Weighted sum of multiple reward functions."""

    def __init__(self, components: list[tuple[RewardFunction, float]] | None = None):
        if components is None:
            components = [
                (DamageDeltaReward(normalizer=20.0), 1.0),
                (LocationDestructionReward(), 1.0),
                (RangeAdvantageReward(), 0.5),
                (CoverReward(), 0.05),
                (PronePenaltyReward(), 0.5),
                (WinLossReward(), 5.0),
            ]
        self.components = components
        self.last_details: list[tuple[str, float, float]] = []

    def compute(self, prev_obs: dict, curr_obs: dict, terminated: bool) -> float:
        self.last_details = []
        total = 0.0
        for fn, weight in self.components:
            raw = fn.compute(prev_obs, curr_obs, terminated)
            weighted = weight * raw
            self.last_details.append((type(fn).__name__, raw, weighted))
            total += weighted
        return total

    def reset(self) -> None:
        for fn, _ in self.components:
            fn.reset()
        self.last_details = []

    def set_rl_owner(self, owner_id: int) -> None:
        for fn, _ in self.components:
            if hasattr(fn, "set_rl_owner"):
                fn.set_rl_owner(owner_id)
