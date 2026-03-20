"""Pluggable reward functions computed from raw JSON observations."""

from __future__ import annotations

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
                total += loc.get("internal", 0) * internal_multiplier
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


def _hex_distance(x1: int, y1: int, x2: int, y2: int) -> int:
    """Hex distance using offset coordinates (even-column offset, MegaMek convention).

    Converts to cube coordinates then computes standard hex distance.
    """
    # Convert even-column offset to cube coordinates
    def _to_cube(x: int, y: int) -> tuple[int, int, int]:
        q = x
        r = y - (x + (x & 1)) // 2
        s = -q - r
        return q, r, s

    q1, r1, s1 = _to_cube(x1, y1)
    q2, r2, s2 = _to_cube(x2, y2)
    return (abs(q1 - q2) + abs(r1 - r2) + abs(s1 - s2)) // 2


def _range_quality(unit: dict, distance: int) -> float:
    """Score how well a unit's weapons perform at the given distance.

    Returns a damage-weighted average of per-weapon range bracket scores:
    - distance < min_range: -0.5
    - distance <= short_range: 1.0
    - distance <= medium_range: 0.5
    - distance <= long_range: 0.0
    - distance > long_range: -0.5
    """
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

    def __init__(self, scale: float = 1.0):
        self.scale = scale
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

        dist = _hex_distance(rl_unit["x"], rl_unit["y"],
                             enemy_unit["x"], enemy_unit["y"])
        range_advantage = (_range_quality(rl_unit, dist)
                           - _range_quality(enemy_unit, dist))
        return self.scale * range_advantage

    def reset(self) -> None:
        self._rl_owner = -1

    def set_rl_owner(self, owner_id: int) -> None:
        self._rl_owner = owner_id


def _cover_value(board_hexes: list[dict], x: int, y: int) -> float:
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
        rl_cover = _cover_value(board_hexes, rl_unit["x"], rl_unit["y"])
        return self.scale * rl_cover

    def reset(self) -> None:
        self._rl_owner = -1

    def set_rl_owner(self, owner_id: int) -> None:
        self._rl_owner = owner_id


class CompositeReward(RewardFunction):
    """Weighted sum of multiple reward functions."""

    def __init__(self, components: list[tuple[RewardFunction, float]] | None = None):
        if components is None:
            components = [
                (DamageDeltaReward(), 1.0),
                (LocationDestructionReward(), 1.0),
                (RangeAdvantageReward(), 0.5),
                (CoverReward(), 0.25),
                (WinLossReward(), 10.0),
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
