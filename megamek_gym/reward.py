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


class CompositeReward(RewardFunction):
    """Weighted sum of multiple reward functions."""

    def __init__(self, components: list[tuple[RewardFunction, float]] | None = None):
        if components is None:
            components = [
                (DamageDeltaReward(), 1.0),
                (LocationDestructionReward(), 1.0),
                (WinLossReward(), 10.0),
            ]
        self.components = components

    def compute(self, prev_obs: dict, curr_obs: dict, terminated: bool) -> float:
        return sum(
            weight * fn.compute(prev_obs, curr_obs, terminated)
            for fn, weight in self.components
        )

    def reset(self) -> None:
        for fn, _ in self.components:
            fn.reset()

    def set_rl_owner(self, owner_id: int) -> None:
        for fn, _ in self.components:
            if hasattr(fn, "set_rl_owner"):
                fn.set_rl_owner(owner_id)
