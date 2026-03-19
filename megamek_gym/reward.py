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


def _total_hp(units: list[dict], owner_id: int) -> float:
    """Sum armor + internal across all locations for units belonging to owner."""
    total = 0.0
    for u in units:
        if u["owner"] == owner_id:
            for loc in u.get("armor", []):
                total += loc.get("armor", 0) + loc.get("internal", 0)
                total += loc.get("rear_armor", 0)
    return total


class DamageDeltaReward(RewardFunction):
    """Reward based on net damage dealt minus damage taken."""

    def __init__(self, scale: float = 1.0, normalizer: float = 100.0):
        self.scale = scale
        self.normalizer = normalizer
        self._prev_own: float | None = None
        self._prev_enemy: float | None = None

    def compute(self, prev_obs: dict, curr_obs: dict, terminated: bool) -> float:
        if terminated and not curr_obs.get("units"):
            # Terminal obs with no units — no delta to compute
            return 0.0

        rl_owner = self._rl_owner
        own_hp = _total_hp(curr_obs.get("units", []), rl_owner)
        enemy_hp = sum(
            _total_hp(curr_obs.get("units", []), u["owner"])
            for u in curr_obs.get("units", [])
            if u["owner"] != rl_owner
        ) if curr_obs.get("units") else 0.0
        # Deduplicate: just compute enemy_hp directly
        enemy_hp = 0.0
        enemy_owners = set()
        for u in curr_obs.get("units", []):
            if u["owner"] != rl_owner:
                enemy_owners.add(u["owner"])
        for eid in enemy_owners:
            enemy_hp += _total_hp(curr_obs.get("units", []), eid)

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


class CompositeReward(RewardFunction):
    """Weighted sum of multiple reward functions."""

    def __init__(self, components: list[tuple[RewardFunction, float]] | None = None):
        if components is None:
            components = [
                (DamageDeltaReward(), 1.0),
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
