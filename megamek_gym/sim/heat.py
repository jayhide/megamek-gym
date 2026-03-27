"""Heat tracking, dissipation, and overheat effects."""

from __future__ import annotations

import random
from typing import TYPE_CHECKING

from megamek_gym.sim.firing import d6

if TYPE_CHECKING:
    from megamek_gym.sim.unit import Unit


def apply_heat(unit: Unit, heat_generated: int) -> None:
    """Add heat from weapons fire and movement."""
    # Running generates +2 heat
    if unit.movement_type == "run":
        heat_generated += 2

    unit.heat += heat_generated


def dissipate_heat(unit: Unit) -> None:
    """Remove heat via heat sinks (end of turn)."""
    sinks = unit.effective_heat_sinks
    unit.heat = max(0, unit.heat - sinks)


def check_overheat(unit: Unit, rng: random.Random | None = None) -> dict:
    """Check for overheat effects: shutdown and ammo explosion.

    Returns a dict describing what happened.
    """
    r = rng or random
    effects: dict = {"shutdown": False, "ammo_explosion": False}

    if unit.heat < 14:
        return effects

    # Ammo explosion check (heat >= 19)
    if unit.heat >= 19:
        # Find ammo bins with remaining ammo
        has_ammo = any(a > 0 for a in unit.ammo_remaining)
        if has_ammo:
            if unit.heat >= 28:
                tn = 4
            elif unit.heat >= 23:
                tn = 6
            else:  # 19-22
                tn = 8
            roll = d6(2, r)
            if roll >= tn:
                effects["ammo_explosion"] = True
                # Explode all remaining ammo
                for i in range(len(unit.ammo_remaining)):
                    unit.ammo_remaining[i] = 0
                unit.destroyed = True
                return effects

    # Shutdown check (heat >= 14)
    if unit.heat >= 14:
        if unit.heat >= 26:
            tn = 6  # Very likely shutdown
        elif unit.heat >= 22:
            tn = 8
        elif unit.heat >= 18:
            tn = 10
        elif unit.heat >= 14:
            tn = 12  # Very unlikely
        else:
            return effects

        roll = d6(2, r)
        if roll < tn:
            unit.shutdown = True
            effects["shutdown"] = True

    return effects
