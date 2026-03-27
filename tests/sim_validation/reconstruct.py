"""Reconstruct a sim.Unit from a Java observation dict."""

from __future__ import annotations

from megamek_gym.sim.unit import UNIT_TEMPLATES, Location, Unit, LOC_NAMES


def reconstruct_unit(java_unit_dict: dict, template_name: str = "Trebuchet TBT-5S") -> Unit:
    """Create a sim.Unit matching the Java unit's current state.

    Reads position, facing, armor, weapon status, heat, and prone flag
    from the Java observation dict and sets them on a fresh Unit.
    """
    tmpl = UNIT_TEMPLATES[template_name]
    unit = Unit(template=tmpl, entity_id=java_unit_dict["id"],
                owner=java_unit_dict["owner"])

    # Position and facing
    unit.x = java_unit_dict["x"]
    unit.y = java_unit_dict["y"]
    unit.facing = java_unit_dict["facing"]

    # Status
    unit.heat = java_unit_dict.get("heat", 0)
    unit.prone = java_unit_dict.get("prone", False)
    unit.destroyed = java_unit_dict.get("destroyed", False)

    # Armor: Java obs has per-location dicts
    java_armor = java_unit_dict.get("armor", [])
    for loc_dict in java_armor:
        loc_name = loc_dict["location"]
        loc_idx = LOC_NAMES.index(loc_name)

        front = loc_dict.get("armor", 0)
        rear = loc_dict.get("rear_armor", 0)
        internal = loc_dict.get("internal", 0)

        unit.armor[loc_idx] = [front, rear, internal]

        # Track location destruction
        internal_max = loc_dict.get("internal_max", tmpl.armor[Location(loc_idx)][2])
        if internal <= 0 and internal_max > 0:
            unit.loc_destroyed[loc_idx] = True

    # Weapon destroyed flags
    java_weapons = java_unit_dict.get("weapons", [])
    for i, jw in enumerate(java_weapons):
        if i < len(unit.weapon_destroyed):
            unit.weapon_destroyed[i] = jw.get("destroyed", False)

    return unit


def extract_unit_state(raw_obs: dict, owner_id: int) -> dict | None:
    """Extract the unit dict for a given owner from a raw observation."""
    for u in raw_obs.get("units", []):
        if u.get("owner") == owner_id:
            return u
    return None


def compute_armor_deltas(prev_unit: dict, curr_unit: dict) -> list[dict]:
    """Compute per-location armor/internal changes between two observations.

    Returns a list of dicts, one per location that changed:
    {
        "location": str,
        "loc_idx": int,
        "front_delta": int,  # positive = damage taken
        "rear_delta": int,
        "internal_delta": int,
        "prev_front": int, "curr_front": int,
        "prev_rear": int, "curr_rear": int,
        "prev_internal": int, "curr_internal": int,
        "location_destroyed": bool,  # internal went to 0
    }
    """
    deltas = []
    prev_armor = prev_unit.get("armor", [])
    curr_armor = curr_unit.get("armor", [])

    for i, (pa, ca) in enumerate(zip(prev_armor, curr_armor)):
        pf = pa.get("armor", 0)
        cf = ca.get("armor", 0)
        pr = pa.get("rear_armor", 0)
        cr = ca.get("rear_armor", 0)
        pi = pa.get("internal", 0)
        ci = ca.get("internal", 0)

        fd = pf - cf
        rd = pr - cr
        id_ = pi - ci

        if fd != 0 or rd != 0 or id_ != 0:
            deltas.append({
                "location": pa.get("location", LOC_NAMES[i]),
                "loc_idx": i,
                "front_delta": fd,
                "rear_delta": rd,
                "internal_delta": id_,
                "prev_front": pf, "curr_front": cf,
                "prev_rear": pr, "curr_rear": cr,
                "prev_internal": pi, "curr_internal": ci,
                "location_destroyed": ci <= 0 and pi > 0,
            })

    return deltas
