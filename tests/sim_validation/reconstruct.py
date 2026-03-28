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
    unit.shutdown = java_unit_dict.get("shutdown", False)
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

    # Crit state (from writeCritState in ObservationBuilder)
    crit = java_unit_dict.get("crit_state")
    if crit:
        unit.engine_hits = crit.get("engine_hits", 0)
        unit.gyro_hits = crit.get("gyro_hits", 0)
        unit.sensor_hits = crit.get("sensor_hits", 0)
        unit.life_support_hits = crit.get("life_support_hits", 0)

        rl = crit.get("right_leg", {})
        ll = crit.get("left_leg", {})
        unit.hip_hits = [rl.get("hip_hits", 0) > 0, ll.get("hip_hits", 0) > 0]
        # Sum upper_leg + lower_leg + foot hits per leg (sim tracks as single counter)
        unit.leg_actuator_hits = [
            rl.get("upper_leg_hits", 0) + rl.get("lower_leg_hits", 0) + rl.get("foot_hits", 0),
            ll.get("upper_leg_hits", 0) + ll.get("lower_leg_hits", 0) + ll.get("foot_hits", 0),
        ]

        ra = crit.get("right_arm", {})
        la = crit.get("left_arm", {})
        unit.shoulder_destroyed = [ra.get("shoulder_hits", 0) > 0, la.get("shoulder_hits", 0) > 0]
        unit.upper_arm_destroyed = [ra.get("upper_arm_hits", 0) > 0, la.get("upper_arm_hits", 0) > 0]
        unit.lower_arm_destroyed = [ra.get("lower_arm_hits", 0) > 0, la.get("lower_arm_hits", 0) > 0]

        unit.heat_sinks_destroyed = crit.get("heat_sinks_destroyed", 0)

    # Ammo state (from writeAmmo in ObservationBuilder)
    java_ammo = java_unit_dict.get("ammo", [])
    for i, ja in enumerate(java_ammo):
        if i < len(unit.ammo_remaining):
            unit.ammo_remaining[i] = ja.get("shots_remaining", 0)

    # Override walk MP from Java's reported value (accounts for heat/damage
    # effects that the sim may not model identically)
    java_walk = java_unit_dict.get("mp_walk")
    if java_walk is not None:
        unit._mp_walk_override = java_walk

    return unit


_JAVA_MOVED_TO_PYTHON = {
    "MOVE_WALK": "walk",
    "MOVE_RUN": "run",
    "MOVE_NONE": "none",
    "MOVE_SPRINT": "run",  # treat sprint as run for modifier purposes
}


def set_firing_state(unit: Unit, firing_entity: dict) -> None:
    """Override unit state with firing-time values from firing_report.

    The firing_report captures position/movement/heat at the start of the
    firing phase (post-movement, pre-firing).  This ensures compute_to_hit()
    uses the same inputs as Java's WeaponAttackAction.toHit().
    """
    unit.x = firing_entity["x"]
    unit.y = firing_entity["y"]
    unit.facing = firing_entity["facing"]
    unit.moved_hexes = firing_entity["delta_distance"]
    unit.heat = firing_entity["heat"]
    moved = firing_entity["moved"]
    unit.movement_type = _JAVA_MOVED_TO_PYTHON.get(moved, "none")

    # Prone flag at firing time (may differ from obs dict if unit stood up)
    if "prone" in firing_entity:
        unit.prone = firing_entity["prone"]

    # Target immobile flag (from Java's Entity.isImmobile())
    unit.immobile = firing_entity.get("immobile", False)
    unit.spotting = firing_entity.get("spotting", False)

    # Arm actuator damage
    if "ra_shoulder_destroyed" in firing_entity:
        unit.shoulder_destroyed[0] = firing_entity["ra_shoulder_destroyed"]
        unit.upper_arm_destroyed[0] = firing_entity["ra_upper_arm_destroyed"]
        unit.lower_arm_destroyed[0] = firing_entity["ra_lower_arm_destroyed"]
    if "la_shoulder_destroyed" in firing_entity:
        unit.shoulder_destroyed[1] = firing_entity["la_shoulder_destroyed"]
        unit.upper_arm_destroyed[1] = firing_entity["la_upper_arm_destroyed"]
        unit.lower_arm_destroyed[1] = firing_entity["la_lower_arm_destroyed"]

    # Sensor damage (for to-hit modifier)
    if "sensor_hits" in firing_entity:
        unit.sensor_hits = firing_entity["sensor_hits"]


def extract_unit_state(raw_obs: dict, owner_id: int) -> dict | None:
    """Extract the unit dict for a given owner from a raw observation."""
    for u in raw_obs.get("units", []):
        if u.get("owner") == owner_id:
            return u
    return None


def extract_crit_state(raw_obs: dict, owner_id: int) -> dict | None:
    """Extract the crit_state dict for a given owner from a raw observation."""
    unit = extract_unit_state(raw_obs, owner_id)
    if unit is None:
        return None
    return unit.get("crit_state")


def extract_ammo_state(raw_obs: dict, owner_id: int) -> list[dict] | None:
    """Extract ammo bin state from a raw observation."""
    unit = extract_unit_state(raw_obs, owner_id)
    if unit is None:
        return None
    return unit.get("ammo")


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
