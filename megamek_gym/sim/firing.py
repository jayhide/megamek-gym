"""Weapon firing resolution: to-hit, hit location, damage, criticals."""

from __future__ import annotations

import random
from typing import TYPE_CHECKING

from megamek_gym.reward import hex_bearing, hex_distance, in_firing_arc_with_twist
from megamek_gym.sim.board import Board
from megamek_gym.sim.los import LosTable
from megamek_gym.sim.unit import (
    TRANSFER_TABLE,
    Location,
    Unit,
    WeaponData,
    cluster_hits,
    HIT_TABLE_FRONT,
    HIT_TABLE_LEFT,
    HIT_TABLE_RIGHT,
    HIT_TABLE_REAR,
)

if TYPE_CHECKING:
    pass


def d6(n: int = 1, rng: random.Random | None = None) -> int:
    """Roll n d6 dice and return the sum."""
    r = rng or random
    return sum(r.randint(1, 6) for _ in range(n))


def target_movement_modifier(hexes_moved: int) -> int:
    """TMM based on number of hexes moved."""
    if hexes_moved <= 2:
        return 0
    elif hexes_moved <= 4:
        return 1
    elif hexes_moved <= 6:
        return 2
    elif hexes_moved <= 9:
        return 3
    else:
        return 4


def compute_to_hit(attacker: Unit, target: Unit, weapon: WeaponData,
                   board: Board, los_table: LosTable) -> int | None:
    """Compute the to-hit target number (2d6 >=).

    Returns None if the shot is impossible.
    """
    if attacker.destroyed or target.destroyed:
        return None
    if not attacker.deployed or not target.deployed:
        return None

    dist = hex_distance(attacker.x, attacker.y, target.x, target.y)

    # Range check
    if dist > weapon.long_range:
        return None
    if dist < weapon.min_range:
        # Min range penalty (not impossible, but penalized)
        pass

    # LOS check
    if not los_table.has_los(attacker.x, attacker.y, target.x, target.y):
        return None

    # Firing arc check (with torso twist)
    bearing = hex_bearing(attacker.x, attacker.y, target.x, target.y)
    if not in_firing_arc_with_twist(attacker.facing, int(weapon.location), bearing):
        return None

    # Base to-hit
    tn = attacker.template.gunnery  # Base gunnery (4)

    # Range modifier
    if dist <= weapon.short_range:
        tn += 0
    elif dist <= weapon.medium_range:
        tn += 2
    elif dist <= weapon.long_range:
        tn += 4

    # Min range penalty
    if weapon.min_range > 0 and dist < weapon.min_range:
        tn += weapon.min_range - dist

    # Attacker movement modifier
    if attacker.movement_type == "run":
        tn += 2
    elif attacker.movement_type == "walk":
        tn += 1

    # Target movement modifier (TMM)
    tn += target_movement_modifier(target.moved_hexes)

    # Heat modifier
    tn += attacker.gunnery_modifier

    # Terrain modifier (woods at target hex)
    from megamek_gym.sim.los import compute_terrain_modifier
    tn += compute_terrain_modifier(board, attacker.x, attacker.y,
                                    target.x, target.y)

    # Attacker prone: leg weapons impossible, arm-destroyed blocks all
    if attacker.prone:
        if weapon.location in (Location.RL, Location.LL):
            return None
        # Standard rules (TAC_OPS_PRONE_FIRE disabled): prone mech needs
        # both arms to brace. If either arm destroyed, all weapons impossible.
        if (attacker.loc_destroyed[Location.RA]
                or attacker.loc_destroyed[Location.LA]):
            return None
        tn += 2

    # Target prone: harder to hit at range, easier up close
    if target.prone:
        if dist <= 1:
            tn -= 2  # Easier to hit prone target adjacent
        else:
            tn += 1  # Harder to hit prone target at range

    # Target immobile: -4 modifier
    if getattr(target, 'immobile', False) or target.shutdown:
        tn -= 4

    # Arm actuator damage modifier
    tn += attacker.arm_actuator_modifier(weapon.location)

    # Sensor damage modifier (+2 per sensor hit, standard cockpit)
    if attacker.sensor_hits > 0:
        tn += 2

    # Spotting for indirect fire: +1 penalty
    if getattr(attacker, 'spotting', False):
        tn += 1

    # Clamp: 2 always misses, 12 always succeeds
    return max(2, tn)


def determine_hit_side(attacker: Unit, target: Unit) -> str:
    """Determine which hit table to use based on relative facing."""
    bearing = hex_bearing(target.x, target.y, attacker.x, attacker.y)
    facing_angle = target.facing * 60
    relative = (round(bearing) - facing_angle) % 360

    if relative >= 300 or relative <= 60:
        return "front"
    elif 60 < relative <= 120:
        return "right"
    elif 120 < relative <= 240:
        return "rear"
    else:
        return "left"


def roll_hit_location(side: str, rng: random.Random | None = None) -> tuple[Location, bool]:
    """Roll hit location on the appropriate table.

    Returns (location, is_tac) where is_tac indicates a through-armor critical.
    """
    roll = d6(2, rng)

    if side == "front":
        table = HIT_TABLE_FRONT
    elif side == "left":
        table = HIT_TABLE_LEFT
    elif side == "right":
        table = HIT_TABLE_RIGHT
    else:
        table = HIT_TABLE_REAR

    loc = table[roll]
    is_tac = (roll == 2)

    return loc, is_tac


def _find_best_twist(attacker: Unit, target: Unit, weapon_indices: list[int]) -> int:
    """Find the best torso twist facing to maximize weapons in arc.

    Returns the twist offset (-1, 0, or +1).
    """
    bearing = hex_bearing(attacker.x, attacker.y, target.x, target.y)
    best_twist = 0
    best_count = 0

    for twist in (-1, 0, 1):
        twisted_facing = (attacker.facing + twist) % 6
        count = 0
        for wi in weapon_indices:
            w = attacker.template.weapons[wi]
            if in_firing_arc_with_twist(twisted_facing, int(w.location), bearing,
                                         max_twist=0):
                count += 1
        if count > best_count:
            best_count = count
            best_twist = twist

    return best_twist


def resolve_firing(attacker: Unit, target: Unit, board: Board,
                   los_table: LosTable,
                   rng: random.Random | None = None) -> dict:
    """Resolve all weapon fire from attacker to target.

    Fires all weapons that can fire. Returns a summary dict.
    """
    r = rng or random
    total_damage = 0
    total_heat = 0
    hits: list[dict] = []

    # Find all fireable weapons
    fireable = []
    for i, w in enumerate(attacker.template.weapons):
        if attacker.weapon_destroyed[i]:
            continue
        if attacker.loc_destroyed[w.location]:
            continue
        if not attacker.has_ammo_for(i):
            continue
        fireable.append(i)

    # Prone arm restriction: biped can only fire arm weapons from one arm
    prone_arm_fired: Location | None = None

    for wi in fireable:
        w = attacker.template.weapons[wi]

        tn = compute_to_hit(attacker, target, w, board, los_table)
        if tn is None:
            continue

        # Prone arm restriction: only one arm can fire, other braces.
        # Check AFTER compute_to_hit so an arm is only claimed when
        # a weapon on it will actually fire (matching Java's
        # isFiringFromArmAlready which checks committed actions).
        if attacker.prone and w.location in (Location.RA, Location.LA):
            if prone_arm_fired is None:
                prone_arm_fired = w.location
            elif prone_arm_fired != w.location:
                continue

        # Fire weapon — accumulate heat
        total_heat += w.heat
        attacker.consume_ammo(wi)

        # Roll to hit
        roll = d6(2, r)

        if roll < tn:
            hits.append({"weapon": w.name, "roll": roll, "tn": tn, "hit": False})
            continue

        # Determine hit side
        side = determine_hit_side(attacker, target)
        use_rear = (side == "rear")

        if w.is_cluster:
            # Cluster weapon: determine how many missiles hit
            cluster_roll = d6(2, r)
            n_hits = cluster_hits(cluster_roll, w.cluster_size)

            # Apply damage in 5-point groups, each to a separate location
            remaining = n_hits
            while remaining > 0:
                group_damage = min(remaining, 5) * w.damage
                remaining -= min(remaining, 5)

                loc, is_tac = roll_hit_location(side, r)
                dmg_dealt = apply_damage(target, loc, group_damage, use_rear, r,
                                         is_tac=is_tac)
                total_damage += dmg_dealt

            hits.append({
                "weapon": w.name, "roll": roll, "tn": tn, "hit": True,
                "missiles": n_hits, "damage": n_hits * w.damage,
            })
        else:
            # Direct fire weapon
            loc, is_tac = roll_hit_location(side, r)
            dmg_dealt = apply_damage(target, loc, w.damage, use_rear, r,
                                     is_tac=is_tac)
            total_damage += dmg_dealt

            hits.append({
                "weapon": w.name, "roll": roll, "tn": tn, "hit": True,
                "location": loc.name, "damage": w.damage,
            })

    return {
        "attacker": attacker.entity_id,
        "target": target.entity_id,
        "heat_generated": total_heat,
        "total_damage": total_damage,
        "hits": hits,
    }


def apply_damage(target: Unit, loc: Location, damage: int,
                 rear: bool = False, rng: random.Random | None = None,
                 is_tac: bool = False) -> int:
    """Apply damage to a specific location, with transfer.

    If is_tac is True, a through-armor critical is guaranteed on the first
    location that takes internal structure damage.

    Returns total damage actually applied.
    """
    r = rng or random
    total_applied = 0
    remaining = damage
    current_loc = loc
    tac_pending = is_tac  # TAC only applies to initial location

    while remaining > 0 and current_loc is not None:
        if target.loc_destroyed[current_loc]:
            # Location already destroyed — transfer
            current_loc = TRANSFER_TABLE.get(current_loc)
            continue

        # Apply to armor first
        if rear and current_loc in (Location.CT, Location.RT, Location.LT):
            armor_idx = 1  # Rear armor
        else:
            armor_idx = 0  # Front armor

        armor = target.armor[current_loc][armor_idx]
        if armor > 0:
            absorbed = min(armor, remaining)
            target.armor[current_loc][armor_idx] -= absorbed
            remaining -= absorbed
            total_applied += absorbed

        if remaining <= 0:
            break

        # Remaining damage goes to internal structure
        internal = target.armor[current_loc][2]
        if internal > 0:
            absorbed = min(internal, remaining)
            target.armor[current_loc][2] -= absorbed
            remaining -= absorbed
            total_applied += absorbed

            # Critical hit check when internals are damaged
            _check_critical(target, current_loc, r, is_tac=tac_pending)
            tac_pending = False  # TAC consumed on first internal damage

        # Check if location is destroyed
        if target.armor[current_loc][2] <= 0:
            target.loc_destroyed[current_loc] = True
            _destroy_location(target, current_loc)

            # Check for unit destruction
            if current_loc in (Location.CT, Location.HD):
                target.destroyed = True
                return total_applied

            # Transfer remaining damage
            current_loc = TRANSFER_TABLE.get(current_loc)
        else:
            break

    return total_applied


def _check_critical(target: Unit, loc: Location,
                    rng: random.Random, is_tac: bool = False) -> None:
    """Roll for and apply critical hits when internal structure takes damage.

    Matches Java TW crit table: 8-9=1 crit, 10-11=2, 12=3 (head: blown off).
    Through-armor crits (is_tac=True) guarantee 1 crit without rolling.
    """
    if is_tac:
        _apply_critical(target, loc, rng)
        return

    roll = d6(2, rng)
    if roll <= 7:
        return
    elif roll <= 9:
        num_crits = 1
    elif roll <= 11:
        num_crits = 2
    else:
        # Roll of 12: head blown off, otherwise 3 crits
        if loc == Location.HD:
            target.armor[loc][2] = 0
            target.loc_destroyed[loc] = True
            _destroy_location(target, loc)
            target.destroyed = True
            return
        num_crits = 3

    for _ in range(num_crits):
        _apply_critical(target, loc, rng)


def _ammo_explosion_damage(target: Unit, ammo_idx: int) -> int:
    """Calculate per-round explosion damage for an ammo bin.

    BattleTech rules: each round of ammo explodes for its full weapon damage.
    For cluster weapons (SRM/LRM): rack_size * per_missile_damage.
    For direct-fire weapons: weapon damage per round.
    """
    ammo_bin = target.template.ammo[ammo_idx]
    for w in target.template.weapons:
        if w.name == ammo_bin.weapon_name:
            if w.is_cluster:
                return w.cluster_size * w.damage
            return w.damage
    return 2  # Fallback


def _apply_critical(target: Unit, loc: Location,
                    rng: random.Random) -> None:
    """Apply a critical hit to a random piece of equipment in the location."""
    # Find weapons in this location
    weapon_indices = [i for i, w in enumerate(target.template.weapons)
                      if w.location == loc and not target.weapon_destroyed[i]]

    # Find ammo bins in this location
    ammo_indices = [i for i, a in enumerate(target.template.ammo)
                    if a.location == loc and target.ammo_remaining[i] > 0]

    # System crits (gyro in CT, engine in CT/RT/LT, actuators in legs/arms,
    # cockpit/sensors/life_support in HD, heat sinks in torsos)
    system_options: list[str] = []
    if loc == Location.HD:
        system_options.extend(["cockpit", "sensors", "life_support"])
    elif loc == Location.CT:
        system_options.extend(["gyro", "engine"])
    elif loc in (Location.RT, Location.LT):
        system_options.append("engine")
    elif loc in (Location.RL, Location.LL):
        leg_idx = 0 if loc == Location.RL else 1
        if not target.hip_hits[leg_idx]:
            system_options.append("hip")
        if target.leg_actuator_hits[leg_idx] < 3:
            system_options.append("leg_actuator")
    elif loc in (Location.RA, Location.LA):
        arm_idx = 0 if loc == Location.RA else 1
        if not target.shoulder_destroyed[arm_idx]:
            system_options.append("shoulder")
        if not target.upper_arm_destroyed[arm_idx]:
            system_options.append("upper_arm")
        if not target.lower_arm_destroyed[arm_idx]:
            system_options.append("lower_arm")

    # Heat sinks as crit targets in torso locations
    # (10 sinks are integral to the engine and not separately destroyable)
    extra_sinks = target.template.heat_sinks - 10
    if (loc in (Location.CT, Location.RT, Location.LT)
            and extra_sinks > 0
            and target.heat_sinks_destroyed < extra_sinks):
        system_options.append("heat_sink")

    all_options = (
        [("weapon", i) for i in weapon_indices]
        + [("ammo", i) for i in ammo_indices]
        + [("system", s) for s in system_options]
    )

    if not all_options:
        return

    choice = rng.choice(all_options)
    crit_type, crit_idx = choice

    if crit_type == "weapon":
        target.weapon_destroyed[crit_idx] = True
    elif crit_type == "ammo":
        # Ammo explosion!
        ammo_remaining = target.ammo_remaining[crit_idx]
        target.ammo_remaining[crit_idx] = 0
        if ammo_remaining > 0:
            per_round = _ammo_explosion_damage(target, crit_idx)
            explosion_damage = ammo_remaining * per_round
            # Apply to the location's internals directly
            internal = target.armor[loc][2]
            target.armor[loc][2] = max(0, internal - explosion_damage)
            if target.armor[loc][2] <= 0:
                target.loc_destroyed[loc] = True
                _destroy_location(target, loc)
                if loc in (Location.CT, Location.HD):
                    target.destroyed = True
    elif crit_type == "system":
        if crit_idx == "engine":
            target.engine_hits += 1
            if target.engine_hits >= 3:
                target.destroyed = True
        elif crit_idx == "gyro":
            target.gyro_hits += 1
            # 1 gyro hit: harder piloting, 2 = fall
            if target.gyro_hits >= 2:
                target.prone = True
        elif crit_idx == "hip":
            leg_idx = 0 if loc == Location.RL else 1
            target.hip_hits[leg_idx] = True
        elif crit_idx == "leg_actuator":
            leg_idx = 0 if loc == Location.RL else 1
            target.leg_actuator_hits[leg_idx] = min(3, target.leg_actuator_hits[leg_idx] + 1)
        elif crit_idx == "shoulder":
            arm_idx = 0 if loc == Location.RA else 1
            target.shoulder_destroyed[arm_idx] = True
        elif crit_idx == "upper_arm":
            arm_idx = 0 if loc == Location.RA else 1
            target.upper_arm_destroyed[arm_idx] = True
        elif crit_idx == "lower_arm":
            arm_idx = 0 if loc == Location.RA else 1
            target.lower_arm_destroyed[arm_idx] = True
        elif crit_idx == "cockpit":
            target.destroyed = True  # Pilot killed
        elif crit_idx == "sensors":
            target.sensor_hits += 1
        elif crit_idx == "life_support":
            target.life_support_hits += 1
        elif crit_idx == "heat_sink":
            target.heat_sinks_destroyed += 1


def _destroy_location(target: Unit, loc: Location) -> None:
    """Handle effects of a location being destroyed."""
    # Destroy all weapons in this location
    for i, w in enumerate(target.template.weapons):
        if w.location == loc:
            target.weapon_destroyed[i] = True

    # Destroy all ammo in this location (no explosion for already-destroyed)
    for i, a in enumerate(target.template.ammo):
        if a.location == loc:
            target.ammo_remaining[i] = 0

    # Leg destruction causes falling
    if loc in (Location.RL, Location.LL):
        target.prone = True

    # If both legs destroyed, mech is destroyed
    if target.loc_destroyed[Location.RL] and target.loc_destroyed[Location.LL]:
        target.destroyed = True

    # Side torso destruction also destroys the corresponding arm
    if loc == Location.RT and not target.loc_destroyed[Location.RA]:
        target.loc_destroyed[Location.RA] = True
        _destroy_location(target, Location.RA)
    elif loc == Location.LT and not target.loc_destroyed[Location.LA]:
        target.loc_destroyed[Location.LA] = True
        _destroy_location(target, Location.LA)
