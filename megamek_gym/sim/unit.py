"""Unit state model for Trebuchet TBT-5S (and future units)."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from enum import IntEnum
from typing import NamedTuple


class Location(IntEnum):
    HD = 0
    CT = 1
    RT = 2
    LT = 3
    RA = 4
    LA = 5
    RL = 6
    LL = 7


LOC_NAMES = ["HD", "CT", "RT", "LT", "RA", "LA", "RL", "LL"]

# Damage transfer: when a location is destroyed, excess damage goes here
TRANSFER_TABLE = {
    Location.RA: Location.RT,
    Location.LA: Location.LT,
    Location.RL: Location.RT,
    Location.LL: Location.LT,
    Location.RT: Location.CT,
    Location.LT: Location.CT,
    Location.HD: Location.CT,
    Location.CT: None,  # CT destruction = mech kill
}


class WeaponData(NamedTuple):
    name: str
    location: Location
    damage: int       # Per-hit damage (5 for ML, 2 for SRM missile)
    heat: int
    min_range: int
    short_range: int
    medium_range: int
    long_range: int
    is_cluster: bool  # True for SRM/LRM
    cluster_size: int  # Number of missiles (6 for SRM-6)


class AmmoBin(NamedTuple):
    weapon_name: str   # Which weapon type this feeds
    location: Location
    shots: int         # Starting ammo count


class UnitTemplate(NamedTuple):
    """Static data for a unit type."""
    name: str
    chassis: str
    model: str
    tonnage: int
    walk_mp: int
    run_mp: int
    jump_mp: int
    heat_sinks: int  # Total heat sink count
    gunnery: int     # Gunnery skill (default 4)
    piloting: int    # Piloting skill (default 5)
    # Per-location: (front_armor, rear_armor, internal_structure)
    # For locations without rear armor, rear_armor = 0
    armor: dict[Location, tuple[int, int, int]]
    weapons: list[WeaponData]
    ammo: list[AmmoBin]


# Trebuchet TBT-5S template
TBT_5S = UnitTemplate(
    name="Trebuchet TBT-5S",
    chassis="Trebuchet",
    model="TBT-5S",
    tonnage=50,
    walk_mp=5,
    run_mp=8,
    jump_mp=0,
    heat_sinks=18,
    gunnery=4,
    piloting=5,
    armor={
        #                  front, rear, internal
        Location.HD: (9, 0, 3),
        Location.CT: (22, 7, 16),
        Location.RT: (11, 5, 12),
        Location.LT: (11, 5, 12),
        Location.RA: (10, 0, 8),
        Location.LA: (10, 0, 8),
        Location.RL: (15, 0, 12),
        Location.LL: (15, 0, 12),
    },
    weapons=[
        WeaponData("Medium Laser", Location.RA, 5, 3, 0, 3, 6, 9, False, 0),
        WeaponData("Medium Laser", Location.RA, 5, 3, 0, 3, 6, 9, False, 0),
        WeaponData("Medium Laser", Location.LA, 5, 3, 0, 3, 6, 9, False, 0),
        WeaponData("SRM 6", Location.LA, 2, 4, 0, 3, 6, 9, True, 6),
        WeaponData("SRM 6", Location.RT, 2, 4, 0, 3, 6, 9, True, 6),
    ],
    ammo=[
        AmmoBin("SRM 6", Location.LT, 15),
        AmmoBin("SRM 6", Location.RT, 15),
    ],
)

UNIT_TEMPLATES = {
    "Trebuchet TBT-5S": TBT_5S,
}


# Cluster hit table (2d6 roll → hits for given rack size)
# Index 0 = roll of 2, index 10 = roll of 12
# Only SRM-6 (rack size 6) needed for TBT-5S
CLUSTER_HITS = {
    2: {2: 1, 4: 1, 5: 1, 6: 2, 10: 3, 15: 5, 20: 6},
    3: {2: 1, 4: 1, 5: 2, 6: 2, 10: 3, 15: 5, 20: 6},
    4: {2: 1, 4: 2, 5: 2, 6: 3, 10: 4, 15: 6, 20: 9},
    5: {2: 1, 4: 2, 5: 3, 6: 3, 10: 6, 15: 9, 20: 12},
    6: {2: 2, 4: 2, 5: 3, 6: 4, 10: 6, 15: 9, 20: 12},
    7: {2: 2, 4: 2, 5: 3, 6: 4, 10: 6, 15: 9, 20: 12},
    8: {2: 2, 4: 3, 5: 3, 6: 4, 10: 6, 15: 9, 20: 12},
    9: {2: 2, 4: 3, 5: 4, 6: 5, 10: 8, 15: 12, 20: 16},
    10: {2: 2, 4: 3, 5: 4, 6: 5, 10: 8, 15: 12, 20: 16},
    11: {2: 2, 4: 4, 5: 5, 6: 6, 10: 10, 15: 15, 20: 20},
    12: {2: 2, 4: 4, 5: 5, 6: 6, 10: 10, 15: 15, 20: 20},
}


def cluster_hits(roll: int, rack_size: int) -> int:
    """Look up number of missiles that hit from cluster table."""
    roll = max(2, min(12, roll))
    row = CLUSTER_HITS.get(roll, CLUSTER_HITS[7])
    # Find the column for this rack size
    if rack_size in row:
        return row[rack_size]
    # Find nearest smaller rack size
    for rs in sorted(row.keys(), reverse=True):
        if rs <= rack_size:
            return row[rs]
    return 1


# Hit location tables (2d6 roll → Location)
HIT_TABLE_FRONT = {
    2: Location.CT,   # TAC
    3: Location.RA,
    4: Location.RA,
    5: Location.RL,
    6: Location.RT,
    7: Location.CT,
    8: Location.LT,
    9: Location.LL,
    10: Location.LA,
    11: Location.LA,
    12: Location.HD,
}

HIT_TABLE_LEFT = {
    2: Location.LT,   # TAC
    3: Location.LL,
    4: Location.LA,
    5: Location.LA,
    6: Location.LL,
    7: Location.LT,
    8: Location.CT,
    9: Location.RT,
    10: Location.RA,
    11: Location.RL,
    12: Location.HD,
}

HIT_TABLE_RIGHT = {
    2: Location.RT,   # TAC
    3: Location.RL,
    4: Location.RA,
    5: Location.RA,
    6: Location.RL,
    7: Location.RT,
    8: Location.CT,
    9: Location.LT,
    10: Location.LA,
    11: Location.LL,
    12: Location.HD,
}

HIT_TABLE_REAR = {
    2: Location.CT,   # TAC
    3: Location.RA,
    4: Location.RA,
    5: Location.RL,
    6: Location.RT,
    7: Location.CT,
    8: Location.LT,
    9: Location.LL,
    10: Location.LA,
    11: Location.LA,
    12: Location.HD,
}


@dataclass
class Unit:
    """Mutable game state for a single mech."""

    template: UnitTemplate
    entity_id: int
    owner: int  # Player ID (0 = RL, 1 = opponent)

    # Position
    x: int = -1
    y: int = -1
    facing: int = 0  # 0-5

    # Per-location state: [front_armor, rear_armor, internal]
    armor: list[list[int]] = field(default_factory=list)
    # Track which locations are destroyed
    loc_destroyed: list[bool] = field(default_factory=list)

    # Weapon states (parallel to template.weapons)
    weapon_destroyed: list[bool] = field(default_factory=list)

    # Ammo remaining (parallel to template.ammo)
    ammo_remaining: list[int] = field(default_factory=list)

    # Heat
    heat: int = 0
    heat_sinks_destroyed: int = 0

    # Status
    prone: bool = False
    destroyed: bool = False
    shutdown: bool = False

    # Movement tracking (reset each turn)
    moved_hexes: int = 0
    movement_type: str = "none"  # "none", "walk", "run"
    mp_used: int = 0

    # Engine/gyro critical tracking
    engine_hits: int = 0
    gyro_hits: int = 0

    # Leg actuator damage (index 0=RL, 1=LL)
    hip_hits: list = field(default_factory=lambda: [False, False])
    leg_actuator_hits: list = field(default_factory=lambda: [0, 0])

    def __post_init__(self) -> None:
        if not self.armor:
            self.reset_state()

    def reset_state(self) -> None:
        """Reset to full health for a new game."""
        t = self.template
        self.armor = []
        for loc in Location:
            front, rear, internal = t.armor[loc]
            self.armor.append([front, rear, internal])
        self.loc_destroyed = [False] * 8
        self.weapon_destroyed = [False] * len(t.weapons)
        self.ammo_remaining = [a.shots for a in t.ammo]
        self.heat = 0
        self.heat_sinks_destroyed = 0
        self.prone = False
        self.destroyed = False
        self.shutdown = False
        self.moved_hexes = 0
        self.movement_type = "none"
        self.mp_used = 0
        self.engine_hits = 0
        self.gyro_hits = 0
        self.hip_hits = [False, False]
        self.leg_actuator_hits = [0, 0]

    def deploy(self, x: int, y: int, facing: int) -> None:
        self.x = x
        self.y = y
        self.facing = facing

    @property
    def deployed(self) -> bool:
        return self.x >= 0 and self.y >= 0

    @property
    def walk_mp(self) -> int:
        """Current walk MP (reduced by leg damage).

        Matches Java BipedMek.getWalkMP():
        - Destroyed leg: MP = 0
        - Hip destroyed: MP = ceil(MP / 2) (applied per leg, sequentially)
        - Each non-hip actuator crit (upper leg, lower leg, foot): MP -= 1
        """
        mp = self.template.walk_mp
        for i, loc in enumerate((Location.RL, Location.LL)):
            if self.loc_destroyed[loc]:
                return 0
            if self.hip_hits[i]:
                mp = math.ceil(mp / 2)
            mp -= self.leg_actuator_hits[i]
        return max(0, mp)

    @property
    def run_mp(self) -> int:
        """Current run MP."""
        return int(self.walk_mp * 1.5) + (self.walk_mp % 2)  # ceil(walk * 1.5)

    @property
    def effective_heat_sinks(self) -> int:
        return max(0, self.template.heat_sinks - self.heat_sinks_destroyed)

    @property
    def gunnery_modifier(self) -> int:
        """Heat-based gunnery penalty (Total Warfare p.153)."""
        if self.heat >= 25:
            return 5
        elif self.heat >= 21:
            return 4
        elif self.heat >= 17:
            return 3
        elif self.heat >= 13:
            return 2
        elif self.heat >= 8:
            return 1
        return 0

    def is_alive(self) -> bool:
        """Unit is still in the fight."""
        return not self.destroyed

    def clear_turn_state(self) -> None:
        """Reset per-turn tracking."""
        self.moved_hexes = 0
        self.movement_type = "none"
        self.mp_used = 0

    def front_armor(self, loc: Location) -> int:
        return self.armor[loc][0]

    def rear_armor(self, loc: Location) -> int:
        return self.armor[loc][1]

    def internal(self, loc: Location) -> int:
        return self.armor[loc][2]

    def max_front_armor(self, loc: Location) -> int:
        return self.template.armor[loc][0]

    def max_rear_armor(self, loc: Location) -> int:
        return self.template.armor[loc][1]

    def max_internal(self, loc: Location) -> int:
        return self.template.armor[loc][2]

    def to_obs_dict(self) -> dict:
        """Produce unit dict matching Java's ObservationBuilder format."""
        armor_list = []
        for loc in Location:
            loc_dict: dict = {
                "location": LOC_NAMES[loc],
                "armor": self.armor[loc][0],
                "armor_max": self.template.armor[loc][0],
                "internal": self.armor[loc][2],
                "internal_max": self.template.armor[loc][2],
            }
            rear_max = self.template.armor[loc][1]
            if rear_max > 0:
                loc_dict["rear_armor"] = self.armor[loc][1]
                loc_dict["rear_armor_max"] = rear_max
            armor_list.append(loc_dict)

        weapons_list = []
        for i, w in enumerate(self.template.weapons):
            eff_damage = w.damage * w.cluster_size if w.is_cluster else w.damage
            weapons_list.append({
                "name": w.name,
                "location": int(w.location),
                "damage": eff_damage,
                "min_range": w.min_range,
                "short_range": w.short_range,
                "medium_range": w.medium_range,
                "long_range": w.long_range,
                "destroyed": self.weapon_destroyed[i],
            })

        return {
            "id": self.entity_id,
            "owner": self.owner,
            "chassis": self.template.chassis,
            "model": self.template.model,
            "x": self.x,
            "y": self.y,
            "facing": self.facing,
            "mp_walk": self.walk_mp,
            "mp_run": self.run_mp,
            "mp_jump": self.template.jump_mp,
            "heat": self.heat,
            "prone": self.prone,
            "destroyed": self.destroyed,
            "deployed": self.deployed,
            "retreated": False,
            "armor": armor_list,
            "weapons": weapons_list,
        }

    def has_ammo_for(self, weapon_idx: int) -> bool:
        """Check if there's ammo available for the given weapon."""
        w = self.template.weapons[weapon_idx]
        if not w.is_cluster:
            return True  # Energy weapons don't need ammo
        # Find ammo bins for this weapon type
        for i, ab in enumerate(self.template.ammo):
            if ab.weapon_name == w.name and self.ammo_remaining[i] > 0:
                return True
        return False

    def consume_ammo(self, weapon_idx: int) -> None:
        """Consume one shot of ammo for the given weapon."""
        w = self.template.weapons[weapon_idx]
        if not w.is_cluster:
            return
        for i, ab in enumerate(self.template.ammo):
            if ab.weapon_name == w.name and self.ammo_remaining[i] > 0:
                self.ammo_remaining[i] -= 1
                return
