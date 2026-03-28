"""Tier 1: Unit template validation (armor, weapons, MP at game start)."""

from __future__ import annotations

from dataclasses import dataclass, field

from megamek_gym.sim.unit import UNIT_TEMPLATES, LOC_NAMES


@dataclass
class UnitTemplateResult:
    unit_name: str = ""
    mismatches: list[str] = field(default_factory=list)

    @property
    def passed(self) -> bool:
        return not self.mismatches


@dataclass
class InitialFacingResult:
    mismatches: list[str] = field(default_factory=list)

    @property
    def passed(self) -> bool:
        return not self.mismatches


def validate_initial_facing(
    raw_obs: dict,
    rl_owner_id: int,
    rl_start: tuple[int, int] = (14, 1),
    opp_start: tuple[int, int] = (1, 15),
    template_name: str = "Trebuchet TBT-5S",
) -> InitialFacingResult:
    """Compare initial unit facings between sim Game and Java observation.

    Creates a sim Game with the same starting positions, resets it, and
    compares the resulting facings against Java's deployment_state (captured
    before any movement, regardless of initiative order).
    """
    from megamek_gym.sim.game import Game

    result = InitialFacingResult()

    deployment = raw_obs.get("deployment_state")
    if deployment is None:
        result.mismatches.append(
            "deployment_state missing from observation "
            "(Java bridge may need updating)"
        )
        return result

    # Run sim game to get its initial facings
    game = Game(
        rl_unit_name=template_name,
        opponent_unit_name=template_name,
        rl_start=rl_start,
        opp_start=opp_start,
    )
    game.reset()

    # Compare against deployment-time facings (pre-movement)
    for u in deployment:
        owner = u.get("owner")
        java_facing = u.get("facing")
        if java_facing is None:
            continue

        if owner == rl_owner_id:
            sim_facing = game.rl_unit.facing
            label = "rl_unit"
        else:
            sim_facing = game.opp_unit.facing
            label = "opp_unit"

        if sim_facing != java_facing:
            result.mismatches.append(
                f"{label} facing: sim={sim_facing} java={java_facing}"
            )

    return result


def validate_unit_template(java_unit_dict: dict,
                           template_name: str = "Trebuchet TBT-5S") -> UnitTemplateResult:
    """Compare sim unit template against Java unit at game start (round 1)."""
    result = UnitTemplateResult(unit_name=template_name)
    tmpl = UNIT_TEMPLATES.get(template_name)

    if tmpl is None:
        result.mismatches.append(f"Template '{template_name}' not found in sim")
        return result

    # Walk/run/jump MP
    java_walk = java_unit_dict.get("mp_walk", -1)
    java_run = java_unit_dict.get("mp_run", -1)
    java_jump = java_unit_dict.get("mp_jump", -1)

    if java_walk != tmpl.walk_mp:
        result.mismatches.append(f"walk_mp: sim={tmpl.walk_mp} java={java_walk}")
    if java_run != tmpl.run_mp:
        result.mismatches.append(f"run_mp: sim={tmpl.run_mp} java={java_run}")
    if java_jump != tmpl.jump_mp:
        result.mismatches.append(f"jump_mp: sim={tmpl.jump_mp} java={java_jump}")

    # Per-location armor
    java_armor = java_unit_dict.get("armor", [])
    for ja in java_armor:
        loc_name = ja.get("location", "")
        if loc_name not in LOC_NAMES:
            result.mismatches.append(f"Unknown location in Java obs: {loc_name}")
            continue

        from megamek_gym.sim.unit import Location
        loc_idx = LOC_NAMES.index(loc_name)
        loc = Location(loc_idx)
        sim_front, sim_rear, sim_internal = tmpl.armor[loc]

        java_front = ja.get("armor_max", -1)
        java_rear = ja.get("rear_armor_max", 0)
        java_internal = ja.get("internal_max", -1)

        if java_front != sim_front:
            result.mismatches.append(
                f"{loc_name} front_armor_max: sim={sim_front} java={java_front}"
            )
        if java_rear != sim_rear:
            result.mismatches.append(
                f"{loc_name} rear_armor_max: sim={sim_rear} java={java_rear}"
            )
        if java_internal != sim_internal:
            result.mismatches.append(
                f"{loc_name} internal_max: sim={sim_internal} java={java_internal}"
            )

    # Weapons
    java_weapons = java_unit_dict.get("weapons", [])
    sim_weapons = tmpl.weapons

    # Java may list weapons in a different order, so match by name+location
    if len(java_weapons) != len(sim_weapons):
        result.mismatches.append(
            f"weapon count: sim={len(sim_weapons)} java={len(java_weapons)}"
        )

    # Validate ranges on matched weapons (ignoring order differences)
    java_by_key: dict[tuple, dict] = {}
    for jw in java_weapons:
        key = (jw.get("name", ""), jw.get("location", -1))
        java_by_key.setdefault(key, []).append(jw)

    sim_by_key: dict[tuple, list] = {}
    for sw in sim_weapons:
        key = (sw.name, int(sw.location))
        sim_by_key.setdefault(key, []).append(sw)

    for key, sim_list in sim_by_key.items():
        java_list = java_by_key.get(key, [])
        if len(java_list) != len(sim_list):
            result.mismatches.append(
                f"weapon '{key[0]}' at loc {key[1]}: "
                f"sim count={len(sim_list)} java count={len(java_list)}"
            )
            continue

        for sw, jw in zip(sim_list, java_list):
            # Java serializes effective damage for all weapons:
            # direct-fire: raw damage, cluster: rackSize * per-missile damage
            java_dmg = jw.get("damage", -1)
            if sw.is_cluster:
                expected = sw.damage * sw.cluster_size
            else:
                expected = sw.damage
            if java_dmg >= 0 and java_dmg != expected:
                result.mismatches.append(
                    f"weapon '{sw.name}' damage: sim={expected} java={java_dmg}"
                )

            # Ranges: Java sends Integer.MIN_VALUE for "no min range"
            for rng_name in ("short_range", "medium_range", "long_range"):
                java_val = jw.get(rng_name, -1)
                sim_val = getattr(sw, rng_name)
                if java_val != sim_val:
                    result.mismatches.append(
                        f"weapon '{sw.name}' {rng_name}: sim={sim_val} java={java_val}"
                    )

            # min_range: Java sends Integer.MIN_VALUE when there's no min range
            java_min = jw.get("min_range", 0)
            if java_min == -2147483648:  # Integer.MIN_VALUE = no min range
                java_min = 0
            if java_min != sw.min_range:
                result.mismatches.append(
                    f"weapon '{sw.name}' min_range: sim={sw.min_range} java={java_min}"
                )

    return result
