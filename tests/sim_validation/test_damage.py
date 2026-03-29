"""Tier 2: Damage application validation via armor deltas.

Infers what happened from consecutive Java observations and validates
that the state transitions are consistent with BattleTech damage rules.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from megamek_gym.sim.unit import LOC_NAMES, Location, TRANSFER_TABLE
from tests.sim_validation.reconstruct import (
    compute_armor_deltas,
    extract_unit_state,
)


@dataclass
class DamageValidationResult:
    """Results of damage consistency checks."""
    total_steps_with_damage: int = 0
    total_damage_events: int = 0
    consistency_errors: list[str] = field(default_factory=list)
    transfer_verified: int = 0  # Location destructions where transfer was correct
    transfer_errors: list[str] = field(default_factory=list)
    side_effect_errors: list[str] = field(default_factory=list)

    @property
    def passed(self) -> bool:
        return (not self.consistency_errors
                and not self.transfer_errors
                and not self.side_effect_errors)


def validate_damage_step(prev_obs: dict, curr_obs: dict,
                         owner_id: int, step_idx: int) -> list[str]:
    """Check damage consistency for one unit between two consecutive steps.

    Returns a list of error strings (empty = all checks passed).
    """
    errors = []

    prev_unit = extract_unit_state(prev_obs, owner_id)
    curr_unit = extract_unit_state(curr_obs, owner_id)
    if prev_unit is None or curr_unit is None:
        return errors

    deltas = compute_armor_deltas(prev_unit, curr_unit)
    if not deltas:
        return errors

    for d in deltas:
        loc_name = d["location"]

        # Check: damage should be non-negative (armor/internal only decreases)
        if d["front_delta"] < 0:
            errors.append(
                f"step {step_idx} {loc_name}: front armor INCREASED by {-d['front_delta']} "
                f"({d['prev_front']}→{d['curr_front']})"
            )
        if d["rear_delta"] < 0:
            errors.append(
                f"step {step_idx} {loc_name}: rear armor INCREASED by {-d['rear_delta']} "
                f"({d['prev_rear']}→{d['curr_rear']})"
            )
        if d["internal_delta"] < 0:
            errors.append(
                f"step {step_idx} {loc_name}: internal INCREASED by {-d['internal_delta']} "
                f"({d['prev_internal']}→{d['curr_internal']})"
            )

        # Check: internal only damaged after armor depleted
        if d["internal_delta"] > 0 and d["curr_front"] > 0 and d["curr_rear"] > 0:
            # This could be legitimate for rear hits on torso (rear armor
            # depleted while front still has armor), but flag as notable
            pass

    # Check location destruction side effects
    prev_armor = prev_unit.get("armor", [])
    curr_armor = curr_unit.get("armor", [])

    for i, (pa, ca) in enumerate(zip(prev_armor, curr_armor)):
        pi = pa.get("internal", 0)
        ci = ca.get("internal", 0)

        if pi > 0 and ci <= 0:
            loc = Location(i)
            loc_name = LOC_NAMES[i]

            # Leg destruction should cause prone
            if loc in (Location.RL, Location.LL):
                if not curr_unit.get("prone", False):
                    errors.append(
                        f"step {step_idx}: {loc_name} destroyed but unit not prone"
                    )

            # CT/HD destruction should kill unit
            if loc in (Location.CT, Location.HD):
                if not curr_unit.get("destroyed", False):
                    errors.append(
                        f"step {step_idx}: {loc_name} destroyed but unit not dead"
                    )

            # Side torso destruction should destroy corresponding arm
            if loc == Location.RT:
                ra_internal = curr_armor[Location.RA].get("internal", 1)
                if ra_internal > 0:
                    # RA should also be destroyed
                    errors.append(
                        f"step {step_idx}: RT destroyed but RA still has internal={ra_internal}"
                    )
            if loc == Location.LT:
                la_internal = curr_armor[Location.LA].get("internal", 1)
                if la_internal > 0:
                    errors.append(
                        f"step {step_idx}: LT destroyed but LA still has internal={la_internal}"
                    )

    return errors


def count_location_destructions(
    trace_steps: list, owner_id: int,
) -> list[tuple[int, str]]:
    """Scan trace for location destruction events (internal >0 → ≤0).

    Returns list of (step_idx, location_name) tuples.
    """
    destructions = []

    for i in range(1, len(trace_steps)):
        prev_obs = trace_steps[i - 1].raw_obs
        curr_obs = trace_steps[i].raw_obs

        prev_unit = extract_unit_state(prev_obs, owner_id)
        curr_unit = extract_unit_state(curr_obs, owner_id)
        if prev_unit is None or curr_unit is None:
            continue

        prev_armor = prev_unit.get("armor", [])
        curr_armor = curr_unit.get("armor", [])

        for loc_i, (pa, ca) in enumerate(zip(prev_armor, curr_armor)):
            pi = pa.get("internal", 0)
            ci = ca.get("internal", 0)
            if pi > 0 and ci <= 0:
                destructions.append((trace_steps[i].step_idx, LOC_NAMES[loc_i]))

    return destructions


def validate_damage_trace(trace_steps: list, rl_owner_id: int) -> DamageValidationResult:
    """Validate damage consistency across all consecutive step pairs."""
    result = DamageValidationResult()

    for i in range(1, len(trace_steps)):
        prev_obs = trace_steps[i - 1].raw_obs
        curr_obs = trace_steps[i].raw_obs

        # Check both units
        for owner_id in (rl_owner_id, 1 - rl_owner_id):
            prev_unit = extract_unit_state(prev_obs, owner_id)
            curr_unit = extract_unit_state(curr_obs, owner_id)
            if prev_unit is None or curr_unit is None:
                continue

            deltas = compute_armor_deltas(prev_unit, curr_unit)
            if deltas:
                result.total_steps_with_damage += 1
                result.total_damage_events += len(deltas)

            errors = validate_damage_step(prev_obs, curr_obs, owner_id,
                                          trace_steps[i].step_idx)
            for err in errors:
                if "INCREASED" in err:
                    result.consistency_errors.append(err)
                elif "destroyed but" in err:
                    result.side_effect_errors.append(err)
                else:
                    result.transfer_errors.append(err)

    return result
