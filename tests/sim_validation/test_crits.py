"""Tier 2: Critical hit state validation via Java observation traces.

Validates that crit-related state transitions in Java observations are
internally consistent (monotonicity, side effects, MP correlation).
Since crits are stochastic, we can't compare exact outcomes — only
check that the rules hold.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from tests.sim_validation.reconstruct import (
    extract_crit_state,
    extract_unit_state,
    compute_armor_deltas,
)


@dataclass
class CritValidationResult:
    """Results of crit consistency checks."""
    total_steps: int = 0
    crit_events: int = 0  # Steps where any crit counter increased
    consistency_errors: list[str] = field(default_factory=list)
    side_effect_errors: list[str] = field(default_factory=list)
    mp_errors: list[str] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)

    @property
    def passed(self) -> bool:
        return (not self.consistency_errors
                and not self.side_effect_errors
                and not self.mp_errors)


# All crit fields that should only increase (monotonic)
_MONOTONIC_FIELDS = [
    "engine_hits", "gyro_hits", "sensor_hits",
    "cockpit_hits", "life_support_hits", "heat_sinks_destroyed",
]

_LEG_ACTUATOR_FIELDS = ["hip_hits", "upper_leg_hits", "lower_leg_hits", "foot_hits"]
_ARM_ACTUATOR_FIELDS = ["shoulder_hits", "upper_arm_hits", "lower_arm_hits"]


def _get_nested(crit: dict, *keys: str, default=0):
    """Get a nested value from crit_state dict."""
    val = crit
    for k in keys:
        if not isinstance(val, dict):
            return default
        val = val.get(k, default)
    return val


def validate_crit_step(prev_obs: dict, curr_obs: dict,
                       owner_id: int, step_idx: int) -> list[str]:
    """Check crit state consistency between two consecutive observations.

    Returns a list of error strings (empty = all checks passed).
    """
    errors = []

    prev_crit = extract_crit_state(prev_obs, owner_id)
    curr_crit = extract_crit_state(curr_obs, owner_id)
    if prev_crit is None or curr_crit is None:
        return errors

    prev_unit = extract_unit_state(prev_obs, owner_id)
    curr_unit = extract_unit_state(curr_obs, owner_id)
    if prev_unit is None or curr_unit is None:
        return errors

    # Skip if unit is destroyed
    if curr_unit.get("destroyed", False):
        return errors

    # 1. Monotonicity: top-level crit counters only increase
    for f in _MONOTONIC_FIELDS:
        pv = prev_crit.get(f, 0)
        cv = curr_crit.get(f, 0)
        if cv < pv:
            errors.append(
                f"step {step_idx}: {f} DECREASED {pv}→{cv}"
            )

    # 2. Monotonicity: per-leg actuator fields
    for leg_name in ("right_leg", "left_leg"):
        prev_leg = prev_crit.get(leg_name, {})
        curr_leg = curr_crit.get(leg_name, {})
        for f in _LEG_ACTUATOR_FIELDS:
            pv = prev_leg.get(f, 0)
            cv = curr_leg.get(f, 0)
            if cv < pv:
                errors.append(
                    f"step {step_idx}: {leg_name}.{f} DECREASED {pv}→{cv}"
                )

    # 3. Monotonicity: per-arm actuator fields
    for arm_name in ("right_arm", "left_arm"):
        prev_arm = prev_crit.get(arm_name, {})
        curr_arm = curr_crit.get(arm_name, {})
        for f in _ARM_ACTUATOR_FIELDS:
            pv = prev_arm.get(f, 0)
            cv = curr_arm.get(f, 0)
            if cv < pv:
                errors.append(
                    f"step {step_idx}: {arm_name}.{f} DECREASED {pv}→{cv}"
                )

    # 4. Pilot hits monotonicity
    pp = prev_crit.get("pilot_hits", 0)
    cp = curr_crit.get("pilot_hits", 0)
    if cp < pp:
        errors.append(f"step {step_idx}: pilot_hits DECREASED {pp}→{cp}")

    # 5. Engine death: 3+ engine hits should mean unit destroyed
    if curr_crit.get("engine_hits", 0) >= 3:
        if not curr_unit.get("destroyed", False):
            errors.append(
                f"step {step_idx}: engine_hits={curr_crit['engine_hits']} "
                f"but unit not destroyed"
            )

    # 6. Walk MP consistency with leg damage
    # Java's BipedMek.getWalkMP() formula (non-PLAYTEST2):
    #   start with base walk MP
    #   for each leg: if destroyed → mp=0; if hip hit → ceil(mp/2);
    #                 actuator hits (upper+lower+foot) → mp -= count
    #   heat penalty: mp -= heat // 5
    java_walk = curr_unit.get("mp_walk")
    if java_walk is not None and not curr_unit.get("destroyed", False):
        import math
        base_mp = 5  # TBT-5S walk MP

        # Check for destroyed legs
        curr_armor = curr_unit.get("armor", [])
        rl_destroyed = False
        ll_destroyed = False
        if len(curr_armor) > 5:
            rl_internal = curr_armor[5].get("internal", 1)
            ll_internal = curr_armor[6].get("internal", 1)
            rl_destroyed = rl_internal <= 0
            ll_destroyed = ll_internal <= 0

        if rl_destroyed or ll_destroyed:
            # With a destroyed leg, Java sets mp to 1 (one leg) or 0 (both)
            if rl_destroyed and ll_destroyed:
                expected = 0
            else:
                expected = 1
        else:
            mp = base_mp
            rl = curr_crit.get("right_leg", {})
            ll = curr_crit.get("left_leg", {})

            # Hip hits halve MP (ceil)
            if rl.get("hip_hits", 0) > 0:
                mp = math.ceil(mp / 2)
            if ll.get("hip_hits", 0) > 0:
                mp = math.ceil(mp / 2)

            # Actuator hits reduce MP
            for leg in (rl, ll):
                mp -= leg.get("upper_leg_hits", 0)
                mp -= leg.get("lower_leg_hits", 0)
                mp -= leg.get("foot_hits", 0)

            # Heat penalty
            heat = curr_unit.get("heat", 0)
            mp -= heat // 5

            expected = max(0, mp)

        if java_walk != expected:
            # Only flag as error if significant (allow for edge cases)
            errors.append(
                f"step {step_idx}: mp_walk={java_walk} but expected={expected} "
                f"(engine={curr_crit.get('engine_hits', 0)} "
                f"rl_hip={_get_nested(curr_crit, 'right_leg', 'hip_hits')} "
                f"ll_hip={_get_nested(curr_crit, 'left_leg', 'hip_hits')} "
                f"heat={curr_unit.get('heat', 0)})"
            )

    return errors


def validate_crit_trace(trace_steps: list, rl_owner_id: int) -> CritValidationResult:
    """Validate crit state consistency across all consecutive step pairs."""
    result = CritValidationResult()

    for i in range(1, len(trace_steps)):
        prev_obs = trace_steps[i - 1].raw_obs
        curr_obs = trace_steps[i].raw_obs

        # Skip terminal observations
        if curr_obs.get("terminated") or curr_obs.get("truncated"):
            continue

        result.total_steps += 1

        for owner_id in (rl_owner_id, 1 - rl_owner_id):
            prev_crit = extract_crit_state(prev_obs, owner_id)
            curr_crit = extract_crit_state(curr_obs, owner_id)

            # Detect crit events (any counter increased)
            if prev_crit and curr_crit:
                for f in _MONOTONIC_FIELDS:
                    if curr_crit.get(f, 0) > prev_crit.get(f, 0):
                        result.crit_events += 1
                        break

            errors = validate_crit_step(prev_obs, curr_obs, owner_id,
                                        trace_steps[i].step_idx)
            for err in errors:
                if "DECREASED" in err:
                    result.consistency_errors.append(err)
                elif "mp_walk" in err:
                    result.mp_errors.append(err)
                else:
                    result.side_effect_errors.append(err)

    return result
