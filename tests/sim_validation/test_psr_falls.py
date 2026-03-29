"""Tier 2: PSR/fall mechanics cross-validation via Java observation traces.

Validates that prone transitions, fall triggers, stand-up conditions, and
movement point reductions from crit damage are consistent with BattleTech
rules. Uses deterministic state injection (gyro_hits, hip actuator damage)
for guaranteed outcomes rather than relying on stochastic dice rolls.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from tests.sim_validation.reconstruct import (
    compute_armor_deltas,
    extract_crit_state,
    extract_unit_state,
)


@dataclass
class PSRFallValidationResult:
    """Results of PSR/fall consistency checks."""
    fall_events: int = 0           # prone transitions (False -> True)
    stand_events: int = 0          # prone transitions (True -> False)
    unexplained_falls: list[str] = field(default_factory=list)
    gyro_destroyed_not_prone: list[str] = field(default_factory=list)
    stand_mp_errors: list[str] = field(default_factory=list)
    damage_formula_errors: list[str] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)

    @property
    def passed(self) -> bool:
        return (not self.unexplained_falls
                and not self.gyro_destroyed_not_prone
                and not self.stand_mp_errors)


def _compute_total_damage(armor_deltas: list[dict]) -> int:
    """Sum all positive damage deltas across all locations."""
    total = 0
    for d in armor_deltas:
        total += max(0, d.get("front_delta", 0))
        total += max(0, d.get("rear_delta", 0))
        total += max(0, d.get("internal_delta", 0))
    return total


def _leg_destroyed_this_step(armor_deltas: list[dict]) -> bool:
    """Check if any leg location was destroyed in this step."""
    for d in armor_deltas:
        loc = d.get("location", "")
        if loc in ("RL", "LL") and d.get("location_destroyed", False):
            return True
    return False


def _classify_fall_trigger(prev_unit: dict, curr_unit: dict,
                           prev_crit: dict | None, curr_crit: dict | None,
                           armor_deltas: list[dict]) -> list[str]:
    """Classify what could have triggered a fall between two observations.

    Returns list of possible trigger reasons (OR logic — multiple can co-occur
    in a single observation step). Empty list means unexplained fall.
    """
    triggers = []

    # 1. Leg destroyed (internal > 0 -> <= 0)
    if _leg_destroyed_this_step(armor_deltas):
        triggers.append("leg_destroyed")

    if prev_crit is None or curr_crit is None:
        return triggers

    # 2. Gyro just destroyed (gyro_hits went from <2 to >=2) — automatic fall
    prev_gyro = prev_crit.get("gyro_hits", 0)
    curr_gyro = curr_crit.get("gyro_hits", 0)
    if prev_gyro < 2 and curr_gyro >= 2:
        triggers.append("gyro_destroyed")

    # 3. New gyro hit (PSR with +3 modifier, may fail)
    if curr_gyro > prev_gyro and curr_gyro < 2:
        triggers.append("new_gyro_hit")

    # 4. 20+ damage in this step (damage PSR)
    total_dmg = _compute_total_damage(armor_deltas)
    if total_dmg >= 20:
        triggers.append("damage_20plus")

    # 5. New hip hit (PSR with +2 modifier)
    for leg_name in ("right_leg", "left_leg"):
        prev_leg = prev_crit.get(leg_name, {})
        curr_leg = curr_crit.get(leg_name, {})
        if curr_leg.get("hip_hits", 0) > prev_leg.get("hip_hits", 0):
            triggers.append(f"hip_hit_{leg_name}")

    # 6. New leg actuator hit (PSR with +1 modifier)
    for leg_name in ("right_leg", "left_leg"):
        prev_leg = prev_crit.get(leg_name, {})
        curr_leg = curr_crit.get(leg_name, {})
        for f in ("upper_leg_hits", "lower_leg_hits", "foot_hits"):
            if curr_leg.get(f, 0) > prev_leg.get(f, 0):
                triggers.append(f"leg_actuator_{leg_name}_{f}")

    return triggers


def validate_gyro_destroyed_implies_prone(
    trace_steps: list, owner_id: int,
) -> list[str]:
    """When gyro_hits transitions from <2 to >=2 in combat, unit must fall.

    This checks that when the gyro becomes destroyed (gyro_hits goes from
    <2 to >=2 between consecutive observations), the unit becomes prone.
    Injected gyro damage (present from step 0) does NOT trigger an immediate
    fall — only combat-induced gyro destruction does.

    Allows one-step lag (fall may appear in next observation).
    """
    errors = []
    pending_check = False
    pending_step = -1

    for i in range(1, len(trace_steps)):
        prev_obs = trace_steps[i - 1].raw_obs
        curr_obs = trace_steps[i].raw_obs

        if curr_obs.get("terminated") or curr_obs.get("truncated"):
            if pending_check:
                errors.append(
                    f"step {pending_step}: gyro just destroyed but unit "
                    f"not prone (end of trace/terminal)"
                )
                pending_check = False
            continue

        curr_unit = extract_unit_state(curr_obs, owner_id)
        if curr_unit is None:
            continue

        is_prone = curr_unit.get("prone", False)
        is_destroyed = curr_unit.get("destroyed", False)

        # Resolve pending check from previous step
        if pending_check:
            if not is_prone and not is_destroyed:
                errors.append(
                    f"step {pending_step}: gyro just destroyed but unit not "
                    f"prone (checked at step {trace_steps[i].step_idx}, "
                    f"still standing)"
                )
            pending_check = False

        # Detect gyro destruction transition (< 2 -> >= 2)
        prev_crit = extract_crit_state(prev_obs, owner_id)
        curr_crit = extract_crit_state(curr_obs, owner_id)
        if prev_crit is None or curr_crit is None:
            continue

        prev_gyro = prev_crit.get("gyro_hits", 0)
        curr_gyro = curr_crit.get("gyro_hits", 0)

        if prev_gyro < 2 and curr_gyro >= 2:
            # Gyro just became destroyed — unit should fall
            if not is_prone and not is_destroyed:
                pending_check = True
                pending_step = trace_steps[i].step_idx

    if pending_check:
        errors.append(
            f"step {pending_step}: gyro just destroyed but unit not prone "
            f"(end of trace)"
        )

    return errors


def validate_gyro_destroyed_auto_fall(
    trace_steps: list, owner_id: int,
) -> dict:
    """For gyro_destroyed_trace: verify that any PSR trigger causes a fall.

    With gyro_hits=2, all PSR checks auto-fail. So if a unit is standing
    and takes damage that would trigger a PSR, it must fall.

    Returns dict with counts for diagnostics.
    """
    result = {
        "standing_with_damage": 0,
        "fell_after_damage": 0,
        "errors": [],
    }

    for i in range(1, len(trace_steps)):
        prev_obs = trace_steps[i - 1].raw_obs
        curr_obs = trace_steps[i].raw_obs

        if curr_obs.get("terminated") or curr_obs.get("truncated"):
            continue

        prev_unit = extract_unit_state(prev_obs, owner_id)
        curr_unit = extract_unit_state(curr_obs, owner_id)
        prev_crit = extract_crit_state(prev_obs, owner_id)
        curr_crit = extract_crit_state(curr_obs, owner_id)

        if prev_unit is None or curr_unit is None:
            continue
        if prev_unit.get("destroyed") or curr_unit.get("destroyed"):
            continue

        prev_prone = prev_unit.get("prone", False)
        curr_prone = curr_unit.get("prone", False)
        gyro_hits = (curr_crit or {}).get("gyro_hits", 0)

        if gyro_hits < 2:
            continue  # Only relevant when gyro is destroyed

        if prev_prone:
            continue  # Already prone, PSR skipped

        # Unit was standing with destroyed gyro — check for actual PSR triggers
        # (events that would cause a PSR check, not just the pre-existing gyro state)
        armor_deltas = compute_armor_deltas(prev_unit, curr_unit)
        total_dmg = _compute_total_damage(armor_deltas)

        triggers = _classify_fall_trigger(
            prev_unit, curr_unit, prev_crit, curr_crit, armor_deltas,
        )
        # Only count triggers that cause a NEW PSR check and are reliably
        # detectable from observation deltas. Exclude:
        # - gyro_destroyed: pre-existing injected state
        # - leg_destroyed: auto-fall (not a PSR check)
        # - damage_20plus: observation deltas span multiple sub-phases,
        #   so total delta may exceed 20 even if no single firing phase did
        psr_triggers = [t for t in triggers
                        if t not in ("leg_destroyed", "gyro_destroyed",
                                     "damage_20plus")]

        if psr_triggers:
            result["standing_with_damage"] += 1
            if curr_prone:
                result["fell_after_damage"] += 1
            else:
                result["errors"].append(
                    f"step {trace_steps[i].step_idx}: gyro_hits=2, standing, "
                    f"PSR triggers={psr_triggers}, dmg={total_dmg}, "
                    f"but unit did NOT fall"
                )

    return result


def validate_hip_mp_reduction(
    trace_steps: list, owner_id: int,
    expected_walk_mp: int = 3,
) -> list[str]:
    """Verify that hip damage reduces walk MP as expected.

    For TBT-5S with one hip hit: ceil(5/2) = 3.
    Checks the first non-terminal observation after injection.
    """
    errors = []

    for step in trace_steps:
        obs = step.raw_obs
        if obs.get("terminated") or obs.get("truncated"):
            continue

        unit = extract_unit_state(obs, owner_id)
        if unit is None:
            continue

        mp_walk = unit.get("mp_walk")
        if mp_walk is None:
            continue

        # Check hip hit is present in crit state
        crit = extract_crit_state(obs, owner_id)
        if crit is None:
            continue

        rl = crit.get("right_leg", {})
        ll = crit.get("left_leg", {})
        has_hip = (rl.get("hip_hits", 0) > 0) or (ll.get("hip_hits", 0) > 0)

        if not has_hip:
            errors.append(
                f"step {step.step_idx}: hip injection not reflected "
                f"in crit_state (rl_hip={rl.get('hip_hits', 0)}, "
                f"ll_hip={ll.get('hip_hits', 0)})"
            )
            break

        # Verify MP reduction (heat may further reduce MP)
        heat = unit.get("heat", 0)
        heat_penalty = heat // 5
        effective_expected = max(0, expected_walk_mp - heat_penalty)

        if mp_walk != effective_expected:
            errors.append(
                f"step {step.step_idx}: mp_walk={mp_walk} but expected "
                f"{effective_expected} (base=5, hip halved to {expected_walk_mp}, "
                f"heat={heat}, heat_penalty={heat_penalty})"
            )

        break  # Only check first observation

    return errors


def validate_psr_fall_trace(
    trace_steps: list, rl_owner_id: int,
) -> PSRFallValidationResult:
    """Validate PSR/fall consistency across all consecutive step pairs.

    Checks both units for:
    - Every fall has a valid trigger
    - Gyro destroyed implies prone
    - Stand-up requires sufficient MP
    """
    result = PSRFallValidationResult()

    for owner_id in (rl_owner_id, 1 - rl_owner_id):
        # Gyro destroyed invariant
        gyro_errors = validate_gyro_destroyed_implies_prone(
            trace_steps, owner_id,
        )
        result.gyro_destroyed_not_prone.extend(gyro_errors)

        # Step-by-step checks
        for i in range(1, len(trace_steps)):
            prev_obs = trace_steps[i - 1].raw_obs
            curr_obs = trace_steps[i].raw_obs

            if curr_obs.get("terminated") or curr_obs.get("truncated"):
                continue

            prev_unit = extract_unit_state(prev_obs, owner_id)
            curr_unit = extract_unit_state(curr_obs, owner_id)
            if prev_unit is None or curr_unit is None:
                continue
            if curr_unit.get("destroyed", False):
                continue

            prev_prone = prev_unit.get("prone", False)
            curr_prone = curr_unit.get("prone", False)

            # Fall detection (not-prone -> prone)
            if not prev_prone and curr_prone:
                result.fall_events += 1

                prev_crit = extract_crit_state(prev_obs, owner_id)
                curr_crit = extract_crit_state(curr_obs, owner_id)
                armor_deltas = compute_armor_deltas(prev_unit, curr_unit)

                triggers = _classify_fall_trigger(
                    prev_unit, curr_unit, prev_crit, curr_crit, armor_deltas,
                )

                if not triggers:
                    result.unexplained_falls.append(
                        f"step {trace_steps[i].step_idx} owner={owner_id}: "
                        f"fell prone with no identifiable trigger "
                        f"(dmg={_compute_total_damage(armor_deltas)}, "
                        f"gyro={curr_crit.get('gyro_hits', '?') if curr_crit else '?'})"
                    )
                else:
                    result.notes.append(
                        f"step {trace_steps[i].step_idx} owner={owner_id}: "
                        f"fall triggered by {triggers}"
                    )

            # Stand-up detection (prone -> not-prone)
            if prev_prone and not curr_prone:
                result.stand_events += 1

                # Check MP was sufficient to stand up
                prev_mp_walk = prev_unit.get("mp_walk", 0)
                if prev_mp_walk < 2:
                    # Edge case: walk_mp=1 allows stand with cost 1
                    if prev_mp_walk < 1:
                        result.stand_mp_errors.append(
                            f"step {trace_steps[i].step_idx} owner={owner_id}: "
                            f"stood up with mp_walk={prev_mp_walk} (need >= 2)"
                        )

    return result
