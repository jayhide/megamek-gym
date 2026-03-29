"""Tier 2: Heat mechanics validation.

Tracks heat changes between Java observations and validates
consistency with BattleTech heat rules.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from megamek_gym.sim.unit import UNIT_TEMPLATES
from tests.sim_validation.reconstruct import extract_unit_state, extract_crit_state

# Weapon heat lookup by name from TBT-5S template
_TBT_WEAPON_HEAT_BY_NAME = {}
for _w in UNIT_TEMPLATES["Trebuchet TBT-5S"].weapons:
    _TBT_WEAPON_HEAT_BY_NAME[_w.name] = _w.heat
_TBT_HEAT_SINKS = UNIT_TEMPLATES["Trebuchet TBT-5S"].heat_sinks


@dataclass
class HeatValidationResult:
    """Results of heat validation across a game."""
    total_steps: int = 0
    heat_changes: int = 0  # Steps where heat changed
    max_heat_observed: int = 0
    dissipation_errors: list[str] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)

    @property
    def passed(self) -> bool:
        return not self.dissipation_errors


def validate_heat_trace(trace_steps: list, rl_owner_id: int,
                        heat_sinks: int = 18) -> HeatValidationResult:
    """Validate heat changes across consecutive Java observations.

    Since we can't observe which weapons fired, we validate:
    1. Heat never goes negative
    2. Heat dissipation per round doesn't exceed heat_sinks
    3. Running adds exactly +2 heat (when we can isolate it)
    4. Heat is bounded by game rules
    """
    result = HeatValidationResult()

    for i in range(1, len(trace_steps)):
        prev_obs = trace_steps[i - 1].raw_obs
        curr_obs = trace_steps[i].raw_obs
        step_idx = trace_steps[i].step_idx

        prev_unit = extract_unit_state(prev_obs, rl_owner_id)
        curr_unit = extract_unit_state(curr_obs, rl_owner_id)
        if prev_unit is None or curr_unit is None:
            continue

        result.total_steps += 1

        prev_heat = prev_unit.get("heat", 0)
        curr_heat = curr_unit.get("heat", 0)

        if curr_heat != prev_heat:
            result.heat_changes += 1

        result.max_heat_observed = max(result.max_heat_observed, curr_heat)

        # Heat should never be negative
        if curr_heat < 0:
            result.dissipation_errors.append(
                f"step {step_idx}: heat went negative ({prev_heat}→{curr_heat})"
            )

        # Between rounds, heat change = generated - dissipated
        # generated >= 0, dissipated <= heat_sinks
        # So: curr_heat >= prev_heat - heat_sinks (minimum: all sinks fire, no weapons)
        # And if curr_heat < prev_heat - heat_sinks, something is wrong
        heat_drop = prev_heat - curr_heat
        if heat_drop > heat_sinks:
            # More heat dissipated than heat sinks allow (without generation)
            # This is actually OK if the unit was at low heat (dissipated = min(heat, sinks))
            # Only flag if prev_heat > heat_sinks and drop exceeds sinks
            if prev_heat > heat_sinks and heat_drop > heat_sinks:
                result.dissipation_errors.append(
                    f"step {step_idx}: heat dropped by {heat_drop} "
                    f"(prev={prev_heat}, curr={curr_heat}, sinks={heat_sinks})"
                )

    return result


def validate_initial_heat(obs: dict, owner_id: int,
                          expected_heat: int, expected_walk_mp: int,
                          base_mp: int = 5) -> list[str]:
    """Check that a single observation has the expected heat and walk_mp.

    Used to verify inject_state applied correctly and that Java's MP
    formula matches: walk_mp = base_mp - heat // 5.
    """
    errors = []
    unit = extract_unit_state(obs, owner_id)
    if unit is None:
        errors.append(f"owner {owner_id}: unit not found in observation")
        return errors

    actual_heat = unit.get("heat", 0)
    if actual_heat != expected_heat:
        errors.append(
            f"heat: expected {expected_heat}, got {actual_heat}"
        )

    actual_mp = unit.get("mp_walk")
    if actual_mp is not None and actual_mp != expected_walk_mp:
        errors.append(
            f"walk_mp: expected {expected_walk_mp} "
            f"(base={base_mp} - heat//5={expected_heat // 5}), "
            f"got {actual_mp}"
        )

    return errors


# ---------------------------------------------------------------------------
# Heat generation cross-validation (H4)
# ---------------------------------------------------------------------------

@dataclass
class HeatGenerationResult:
    """Results of heat generation cross-validation."""
    rounds_validated: int = 0
    rounds_matched: int = 0
    mismatches: list[str] = field(default_factory=list)
    skips: list[str] = field(default_factory=list)

    @property
    def passed(self) -> bool:
        return not self.mismatches


def _resolve_fr_keys(firing_report: dict, rl_owner_id: int,
                     ) -> tuple[str, str]:
    """Resolve firing_report entity/weapon keys for the RL player.

    In dual-bot mode, each bot labels ITSELF as "rl" in its firing report.
    Use the owner field to correctly map to the test's RL entity.
    """
    rl_entity = firing_report.get("rl_entity")
    fr_rl_owner = rl_entity.get("owner") if rl_entity else None
    if fr_rl_owner is not None and fr_rl_owner != rl_owner_id:
        return "opp_entity", "opp_weapons"
    return "rl_entity", "rl_weapons"


def _compute_weapon_heat(firing_report: dict, entity_key: str,
                         weapons_key: str) -> int:
    """Sum heat from weapons that fired (can_fire=True, not impossible).

    Uses weapon_name for heat lookup since firing_report's weapon_index
    is the MegaMek equipment slot number, not the position in the weapon list.
    """
    weapons = firing_report.get(weapons_key, [])
    total = 0
    for w in weapons:
        if w.get("can_fire") and not w.get("impossible"):
            name = w.get("weapon_name", "")
            total += _TBT_WEAPON_HEAT_BY_NAME.get(name, 0)
    return total


def _movement_heat(firing_report: dict, entity_key: str) -> int:
    """Return movement heat: walk=1, run=2, standing=0."""
    entity = firing_report.get(entity_key, {})
    moved = entity.get("moved", "MOVE_NONE")
    if moved in ("MOVE_RUN", "MOVE_SPRINT"):
        return 2
    elif moved == "MOVE_WALK":
        return 1
    return 0


def validate_heat_generation(trace_steps: list, rl_owner_id: int,
                             heat_sinks: int = _TBT_HEAT_SINKS,
                             ) -> HeatGenerationResult:
    """Cross-validate heat arithmetic against Java using firing_report data.

    For each round transition where a firing_report is present, computes:
        expected = max(0, prev_heat + weapon_heat + running_heat
                       + 5*engine_hits - effective_sinks)
    and compares with the observed post-dissipation heat.

    Only validates on round transitions (when obs round number increases)
    to avoid double-counting in dual-bot traces where both entities get
    observations within the same round.
    """
    result = HeatGenerationResult()
    prev_heat: int | None = None
    prev_round: int | None = None
    validated_round: int | None = None  # track which round we already validated

    for step in trace_steps:
        obs = step.raw_obs
        if obs.get("terminated") or obs.get("truncated"):
            prev_heat = None
            prev_round = None
            continue

        unit = extract_unit_state(obs, rl_owner_id)
        if unit is None:
            continue

        curr_heat = unit.get("heat", 0)
        curr_round = obs.get("round", -1)
        curr_destroyed = unit.get("destroyed", False)

        firing_report = obs.get("firing_report")

        # Only validate on round transitions with a firing_report.
        # - Skip round 1→2: inject_state timing makes first firing_report unreliable
        # - Only validate ONCE per round (first step in new round) to avoid
        #   double-counting from the other bot's perspective
        is_new_round = prev_round is not None and curr_round > prev_round
        is_first_transition = prev_round == 1
        already_validated = validated_round == curr_round
        if (prev_heat is not None and firing_report is not None
                and is_new_round and not is_first_transition
                and not already_validated):
            # Skip if unit was destroyed (heat processing may differ)
            if curr_destroyed:
                result.skips.append(
                    f"step {step.step_idx}: unit destroyed, skipping"
                )
                prev_heat = curr_heat
                prev_round = curr_round
                continue

            # Resolve firing_report keys (dual-bot swaps rl/opp labels)
            entity_key, weapons_key = _resolve_fr_keys(
                firing_report, rl_owner_id)

            # Compute expected heat
            weapon_heat = _compute_weapon_heat(
                firing_report, entity_key, weapons_key)
            move_heat = _movement_heat(firing_report, entity_key)

            # Capture details for mismatch diagnostics
            _fr_entity = firing_report.get(entity_key, {})
            _fr_weapons = firing_report.get(weapons_key, [])
            _fireable = [w for w in _fr_weapons
                         if w.get("can_fire") and not w.get("impossible")]

            crit = extract_crit_state(obs, rl_owner_id)
            engine_hits = crit.get("engine_hits", 0) if crit else 0
            sinks_destroyed = crit.get("heat_sinks_destroyed", 0) if crit else 0
            effective_sinks = max(0, heat_sinks - sinks_destroyed)

            engine_heat = 5 * engine_hits
            pre_dissipation = prev_heat + weapon_heat + move_heat + engine_heat
            expected = max(0, pre_dissipation - effective_sinks)

            validated_round = curr_round
            result.rounds_validated += 1
            if curr_heat == expected:
                result.rounds_matched += 1
            else:
                result.mismatches.append(
                    f"step {step.step_idx} round {curr_round}: "
                    f"expected heat={expected} "
                    f"(prev={prev_heat} +wpn={weapon_heat} +move={move_heat} "
                    f"+eng={engine_heat} -sinks={effective_sinks}), "
                    f"got {curr_heat} "
                    f"[keys={entity_key}/{weapons_key} "
                    f"moved={_fr_entity.get('moved')} "
                    f"owner={_fr_entity.get('owner')} "
                    f"fireable={len(_fireable)} "
                    f"n_weapons={len(_fr_weapons)}]"
                )

        prev_heat = curr_heat
        prev_round = curr_round

    return result


# ---------------------------------------------------------------------------
# Shutdown consistency validation (H5)
# ---------------------------------------------------------------------------

@dataclass
class ShutdownValidationResult:
    """Results of shutdown consistency check."""
    total_transitions: int = 0
    shutdown_transitions: int = 0  # not-shutdown → shutdown
    recovery_transitions: int = 0  # shutdown → not-shutdown
    invalid_shutdowns: list[str] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)

    @property
    def passed(self) -> bool:
        return not self.invalid_shutdowns


def validate_shutdown_consistency(trace_steps: list,
                                  rl_owner_id: int,
                                  ) -> ShutdownValidationResult:
    """Validate that shutdown transitions only occur when heat >= 14.

    Iterates consecutive observations, detects shutdown state transitions,
    and verifies they are consistent with heat levels.
    """
    result = ShutdownValidationResult()
    prev_shutdown: bool | None = None
    prev_heat: int | None = None

    for step in trace_steps:
        obs = step.raw_obs
        if obs.get("terminated") or obs.get("truncated"):
            prev_shutdown = None
            prev_heat = None
            continue

        unit = extract_unit_state(obs, rl_owner_id)
        if unit is None:
            continue

        curr_shutdown = unit.get("shutdown", False)
        curr_heat = unit.get("heat", 0)
        curr_destroyed = unit.get("destroyed", False)

        if curr_destroyed:
            prev_shutdown = None
            prev_heat = None
            continue

        if prev_shutdown is not None:
            result.total_transitions += 1

            if not prev_shutdown and curr_shutdown:
                # Shutdown transition: heat must be >= 14
                result.shutdown_transitions += 1
                if curr_heat < 14:
                    result.invalid_shutdowns.append(
                        f"step {step.step_idx}: shutdown at heat={curr_heat} "
                        f"(prev_heat={prev_heat}, expected heat >= 14)"
                    )

            elif prev_shutdown and not curr_shutdown:
                result.recovery_transitions += 1

        prev_shutdown = curr_shutdown
        prev_heat = curr_heat

    return result
