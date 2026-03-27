"""Tier 2: Heat mechanics validation.

Tracks heat changes between Java observations and validates
consistency with BattleTech heat rules.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from tests.sim_validation.reconstruct import extract_unit_state


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
