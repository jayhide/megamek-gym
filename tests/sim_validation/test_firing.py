"""Firing behavior cross-validation: Python sim vs Java MegaMek.

Compares per-weapon fireability and to-hit target numbers (TNs) computed by
the Python sim's compute_to_hit() against Java's authoritative
WeaponAttackAction.toHit().  Hit/miss outcomes are non-deterministic and
are NOT compared.

The firing_report field in Java observations captures both entities'
positions, movement state, and per-weapon TNs at the start of each
firing phase (post-movement, pre-firing).
"""

from __future__ import annotations

from dataclasses import dataclass, field

from megamek_gym.sim.board import BOARD
from megamek_gym.sim.firing import compute_to_hit
from megamek_gym.sim.los import LosTable
from tests.sim_validation.reconstruct import (
    extract_unit_state,
    reconstruct_unit,
    set_firing_state,
)


@dataclass
class WeaponComparison:
    """Single weapon comparison result."""
    weapon_name: str
    weapon_index: int
    location: int
    # Java side
    java_can_fire: bool
    java_impossible: bool
    java_tn: int | None  # None if not fireable or impossible
    java_desc: str
    # Python side
    python_can_fire: bool
    python_tn: int | None  # None if not fireable
    # Match results
    fireable_match: bool = False
    tn_match: bool | None = None  # None if not mutually fireable
    tn_delta: int | None = None

    def __post_init__(self):
        # A weapon is "effectively fireable" in Java if can_fire and not impossible
        java_fireable = self.java_can_fire and not self.java_impossible
        self.fireable_match = (java_fireable == self.python_can_fire)
        if java_fireable and self.python_can_fire:
            self.tn_delta = (self.python_tn or 0) - (self.java_tn or 0)
            self.tn_match = (self.tn_delta == 0)
        else:
            self.tn_match = None
            self.tn_delta = None


@dataclass
class FiringRoundComparison:
    """Per-round comparison for one entity."""
    round_num: int
    entity_label: str  # "rl" or "opp"
    weapons: list[WeaponComparison] = field(default_factory=list)
    skipped: bool = False
    skip_reason: str = ""

    @property
    def fireable_match_count(self) -> int:
        return sum(1 for w in self.weapons if w.fireable_match)

    @property
    def fireable_total(self) -> int:
        return len(self.weapons)

    @property
    def tn_match_count(self) -> int:
        return sum(1 for w in self.weapons if w.tn_match is True)

    @property
    def tn_total(self) -> int:
        return sum(1 for w in self.weapons if w.tn_match is not None)


@dataclass
class FiringValidationSummary:
    """Aggregate metrics across a full game."""
    total_rounds: int = 0
    compared_rounds: int = 0
    skipped_rounds: int = 0
    skip_reasons: dict[str, int] = field(default_factory=dict)
    # Opponent aggregate
    opp_fireable_matches: int = 0
    opp_fireable_total: int = 0
    opp_tn_matches: int = 0
    opp_tn_total: int = 0
    # RL aggregate
    rl_fireable_matches: int = 0
    rl_fireable_total: int = 0
    rl_tn_matches: int = 0
    rl_tn_total: int = 0
    # Detailed
    tn_delta_histogram: dict[int, int] = field(default_factory=dict)
    comparisons: list[FiringRoundComparison] = field(default_factory=list)

    @property
    def opp_fireable_match_rate(self) -> float:
        return self.opp_fireable_matches / self.opp_fireable_total if self.opp_fireable_total else 1.0

    @property
    def opp_tn_match_rate(self) -> float:
        return self.opp_tn_matches / self.opp_tn_total if self.opp_tn_total else 1.0

    @property
    def rl_fireable_match_rate(self) -> float:
        return self.rl_fireable_matches / self.rl_fireable_total if self.rl_fireable_total else 1.0

    @property
    def rl_tn_match_rate(self) -> float:
        return self.rl_tn_matches / self.rl_tn_total if self.rl_tn_total else 1.0


def _compare_weapons_for_entity(
    entity_label: str,
    round_num: int,
    unit,
    target,
    java_weapons: list[dict],
    board,
    los_table,
) -> FiringRoundComparison:
    """Compare Python fireability + TNs against Java for one entity."""
    comparisons = []

    for jw in java_weapons:
        weapon_name = jw["weapon_name"]
        weapon_index = jw["weapon_index"]
        location = jw["location"]
        java_destroyed = jw["destroyed"]
        java_can_fire = jw["can_fire"]
        java_impossible = jw.get("impossible", False)
        java_tn = jw.get("to_hit_value") if java_can_fire and not java_impossible else None
        java_desc = jw.get("to_hit_desc", "")

        # Find matching Python weapon by index in template weapons list
        # Java weapon_index is getEquipmentNum, but weapons are listed in
        # the same order as getWeaponList(), which matches template.weapons order
        py_weapon_idx = len(comparisons)  # weapons come in order
        python_can_fire = False
        python_tn = None

        if py_weapon_idx < len(unit.template.weapons):
            w = unit.template.weapons[py_weapon_idx]
            # Check Python fireability
            if (not unit.weapon_destroyed[py_weapon_idx]
                    and not unit.loc_destroyed[w.location]
                    and unit.has_ammo_for(py_weapon_idx)):
                tn = compute_to_hit(unit, target, w, board, los_table)
                if tn is not None:
                    python_can_fire = True
                    python_tn = tn

        comparisons.append(WeaponComparison(
            weapon_name=weapon_name,
            weapon_index=weapon_index,
            location=location,
            java_can_fire=java_can_fire,
            java_impossible=java_impossible,
            java_tn=java_tn,
            java_desc=java_desc,
            python_can_fire=python_can_fire,
            python_tn=python_tn,
        ))

    return FiringRoundComparison(
        round_num=round_num,
        entity_label=entity_label,
        weapons=comparisons,
    )


def validate_firing_single_round(
    raw_obs: dict,
    rl_owner_id: int,
    board=None,
    los_table=None,
    template_name: str = "Trebuchet TBT-5S",
) -> list[FiringRoundComparison]:
    """Compare Python vs Java firing for both entities in one round.

    Returns a list of FiringRoundComparison (one for RL, one for opponent),
    or comparisons with skipped=True if the round should be skipped.
    """
    if board is None:
        board = BOARD
    if los_table is None:
        los_table = LosTable(board)

    firing_report = raw_obs.get("firing_report")
    if not firing_report:
        skip = FiringRoundComparison(
            round_num=raw_obs.get("round", -1),
            entity_label="both",
            skipped=True,
            skip_reason="no firing_report",
        )
        return [skip]

    round_num = raw_obs.get("round", -1)

    # Determine which owner is opponent
    opp_owner_id = None
    for u in raw_obs.get("units", []):
        if u.get("owner") != rl_owner_id:
            opp_owner_id = u["owner"]
            break

    results = []
    for entity_label, entity_key, owner_id, target_owner_id in [
        ("rl", "rl_entity", rl_owner_id, opp_owner_id),
        ("opp", "opp_entity", opp_owner_id, rl_owner_id),
    ]:
        weapons_key = f"{entity_label}_weapons"
        java_weapons = firing_report.get(weapons_key, [])
        entity_state = firing_report.get(entity_key)

        if not java_weapons or not entity_state:
            results.append(FiringRoundComparison(
                round_num=round_num,
                entity_label=entity_label,
                skipped=True,
                skip_reason=f"no {weapons_key} or {entity_key} in firing_report",
            ))
            continue

        # Get unit dicts from observation (for armor/weapon state)
        unit_dict = extract_unit_state(raw_obs, owner_id)
        target_dict = extract_unit_state(raw_obs, target_owner_id)
        if not unit_dict or not target_dict:
            results.append(FiringRoundComparison(
                round_num=round_num,
                entity_label=entity_label,
                skipped=True,
                skip_reason="missing unit dict in observation",
            ))
            continue

        # Check for destroyed/shutdown units
        if unit_dict.get("destroyed") or target_dict.get("destroyed"):
            results.append(FiringRoundComparison(
                round_num=round_num,
                entity_label=entity_label,
                skipped=True,
                skip_reason="destroyed unit",
            ))
            continue

        # Reconstruct units and override with firing-time state
        unit = reconstruct_unit(unit_dict, template_name)
        target = reconstruct_unit(target_dict, template_name)
        set_firing_state(unit, entity_state)

        target_state = firing_report.get(
            "opp_entity" if entity_label == "rl" else "rl_entity"
        )
        if target_state:
            set_firing_state(target, target_state)

        result = _compare_weapons_for_entity(
            entity_label, round_num,
            unit, target, java_weapons, board, los_table,
        )
        results.append(result)

    return results


def validate_firing_trace(
    steps: list,
    rl_owner_id: int,
    template_name: str = "Trebuchet TBT-5S",
) -> FiringValidationSummary:
    """Validate firing across all rounds in a Java game trace.

    Returns aggregate metrics comparing Python vs Java firing decisions.
    """
    board = BOARD
    los_table = LosTable(board)
    summary = FiringValidationSummary()

    for step in steps:
        raw_obs = step.raw_obs
        if raw_obs.get("terminated") or raw_obs.get("truncated"):
            continue

        firing_report = raw_obs.get("firing_report")
        if not firing_report:
            continue

        summary.total_rounds += 1

        round_comparisons = validate_firing_single_round(
            raw_obs, rl_owner_id,
            board=board, los_table=los_table,
            template_name=template_name,
        )

        any_compared = False
        for comp in round_comparisons:
            summary.comparisons.append(comp)
            if comp.skipped:
                reason = comp.skip_reason
                summary.skip_reasons[reason] = summary.skip_reasons.get(reason, 0) + 1
                continue

            any_compared = True

            if comp.entity_label == "opp":
                summary.opp_fireable_matches += comp.fireable_match_count
                summary.opp_fireable_total += comp.fireable_total
                summary.opp_tn_matches += comp.tn_match_count
                summary.opp_tn_total += comp.tn_total
            elif comp.entity_label == "rl":
                summary.rl_fireable_matches += comp.fireable_match_count
                summary.rl_fireable_total += comp.fireable_total
                summary.rl_tn_matches += comp.tn_match_count
                summary.rl_tn_total += comp.tn_total

            # Accumulate TN delta histogram
            for w in comp.weapons:
                if w.tn_delta is not None:
                    summary.tn_delta_histogram[w.tn_delta] = (
                        summary.tn_delta_histogram.get(w.tn_delta, 0) + 1
                    )

        if any_compared:
            summary.compared_rounds += 1
        else:
            summary.skipped_rounds += 1

    _print_summary(summary)
    return summary


def _print_summary(summary: FiringValidationSummary) -> None:
    """Print a human-readable summary of firing validation results."""
    print(f"\n=== FIRING VALIDATION SUMMARY ===")
    print(f"Rounds: {summary.total_rounds} total, "
          f"{summary.compared_rounds} compared, "
          f"{summary.skipped_rounds} skipped")

    if summary.skip_reasons:
        for reason, count in summary.skip_reasons.items():
            print(f"  Skip: {reason} ({count})")

    if summary.opp_fireable_total:
        print(f"\nOPPONENT:")
        print(f"  Fireability: {summary.opp_fireable_matches}/{summary.opp_fireable_total} "
              f"({summary.opp_fireable_match_rate:.1%})")
        print(f"  To-hit TNs:  {summary.opp_tn_matches}/{summary.opp_tn_total} "
              f"({summary.opp_tn_match_rate:.1%})")

    if summary.rl_fireable_total:
        print(f"\nRL UNIT:")
        print(f"  Fireability: {summary.rl_fireable_matches}/{summary.rl_fireable_total} "
              f"({summary.rl_fireable_match_rate:.1%})")
        print(f"  To-hit TNs:  {summary.rl_tn_matches}/{summary.rl_tn_total} "
              f"({summary.rl_tn_match_rate:.1%})")

    if summary.tn_delta_histogram:
        print(f"\n  TN delta histogram (python - java):")
        for delta in sorted(summary.tn_delta_histogram.keys()):
            count = summary.tn_delta_histogram[delta]
            print(f"    {delta:+d}: {count}")

    # Print details for mismatches
    mismatches = []
    for comp in summary.comparisons:
        if comp.skipped:
            continue
        for w in comp.weapons:
            if not w.fireable_match:
                java_f = w.java_can_fire and not w.java_impossible
                mismatches.append(
                    f"  Round {comp.round_num} {comp.entity_label} "
                    f"{w.weapon_name}: java_fireable={java_f} "
                    f"python_fireable={w.python_can_fire}"
                )
            elif w.tn_match is False:
                mismatches.append(
                    f"  Round {comp.round_num} {comp.entity_label} "
                    f"{w.weapon_name}: java_tn={w.java_tn} "
                    f"python_tn={w.python_tn} (delta={w.tn_delta:+d}) "
                    f"java_desc=\"{w.java_desc}\""
                )

    if mismatches:
        print(f"\n  MISMATCHES ({len(mismatches)}):")
        for m in mismatches[:20]:  # cap output
            print(m)
        if len(mismatches) > 20:
            print(f"  ... and {len(mismatches) - 20} more")
