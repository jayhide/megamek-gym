"""Pytest wrappers for sim-vs-Java cross-validation tests.

These tests collect a Java game trace (shared via session fixture) and compare
the Python sim's outputs against Java MegaMek's outputs.

Deselected by default; run with:

    poetry run pytest -m validation --megamek-dir ../megamek
"""

from __future__ import annotations

import pytest


@pytest.mark.validation
class TestBoard:
    def test_board(self, java_trace):
        from tests.sim_validation.test_board import validate_board

        first_obs = java_trace.steps[0].raw_obs
        result = validate_board(first_obs)

        assert result.dimension_mismatch is None, (
            f"Board dimension mismatch: {result.dimension_mismatch}"
        )
        assert not result.elevation_mismatches, (
            f"{len(result.elevation_mismatches)} elevation mismatches. "
            f"First: {result.elevation_mismatches[0]}"
        )
        assert not result.terrain_mismatches, (
            f"{len(result.terrain_mismatches)} terrain mismatches. "
            f"First: {result.terrain_mismatches[0]}"
        )


@pytest.mark.validation
class TestUnitTemplate:
    def test_unit_template(self, java_trace):
        from tests.sim_validation.test_unit_template import validate_unit_template
        from tests.sim_validation.reconstruct import extract_unit_state

        first_obs = java_trace.steps[0].raw_obs
        rl_unit = extract_unit_state(first_obs, java_trace.rl_owner_id)
        assert rl_unit is not None, "No RL unit in first observation"

        result = validate_unit_template(rl_unit)
        assert not result.mismatches, (
            f"{len(result.mismatches)} mismatches. First: {result.mismatches[0]}"
        )


@pytest.mark.validation
class TestInitialFacing:
    def test_initial_facing(self, java_trace):
        from tests.sim_validation.test_unit_template import validate_initial_facing

        first_obs = java_trace.steps[0].raw_obs
        result = validate_initial_facing(first_obs, java_trace.rl_owner_id)
        assert not result.mismatches, (
            f"Initial facing mismatch: {', '.join(result.mismatches)}"
        )


@pytest.mark.validation
class TestLosWalkPatrol:
    """Both mechs walk scripted waypoints; validate LOS at every step."""

    def test_los_walk_patrol(self, walk_patrol_trace):
        from megamek_gym.sim.board import BOARD
        from megamek_gym.sim.los import LosTable
        from tests.sim_validation.test_los import validate_los

        trace = walk_patrol_trace
        los_table = LosTable(BOARD)
        total_checks = 0
        all_mismatches = []

        for step in trace.steps:
            obs = step.raw_obs
            if obs.get("terminated") or obs.get("truncated"):
                continue
            if not obs.get("legal_moves"):
                continue

            active_id = obs.get("active_entity_id", -1)
            owner = trace.entity_owners.get(active_id, -1)
            if owner < 0:
                continue

            result = validate_los(obs, los_table, owner)
            total_checks += result.total_checks
            all_mismatches.extend(result.mismatches)

        assert total_checks > 0, "No LOS checks performed"
        assert not all_mismatches, (
            f"{len(all_mismatches)} LOS mismatches out of {total_checks}. "
            f"First: {all_mismatches[0]}"
        )


@pytest.mark.validation
class TestLegalMovesWalkPatrol:
    """Both mechs walk scripted waypoints; validate legal moves at every step."""

    def test_walk_patrol(self, walk_patrol_trace):
        from tests.sim_validation.test_legal_moves import validate_legal_moves

        trace = walk_patrol_trace
        failures = []
        validated = 0

        for step in trace.steps:
            obs = step.raw_obs
            if obs.get("terminated") or obs.get("truncated"):
                continue
            if not obs.get("legal_moves"):
                continue

            active_id = obs.get("active_entity_id", -1)
            owner = trace.entity_owners.get(active_id, -1)
            if owner < 0:
                continue

            result = validate_legal_moves(obs, owner, step.step_idx,
                                            algorithm="deque")
            if result.skipped_partial_move:
                continue
            validated += 1
            if not result.passed:
                failures.append(
                    f"step {step.step_idx} entity {active_id} at "
                    f"({result.unit_pos[0]},{result.unit_pos[1]},f={result.unit_pos[2]}): "
                    f"{len(result.java_only_hexes)} java-only hexes, "
                    f"{len(result.sim_only_hexes)} sim-only hexes, "
                    f"{len(result.java_only_moves)} java-only moves, "
                    f"{len(result.sim_only_moves)} sim-only moves"
                )

        assert validated > 0, "No movement steps validated"
        assert not failures, (
            f"{len(failures)}/{validated} steps failed.\n"
            + "\n".join(failures[:10])
        )


@pytest.mark.validation
class TestLegalMovesRunPatrol:
    """Both mechs run scripted waypoints; validate legal moves at every step."""

    def test_run_patrol(self, run_patrol_trace):
        from tests.sim_validation.test_legal_moves import validate_legal_moves

        trace = run_patrol_trace
        failures = []
        validated = 0

        for step in trace.steps:
            obs = step.raw_obs
            if obs.get("terminated") or obs.get("truncated"):
                continue
            if not obs.get("legal_moves"):
                continue

            active_id = obs.get("active_entity_id", -1)
            owner = trace.entity_owners.get(active_id, -1)
            if owner < 0:
                continue

            result = validate_legal_moves(obs, owner, step.step_idx,
                                            algorithm="deque")
            if result.skipped_partial_move:
                continue
            validated += 1
            if not result.passed:
                failures.append(
                    f"step {step.step_idx} entity {active_id} at "
                    f"({result.unit_pos[0]},{result.unit_pos[1]},f={result.unit_pos[2]}): "
                    f"{len(result.java_only_hexes)} java-only hexes, "
                    f"{len(result.sim_only_hexes)} sim-only hexes, "
                    f"{len(result.java_only_moves)} java-only moves, "
                    f"{len(result.sim_only_moves)} sim-only moves"
                )

        assert validated > 0, "No movement steps validated"
        assert not failures, (
            f"{len(failures)}/{validated} steps failed.\n"
            + "\n".join(failures[:10])
        )


@pytest.mark.validation
class TestProneMoves:
    """Validate legal moves when the RL unit is prone.

    Uses prone_patrol_trace: a dual-bot trace where the RL unit (owner 0) starts
    prone via inject_state. This guarantees prone steps every run without relying
    on stochastic falls.
    """

    def test_prone_moves(self, prone_patrol_trace):
        from tests.sim_validation.test_legal_moves import (
            validate_legal_moves, diagnose_prone_extras,
        )
        from tests.sim_validation.reconstruct import extract_unit_state

        trace = prone_patrol_trace
        rl_owner = trace.rl_owner_id
        prone_steps = []
        for step in trace.steps:
            raw_obs = step.raw_obs
            if raw_obs.get("terminated") or raw_obs.get("truncated"):
                continue
            if not raw_obs.get("legal_moves"):
                continue
            active_id = raw_obs.get("active_entity_id", -1)
            owner = trace.entity_owners.get(active_id, -1)
            if owner != rl_owner:
                continue  # Only check the prone RL unit
            java_unit = extract_unit_state(raw_obs, owner)
            if java_unit and java_unit.get("prone", False):
                prone_steps.append((step, owner))

        assert prone_steps, (
            "No prone steps found in prone_patrol_trace. "
            "inject_state may not have applied correctly."
        )

        diagnostics = []
        total_sim_only = 0
        steps_with_extras = 0
        java_only_total = 0

        for step, owner in prone_steps:
            result = validate_legal_moves(
                step.raw_obs, owner,
                step_idx=step.step_idx, algorithm="deque",
            )
            if result.skipped_partial_move:
                continue
            if result.sim_only_hexes:
                steps_with_extras += 1
                total_sim_only += len(result.sim_only_hexes)
                diag = diagnose_prone_extras(
                    result, result._sim_moves, result._walk_mp,
                )
                if diag:
                    diagnostics.append(diag)
            java_only_total += len(result.java_only_hexes)

        summary = (
            f"Prone steps: {len(prone_steps)}, "
            f"with extras: {steps_with_extras}, "
            f"total sim-only hexes: {total_sim_only}"
        )
        print(f"\n{summary}")
        for d in diagnostics:
            print(d)

        assert java_only_total == 0, (
            f"{java_only_total} java-only hexes in prone steps "
            f"(Python sim is MISSING moves)"
        )


@pytest.mark.validation
class TestDistancesWalkPatrol:
    """Both mechs walk scripted waypoints; validate distances at every step."""

    def test_distances_walk_patrol(self, walk_patrol_trace):
        from tests.sim_validation.test_to_hit import validate_distances

        trace = walk_patrol_trace
        total_checks = 0
        all_mismatches = []

        for step in trace.steps:
            obs = step.raw_obs
            if obs.get("terminated") or obs.get("truncated"):
                continue
            if not obs.get("legal_moves"):
                continue

            active_id = obs.get("active_entity_id", -1)
            owner = trace.entity_owners.get(active_id, -1)
            if owner < 0:
                continue

            result = validate_distances(obs, owner)
            total_checks += result.distance_checks
            all_mismatches.extend(result.distance_mismatches)

        assert total_checks > 0, "No distance checks performed"
        assert not all_mismatches, (
            f"{len(all_mismatches)} distance mismatches out of {total_checks}. "
            f"First: {all_mismatches[0]}"
        )


@pytest.mark.validation
class TestToHitComponents:
    def test_to_hit_components(self, java_trace):
        from tests.sim_validation.test_to_hit import validate_to_hit_components

        result = validate_to_hit_components()
        assert not result.component_mismatches, (
            f"{len(result.component_mismatches)} mismatches. "
            f"First: {result.component_mismatches[0]}"
        )


@pytest.mark.validation
class TestMoveFeatures:
    """Per-move feature cross-validation (distance, weapon arcs) on walk patrol."""

    def test_move_features_walk_patrol(self, walk_patrol_trace):
        from tests.sim_validation.test_move_features import validate_move_features

        trace = walk_patrol_trace
        total_dist = 0
        total_arc = 0
        all_dist_mismatches = []
        all_arc_mismatches = []

        for step in trace.steps:
            obs = step.raw_obs
            if obs.get("terminated") or obs.get("truncated"):
                continue
            if not obs.get("legal_moves"):
                continue

            active_id = obs.get("active_entity_id", -1)
            owner = trace.entity_owners.get(active_id, -1)
            if owner < 0:
                continue

            result = validate_move_features(obs, owner)
            total_dist += result.distance_checks
            total_arc += result.arc_checks
            all_dist_mismatches.extend(result.distance_mismatches)
            all_arc_mismatches.extend(result.arc_mismatches)

        assert total_dist > 0, "No distance checks performed"
        assert total_arc > 0, "No arc checks performed"
        assert not all_dist_mismatches, (
            f"{len(all_dist_mismatches)} distance mismatches out of {total_dist}. "
            f"First: {all_dist_mismatches[0]}"
        )
        assert not all_arc_mismatches, (
            f"{len(all_arc_mismatches)} arc mismatches out of {total_arc}. "
            f"First: {all_arc_mismatches[0]}"
        )


@pytest.mark.validation
class TestObsFeatures:
    """127-dim observation feature vector end-to-end validation on walk patrol."""

    def test_obs_features_walk_patrol(self, walk_patrol_trace):
        from tests.sim_validation.test_obs_features import validate_obs_features

        trace = walk_patrol_trace
        validated = 0
        all_mismatches = []

        for step in trace.steps:
            obs = step.raw_obs
            if obs.get("terminated") or obs.get("truncated"):
                continue
            if not obs.get("legal_moves"):
                continue

            active_id = obs.get("active_entity_id", -1)
            owner = trace.entity_owners.get(active_id, -1)
            if owner < 0:
                continue

            result = validate_obs_features(obs, owner)
            if result.skipped:
                continue
            validated += 1
            for m in result.mismatches:
                all_mismatches.append(f"step {step.step_idx}: {m}")

        assert validated > 0, "No observation steps validated"
        assert not all_mismatches, (
            f"{len(all_mismatches)} feature mismatches across {validated} steps. "
            f"First: {all_mismatches[0]}"
        )


@pytest.mark.validation
class TestDamage:
    """Damage rule validation on stochastic java_trace.

    Validates rules hold when damage occurs, but damage events are not
    guaranteed (depends on Princess firing and hit rolls).
    TestDamageInjected is the deterministic counterpart that guarantees
    damage events via low-armor injection.
    Best with --random-actions to increase combat activity.
    """

    def test_damage(self, java_trace):
        from tests.sim_validation.test_damage import validate_damage_trace

        result = validate_damage_trace(
            java_trace.steps, java_trace.rl_owner_id,
        )
        all_errors = (
            result.consistency_errors
            + result.transfer_errors
            + result.side_effect_errors
        )
        assert not all_errors, (
            f"{len(all_errors)} damage errors. First: {all_errors[0]}"
        )
        if result.total_steps_with_damage == 0:
            print("  NOTE: No damage events in trace — use --random-actions "
                  "for more combat activity")


@pytest.mark.validation
class TestHeat:
    """Heat rule validation on stochastic java_trace.

    Validates rules hold when heat changes, but changes are not guaranteed
    (depends on weapons firing). TestHeatInjected and TestHeatGeneration
    are the deterministic counterparts.
    Best with --random-actions to increase combat activity.
    """

    def test_heat(self, java_trace):
        from tests.sim_validation.test_heat import validate_heat_trace

        result = validate_heat_trace(
            java_trace.steps, java_trace.rl_owner_id,
        )
        assert not result.dissipation_errors, (
            f"{len(result.dissipation_errors)} heat errors. "
            f"First: {result.dissipation_errors[0]}"
        )
        if result.heat_changes == 0:
            print("  NOTE: No heat changes in trace — use --random-actions "
                  "for more combat activity")


@pytest.mark.validation
class TestCrits:
    """Crit rule validation on stochastic java_trace.

    Validates rules hold when crits occur, but crit events are not
    guaranteed (depends on internal damage from stochastic combat).
    TestCritsInjected is the deterministic counterpart that guarantees
    internal damage via low-armor injection.
    Best with --random-actions --max-rounds 20.
    """

    def test_crits(self, java_trace):
        """Validate crit state consistency across Java game trace."""
        from tests.sim_validation.test_crits import validate_crit_trace

        result = validate_crit_trace(
            java_trace.steps, java_trace.rl_owner_id,
        )
        if result.total_steps == 0:
            pytest.skip("No non-terminal steps in trace")

        all_errors = (
            result.consistency_errors
            + result.side_effect_errors
            + result.mp_errors
        )

        # Print diagnostics
        print(f"\nCrit validation: {result.total_steps} steps, "
              f"{result.crit_events} crit events")
        if result.mp_errors:
            print(f"  MP errors: {len(result.mp_errors)}")
            for err in result.mp_errors[:5]:
                print(f"    {err}")

        assert not result.consistency_errors, (
            f"{len(result.consistency_errors)} consistency errors. "
            f"First: {result.consistency_errors[0]}"
        )
        assert not result.side_effect_errors, (
            f"{len(result.side_effect_errors)} side-effect errors. "
            f"First: {result.side_effect_errors[0]}"
        )
        assert not result.mp_errors, (
            f"{len(result.mp_errors)} MP consistency errors. "
            f"First: {result.mp_errors[0]}"
        )
        if result.crit_events == 0:
            print("  NOTE: No crit events in trace — use --random-actions "
                  "--max-rounds 20 for more internal damage")


@pytest.mark.validation
class TestFiring:
    """DIAGNOSTIC ONLY — no assertions, always passes.

    Prints firing match rates on the stochastic java_trace for manual
    inspection. TestFiringPatrol is the asserting regression test that
    uses scripted waypoints for deterministic positions.
    """

    def test_firing(self, java_trace):
        """Print firing match rates (diagnostic — always passes).

        Known gaps causing mismatches in edge cases:
        - Prone arm restriction ("firing from other arm already")
        - Target immobile modifier (-4)
        - Shoulder actuator damage penalty (+4)
        - LOS/arc edge cases (~2% mismatch rate)
        """
        from tests.sim_validation.test_firing import validate_firing_trace

        result = validate_firing_trace(
            java_trace.steps, java_trace.rl_owner_id,
        )
        if result.opp_fireable_total == 0 and result.rl_fireable_total == 0:
            pytest.skip("No firing rounds with firing_report (Java may lack telemetry)")


@pytest.mark.validation
class TestFiringPatrol:
    """Scripted waypoints designed for firing validation.

    Units converge to positions with light woods between them at (6,8),
    producing reliable partial cover scenarios alongside standard arc/range
    coverage. Uses a dedicated firing_patrol_trace fixture.
    """

    def test_firing_patrol(self, firing_patrol_trace):
        from megamek_gym.sim.board import BOARD
        from megamek_gym.sim.los import LosTable
        from tests.sim_validation.test_firing import validate_firing_single_round

        trace = firing_patrol_trace
        rl_owner_id = trace.rl_owner_id
        assert rl_owner_id >= 0, "Could not determine RL owner ID"

        board = BOARD
        los_table = LosTable(board)

        rounds_with_firing = 0
        fireable_matches = 0
        fireable_total = 0
        tn_matches = 0
        tn_total = 0
        mismatches = []

        for step in trace.steps:
            obs = step.raw_obs
            if obs.get("terminated") or obs.get("truncated"):
                continue
            if not obs.get("firing_report"):
                continue

            rounds_with_firing += 1
            comparisons = validate_firing_single_round(
                obs, rl_owner_id, board=board, los_table=los_table,
            )

            for comp in comparisons:
                if comp.skipped:
                    continue
                fireable_matches += comp.fireable_match_count
                fireable_total += comp.fireable_total
                tn_matches += comp.tn_match_count
                tn_total += comp.tn_total

                for w in comp.weapons:
                    if not w.fireable_match and len(mismatches) < 3:
                        fr = obs["firing_report"]
                        entity_key = "rl_entity" if comp.entity_label == "rl" else "opp_entity"
                        weapons_key = f"{comp.entity_label}_weapons"
                        print(f"\n  DEBUG Round {comp.round_num} {comp.entity_label} {w.weapon_name} loc={w.location}:")
                        print(f"    entity state: {fr.get(entity_key)}")
                        for jw in fr.get(weapons_key, []):
                            if jw['location'] == w.location:
                                print(f"    java weapon: {jw}")
                    if not w.fireable_match:
                        java_f = w.java_can_fire and not w.java_impossible
                        mismatches.append(
                            f"Round {comp.round_num} {comp.entity_label} "
                            f"{w.weapon_name} loc={w.location}: "
                            f"java_fireable={java_f} (can_fire={w.java_can_fire} "
                            f"impossible={w.java_impossible} desc=\"{w.java_desc}\") "
                            f"python_fireable={w.python_can_fire}"
                        )
                    elif w.tn_match is False:
                        mismatches.append(
                            f"Round {comp.round_num} {comp.entity_label} "
                            f"{w.weapon_name}: java_tn={w.java_tn} "
                            f"python_tn={w.python_tn} (delta={w.tn_delta:+d}) "
                            f"desc=\"{w.java_desc}\""
                        )

        print(f"\nFiring patrol: {rounds_with_firing} rounds with firing_report, "
              f"fireability {fireable_matches}/{fireable_total}, "
              f"TNs {tn_matches}/{tn_total}")
        if mismatches:
            print(f"\nMISMATCHES ({len(mismatches)}):")
            for m in mismatches:
                print(f"  {m}")

        assert rounds_with_firing > 0, (
            "No rounds with firing_report in firing patrol trace"
        )
        assert fireable_total > 0, (
            "No weapons compared across all firing rounds"
        )
        assert not mismatches, (
            f"{len(mismatches)} mismatches.\n"
            + "\n".join(mismatches[:10])
        )


@pytest.mark.validation
class TestHeatInjected:
    """Validate heat mechanics with deterministic inject_state.

    Uses heated_patrol_trace: RL unit starts at heat=15 via inject_state.
    Cross-validates Java's heat MP penalty formula on the first step.
    """

    def test_initial_heat(self, heated_patrol_trace):
        """Verify inject_state applied correctly and Java MP formula matches."""
        from tests.sim_validation.test_heat import validate_initial_heat

        trace = heated_patrol_trace
        rl_owner = trace.rl_owner_id
        assert rl_owner >= 0, "Could not determine RL owner ID"

        # First non-terminal step for the RL unit
        for step in trace.steps:
            obs = step.raw_obs
            if obs.get("terminated") or obs.get("truncated"):
                continue
            active_id = obs.get("active_entity_id", -1)
            owner = trace.entity_owners.get(active_id, -1)
            if owner != rl_owner:
                continue

            errors = validate_initial_heat(
                obs, rl_owner,
                expected_heat=15,
                expected_walk_mp=2,  # base 5 - heat//5 (15//5=3) = 2
            )
            assert not errors, (
                f"Initial heat validation failed: {'; '.join(errors)}"
            )
            break
        else:
            pytest.fail("No RL movement step found in heated_patrol_trace")

    def test_heat_dissipation(self, heated_patrol_trace):
        """Validate heat rules across the trace (dissipation, bounds)."""
        from tests.sim_validation.test_heat import validate_heat_trace

        trace = heated_patrol_trace
        result = validate_heat_trace(trace.steps, trace.rl_owner_id)

        assert result.heat_changes > 0, (
            "No heat changes observed despite starting at heat=15 and naive firing"
        )
        assert not result.dissipation_errors, (
            f"{len(result.dissipation_errors)} heat errors. "
            f"First: {result.dissipation_errors[0]}"
        )


@pytest.mark.validation
class TestHeatGeneration:
    """Cross-validate Python sim heat generation against Java MegaMek.

    Verifies: expected = max(0, prev_heat + weapon_heat + running_heat
                             + 5*engine_hits - effective_sinks)
    using firing_report data to determine which weapons fired.
    """

    def test_heat_generation_heated(self, heated_patrol_trace):
        """Validate on high-heat trace with naive firing."""
        from tests.sim_validation.test_heat import validate_heat_generation

        trace = heated_patrol_trace
        result = validate_heat_generation(trace.steps, trace.rl_owner_id)

        print(f"\nHeat generation (heated): {result.rounds_validated} rounds, "
              f"{result.rounds_matched} matched, "
              f"{len(result.mismatches)} mismatches, "
              f"{len(result.skips)} skips")
        for m in result.mismatches:
            print(f"  {m}")

        assert result.rounds_validated > 0, (
            "No rounds validated for heat generation"
        )
        assert not result.mismatches, (
            f"{len(result.mismatches)} heat generation mismatches. "
            f"First: {result.mismatches[0]}"
        )

    def test_heat_generation_damaged(self, damaged_patrol_trace):
        """Validate on damaged trace (engine crit heat)."""
        from tests.sim_validation.test_heat import validate_heat_generation

        trace = damaged_patrol_trace
        result = validate_heat_generation(trace.steps, trace.rl_owner_id)

        print(f"\nHeat generation (damaged): {result.rounds_validated} rounds, "
              f"{result.rounds_matched} matched, "
              f"{len(result.mismatches)} mismatches, "
              f"{len(result.skips)} skips")
        for m in result.mismatches:
            print(f"  {m}")

        assert not result.mismatches, (
            f"{len(result.mismatches)} heat generation mismatches. "
            f"First: {result.mismatches[0]}"
            )


@pytest.mark.validation
class TestShutdownConsistency:
    """Validate shutdown mechanics consistency in Java traces.

    Uses shutdown_patrol_trace: RL at heat=35, disarmed opponent.
    Verifies shutdown transitions only occur when heat >= 14.

    Inherently probabilistic: with heat=35 (walk), post-dissipation=18,
    shutdown TN=6 → 27.8% per round, and only 1 round in shutdown zone
    before heat dissipates below threshold. Cannot raise heat further
    without triggering ammo explosion checks (TN=6 at heat 23+).
    The rule validation (shutdown only at heat >= 14) is the primary value;
    observing an actual shutdown is a bonus.
    """

    def test_shutdown(self, shutdown_patrol_trace):
        from tests.sim_validation.test_heat import validate_shutdown_consistency
        from tests.sim_validation.reconstruct import extract_unit_state

        trace = shutdown_patrol_trace
        rl_owner = trace.rl_owner_id
        print(f"\nShutdown trace: {len(trace.steps)} steps, rl_owner={rl_owner}")
        for step in trace.steps[:20]:
            obs = step.raw_obs
            unit = extract_unit_state(obs, rl_owner)
            if unit:
                print(f"  step {step.step_idx} round {obs.get('round')} "
                      f"heat={unit.get('heat')} shutdown={unit.get('shutdown')} "
                      f"destroyed={unit.get('destroyed')}")
            elif obs.get("terminated"):
                print(f"  step {step.step_idx}: TERMINAL "
                      f"outcome={obs.get('game_outcome')}")

        result = validate_shutdown_consistency(trace.steps, trace.rl_owner_id)

        print(f"\nShutdown consistency: {result.total_transitions} transitions, "
              f"{result.shutdown_transitions} shutdowns, "
              f"{result.recovery_transitions} recoveries")

        assert not result.invalid_shutdowns, (
            f"{len(result.invalid_shutdowns)} invalid shutdowns. "
            f"First: {result.invalid_shutdowns[0]}"
        )
        if result.shutdown_transitions == 0:
            print("  NOTE: No shutdown observed (probabilistic — 27.8% chance)")


@pytest.mark.validation
class TestDamageInjected:
    """Validate damage mechanics with deterministic inject_state.

    Uses damaged_patrol_trace: both units start with low armor (RT/LL=2 on RL,
    LT/RL=2 on opponent) and naive firing. Guarantees damage events and
    location destructions every run.
    """

    def test_initial_armor(self, damaged_patrol_trace):
        """Verify inject_state applied the low armor correctly."""
        from tests.sim_validation.reconstruct import extract_unit_state

        trace = damaged_patrol_trace
        rl_owner = trace.rl_owner_id
        assert rl_owner >= 0, "Could not determine RL owner ID"

        # Check the first observation where the RL unit is active
        # (deferred injection applies at the start of each entity's turn)
        for step in trace.steps:
            obs = step.raw_obs
            if obs.get("terminated") or obs.get("truncated"):
                continue
            active_id = obs.get("active_entity_id", -1)
            owner = trace.entity_owners.get(active_id, -1)
            if owner != rl_owner:
                continue

            rl_unit = extract_unit_state(obs, rl_owner)
            assert rl_unit is not None, "No RL unit in observation"

            # Check that armor was reduced on at least some limb locations.
            # RT is location index 2 in BipedMek ordering.
            armor_list = rl_unit.get("armor", [])
            assert len(armor_list) >= 7, "Armor list too short"
            rt_armor = armor_list[2].get("armor", -1)
            rt_internal = armor_list[2].get("internal", -1)
            assert rt_armor <= 0, f"RT armor {rt_armor} > 0 — inject_state may not have applied"
            assert rt_internal <= 1, f"RT internal {rt_internal} > 1 — inject_state may not have applied"
            break
        else:
            pytest.fail("No RL movement step found in damaged_patrol_trace")

    def test_damage_consistency(self, damaged_patrol_trace):
        """Validate damage rules and assert damage actually occurred."""
        from tests.sim_validation.test_damage import validate_damage_trace

        trace = damaged_patrol_trace
        result = validate_damage_trace(trace.steps, trace.rl_owner_id)

        all_errors = (
            result.consistency_errors
            + result.transfer_errors
            + result.side_effect_errors
        )
        assert not all_errors, (
            f"{len(all_errors)} damage errors. First: {all_errors[0]}"
        )
        assert result.total_steps_with_damage > 0, (
            "No damage events despite low armor and naive firing"
        )

    def test_location_destructions(self, damaged_patrol_trace):
        """Check that location destructions occur with low-armor injection.

        With armor=0/internal=1 on all limbs and naive firing at close range,
        at least one location destruction should occur within the trace.
        """
        from tests.sim_validation.test_damage import count_location_destructions

        trace = damaged_patrol_trace
        destructions = []
        for owner_id in (trace.rl_owner_id, 1 - trace.rl_owner_id):
            destructions.extend(
                count_location_destructions(trace.steps, owner_id)
            )

        print(f"\nLocation destructions: {len(destructions)}")
        for step_idx, loc_name in destructions:
            print(f"  step {step_idx}: {loc_name}")

        assert len(destructions) > 0, (
            "No location destructions despite armor=0/internal=1 on limbs "
            "with naive firing — trace may be too short or injection failed"
        )


@pytest.mark.validation
class TestCritsInjected:
    """Validate crit mechanics with deterministic inject_state.

    Uses damaged_patrol_trace: low armor guarantees internal damage and
    crit rolls occur during combat.
    """

    def test_crits_with_damage(self, damaged_patrol_trace):
        from tests.sim_validation.test_crits import validate_crit_trace

        trace = damaged_patrol_trace
        result = validate_crit_trace(trace.steps, trace.rl_owner_id)

        if result.total_steps == 0:
            pytest.skip("No non-terminal steps in trace")

        print(f"\nCrit validation (injected): {result.total_steps} steps, "
              f"{result.crit_events} crit events")

        assert not result.consistency_errors, (
            f"{len(result.consistency_errors)} consistency errors. "
            f"First: {result.consistency_errors[0]}"
        )
        assert not result.side_effect_errors, (
            f"{len(result.side_effect_errors)} side-effect errors. "
            f"First: {result.side_effect_errors[0]}"
        )
        assert not result.mp_errors, (
            f"{len(result.mp_errors)} MP consistency errors. "
            f"First: {result.mp_errors[0]}"
        )


@pytest.mark.validation
class TestFiringDestroyedWeapons:
    """Validate that destroyed weapons correctly show as unfireable.

    Uses weapons_damaged_patrol_trace: RL unit has weapons 0 and 2
    destroyed via inject_state (Medium Lasers in RA and LA on TBT-5S).
    """

    def test_initial_weapons_destroyed(self, weapons_damaged_patrol_trace):
        """Verify inject_state destroyed the right weapons."""
        from tests.sim_validation.reconstruct import extract_unit_state

        trace = weapons_damaged_patrol_trace
        rl_owner = trace.rl_owner_id
        assert rl_owner >= 0, "Could not determine RL owner ID"

        # Check the first observation where the RL unit is active
        # (deferred injection applies at the start of each entity's turn)
        for step in trace.steps:
            obs = step.raw_obs
            if obs.get("terminated") or obs.get("truncated"):
                continue
            active_id = obs.get("active_entity_id", -1)
            owner = trace.entity_owners.get(active_id, -1)
            if owner != rl_owner:
                continue

            rl_unit = extract_unit_state(obs, rl_owner)
            assert rl_unit is not None, "No RL unit in observation"

            weapons = rl_unit.get("weapons", [])
            assert len(weapons) >= 3, f"Expected >= 3 weapons, got {len(weapons)}"
            assert weapons[0].get("destroyed", False), (
                "Weapon 0 should be destroyed after inject_state"
            )
            assert weapons[2].get("destroyed", False), (
                "Weapon 2 should be destroyed after inject_state"
            )
            assert not weapons[1].get("destroyed", False), (
                "Weapon 1 should NOT be destroyed"
            )
            break
        else:
            pytest.fail("No RL movement step found in weapons_damaged_patrol_trace")

    def test_destroyed_weapons_unfireable(self, weapons_damaged_patrol_trace):
        """Verify destroyed weapons have can_fire=false in firing_report."""
        trace = weapons_damaged_patrol_trace
        rl_owner = trace.rl_owner_id

        rounds_checked = 0
        destroyed_unfireable = 0
        destroyed_fireable_errors = []

        for step in trace.steps:
            obs = step.raw_obs
            if obs.get("terminated") or obs.get("truncated"):
                continue
            firing_report = obs.get("firing_report")
            if not firing_report:
                continue

            rl_weapons = firing_report.get("rl_weapons", [])
            if not rl_weapons:
                continue

            rounds_checked += 1
            for jw in rl_weapons:
                if jw.get("destroyed", False):
                    if jw.get("can_fire", True):
                        destroyed_fireable_errors.append(
                            f"round {obs.get('round', '?')}: "
                            f"{jw['weapon_name']} idx={jw['weapon_index']} "
                            f"is destroyed but can_fire=true"
                        )
                    else:
                        destroyed_unfireable += 1

        print(f"\nFiring destroyed weapons: {rounds_checked} rounds checked, "
              f"{destroyed_unfireable} destroyed+unfireable confirmations")

        assert rounds_checked > 0, (
            "No rounds with firing_report in weapons_damaged_patrol_trace"
        )
        assert not destroyed_fireable_errors, (
            f"{len(destroyed_fireable_errors)} errors. "
            f"First: {destroyed_fireable_errors[0]}"
        )
        assert destroyed_unfireable > 0, (
            "No destroyed weapons found in firing_report — "
            "inject_state may not have persisted to the server"
        )


@pytest.mark.validation
class TestStatistical:
    def test_statistical(self, megamek_dir, base_port, java_trace):
        """Statistical distribution comparison (Tier 3).

        This test runs its own Java games (separate from the shared trace).
        """
        from tests.sim_validation.test_statistical import (
            run_sim_games,
            run_java_games,
            compare_distributions,
        )
        from tests.sim_validation.collector import collect_game_trace

        n_games = 10
        sim_games = run_sim_games(n_games)
        java_games = run_java_games(
            collect_game_trace, n_games,
            megamek_dir=megamek_dir,
            base_port=base_port,
        )

        comparison = compare_distributions(sim_games, java_games)
        failures = []
        for metric, data in comparison.items():
            if isinstance(data, dict) and not data.get("ok"):
                failures.append(metric)

        assert not failures, (
            f"Statistical distributions differ for: {', '.join(failures)}"
        )


# ---------------------------------------------------------------------------
# PSR / Fall mechanics
# ---------------------------------------------------------------------------

@pytest.mark.validation
class TestPSRFalls:
    """Deterministic PSR/fall cross-validation.

    Uses state injection (gyro_hits, hip actuator damage, low armor) to
    create guaranteed fall/PSR scenarios and validates Java state transitions.
    """

    def test_gyro_destroyed_injection(self, gyro_destroyed_trace):
        """Verify gyro_hits=2 injection is reflected in first observation."""
        from tests.sim_validation.reconstruct import extract_crit_state

        trace = gyro_destroyed_trace
        first_obs = trace.steps[0].raw_obs
        for owner_id in (trace.rl_owner_id, 1 - trace.rl_owner_id):
            crit = extract_crit_state(first_obs, owner_id)
            assert crit is not None, f"owner={owner_id}: no crit_state"
            assert crit.get("gyro_hits", 0) == 2, (
                f"owner={owner_id}: gyro_hits={crit.get('gyro_hits')} "
                f"(expected 2 from injection)"
            )

    def test_gyro_destroyed_auto_fall(self, gyro_destroyed_trace):
        """Gyro destroyed + any PSR trigger -> must fall (auto-fail)."""
        from tests.sim_validation.test_psr_falls import (
            validate_gyro_destroyed_auto_fall,
        )

        trace = gyro_destroyed_trace
        for owner_id in (trace.rl_owner_id, 1 - trace.rl_owner_id):
            result = validate_gyro_destroyed_auto_fall(
                trace.steps, owner_id,
            )
            print(f"\n  owner={owner_id}: standing_with_damage="
                  f"{result['standing_with_damage']}, "
                  f"fell_after_damage={result['fell_after_damage']}")

            assert not result["errors"], (
                f"owner={owner_id}: {len(result['errors'])} auto-fall errors. "
                f"First: {result['errors'][0]}"
            )

    def test_gyro_destroyed_implies_prone(self, gyro_destroyed_trace):
        """Combat gyro destruction (transition to 2 hits) -> unit falls."""
        from tests.sim_validation.test_psr_falls import (
            validate_gyro_destroyed_implies_prone,
        )

        trace = gyro_destroyed_trace
        for owner_id in (trace.rl_owner_id, 1 - trace.rl_owner_id):
            errors = validate_gyro_destroyed_implies_prone(
                trace.steps, owner_id,
            )
            assert not errors, (
                f"owner={owner_id}: {len(errors)} violations. "
                f"First: {errors[0]}"
            )

    def test_hip_injection(self, hip_damaged_trace):
        """Verify hip_hits injection is reflected in first observation."""
        from tests.sim_validation.reconstruct import extract_crit_state

        trace = hip_damaged_trace
        first_obs = trace.steps[0].raw_obs
        crit = extract_crit_state(first_obs, trace.rl_owner_id)
        assert crit is not None, "no crit_state in first observation"

        rl = crit.get("right_leg", {})
        assert rl.get("hip_hits", 0) == 1, (
            f"right_leg hip_hits={rl.get('hip_hits')} (expected 1 from injection)"
        )

    def test_hip_reduces_mp(self, hip_damaged_trace):
        """Hip hit halves walk MP: ceil(5/2) = 3 for TBT-5S."""
        from tests.sim_validation.test_psr_falls import validate_hip_mp_reduction

        trace = hip_damaged_trace
        errors = validate_hip_mp_reduction(
            trace.steps, trace.rl_owner_id,
            expected_walk_mp=3,
        )
        assert not errors, (
            f"{len(errors)} MP errors. First: {errors[0]}"
        )

    def test_falls_have_triggers(self, damaged_patrol_trace):
        """Every fall in damaged trace has a valid PSR trigger."""
        from tests.sim_validation.test_psr_falls import validate_psr_fall_trace

        trace = damaged_patrol_trace
        result = validate_psr_fall_trace(trace.steps, trace.rl_owner_id)

        print(f"\nPSR/Fall: {result.fall_events} falls, "
              f"{result.stand_events} stand-ups")
        for note in result.notes[:10]:
            print(f"  {note}")

        assert not result.unexplained_falls, (
            f"{len(result.unexplained_falls)} unexplained falls. "
            f"First: {result.unexplained_falls[0]}"
        )

    def test_stand_requires_mp(self, damaged_patrol_trace):
        """Stand-up requires walk_mp >= 2."""
        from tests.sim_validation.test_psr_falls import validate_psr_fall_trace

        trace = damaged_patrol_trace
        result = validate_psr_fall_trace(trace.steps, trace.rl_owner_id)

        assert not result.stand_mp_errors, (
            f"{len(result.stand_mp_errors)} stand-up MP errors. "
            f"First: {result.stand_mp_errors[0]}"
        )

    def test_falls_occurred(self, gyro_destroyed_trace):
        """Liveness: at least 1 fall in the gyro-destroyed trace."""
        from tests.sim_validation.test_psr_falls import validate_psr_fall_trace

        trace = gyro_destroyed_trace
        result = validate_psr_fall_trace(trace.steps, trace.rl_owner_id)

        print(f"\nPSR/Fall liveness: {result.fall_events} falls, "
              f"{result.stand_events} stand-ups")

        assert result.fall_events > 0, (
            "No falls observed despite destroyed gyros and low armor — "
            "test fixture may not be generating combat"
        )
