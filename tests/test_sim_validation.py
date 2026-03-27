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
class TestLos:
    def test_los(self, java_trace):
        from megamek_gym.sim.board import BOARD
        from megamek_gym.sim.los import LosTable
        from tests.sim_validation.test_los import validate_los

        los_table = LosTable(BOARD)
        total_checks = 0
        all_mismatches = []

        for step in java_trace.steps:
            raw_obs = step.raw_obs
            if raw_obs.get("terminated") or raw_obs.get("truncated"):
                continue
            if not raw_obs.get("legal_moves"):
                continue

            result = validate_los(raw_obs, los_table, java_trace.rl_owner_id)
            total_checks += result.total_checks
            all_mismatches.extend(result.mismatches)

        assert total_checks > 0, "No LOS checks performed"
        assert not all_mismatches, (
            f"{len(all_mismatches)} LOS mismatches out of {total_checks}. "
            f"First: {all_mismatches[0]}"
        )


@pytest.mark.validation
class TestLegalMoves:
    def test_legal_moves(self, java_trace):
        from tests.sim_validation.test_legal_moves import validate_legal_moves_trace

        summary = validate_legal_moves_trace(
            java_trace.steps, java_trace.rl_owner_id,
        )
        assert summary.passed, (
            f"Legal moves mismatch: {summary.total_java_only_hexes} java-only + "
            f"{summary.total_sim_only_hexes} sim-only hexes "
            f"(hex_match={summary.hex_match_rate:.1%}, "
            f"move_match={summary.move_match_rate:.1%})"
        )


@pytest.mark.validation
class TestDistances:
    def test_distances(self, java_trace):
        from tests.sim_validation.test_to_hit import validate_distances

        total_checks = 0
        all_mismatches = []

        for step in java_trace.steps:
            raw_obs = step.raw_obs
            if raw_obs.get("terminated") or raw_obs.get("truncated"):
                continue
            if not raw_obs.get("legal_moves"):
                continue

            result = validate_distances(raw_obs, java_trace.rl_owner_id)
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
class TestDamage:
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


@pytest.mark.validation
class TestHeat:
    def test_heat(self, java_trace):
        from tests.sim_validation.test_heat import validate_heat_trace

        result = validate_heat_trace(
            java_trace.steps, java_trace.rl_owner_id,
        )
        assert not result.dissipation_errors, (
            f"{len(result.dissipation_errors)} heat errors. "
            f"First: {result.dissipation_errors[0]}"
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
