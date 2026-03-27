#!/usr/bin/env python3
"""Comprehensive validation of Python sim against Java MegaMek.

Runs tiered validation tests:
  Tier 1 (deterministic): board, unit template, LOS, legal moves, distances
  Tier 2 (delta inference): damage consistency, heat consistency
  Tier 3 (statistical): game length/win rate distributions

Usage:
    poetry run python validate_sim.py --megamek-dir ../megamek
    poetry run python validate_sim.py --megamek-dir ../megamek --tier 1
    poetry run python validate_sim.py --megamek-dir ../megamek --only legal_moves
    poetry run python validate_sim.py --megamek-dir ../megamek --verbose
"""

from __future__ import annotations

import argparse
import sys
import time
import traceback

# ---------------------------------------------------------------------------
# Test registry
# ---------------------------------------------------------------------------

ALL_TESTS = [
    # (name, tier, description)
    ("board", 1, "Board terrain & elevation"),
    ("unit_template", 1, "Unit template (armor, weapons, MP)"),
    ("los", 1, "Line of sight"),
    ("legal_moves", 1, "Legal move enumeration"),
    ("distances", 1, "Hex distance cross-validation"),
    ("to_hit_components", 1, "To-hit modifier tables"),
    ("damage", 2, "Damage consistency (armor deltas)"),
    ("heat", 2, "Heat consistency"),
    ("statistical", 3, "Statistical distribution comparison"),
]

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _print_header(name: str, description: str, tier: int) -> None:
    print(f"\n{'='*60}")
    print(f"  Tier {tier}: {description}")
    print(f"  Test: {name}")
    print(f"{'='*60}")


def _print_pass(name: str, detail: str = "") -> None:
    suffix = f" — {detail}" if detail else ""
    print(f"  PASS: {name}{suffix}")


def _print_fail(name: str, detail: str = "") -> None:
    suffix = f" — {detail}" if detail else ""
    print(f"  FAIL: {name}{suffix}")


def _print_info(msg: str) -> None:
    print(f"  {msg}")

# ---------------------------------------------------------------------------
# Test implementations
# ---------------------------------------------------------------------------

def test_board(trace, verbose: bool) -> bool:
    """Tier 1: Board terrain and elevation."""
    from tests.sim_validation.test_board import validate_board

    first_obs = trace.steps[0].raw_obs
    result = validate_board(first_obs)

    if result.dimension_mismatch:
        _print_fail("board dimensions", result.dimension_mismatch)
        return False

    _print_info(f"Checked {result.total_hexes} hexes")

    if result.elevation_mismatches:
        _print_fail("elevation", f"{len(result.elevation_mismatches)} mismatches")
        if verbose:
            for m in result.elevation_mismatches[:10]:
                _print_info(f"  {m}")
        return False

    if result.terrain_mismatches:
        _print_fail("terrain", f"{len(result.terrain_mismatches)} mismatches")
        if verbose:
            for m in result.terrain_mismatches[:10]:
                _print_info(f"  {m}")
        return False

    _print_pass("board", f"{result.total_hexes} hexes match")
    return True


def test_unit_template(trace, verbose: bool) -> bool:
    """Tier 1: Unit template validation."""
    from tests.sim_validation.test_unit_template import validate_unit_template
    from tests.sim_validation.reconstruct import extract_unit_state

    first_obs = trace.steps[0].raw_obs
    rl_unit = extract_unit_state(first_obs, trace.rl_owner_id)
    if rl_unit is None:
        _print_fail("unit_template", "No RL unit in first observation")
        return False

    result = validate_unit_template(rl_unit)

    if result.mismatches:
        _print_fail("unit_template", f"{len(result.mismatches)} mismatches")
        for m in result.mismatches:
            _print_info(f"  {m}")
        return False

    _print_pass("unit_template", "All armor, weapon, and MP values match")
    return True


def test_los(trace, verbose: bool) -> bool:
    """Tier 1: Line of sight comparison."""
    from megamek_gym.sim.board import BOARD
    from megamek_gym.sim.los import LosTable
    from tests.sim_validation.test_los import validate_los

    _print_info("Computing LOS table...")
    los_table = LosTable(BOARD)

    total_checks = 0
    total_mismatches = 0
    all_mismatches = []

    for step in trace.steps:
        raw_obs = step.raw_obs
        if raw_obs.get("terminated") or raw_obs.get("truncated"):
            continue
        if not raw_obs.get("legal_moves"):
            continue

        result = validate_los(raw_obs, los_table, trace.rl_owner_id)
        total_checks += result.total_checks
        total_mismatches += len(result.mismatches)
        all_mismatches.extend(result.mismatches)

    _print_info(f"Checked {total_checks} LOS queries across {len(trace.steps)} steps")

    if total_mismatches > 0:
        _print_fail("los", f"{total_mismatches} mismatches out of {total_checks}")
        if verbose:
            for m in all_mismatches[:20]:
                _print_info(f"  {m}")
        return False

    _print_pass("los", f"{total_checks} queries match")
    return True


def test_legal_moves(trace, verbose: bool) -> bool:
    """Tier 1: Legal move enumeration comparison."""
    from tests.sim_validation.test_legal_moves import validate_legal_moves_trace

    summary = validate_legal_moves_trace(trace.steps, trace.rl_owner_id)

    _print_info(f"Checked {len(summary.per_step)} steps")
    _print_info(f"Java total moves: {summary.total_java_moves}")
    _print_info(f"Sim total moves:  {summary.total_sim_moves}")
    _print_info(f"Hex match rate:   {summary.hex_match_rate:.1%}")
    _print_info(f"Move match rate:  {summary.move_match_rate:.1%}")
    _print_info(
        f"Hexes: shared={summary.total_shared_hexes} "
        f"java_only={summary.total_java_only_hexes} "
        f"sim_only={summary.total_sim_only_hexes}"
    )

    if verbose or not summary.passed:
        for r in summary.per_step:
            if r.java_only_hexes or r.sim_only_hexes:
                _print_info(
                    f"  Step {r.step_idx} at ({r.unit_pos[0]},{r.unit_pos[1]},f={r.unit_pos[2]}) "
                    f"prone={r.unit_prone}: "
                    f"java={r.java_move_count} sim={r.sim_move_count} "
                    f"java_only_hex={len(r.java_only_hexes)} sim_only_hex={len(r.sim_only_hexes)}"
                )
                if verbose:
                    if r.java_only_hexes:
                        sample = sorted(r.java_only_hexes)[:5]
                        _print_info(f"    Java-only hexes: {sample}")
                    if r.sim_only_hexes:
                        sample = sorted(r.sim_only_hexes)[:5]
                        _print_info(f"    Sim-only hexes: {sample}")
                    if r.walk_run_mismatches:
                        for m in r.walk_run_mismatches[:5]:
                            _print_info(f"    Walk/run: {m}")
                    if r.mp_mismatches and verbose:
                        for m in r.mp_mismatches[:5]:
                            _print_info(f"    MP: {m}")

    if summary.passed:
        _print_pass("legal_moves", f"{summary.total_shared_moves} moves match exactly")
        return True
    else:
        _print_fail(
            "legal_moves",
            f"{summary.total_java_only_hexes} java-only + "
            f"{summary.total_sim_only_hexes} sim-only hexes"
        )
        return False


def test_distances(trace, verbose: bool) -> bool:
    """Tier 1: Hex distance cross-validation."""
    from tests.sim_validation.test_to_hit import validate_distances

    total_checks = 0
    total_mismatches = 0
    all_mismatches = []

    for step in trace.steps:
        raw_obs = step.raw_obs
        if raw_obs.get("terminated") or raw_obs.get("truncated"):
            continue
        if not raw_obs.get("legal_moves"):
            continue

        result = validate_distances(raw_obs, trace.rl_owner_id)
        total_checks += result.distance_checks
        total_mismatches += len(result.distance_mismatches)
        all_mismatches.extend(result.distance_mismatches)

    _print_info(f"Checked {total_checks} distances")

    if total_mismatches > 0:
        _print_fail("distances", f"{total_mismatches} mismatches")
        if verbose:
            for m in all_mismatches[:10]:
                _print_info(f"  {m}")
        return False

    _print_pass("distances", f"{total_checks} distances match")
    return True


def test_to_hit_components(trace, verbose: bool) -> bool:
    """Tier 1: To-hit modifier table validation."""
    from tests.sim_validation.test_to_hit import validate_to_hit_components

    result = validate_to_hit_components()
    _print_info(f"Checked {result.component_checks} component values")

    if result.component_mismatches:
        _print_fail("to_hit_components", f"{len(result.component_mismatches)} mismatches")
        for m in result.component_mismatches:
            _print_info(f"  {m}")
        return False

    _print_pass("to_hit_components", f"{result.component_checks} values correct")
    return True


def test_damage(trace, verbose: bool) -> bool:
    """Tier 2: Damage consistency from armor deltas."""
    from tests.sim_validation.test_damage import validate_damage_trace

    result = validate_damage_trace(trace.steps, trace.rl_owner_id)

    _print_info(f"Steps with damage: {result.total_steps_with_damage}")
    _print_info(f"Total damage events: {result.total_damage_events}")

    all_errors = result.consistency_errors + result.transfer_errors + result.side_effect_errors
    if all_errors:
        _print_fail("damage", f"{len(all_errors)} errors")
        for e in all_errors[:10]:
            _print_info(f"  {e}")
        if verbose:
            for e in all_errors[10:]:
                _print_info(f"  {e}")
        return False

    _print_pass("damage", f"{result.total_damage_events} damage events consistent")
    return True


def test_heat(trace, verbose: bool) -> bool:
    """Tier 2: Heat consistency."""
    from tests.sim_validation.test_heat import validate_heat_trace

    result = validate_heat_trace(trace.steps, trace.rl_owner_id)

    _print_info(f"Steps checked: {result.total_steps}")
    _print_info(f"Steps with heat change: {result.heat_changes}")
    _print_info(f"Max heat observed: {result.max_heat_observed}")

    if result.dissipation_errors:
        _print_fail("heat", f"{len(result.dissipation_errors)} errors")
        for e in result.dissipation_errors:
            _print_info(f"  {e}")
        return False

    _print_pass("heat", f"{result.heat_changes} heat changes validated")
    return True


def test_statistical(args, verbose: bool) -> bool:
    """Tier 3: Statistical distribution comparison."""
    from tests.sim_validation.test_statistical import (
        run_sim_games, run_java_games, compare_distributions,
    )
    from tests.sim_validation.collector import collect_game_trace

    n_games = 10  # Balance between statistical power and runtime

    _print_info(f"Running {n_games} sim games...")
    t0 = time.time()
    sim_games = run_sim_games(n_games)
    sim_time = time.time() - t0
    _print_info(f"  Sim: {sim_time:.1f}s ({sim_time/n_games:.1f}s/game)")

    _print_info(f"Running {n_games} Java games...")
    t0 = time.time()
    java_games = run_java_games(
        collect_game_trace, n_games,
        megamek_dir=args.megamek_dir,
        base_port=args.port,
    )
    java_time = time.time() - t0
    _print_info(f"  Java: {java_time:.1f}s ({java_time/n_games:.1f}s/game)")

    comparison = compare_distributions(sim_games, java_games)

    all_ok = True
    for metric, data in comparison.items():
        if isinstance(data, dict):
            status = "ok" if data.get("ok") else "MISMATCH"
            detail = " | ".join(f"{k}={v}" for k, v in data.items() if k != "ok")
            _print_info(f"  {metric}: [{status}] {detail}")
            if not data.get("ok"):
                all_ok = False

    if all_ok:
        _print_pass("statistical", "All distributions within tolerance")
    else:
        _print_fail("statistical", "Some distributions differ significantly")

    return all_ok


# ---------------------------------------------------------------------------
# Main orchestrator
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Validate Python sim against Java MegaMek")
    parser.add_argument("--megamek-dir", default="../megamek", help="Path to MegaMek checkout")
    parser.add_argument("--port", type=int, default=9999, help="Base TCP port")
    parser.add_argument("--tier", type=int, choices=[1, 2, 3], help="Run only this tier")
    parser.add_argument("--only", type=str, help="Run only this test (e.g., legal_moves)")
    parser.add_argument("--verbose", action="store_true", help="Detailed mismatch output")
    parser.add_argument("--max-rounds", type=int, default=10,
                        help="Max game rounds for Java trace (default 10)")
    parser.add_argument("--random-actions", action="store_true",
                        help="Use random actions instead of stand-still (better for damage/heat)")
    args = parser.parse_args()

    # Filter tests
    tests_to_run = []
    for name, tier, desc in ALL_TESTS:
        if args.only and name != args.only:
            continue
        if args.tier and tier != args.tier:
            continue
        tests_to_run.append((name, tier, desc))

    if not tests_to_run:
        print(f"No tests match filters (tier={args.tier}, only={args.only})")
        sys.exit(1)

    print(f"Sim Validation: {len(tests_to_run)} tests to run")
    print(f"MegaMek dir: {args.megamek_dir}")
    print(f"Port: {args.port}")

    # Collect Java game trace (shared by Tier 1 and 2 tests)
    needs_java_trace = any(
        name not in ("to_hit_components", "statistical")
        for name, _, _ in tests_to_run
    )

    trace = None
    if needs_java_trace:
        print(f"\nCollecting Java game trace ({args.max_rounds} rounds)...")
        from tests.sim_validation.collector import collect_game_trace, random_action_fn

        action_fn = random_action_fn if args.random_actions else None
        t0 = time.time()
        trace = collect_game_trace(
            megamek_dir=args.megamek_dir,
            port=args.port,
            max_rounds=args.max_rounds,
            action_fn=action_fn,
        )
        elapsed = time.time() - t0
        print(f"  Collected {len(trace.steps)} steps in {elapsed:.1f}s")
        if trace.steps:
            last = trace.steps[-1].raw_obs
            print(f"  Game ended: round={last.get('round')}, "
                  f"outcome={last.get('game_outcome', 'in_progress')}")

    # Run tests
    passed = 0
    failed = 0
    errors = 0

    test_dispatch = {
        "board": lambda: test_board(trace, args.verbose),
        "unit_template": lambda: test_unit_template(trace, args.verbose),
        "los": lambda: test_los(trace, args.verbose),
        "legal_moves": lambda: test_legal_moves(trace, args.verbose),
        "distances": lambda: test_distances(trace, args.verbose),
        "to_hit_components": lambda: test_to_hit_components(trace, args.verbose),
        "damage": lambda: test_damage(trace, args.verbose),
        "heat": lambda: test_heat(trace, args.verbose),
        "statistical": lambda: test_statistical(args, args.verbose),
    }

    for name, tier, desc in tests_to_run:
        _print_header(name, desc, tier)
        try:
            t0 = time.time()
            ok = test_dispatch[name]()
            elapsed = time.time() - t0
            _print_info(f"({elapsed:.1f}s)")
            if ok:
                passed += 1
            else:
                failed += 1
        except Exception as e:
            _print_fail(name, f"ERROR: {e}")
            if args.verbose:
                traceback.print_exc()
            errors += 1

    # Summary
    print(f"\n{'='*60}")
    print(f"  SUMMARY: {passed} passed, {failed} failed, {errors} errors")
    print(f"{'='*60}")

    sys.exit(0 if failed == 0 and errors == 0 else 1)


if __name__ == "__main__":
    main()
