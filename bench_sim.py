"""Profile latency and memory of the pure-Python MegaMek simulator.

Monkey-patches sim internals with timing wrappers (zero sim code changes),
runs N games, and reports per-phase latency breakdown + memory usage.

Usage:
    poetry run python bench_sim.py
    poetry run python bench_sim.py --num-games 5 --verbose
    poetry run python bench_sim.py --scaling --env-counts 1,2,4
"""

from __future__ import annotations

import argparse
import os
import sys
import time
import tracemalloc
from collections import defaultdict

import numpy as np


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def read_rss_mb() -> float:
    """Read current process RSS in MB from /proc/self/status."""
    try:
        with open("/proc/self/status") as f:
            for line in f:
                if line.startswith("VmRSS:"):
                    return int(line.split()[1]) / 1024.0
    except (OSError, ValueError):
        pass
    return 0.0


def deep_getsizeof(obj, seen: set | None = None) -> int:
    """Recursively measure object size in bytes."""
    if seen is None:
        seen = set()
    obj_id = id(obj)
    if obj_id in seen:
        return 0
    seen.add(obj_id)
    size = sys.getsizeof(obj)
    if isinstance(obj, dict):
        for k, v in obj.items():
            size += deep_getsizeof(k, seen) + deep_getsizeof(v, seen)
    elif isinstance(obj, (list, tuple, set, frozenset)):
        for item in obj:
            size += deep_getsizeof(item, seen)
    elif hasattr(obj, "__dict__"):
        size += deep_getsizeof(obj.__dict__, seen)
    return size


# ---------------------------------------------------------------------------
# Timing infrastructure
# ---------------------------------------------------------------------------

# Global accumulator: phase_name -> list of elapsed_ns values
_phase_acc: dict[str, list[int]] = defaultdict(list)
# Per-step snapshot buffer
_step_phases: dict[str, int] = {}
_originals: dict[str, object] = {}


def _start_step():
    """Mark start of a new step — snapshot accumulator lengths."""
    global _step_phases
    _step_phases = {k: len(v) for k, v in _phase_acc.items()}


def _end_step() -> dict[str, float]:
    """Collect timings added during this step, return as ms dict."""
    result = {}
    for k, v in _phase_acc.items():
        prev_len = _step_phases.get(k, 0)
        new_entries = v[prev_len:]
        result[k] = sum(new_entries) / 1e6  # ns -> ms
    return result


def _wrap(module, attr_name: str, phase_name: str, owner_split: bool = False):
    """Replace module.attr_name with a timed wrapper.

    If owner_split=True, split into "{phase_name}_rl" and "{phase_name}_opp"
    based on args[0].owner.
    """
    original = getattr(module, attr_name)
    key = f"_orig_{id(module)}_{attr_name}"
    _originals[key] = original

    if owner_split:
        def wrapper(*args, **kwargs):
            t0 = time.perf_counter_ns()
            result = original(*args, **kwargs)
            elapsed = time.perf_counter_ns() - t0
            owner = getattr(args[0], "owner", -1) if args else -1
            suffix = "rl" if owner == 0 else "opp"
            _phase_acc[f"{phase_name}_{suffix}"].append(elapsed)
            return result
    else:
        def wrapper(*args, **kwargs):
            t0 = time.perf_counter_ns()
            result = original(*args, **kwargs)
            elapsed = time.perf_counter_ns() - t0
            _phase_acc[phase_name].append(elapsed)
            return result

    setattr(module, attr_name, wrapper)


def _wrap_method(cls, method_name: str, phase_name: str):
    """Replace a class method with a timed wrapper."""
    original = getattr(cls, method_name)
    key = f"_orig_{id(cls)}_{method_name}"
    _originals[key] = original

    def wrapper(self, *args, **kwargs):
        t0 = time.perf_counter_ns()
        result = original(self, *args, **kwargs)
        elapsed = time.perf_counter_ns() - t0
        _phase_acc[phase_name].append(elapsed)
        return result

    setattr(cls, method_name, wrapper)


def install_hooks():
    """Monkey-patch sim internals with timing wrappers."""
    import megamek_gym.sim.game as game_mod
    import megamek_gym.sim.env as env_mod
    from megamek_gym.reward import CompositeReward

    # Game-internal functions (patched on game module where they're imported)
    _wrap(game_mod, "enumerate_moves", "enumerate", owner_split=True)
    _wrap(game_mod, "select_move", "princess")
    _wrap(game_mod, "resolve_firing", "firing")
    _wrap(game_mod, "apply_heat", "apply_heat")
    _wrap(game_mod, "dissipate_heat", "dissipate_heat")
    _wrap(game_mod, "check_overheat", "check_overheat")

    # Observation flattening (patched on env module)
    _wrap(env_mod, "flatten_observation_hierarchical", "flatten_obs")

    # Reward computation (method on class)
    _wrap_method(CompositeReward, "compute", "reward")


def clear_acc():
    """Clear all accumulated timings."""
    _phase_acc.clear()


# ---------------------------------------------------------------------------
# Statistics
# ---------------------------------------------------------------------------

def compute_stats(values: list[float]) -> dict[str, float]:
    """Compute mean, median, p95, p99 from a list of floats."""
    if not values:
        return {"mean": 0, "median": 0, "p95": 0, "p99": 0}
    a = np.array(values)
    return {
        "mean": float(np.mean(a)),
        "median": float(np.median(a)),
        "p95": float(np.percentile(a, 95)),
        "p99": float(np.percentile(a, 99)),
    }


# ---------------------------------------------------------------------------
# Benchmark runner
# ---------------------------------------------------------------------------

def run_benchmark(
    num_games: int,
    seed: int,
    use_tracemalloc: bool,
    verbose: bool,
) -> dict:
    """Run num_games games, return timing and memory data."""
    from megamek_gym.sim.env import MegaMekSimEnv
    from megamek_gym.sim.los import LosTable
    from megamek_gym.sim.board import BOARD

    results: dict = {}

    # --- Memory checkpoint: before env ---
    rss_before = read_rss_mb()

    if use_tracemalloc:
        tracemalloc.start()

    # --- LOS table timing ---
    t0 = time.perf_counter_ns()
    los = LosTable(BOARD)
    los_time_ms = (time.perf_counter_ns() - t0) / 1e6
    los_size_kb = deep_getsizeof(los) / 1024
    results["los_time_ms"] = los_time_ms
    results["los_size_kb"] = los_size_kb
    results["los_entries"] = BOARD.width * BOARD.height * BOARD.width * BOARD.height

    rss_after_los = read_rss_mb()

    # --- Create env ---
    env = MegaMekSimEnv(seed=seed)
    rss_after_env = read_rss_mb()

    # --- Run games ---
    install_hooks()
    clear_acc()

    step_timings: list[dict[str, float]] = []
    reset_times_ms: list[float] = []
    total_steps = 0
    game_lengths: list[int] = []
    max_legal_moves_seen = 0

    # Warm up: first reset triggers LOS table build inside the env's Game.
    # Exclude this from reset timing.
    env.reset(seed=seed)

    wall_start = time.perf_counter_ns()

    for game_i in range(num_games):
        t0 = time.perf_counter_ns()
        obs, info = env.reset(seed=seed + game_i)
        reset_ms = (time.perf_counter_ns() - t0) / 1e6
        reset_times_ms.append(reset_ms)

        steps_this_game = 0
        done = False
        while not done:
            masks = env.action_masks()
            # Pick a random valid action
            dest_mask = masks["dest_mask"]
            valid_dests = np.where(dest_mask)[0]
            if len(valid_dests) == 0:
                dest_idx = 0
            else:
                dest_idx = np.random.choice(valid_dests)
            facing_mask = masks["facing_mask"][dest_idx]
            valid_facings = np.where(facing_mask)[0]
            if len(valid_facings) == 0:
                facing_idx = 0
            else:
                facing_idx = np.random.choice(valid_facings)
            action = np.array([dest_idx, facing_idx])

            _start_step()
            t0 = time.perf_counter_ns()
            obs, reward, terminated, truncated, info = env.step(action)
            step_wall_ms = (time.perf_counter_ns() - t0) / 1e6

            phase_ms = _end_step()
            phase_ms["step_total"] = step_wall_ms
            step_timings.append(phase_ms)

            steps_this_game += 1
            total_steps += 1
            if not (terminated or truncated):
                n_moves = info.get("n_legal_moves", 0)
                if n_moves > max_legal_moves_seen:
                    max_legal_moves_seen = n_moves
            done = terminated or truncated

        game_lengths.append(steps_this_game)
        if verbose:
            outcome = "?"
            raw = env._last_raw_obs
            if raw:
                outcome = raw.get("game_outcome", "?")
            print(f"  Game {game_i+1}/{num_games}: {steps_this_game} steps, outcome={outcome}")

    wall_elapsed_s = (time.perf_counter_ns() - wall_start) / 1e9

    rss_after_games = read_rss_mb()

    # --- Collect results ---
    results["rss_before"] = rss_before
    results["rss_after_los"] = rss_after_los
    results["rss_after_env"] = rss_after_env
    results["rss_after_games"] = rss_after_games

    results["total_steps"] = total_steps
    results["num_games"] = num_games
    results["wall_elapsed_s"] = wall_elapsed_s
    results["sps"] = total_steps / wall_elapsed_s if wall_elapsed_s > 0 else 0
    results["games_per_min"] = num_games / wall_elapsed_s * 60 if wall_elapsed_s > 0 else 0
    results["game_lengths"] = game_lengths

    # Per-phase stats
    phases = [
        "step_total",
        "enumerate_rl",
        "enumerate_opp",
        "princess",
        "firing",
        "apply_heat",
        "dissipate_heat",
        "check_overheat",
        "flatten_obs",
        "reward",
    ]
    phase_stats = {}
    for phase in phases:
        values = [s.get(phase, 0.0) for s in step_timings]
        phase_stats[phase] = compute_stats(values)
    results["phase_stats"] = phase_stats

    # Reset stats
    results["reset_stats"] = compute_stats(reset_times_ms)

    # Object sizes — reset to get a non-terminal state with legal moves
    env.reset(seed=seed)
    game = env._game
    results["obj_sizes"] = {
        "LosTable": deep_getsizeof(game._los_table) / 1024,
        "Board": deep_getsizeof(game.board) / 1024,
        "Unit_rl": deep_getsizeof(game.rl_unit) / 1024,
        "Unit_opp": deep_getsizeof(game.opp_unit) / 1024,
        "legal_moves": deep_getsizeof(game._cached_rl_moves) / 1024,
        "n_legal_moves": len(game._cached_rl_moves),
        "max_legal_moves": max_legal_moves_seen,
    }

    # tracemalloc
    if use_tracemalloc:
        snapshot = tracemalloc.take_snapshot()
        top = snapshot.statistics("filename")[:10]
        results["tracemalloc_top"] = [
            (str(s.traceback), s.size / 1024) for s in top
        ]
        tracemalloc.stop()
    else:
        results["tracemalloc_top"] = []

    return results


# ---------------------------------------------------------------------------
# Scaling benchmark
# ---------------------------------------------------------------------------

def run_scaling(env_counts: list[int], num_games: int, seed: int) -> list[dict]:
    """Run benchmark with varying numbers of in-process envs."""
    from megamek_gym.sim.env import MegaMekSimEnv

    scaling_results = []
    for n_envs in env_counts:
        envs = [MegaMekSimEnv() for _ in range(n_envs)]

        # Reset all
        for i, env in enumerate(envs):
            env.reset(seed=seed + i)

        total_steps = 0
        games_done = [0] * n_envs
        t0 = time.perf_counter_ns()

        while min(games_done) < num_games:
            for i, env in enumerate(envs):
                if games_done[i] >= num_games:
                    continue
                masks = env.action_masks()
                dest_mask = masks["dest_mask"]
                valid_dests = np.where(dest_mask)[0]
                dest_idx = np.random.choice(valid_dests) if len(valid_dests) else 0
                facing_mask = masks["facing_mask"][dest_idx]
                valid_facings = np.where(facing_mask)[0]
                facing_idx = np.random.choice(valid_facings) if len(valid_facings) else 0

                obs, reward, terminated, truncated, info = env.step(
                    np.array([dest_idx, facing_idx])
                )
                total_steps += 1
                if terminated or truncated:
                    games_done[i] += 1
                    if games_done[i] < num_games:
                        env.reset(seed=seed + i + games_done[i] * 100)

        elapsed_s = (time.perf_counter_ns() - t0) / 1e9
        rss = read_rss_mb()
        sps = total_steps / elapsed_s if elapsed_s > 0 else 0

        scaling_results.append({
            "n_envs": n_envs,
            "total_steps": total_steps,
            "elapsed_s": elapsed_s,
            "sps": sps,
            "sps_per_env": sps / n_envs,
            "rss_mb": rss,
        })

        # Clean up
        for env in envs:
            env.close()

        print(f"  {n_envs} envs: {sps:.1f} SPS ({sps/n_envs:.1f}/env), RSS={rss:.1f} MB")

    return scaling_results


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------

def print_report(results: dict, scaling: list[dict] | None = None):
    """Print formatted profiling report."""
    print("=" * 70)
    print("  MegaMek Sim Profiler")
    print(f"  Games: {results['num_games']} | Steps: {results['total_steps']}"
          f" | Wall: {results['wall_elapsed_s']:.1f}s")
    print("=" * 70)

    # LOS table
    print(f"\n--- LOS Table ---")
    print(f"  Precomputation:  {results['los_time_ms']:.1f} ms")
    print(f"  Table size:      {results['los_size_kb']:.0f} KB")

    # Game lengths
    gl = results["game_lengths"]
    print(f"\n--- Game Lengths ---")
    print(f"  Mean: {np.mean(gl):.1f}  Median: {np.median(gl):.0f}"
          f"  Min: {min(gl)}  Max: {max(gl)}")

    # Latency breakdown
    print(f"\n--- Latency Breakdown (per step, ms) ---")
    labels = {
        "step_total":      "env.step() total",
        "enumerate_rl":    "  RL move enum",
        "enumerate_opp":   "  Opp move enum",
        "princess":        "  Princess select",
        "firing":          "  Firing (both)",
        "apply_heat":      "  Apply heat",
        "dissipate_heat":  "  Dissipate heat",
        "check_overheat":  "  Check overheat",
        "flatten_obs":     "  Obs flatten",
        "reward":          "  Reward compute",
    }
    header = f"{'Phase':<24s} {'Mean':>8s} {'Median':>8s} {'p95':>8s} {'p99':>8s}"
    print(header)
    print("-" * len(header))
    for key, label in labels.items():
        s = results["phase_stats"].get(key, {})
        print(f"{label:<24s} {s.get('mean',0):>7.2f}  {s.get('median',0):>7.2f}"
              f"  {s.get('p95',0):>7.2f}  {s.get('p99',0):>7.2f}")

    # Reset
    rs = results["reset_stats"]
    print(f"\n{'env.reset()':<24s} {rs['mean']:>7.2f}  {rs['median']:>7.2f}"
          f"  {rs['p95']:>7.2f}  {rs['p99']:>7.2f}")

    # Throughput
    print(f"\n--- Throughput ---")
    print(f"  Steps/sec:       {results['sps']:.1f}")
    print(f"  Games/min:       {results['games_per_min']:.1f}")

    # Memory
    print(f"\n--- Memory (RSS, MB) ---")
    print(f"  Before env:      {results['rss_before']:.1f}")
    print(f"  After LOS table: {results['rss_after_los']:.1f}")
    print(f"  After env:       {results['rss_after_env']:.1f}")
    print(f"  After {results['num_games']} games:    {results['rss_after_games']:.1f}")

    # Object sizes
    print(f"\n--- Key Object Sizes (KB) ---")
    obj = results["obj_sizes"]
    for name in ("LosTable", "Board", "Unit_rl", "Unit_opp"):
        print(f"  {name:<16s} {obj[name]:>8.1f}")
    print(f"  {'legal_moves':<16s} {obj['legal_moves']:>8.1f}"
          f"  ({obj['n_legal_moves']} moves at reset,"
          f" max {obj['max_legal_moves']} seen)")

    # tracemalloc
    if results["tracemalloc_top"]:
        print(f"\n--- tracemalloc Top 10 (KB) ---")
        for tb, size_kb in results["tracemalloc_top"]:
            # Shorten path
            short = tb.split("/")[-1] if "/" in tb else tb
            print(f"  {short:<40s} {size_kb:>8.1f}")

    # Scaling
    if scaling:
        print(f"\n--- Scaling (in-process) ---")
        header = f"{'Envs':>4s}  {'SPS':>8s}  {'SPS/env':>8s}  {'RSS (MB)':>8s}"
        print(header)
        print("-" * len(header))
        for s in scaling:
            print(f"{s['n_envs']:>4d}  {s['sps']:>8.1f}  {s['sps_per_env']:>8.1f}"
                  f"  {s['rss_mb']:>8.1f}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Profile Python MegaMek sim")
    parser.add_argument("--num-games", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--scaling", action="store_true",
                        help="Run scaling benchmark with multiple envs")
    parser.add_argument("--env-counts", default="1,2,4,8",
                        help="Comma-separated env counts for scaling")
    parser.add_argument("--no-tracemalloc", action="store_true")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    print("Running sim profiler...")
    results = run_benchmark(
        num_games=args.num_games,
        seed=args.seed,
        use_tracemalloc=not args.no_tracemalloc,
        verbose=args.verbose,
    )

    scaling = None
    if args.scaling:
        env_counts = [int(x) for x in args.env_counts.split(",")]
        print(f"\nRunning scaling benchmark: {env_counts} envs...")
        scaling = run_scaling(env_counts, args.num_games, args.seed)

    print()
    print_report(results, scaling)


if __name__ == "__main__":
    main()
