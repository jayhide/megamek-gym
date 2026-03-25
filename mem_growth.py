"""Track JVM memory growth over many games.

Runs a single JVM through N games via persistent reset, capturing RSS and heap
usage after each game. Parses Java-side [rl-mem] log lines to produce a
time-series showing how memory grows game-over-game.

Usage:
    # Basic: 30 games
    poetry run python mem_growth.py --megamek-dir ../megamek

    # Quick test
    poetry run python mem_growth.py --megamek-dir ../megamek --num-games 5 --verbose
"""

from __future__ import annotations

import argparse
import os
import re
import signal
import sys
from pathlib import Path

import numpy as np

from megamek_gym.config import MegaMekConfig
from megamek_gym.env import MegaMekEnv
from mem_benchmark import read_rss_mb, run_episode


def parse_mem_log(log_path: Path) -> list[dict]:
    """Parse [rl-mem] post-game lines from Java log.

    Returns list of dicts with keys: game, used_mb, committed_mb, max_mb, delta_mb.
    delta_mb is None for the first game.
    """
    if not log_path.exists():
        return []
    results = []
    for line in log_path.read_text().splitlines():
        m = re.search(
            r"\[rl-mem\] game=(\d+) post: used=(\d+)MB committed=(\d+)MB max=(\d+)MB"
            r"(?: delta=([+-]?\d+)MB)?",
            line,
        )
        if m:
            results.append({
                "game": int(m.group(1)),
                "used_mb": int(m.group(2)),
                "committed_mb": int(m.group(3)),
                "max_mb": int(m.group(4)),
                "delta_mb": int(m.group(5)) if m.group(5) else None,
            })
    return results


def run_growth_benchmark(
    megamek_dir: str,
    num_games: int,
    port: int,
    max_game_rounds: int,
    verbose: bool,
) -> list[dict]:
    """Run num_games on a single JVM, collecting per-game metrics.

    Returns list of dicts, one per game, with keys:
        game, rss_mb, steps, heap_used_mb, heap_delta_mb
    """
    megamek_dir_abs = os.path.abspath(megamek_dir)
    log_path = Path(megamek_dir_abs) / f"rl_java_{port}.log"

    config = MegaMekConfig(
        megamek_dir=megamek_dir_abs,
        rl_port=port,
        max_game_rounds=max_game_rounds,
        firing_strategy="naive",
        max_rotating_round_saves=0,
        force_gc=True,
        mem_log=1,
    )

    env = MegaMekEnv(config=config)
    rss_readings = []  # (game_number, rss_mb, steps)

    try:
        # First game: cold start
        print(f"  Starting env on port {port}...", end=" ", flush=True)
        env.reset()
        print("ok")

        for game_num in range(1, num_games + 1):
            if verbose:
                print(f"  Game {game_num}/{num_games}...", end=" ", flush=True)
            steps = run_episode(env)

            # Capture RSS while JVM is alive (just finished game, before reset)
            pid = env._java._process.pid
            rss = read_rss_mb(pid) or 0.0

            rss_readings.append((game_num, rss, steps))
            if verbose:
                print(f"{steps} steps, RSS={rss:.0f} MB")

            # Reset for next game (persistent reset, JVM stays alive)
            if game_num < num_games:
                env.reset()

    finally:
        env.close()

    # Parse Java logs for heap data
    mem_entries = parse_mem_log(log_path)

    # Index mem entries by game number
    mem_by_game = {e["game"]: e for e in mem_entries}

    # Build merged results
    results = []
    for i, (game_num, rss, steps) in enumerate(rss_readings):
        row = {
            "game": game_num,
            "rss_mb": rss,
            "steps": steps,
            "heap_used_mb": None,
            "heap_delta_mb": None,
        }

        # Merge heap data
        mem = mem_by_game.get(game_num)
        if mem:
            row["heap_used_mb"] = mem["used_mb"]
            row["heap_delta_mb"] = mem["delta_mb"]

        results.append(row)

    return results


def fmt_or_dash(val, fmt="{:>7.0f}"):
    """Format a value or return dashes if None."""
    if val is None:
        return "      -"
    return fmt.format(val)


def fmt_delta(val):
    """Format a delta value with +/- prefix or dashes if None."""
    if val is None:
        return "      -"
    return f"{val:>+7d}"


def print_timeseries(results: list[dict], label: str) -> None:
    """Print a formatted table of per-game metrics."""
    print(f"\n{'=' * 72}")
    print(f"  {label}")
    print(f"{'=' * 72}")

    header = f"{'Game':>4}  {'RSS':>7}  {'Heap':>7}  {'Delta':>7}  {'Steps':>5}"

    print(header)
    print("-" * len(header))

    for r in results:
        parts = [
            f"{r['game']:>4}",
            f"{r['rss_mb']:>5.0f}MB",
            fmt_or_dash(r["heap_used_mb"], "{:>5.0f}MB"),
            fmt_delta(r["heap_delta_mb"]),
            f"{r['steps']:>5}",
        ]
        print("  ".join(parts))

    # Summary
    if results:
        rss_growth = results[-1]["rss_mb"] - results[0]["rss_mb"]
        heap_first = results[0].get("heap_used_mb")
        heap_last = results[-1].get("heap_used_mb")
        heap_growth = (heap_last - heap_first) if (heap_first and heap_last) else None

        print()
        print(f"  RSS growth: {rss_growth:+.0f} MB over {len(results)} games")
        if heap_growth is not None:
            print(f"  Heap growth: {heap_growth:+d} MB over {len(results)} games")
    print()


def main():
    parser = argparse.ArgumentParser(
        description="Track JVM memory growth over many games"
    )
    parser.add_argument("--megamek-dir", default="../megamek")
    parser.add_argument("--num-games", type=int, default=30,
                        help="Number of games to play (default: 30)")
    parser.add_argument("--port", type=int, default=9999)
    parser.add_argument("--max-game-rounds", type=int, default=10,
                        help="Max rounds per game before truncation (default: 10)")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    signal.signal(signal.SIGINT, lambda *_: (print("\nInterrupted."), sys.exit(1)))

    print("=" * 72)
    print("  MegaMek Memory Growth Tracker")
    print("=" * 72)
    print(f"Games: {args.num_games} | Max rounds: {args.max_game_rounds} | "
          f"Force GC: on | Mem log: 1")

    results = run_growth_benchmark(
        args.megamek_dir, args.num_games, args.port,
        max_game_rounds=args.max_game_rounds,
        verbose=args.verbose,
    )
    print_timeseries(results, f"Memory Growth: {args.num_games} games")


if __name__ == "__main__":
    main()
