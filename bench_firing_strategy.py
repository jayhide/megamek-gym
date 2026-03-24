#!/usr/bin/env python3
"""Benchmark firing strategy latency: naive vs princess.

Runs several games with each firing strategy using Java-side perf logging,
then parses the logs and reports a comparison.

Usage:
    poetry run python bench_firing_strategy.py --megamek-dir ../megamek --games 3
"""

import argparse
import os
import re
import time

import gymnasium  # noqa: F401
import numpy as np

import megamek_gym  # noqa: F401
from megamek_gym.config import MegaMekConfig


def run_games(config: MegaMekConfig, n_games: int) -> list[dict]:
    """Run n_games with random actions, return per-game timing info."""
    env = gymnasium.make("MegaMekGym/MegaMek-v0", config=config)
    game_results = []

    for game_num in range(n_games):
        obs, info = env.reset()
        steps = 0
        t0 = time.monotonic()

        while True:
            n_legal = info.get("n_legal_moves", 0)
            action = np.random.randint(0, max(n_legal, 1))
            obs, reward, terminated, truncated, info = env.step(action)
            steps += 1

            if terminated or truncated:
                elapsed = time.monotonic() - t0
                outcome = info.get("game_outcome", "?")
                print(f"  Game {game_num + 1}/{n_games}: {steps} steps, "
                      f"{elapsed:.1f}s, outcome={outcome}")
                game_results.append({"steps": steps, "elapsed": elapsed})
                break

    env.close()
    return game_results


def parse_perf_log(log_path: str) -> dict:
    """Parse [rl-perf] lines from a Java log file."""
    fire_times = []
    move_times = []
    game_summaries = []

    if not os.path.exists(log_path):
        print(f"  Warning: log file not found: {log_path}")
        return {"fire_times": [], "move_times": [], "game_summaries": []}

    with open(log_path, "r") as f:
        for line in f:
            # Per-fire-turn: [rl-perf] fire_turn #N: Xms
            m = re.search(r"\[rl-perf\] fire_turn #\d+: ([\d.]+)ms", line)
            if m:
                fire_times.append(float(m.group(1)))
                continue

            # Per-move-turn: [rl-perf] move_turn #N: total=Xms ...
            m = re.search(r"\[rl-perf\] move_turn #\d+: total=([\d.]+)ms", line)
            if m:
                move_times.append(float(m.group(1)))
                continue

            # Game summary
            m = re.search(
                r"\[rl-perf\] game_summary: total=([\d.]+)ms.*"
                r"avg_move=([\d.]+)ms.*avg_fire=([\d.]+)ms",
                line,
            )
            if m:
                game_summaries.append({
                    "total_ms": float(m.group(1)),
                    "avg_move_ms": float(m.group(2)),
                    "avg_fire_ms": float(m.group(3)),
                })

    return {
        "fire_times": fire_times,
        "move_times": move_times,
        "game_summaries": game_summaries,
    }


def print_strategy_stats(name: str, game_results: list[dict], perf: dict):
    """Print stats for one strategy."""
    print(f"\n{'=' * 50}")
    print(f"  {name.upper()} firing strategy")
    print(f"{'=' * 50}")

    if game_results:
        elapsed = [g["elapsed"] for g in game_results]
        steps = [g["steps"] for g in game_results]
        print(f"  Games:          {len(game_results)}")
        print(f"  Avg game time:  {np.mean(elapsed):.1f}s "
              f"(std={np.std(elapsed):.1f}s)")
        print(f"  Avg steps/game: {np.mean(steps):.0f}")

    if perf["fire_times"]:
        ft = perf["fire_times"]
        print(f"  Fire turns:     {len(ft)}")
        print(f"  Avg fire time:  {np.mean(ft):.1f}ms "
              f"(median={np.median(ft):.1f}ms, "
              f"p95={np.percentile(ft, 95):.1f}ms)")

    if perf["move_times"]:
        mt = perf["move_times"]
        print(f"  Move turns:     {len(mt)}")
        print(f"  Avg move time:  {np.mean(mt):.1f}ms "
              f"(median={np.median(mt):.1f}ms)")

    if perf["game_summaries"]:
        for i, gs in enumerate(perf["game_summaries"]):
            print(f"  Game {i + 1} summary: total={gs['total_ms']:.0f}ms "
                  f"avg_move={gs['avg_move_ms']:.1f}ms "
                  f"avg_fire={gs['avg_fire_ms']:.1f}ms")


def main():
    parser = argparse.ArgumentParser(
        description="Benchmark naive vs princess firing strategy latency")
    parser.add_argument("--megamek-dir", type=str, default="../megamek")
    parser.add_argument("--games", type=int, default=3,
                        help="Number of games per strategy")
    parser.add_argument("--config", type=str, default="configs/sanity_check.yaml",
                        help="Base config to use")
    parser.add_argument("--port", type=int, default=9999)
    args = parser.parse_args()

    strategies = ["naive", "princess"]
    all_results = {}

    for strategy in strategies:
        print(f"\n{'#' * 60}")
        print(f"# Running {args.games} games with firing_strategy={strategy}")
        print(f"{'#' * 60}")

        config = MegaMekConfig.load(args.config)
        config.megamek_dir = args.megamek_dir
        config.rl_port = args.port
        config.firing_strategy = strategy
        config.perf_log = True

        log_path = os.path.join(args.megamek_dir, f"rl_java_{args.port}.log")
        # Delete old log so we only parse this run's data
        if os.path.exists(log_path):
            os.remove(log_path)

        game_results = run_games(config, args.games)
        perf = parse_perf_log(log_path)
        print_strategy_stats(strategy, game_results, perf)
        all_results[strategy] = {"games": game_results, "perf": perf}

    # Comparison
    print(f"\n{'#' * 60}")
    print("# COMPARISON")
    print(f"{'#' * 60}")

    for metric, key in [("Fire turn", "fire_times"), ("Move turn", "move_times")]:
        naive_vals = all_results["naive"]["perf"][key]
        princess_vals = all_results["princess"]["perf"][key]
        if naive_vals and princess_vals:
            n_avg = np.mean(naive_vals)
            p_avg = np.mean(princess_vals)
            diff = p_avg - n_avg
            pct = (diff / n_avg * 100) if n_avg > 0 else 0
            print(f"  {metric}: naive={n_avg:.1f}ms  princess={p_avg:.1f}ms  "
                  f"delta={diff:+.1f}ms ({pct:+.0f}%)")

    naive_games = all_results["naive"]["games"]
    princess_games = all_results["princess"]["games"]
    if naive_games and princess_games:
        n_avg = np.mean([g["elapsed"] for g in naive_games])
        p_avg = np.mean([g["elapsed"] for g in princess_games])
        diff = p_avg - n_avg
        pct = (diff / n_avg * 100) if n_avg > 0 else 0
        print(f"  Game time: naive={n_avg:.1f}s  princess={p_avg:.1f}s  "
              f"delta={diff:+.1f}s ({pct:+.0f}%)")


if __name__ == "__main__":
    main()
