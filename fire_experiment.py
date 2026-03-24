#!/usr/bin/env python3
"""Measure fire ignition and smoke frequency with fire mechanics enabled.

Requires tacops_start_fire and woods_burn_down to be set to true in
RLGameRunner.java (normally disabled for training).

Usage:
    poetry run python fire_experiment.py --megamek-dir ../megamek --games 20
"""

import argparse
import re
import time

import gymnasium  # noqa: F401
import numpy as np

import megamek_gym  # noqa: F401
from megamek_gym.config import MegaMekConfig


def count_terrain(hexes: list[dict], pattern: str) -> set[tuple[int, int]]:
    """Return set of (x, y) coords where terrain string matches pattern."""
    result = set()
    for h in hexes:
        terrain = h.get("terrain", "")
        if pattern in terrain:
            result.add((h["x"], h["y"]))
    return result


def main():
    parser = argparse.ArgumentParser(description="Fire/smoke frequency experiment")
    parser.add_argument("--games", type=int, default=20)
    parser.add_argument("--config", type=str, default="configs/sanity_check.yaml")
    parser.add_argument("--megamek-dir", type=str, default="../megamek")
    parser.add_argument("--port", type=int, default=None)
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    config = MegaMekConfig.load(args.config)
    config.megamek_dir = args.megamek_dir
    if args.port is not None:
        config.rl_port = args.port

    print(f"Fire Experiment: {args.games} games")
    print(f"  Unit:  {config.rl_unit} vs {config.opponent_unit}")
    print(f"  Board: {config.board}")
    print(f"  Max rounds: {config.max_game_rounds}")
    print()

    env = gymnasium.make("MegaMekGym/MegaMek-v0", config=config)

    # Per-game stats
    game_stats = []

    for game_idx in range(args.games):
        obs, info = env.reset()
        t0 = time.time()

        # Track per-round board state
        fire_hexes_by_round = {}  # round -> set of (x,y)
        smoke_hexes_by_round = {}
        peak_fire = 0
        peak_smoke = 0
        first_fire_round = None
        first_smoke_round = None
        total_steps = 0
        game_outcome = None

        while True:
            n_legal = info.get("n_legal_moves", 0)
            action = np.random.randint(0, max(1, n_legal))

            obs, reward, terminated, truncated, info = env.step(action)
            total_steps += 1

            # Inspect board for fire/smoke
            raw = env.unwrapped._last_raw_obs
            if raw and "board" in raw:
                hexes = raw["board"].get("hexes", [])
                rnd = raw.get("round", 0)

                fire_coords = count_terrain(hexes, "Fire")
                smoke_coords = count_terrain(hexes, "Smoke")

                if fire_coords:
                    fire_hexes_by_round[rnd] = fire_coords
                    peak_fire = max(peak_fire, len(fire_coords))
                    if first_fire_round is None:
                        first_fire_round = rnd

                if smoke_coords:
                    smoke_hexes_by_round[rnd] = smoke_coords
                    peak_smoke = max(peak_smoke, len(smoke_coords))
                    if first_smoke_round is None:
                        first_smoke_round = rnd

                if args.verbose and (fire_coords or smoke_coords):
                    print(f"  Game {game_idx+1} R{rnd}: "
                          f"fire={len(fire_coords)} smoke={len(smoke_coords)} "
                          f"fire_hexes={sorted(fire_coords)} "
                          f"smoke_hexes={sorted(smoke_coords)}")

            if terminated or truncated:
                game_outcome = info.get("game_outcome", 0)
                break

        elapsed = time.time() - t0
        has_fire = first_fire_round is not None
        has_smoke = first_smoke_round is not None

        # Count total unique fire/smoke hexes across all rounds
        all_fire_hexes = set()
        for coords in fire_hexes_by_round.values():
            all_fire_hexes.update(coords)
        all_smoke_hexes = set()
        for coords in smoke_hexes_by_round.values():
            all_smoke_hexes.update(coords)

        stats = {
            "game": game_idx + 1,
            "outcome": game_outcome,
            "rounds": max(fire_hexes_by_round.keys(), default=0)
                      if fire_hexes_by_round else info.get("round", 0),
            "steps": total_steps,
            "has_fire": has_fire,
            "has_smoke": has_smoke,
            "first_fire_round": first_fire_round,
            "first_smoke_round": first_smoke_round,
            "peak_fire_hexes": peak_fire,
            "peak_smoke_hexes": peak_smoke,
            "unique_fire_hexes": len(all_fire_hexes),
            "unique_smoke_hexes": len(all_smoke_hexes),
            "fire_rounds": len(fire_hexes_by_round),
            "smoke_rounds": len(smoke_hexes_by_round),
            "elapsed": elapsed,
        }
        game_stats.append(stats)

        fire_str = (f"fire R{first_fire_round}+ peak={peak_fire} unique={len(all_fire_hexes)}"
                    if has_fire else "no fire")
        smoke_str = (f"smoke R{first_smoke_round}+ peak={peak_smoke} unique={len(all_smoke_hexes)}"
                     if has_smoke else "no smoke")
        print(f"Game {game_idx+1:2d}/{args.games}: "
              f"outcome={game_outcome} steps={total_steps:3d} {elapsed:.1f}s | "
              f"{fire_str} | {smoke_str}")

    env.close()

    # Summary
    n = len(game_stats)
    games_with_fire = sum(1 for s in game_stats if s["has_fire"])
    games_with_smoke = sum(1 for s in game_stats if s["has_smoke"])

    print(f"\n{'='*70}")
    print(f"SUMMARY ({n} games)")
    print(f"{'='*70}")
    print(f"Games with fire:  {games_with_fire}/{n} ({100*games_with_fire/n:.0f}%)")
    print(f"Games with smoke: {games_with_smoke}/{n} ({100*games_with_smoke/n:.0f}%)")

    if games_with_fire > 0:
        fire_games = [s for s in game_stats if s["has_fire"]]
        avg_first = np.mean([s["first_fire_round"] for s in fire_games])
        avg_peak = np.mean([s["peak_fire_hexes"] for s in fire_games])
        avg_unique = np.mean([s["unique_fire_hexes"] for s in fire_games])
        max_peak = max(s["peak_fire_hexes"] for s in fire_games)
        max_unique = max(s["unique_fire_hexes"] for s in fire_games)
        print(f"\nFire stats (games with fire only):")
        print(f"  First fire round:  avg={avg_first:.1f} "
              f"range=[{min(s['first_fire_round'] for s in fire_games)}, "
              f"{max(s['first_fire_round'] for s in fire_games)}]")
        print(f"  Peak fire hexes:   avg={avg_peak:.1f} max={max_peak}")
        print(f"  Unique fire hexes: avg={avg_unique:.1f} max={max_unique}")

    if games_with_smoke > 0:
        smoke_games = [s for s in game_stats if s["has_smoke"]]
        avg_first = np.mean([s["first_smoke_round"] for s in smoke_games])
        avg_peak = np.mean([s["peak_smoke_hexes"] for s in smoke_games])
        avg_unique = np.mean([s["unique_smoke_hexes"] for s in smoke_games])
        max_peak = max(s["peak_smoke_hexes"] for s in smoke_games)
        max_unique = max(s["unique_smoke_hexes"] for s in smoke_games)
        print(f"\nSmoke stats (games with smoke only):")
        print(f"  First smoke round:  avg={avg_first:.1f} "
              f"range=[{min(s['first_smoke_round'] for s in smoke_games)}, "
              f"{max(s['first_smoke_round'] for s in smoke_games)}]")
        print(f"  Peak smoke hexes:   avg={avg_peak:.1f} max={max_peak}")
        print(f"  Unique smoke hexes: avg={avg_unique:.1f} max={max_unique}")

    # Overall board impact
    total_board_hexes = config.resolved_board_width * config.resolved_board_height
    if games_with_fire > 0:
        max_pct = max(s["peak_fire_hexes"] for s in game_stats) / total_board_hexes * 100
        print(f"\nBoard impact: max {max_pct:.1f}% of hexes on fire at once "
              f"({total_board_hexes} total hexes)")

    # Win/loss breakdown
    outcomes = [s["outcome"] for s in game_stats]
    wins = outcomes.count(1)
    losses = outcomes.count(-1)
    draws = outcomes.count(0)
    print(f"\nOutcomes: W={wins} L={losses} D={draws}")

    print(f"\nTotal time: {sum(s['elapsed'] for s in game_stats):.0f}s")


if __name__ == "__main__":
    main()
