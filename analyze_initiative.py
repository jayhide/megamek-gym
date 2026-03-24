#!/usr/bin/env python3
"""Analyze initiative (move-first) distribution across multiple games.

Usage:
    poetry run python analyze_initiative.py --megamek-dir ../megamek --games 5
    poetry run python analyze_initiative.py --config configs/sanity_check.yaml --games 10
"""

import argparse
import math

import gymnasium  # noqa: F401
import numpy as np

import megamek_gym  # noqa: F401
from megamek_gym.config import MegaMekConfig


def main():
    parser = argparse.ArgumentParser(description="Initiative analysis")
    parser.add_argument("--config", type=str, default="configs/sanity_check.yaml")
    parser.add_argument("--megamek-dir", type=str, default=None)
    parser.add_argument("--port", type=int, default=None)
    parser.add_argument("--games", type=int, default=5)
    args = parser.parse_args()

    config = MegaMekConfig.load(args.config)
    if args.megamek_dir:
        config.megamek_dir = args.megamek_dir
    if args.port:
        config.rl_port = args.port

    print(f"Config: {args.config}")
    print(f"Unit: {config.rl_unit} vs {config.opponent_unit}")
    print(f"Games: {args.games}\n")

    env = gymnasium.make("MegaMekGym/MegaMek-v0", config=config)
    inner = env.unwrapped

    all_rounds = []  # list of (game_idx, round_num, rl_moves_first)
    game_outcomes = []

    for game in range(args.games):
        obs, info = env.reset()
        last_round = -1
        game_initiative = []  # (round, rl_first) for this game

        while True:
            # Record initiative for each new round
            raw = inner._last_raw_obs
            rnd = raw.get("round", 0)
            if rnd != last_round:
                rl_first = raw.get("rl_moves_first", False)
                game_initiative.append((rnd, rl_first))
                last_round = rnd

            # Random action
            n_legal = info.get("n_legal_moves", 0)
            action = np.random.randint(0, max(1, n_legal))
            obs, reward, terminated, truncated, info = env.step(action)

            if terminated or truncated:
                outcome = info.get("game_outcome", 0)
                outcome_str = {1: "WIN", -1: "LOSS", 0: "DRAW"}[outcome]
                game_outcomes.append(outcome_str)

                rl_first_count = sum(1 for _, f in game_initiative if f)
                total = len(game_initiative)
                print(
                    f"Game {game+1}: {total} rounds, "
                    f"RL first {rl_first_count}/{total} "
                    f"({100*rl_first_count/total:.0f}%), "
                    f"outcome={outcome_str}"
                )
                for rnd, rl_first in game_initiative:
                    all_rounds.append((game + 1, rnd, rl_first))
                break

    env.close()

    # Aggregate stats
    total = len(all_rounds)
    rl_first_total = sum(1 for _, _, f in all_rounds if f)
    princess_first_total = total - rl_first_total

    pct = 100 * rl_first_total / total if total > 0 else 0
    # Wilson score 95% CI
    z = 1.96
    p_hat = rl_first_total / total if total > 0 else 0.5
    denom = 1 + z**2 / total
    center = (p_hat + z**2 / (2 * total)) / denom
    spread = z * math.sqrt((p_hat * (1 - p_hat) + z**2 / (4 * total)) / total) / denom
    ci_lo, ci_hi = max(0, center - spread), min(1, center + spread)

    print(f"\n{'='*50}")
    print(f"INITIATIVE SUMMARY ({total} rounds across {args.games} games)")
    print(f"{'='*50}")
    print(f"RL moves first:       {rl_first_total:3d} ({pct:.1f}%)")
    print(f"Princess moves first: {princess_first_total:3d} ({100-pct:.1f}%)")
    print(f"95% CI for RL-first:  [{100*ci_lo:.1f}%, {100*ci_hi:.1f}%]")
    print(f"\nGame outcomes: {', '.join(game_outcomes)}")

    wins = game_outcomes.count("WIN")
    losses = game_outcomes.count("LOSS")
    draws = game_outcomes.count("DRAW")
    print(f"W/L/D: {wins}/{losses}/{draws}")


if __name__ == "__main__":
    main()
