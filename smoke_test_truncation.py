#!/usr/bin/env python3
"""Smoke test for max_game_rounds truncation.

The RL bot always picks action 0 (stand still). With a low max_game_rounds
limit, the game should be truncated by Java before either unit is destroyed.

Usage:
    poetry run python smoke_test_truncation.py --megamek-dir ../megamek
"""

import argparse
import sys
import time

import gymnasium  # noqa: F401
import megamek_gym  # noqa: F401
from megamek_gym.config import MegaMekConfig


def main():
    parser = argparse.ArgumentParser(description="Truncation smoke test")
    parser.add_argument("--megamek-dir", type=str, default="../megamek")
    parser.add_argument("--port", type=int, default=9999)
    parser.add_argument("--max-game-rounds", type=int, default=3,
                        help="Round limit to test truncation (default: 3)")
    parser.add_argument("--episodes", type=int, default=2,
                        help="Number of episodes to run (tests persistent reset after truncation)")
    args = parser.parse_args()

    config = MegaMekConfig(
        megamek_dir=args.megamek_dir,
        rl_port=args.port,
        max_game_rounds=args.max_game_rounds,
        max_rotating_round_saves=0,
        firing_strategy="naive",
    )

    print(f"Truncation smoke test: max_game_rounds={config.max_game_rounds}")
    print(f"  RL unit:       {config.rl_unit}")
    print(f"  Opponent unit: {config.opponent_unit}")
    print(f"  Port:          {config.rl_port}")
    print(f"  Episodes:      {args.episodes}")

    env = gymnasium.make("MegaMekGym/MegaMek-v0", config=config)

    for ep in range(args.episodes):
        print(f"\n{'='*60}")
        print(f"Episode {ep + 1}/{args.episodes}")
        print(f"{'='*60}")

        obs, info = env.reset()
        print(f"Reset complete. Legal moves: {info.get('n_legal_moves', 0)}")

        step = 0
        total_reward = 0.0
        terminated = False
        truncated = False
        t0 = time.time()

        while True:
            # Always stand still (action 0)
            obs, reward, terminated, truncated, info = env.step(0)
            step += 1
            total_reward += reward
            game_round = info.get("round", "?")
            phase = info.get("phase", "?")

            print(
                f"  Step {step:3d} r={game_round:>2} {phase:<14s} "
                f"reward={reward:+.3f} moves={info.get('n_legal_moves', 0):>4d}"
            )

            if terminated or truncated:
                break

        elapsed = time.time() - t0
        print(f"\nEpisode {ep + 1} finished: {step} steps, {elapsed:.1f}s")
        print(f"  terminated={terminated}, truncated={truncated}")
        print(f"  total_reward={total_reward:+.4f}")
        print(f"  final_round={info.get('round', '?')}")
        print(f"  game_outcome={info.get('game_outcome', '?')}")

        if not truncated:
            print(f"\nFAIL: Expected truncated=True but got truncated={truncated}")
            if terminated:
                print("  Game ended via termination (unit destroyed) before round limit hit.")
                print("  Try increasing --max-game-rounds.")
            env.close()
            sys.exit(1)

        print(f"  PASS: Game truncated at round limit as expected")

    env.close()
    print(f"\n{'='*60}")
    print(f"All {args.episodes} episodes truncated correctly. Smoke test passed.")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
