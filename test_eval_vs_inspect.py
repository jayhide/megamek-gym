"""Compare eval.py and inspect_games.py random-agent behavior.

Runs both approaches with identical configs to determine if there's a
systematic difference, or if the user's 0/50 vs 1/1 result was just variance.

Usage:
    poetry run python test_eval_vs_inspect.py --megamek-dir ../megamek --num-games 20
"""

import argparse
import random
import time

import numpy as np
import gymnasium as gym

from megamek_gym.agent import OUTCOME_MAP
from megamek_gym.config import MegaMekConfig


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--megamek-dir", type=str, default="../megamek")
    parser.add_argument("--config", type=str, default=None)
    parser.add_argument("--port", type=int, default=9999)
    parser.add_argument("--num-games", type=int, default=20)
    parser.add_argument("--seed", type=int, default=1)
    return parser.parse_args()


def run_random_game(env):
    """Play one game with uniform random actions. Returns (outcome, steps, return)."""
    obs, info = env.reset()
    done = False
    episode_return = 0.0
    steps = 0

    while not done:
        n_legal = info.get("n_legal_moves", 1)
        action = np.random.randint(0, max(n_legal, 1))
        obs, reward, terminated, truncated, info = env.step(action)
        episode_return += reward
        steps += 1
        done = terminated or truncated

    outcome = OUTCOME_MAP.get(info.get("game_outcome", 0), "DRAW")
    return outcome, steps, episode_return


def main():
    args = parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)

    # Test with default MegaMekConfig (what eval.py uses without --config)
    default_cfg = MegaMekConfig()
    default_cfg.megamek_dir = args.megamek_dir
    default_cfg.rl_port = args.port
    default_cfg.env_index = 0

    # Test with configs/default.yaml (what inspect_games.py likely uses)
    if args.config:
        yaml_cfg = MegaMekConfig.load(args.config)
    else:
        yaml_cfg = MegaMekConfig.load("configs/default.yaml")
    yaml_cfg.megamek_dir = args.megamek_dir
    yaml_cfg.rl_port = args.port
    yaml_cfg.env_index = 0

    # Print the gameplay-affecting differences
    print("=== Config Comparison (gameplay-affecting fields only) ===")
    print(f"  {'Field':<25} {'MegaMekConfig()':<20} {'default.yaml':<20}")
    print(f"  {'max_game_rounds':<25} {default_cfg.max_game_rounds:<20} {yaml_cfg.max_game_rounds:<20}")
    print(f"  {'firing_strategy':<25} {default_cfg.firing_strategy:<20} {yaml_cfg.firing_strategy:<20}")
    print(f"  {'rl_unit':<25} {default_cfg.rl_unit:<20} {yaml_cfg.rl_unit:<20}")
    print(f"  {'opponent_unit':<25} {default_cfg.opponent_unit:<20} {yaml_cfg.opponent_unit:<20}")
    print()

    configs_to_test = [
        ("MegaMekConfig() defaults (eval.py style)", default_cfg),
        ("default.yaml (inspect_games.py style)", yaml_cfg),
    ]

    for label, cfg in configs_to_test:
        print(f"=== {label} ===")
        print(f"  max_game_rounds={cfg.max_game_rounds}")

        # Re-seed before each config so results are comparable
        random.seed(args.seed)
        np.random.seed(args.seed)

        env = gym.make("MegaMekGym/MegaMek-v0", config=cfg)

        wins, losses, draws = 0, 0, 0
        returns = []

        for ep in range(1, args.num_games + 1):
            t0 = time.monotonic()
            outcome, steps, ret = run_random_game(env)
            elapsed = time.monotonic() - t0

            if outcome == "WIN":
                wins += 1
            elif outcome == "LOSS":
                losses += 1
            else:
                draws += 1
            returns.append(ret)

            print(f"  Game {ep:3d}: {outcome:4s}  steps={steps:4d}  "
                  f"return={ret:7.2f}  ({elapsed:.1f}s)")

        env.close()

        n = len(returns)
        print(f"\n  Results: {wins}W / {losses}L / {draws}D")
        print(f"  Win rate: {100 * wins / n:.1f}%")
        print(f"  Avg return: {np.mean(returns):.2f} +/- {np.std(returns):.2f}")
        print()


if __name__ == "__main__":
    main()
