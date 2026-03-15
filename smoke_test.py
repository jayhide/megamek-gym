#!/usr/bin/env python3
"""Smoke test: run random episodes through the MegaMekEnv.

Usage:
    python smoke_test.py [--episodes N] [--megamek-dir PATH] [--config PATH]
"""

import argparse
import time

import gymnasium  # noqa: F401 — triggers env registration
import numpy as np

import megamek_gym  # noqa: F401 — registers MegaMekGym/MegaMek-v0
from megamek_gym.config import MegaMekConfig


def _unit_summary(unit: dict | None, label: str) -> str:
    if unit is None:
        return f"{label}: --"
    x, y = unit.get("x", -1), unit.get("y", -1)
    facing = unit.get("facing", 0)
    heat = unit.get("heat", 0)
    destroyed = unit.get("destroyed", False)

    armor_locs = unit.get("armor", [])
    armor_cur = sum(
        loc.get("armor", 0) + loc.get("rear_armor", 0) for loc in armor_locs
    )
    armor_max = sum(
        loc.get("armor_max", 0) + loc.get("rear_armor_max", 0) for loc in armor_locs
    )
    internal_cur = sum(loc.get("internal", 0) for loc in armor_locs)
    internal_max = sum(loc.get("internal_max", 0) for loc in armor_locs)

    status = "DEAD" if destroyed else "ok"
    return (
        f"{label}: ({x:2d},{y:2d}) f={facing} "
        f"arm={armor_cur}/{armor_max} int={internal_cur}/{internal_max} "
        f"heat={heat} [{status}]"
    )


def main():
    parser = argparse.ArgumentParser(description="MegaMek-Gym smoke test")
    parser.add_argument("--episodes", type=int, default=1)
    parser.add_argument("--config", type=str, default=None,
                        help="Path to YAML config file")
    parser.add_argument("--megamek-dir", type=str, default=None)
    parser.add_argument("--port", type=int, default=None)
    args = parser.parse_args()

    if args.config:
        config = MegaMekConfig.load(args.config)
    else:
        config = MegaMekConfig()

    # Apply CLI overrides
    if args.megamek_dir is not None:
        config.megamek_dir = args.megamek_dir
    if args.port is not None:
        config.rl_port = args.port

    env = gymnasium.make("MegaMekGym/MegaMek-v0", config=config)

    for ep in range(args.episodes):
        print(f"\n{'='*70}")
        print(f"Episode {ep + 1}/{args.episodes}")
        print(f"{'='*70}")

        obs, info = env.reset()
        print(f"Reset complete. Obs shape: {obs.shape}, "
              f"Legal moves: {len(info['legal_moves'])}")
        print(f"  {_unit_summary(info.get('rl_unit'), 'RL ')}")
        print(f"  {_unit_summary(info.get('enemy_unit'), 'Opp')}")

        step = 0
        total_reward = 0.0
        t0 = time.time()

        while True:
            legal = info.get("legal_moves", [])
            if legal:
                action = np.random.randint(0, len(legal))
            else:
                action = 0

            obs, reward, terminated, truncated, info = env.step(action)
            step += 1
            total_reward += reward

            print(
                f"  Step {step:3d} r={info.get('round', '?'):>2} "
                f"{info.get('phase', '?'):<14s} "
                f"reward={reward:+.3f} total={total_reward:+.3f} "
                f"moves={len(info.get('legal_moves', [])): >4d}"
            )
            print(f"    {_unit_summary(info.get('rl_unit'), 'RL ')}")
            print(f"    {_unit_summary(info.get('enemy_unit'), 'Opp')}")

            if terminated or truncated:
                elapsed = time.time() - t0
                print(f"\nEpisode finished in {step} steps, {elapsed:.1f}s")
                print(f"Total reward: {total_reward:+.4f}")
                break

    env.close()
    print("\nSmoke test complete.")


if __name__ == "__main__":
    main()
