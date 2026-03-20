import argparse
from distutils.util import strtobool
import random

import numpy as np
import torch
import gymnasium as gym

from megamek_gym.agent import Agent, load_agent, select_action, OUTCOME_MAP
from megamek_gym.config import MegaMekConfig


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate a trained PPO agent in MegaMek")

    parser.add_argument("--checkpoint", type=str, default=None, help="Path to .pt checkpoint")
    parser.add_argument("--random", type=lambda x: bool(strtobool(x)), default=False, nargs="?", const=True,
                        help="Use random policy (uniform over legal moves) as baseline")
    parser.add_argument("--num-episodes", type=int, default=10)
    parser.add_argument("--deterministic", type=lambda x: bool(strtobool(x)), default=False, nargs="?", const=True)
    parser.add_argument("--megamek-dir", type=str, default="../megamek")
    parser.add_argument("--config", type=str, default=None)
    parser.add_argument("--port", type=int, default=9999)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--cuda", type=lambda x: bool(strtobool(x)), default=True, nargs="?", const=True)
    parser.add_argument("--verbose", type=lambda x: bool(strtobool(x)), default=False, nargs="?", const=True)

    args = parser.parse_args()
    if not args.random and args.checkpoint is None:
        parser.error("Either --checkpoint or --random is required")
    if args.random and args.checkpoint is not None:
        parser.error("Cannot use both --checkpoint and --random")
    return args


if __name__ == "__main__":
    args = parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")

    # Create environment
    cfg = MegaMekConfig.load(args.config) if args.config else MegaMekConfig()
    cfg.megamek_dir = args.megamek_dir
    cfg.rl_port = args.port
    cfg.env_index = 0
    env = gym.make("MegaMekGym/MegaMek-v0", config=cfg)

    obs_size = env.observation_space.shape[0]
    action_size = env.action_space.n

    if args.random:
        agent = None
        print(f"Random baseline (uniform over legal moves)")
        print(f"  num_episodes={args.num_episodes}")
    else:
        agent, checkpoint, device = load_agent(args.checkpoint, obs_size, action_size, device=device)
        print(f"Loaded checkpoint: {args.checkpoint}")
        print(f"  global_step={checkpoint.get('global_step', '?')}, update={checkpoint.get('update', '?')}")
        print(f"  obs_size={obs_size}, action_size={action_size}")
        print(f"  deterministic={args.deterministic}, num_episodes={args.num_episodes}")
    print()

    # Run evaluation episodes
    returns = []
    lengths = []
    results = []

    for ep in range(1, args.num_episodes + 1):
        obs, info = env.reset()
        done = False
        episode_return = 0.0
        episode_length = 0

        while not done:
            if args.random:
                n_legal = info.get("n_legal_moves", 0)
                action = np.random.randint(0, max(n_legal, 1))
            else:
                action = select_action(agent, obs, info["action_mask"], device, args.deterministic)

            obs, reward, terminated, truncated, info = env.step(action)
            episode_return += reward
            episode_length += 1
            done = terminated or truncated

            if args.verbose:
                print(f"  step {episode_length}: action={action}, reward={reward:.4f}")

        result = OUTCOME_MAP.get(info.get("game_outcome", 0), "DRAW")
        returns.append(episode_return)
        lengths.append(episode_length)
        results.append(result)

        print(f"Episode {ep:3d}: return={episode_return:7.2f}, length={episode_length:4d}, result={result}")

    env.close()

    # Summary
    wins = results.count("WIN")
    losses = results.count("LOSS")
    draws = results.count("DRAW")
    n = len(returns)

    print()
    print(f"=== Evaluation Summary ({n} episodes) ===")
    print(f"Win rate:       {100 * wins / n:.1f}% ({wins}W / {losses}L / {draws}D)")
    print(f"Avg return:     {np.mean(returns):.2f} +/- {np.std(returns):.2f}")
    print(f"Avg length:     {np.mean(lengths):.1f} +/- {np.std(lengths):.1f}")
