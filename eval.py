import argparse
from distutils.util import strtobool
import random

import numpy as np
import torch
import gymnasium as gym

from megamek_gym.config import MegaMekConfig
from train_ppo import Agent


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate a trained PPO agent in MegaMek")

    parser.add_argument("--checkpoint", type=str, required=True, help="Path to .pt checkpoint")
    parser.add_argument("--num-episodes", type=int, default=10)
    parser.add_argument("--deterministic", type=lambda x: bool(strtobool(x)), default=False, nargs="?", const=True)
    parser.add_argument("--megamek-dir", type=str, default="../megamek")
    parser.add_argument("--config", type=str, default=None)
    parser.add_argument("--port", type=int, default=9999)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--cuda", type=lambda x: bool(strtobool(x)), default=True, nargs="?", const=True)
    parser.add_argument("--verbose", type=lambda x: bool(strtobool(x)), default=False, nargs="?", const=True)

    return parser.parse_args()


def detect_result(info):
    """Determine WIN/LOSS/DRAW from terminal info dict."""
    rl = info.get("rl_unit")
    enemy = info.get("enemy_unit")

    rl_alive = rl is not None and not rl.get("destroyed", False) and not rl.get("retreated", False)
    enemy_alive = enemy is not None and not enemy.get("destroyed", False) and not enemy.get("retreated", False)

    if rl_alive and not enemy_alive:
        return "WIN"
    elif not rl_alive and enemy_alive:
        return "LOSS"
    return "DRAW"


if __name__ == "__main__":
    args = parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")

    # Load checkpoint
    checkpoint = torch.load(args.checkpoint, map_location=device)
    saved_args = checkpoint.get("args", {})
    obs_size = saved_args.get("obs_size", 382)
    action_size = saved_args.get("action_size", 1000)

    # Create environment
    cfg = MegaMekConfig.load(args.config) if args.config else MegaMekConfig()
    cfg.megamek_dir = args.megamek_dir
    cfg.rl_port = args.port
    cfg.env_index = 0
    env = gym.make("MegaMekGym/MegaMek-v0", config=cfg)

    # Infer dimensions from env
    obs_size = env.observation_space.shape[0]
    action_size = env.action_space.n

    # Load agent
    agent = Agent(obs_size, action_size).to(device)
    agent.load_state_dict(checkpoint["model"])
    agent.eval()

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
            obs_tensor = torch.tensor(obs, dtype=torch.float32).unsqueeze(0).to(device)
            mask_tensor = torch.tensor(info["action_mask"], dtype=torch.bool).unsqueeze(0).to(device)

            with torch.no_grad():
                if args.deterministic:
                    logits = agent.actor(obs_tensor)
                    logits = logits.masked_fill(~mask_tensor, -1e8)
                    action = logits.argmax(dim=1).item()
                else:
                    action, _, _, _ = agent.get_action_and_value(obs_tensor, mask_tensor)
                    action = action.item()

            obs, reward, terminated, truncated, info = env.step(action)
            episode_return += reward
            episode_length += 1
            done = terminated or truncated

            if args.verbose:
                print(f"  step {episode_length}: action={action}, reward={reward:.4f}")

        result = detect_result(info)
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
