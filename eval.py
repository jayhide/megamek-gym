import argparse
import dataclasses
from distutils.util import strtobool
import random

import numpy as np
import torch
import gymnasium as gym

from megamek_gym.agent import Agent, HierarchicalAgent, SpatialHierarchicalAgent, load_agent, load_config_from_checkpoint, load_hierarchical_agent, load_spatial_agent, select_action, OUTCOME_MAP
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


def _select_twostage(agent, obs, action_mask, device, deterministic, is_random):
    """Select a (dest, facing) action for hierarchical or spatial action space.

    Returns a numpy array [dest_idx, facing].
    """
    dest_mask = action_mask["dest_mask"]
    facing_mask = action_mask["facing_mask"]

    if is_random:
        valid_dests = np.where(dest_mask)[0]
        dest = np.random.choice(valid_dests) if len(valid_dests) > 0 else 0
        valid_facings = np.where(facing_mask[dest])[0]
        facing = np.random.choice(valid_facings) if len(valid_facings) > 0 else 0
        return np.array([dest, facing])

    obs_t = torch.tensor(obs, dtype=torch.float32).unsqueeze(0).to(device)
    dm_t = torch.tensor(dest_mask, dtype=torch.bool).unsqueeze(0).to(device)
    fm_t = torch.tensor(facing_mask, dtype=torch.bool).unsqueeze(0).to(device)

    with torch.no_grad():
        if deterministic:
            if isinstance(agent, SpatialHierarchicalAgent):
                return _select_spatial_deterministic(agent, obs_t, dm_t, fm_t)
            else:
                return _select_hierarchical_deterministic(agent, obs_t, dm_t, fm_t)
        else:
            action, _, _, _ = agent.get_action_and_value(obs_t, dm_t, fm_t)
            return action[0].cpu().numpy()


def _select_spatial_deterministic(agent, obs_t, dm_t, fm_t):
    """Greedy action selection for SpatialHierarchicalAgent."""
    base, board = agent._split_obs(obs_t)
    hex_features = agent.spatial_encoder(board)

    dest_logits = agent.dest_head(hex_features).view(-1, agent.n_hexes)
    dest_logits = dest_logits.masked_fill(~dm_t, -1e8)
    dest = dest_logits.argmax(dim=1)

    hex_y = dest // agent.board_w
    hex_x = dest % agent.board_w
    selected_hex_feats = hex_features[0, :, hex_y, hex_x].view(1, -1)

    facing_input = torch.cat([selected_hex_feats, base], dim=-1)
    facing_logits = agent.facing_head(facing_input)
    batch_fm = fm_t[torch.arange(1), dest]
    facing_logits = facing_logits.masked_fill(~batch_fm, -1e8)
    facing = facing_logits.argmax(dim=1)

    return np.array([dest.item(), facing.item()])


def _select_hierarchical_deterministic(agent, obs_t, dm_t, fm_t):
    """Greedy action selection for HierarchicalAgent."""
    from megamek_gym.observation import DEST_FEATURES

    features = agent.feature_net(obs_t)
    dest_logits = agent.dest_head(features)
    dest_logits = dest_logits.masked_fill(~dm_t, -1e8)
    dest = dest_logits.argmax(dim=1)

    off = agent.dest_block_offset
    dest_start = off + dest * DEST_FEATURES
    idx = dest_start.unsqueeze(1) + torch.arange(DEST_FEATURES, device=obs_t.device)
    dest_feats = obs_t.gather(1, idx)
    facing_input = torch.cat([features, dest_feats], dim=-1)
    facing_logits = agent.facing_head(facing_input)
    batch_fm = fm_t[torch.arange(1), dest]
    facing_logits = facing_logits.masked_fill(~batch_fm, -1e8)
    facing = facing_logits.argmax(dim=1)

    return np.array([dest.item(), facing.item()])


if __name__ == "__main__":
    args = parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")

    # Create environment
    if args.config:
        cfg = MegaMekConfig.load(args.config)
    elif args.checkpoint and not args.random:
        # Reconstruct config from checkpoint so env matches the trained agent
        ckpt_config = load_config_from_checkpoint(args.checkpoint)
        valid_fields = {f.name for f in dataclasses.fields(MegaMekConfig)}
        ckpt_config = {k: v for k, v in ckpt_config.items() if k in valid_fields}
        cfg = MegaMekConfig(**ckpt_config)
    else:
        cfg = MegaMekConfig()
    cfg.megamek_dir = args.megamek_dir
    cfg.rl_port = args.port
    cfg.env_index = 0

    if cfg.backend == "sim":
        from megamek_gym.sim.env import MegaMekSimEnv
        env = MegaMekSimEnv(config=cfg)
    else:
        env = gym.make("MegaMekGym/MegaMek-v0", config=cfg)

    obs_size = env.observation_space.shape[0]
    spatial = cfg.use_cnn
    hierarchical = cfg.action_space_type == "hierarchical" and not spatial

    if args.random:
        agent = None
        print(f"Random baseline (uniform over legal moves)")
        print(f"  num_episodes={args.num_episodes}")
    else:
        if spatial:
            agent, checkpoint, device = load_spatial_agent(args.checkpoint, device=device)
        elif hierarchical:
            agent, checkpoint, device = load_hierarchical_agent(
                args.checkpoint, obs_size, cfg.max_destinations, device)
        else:
            agent, checkpoint, device = load_agent(
                args.checkpoint, obs_size, env.action_space.n, device)
        print(f"Loaded checkpoint: {args.checkpoint}")
        print(f"  global_step={checkpoint.get('global_step', '?')}, update={checkpoint.get('update', '?')}")
        print(f"  obs_size={obs_size}, hierarchical={hierarchical}, spatial={spatial}")
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
            if spatial or hierarchical:
                action = _select_twostage(
                    agent, obs, info["action_mask"], device,
                    args.deterministic, args.random,
                )
            elif args.random:
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
