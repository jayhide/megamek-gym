"""Diagnostic tool for inspecting trained PPO checkpoints.

Static mode (no Java needed):
    poetry run python diagnose_checkpoint.py --checkpoint runs/.../latest.pt

Live mode (runs 1 episode):
    poetry run python diagnose_checkpoint.py --checkpoint runs/.../latest.pt --megamek-dir ../megamek --live

Compare checkpoints:
    poetry run python diagnose_checkpoint.py --checkpoint runs/.../step_2048.pt runs/.../latest.pt
"""

import argparse
from distutils.util import strtobool
import os

import numpy as np
import torch
import torch.nn as nn

from megamek_gym.agent import Agent, load_agent, OUTCOME_MAP
from megamek_gym.observation import (
    UNIT_FEATURES, GLOBAL_FEATURES, MOVE_FEATURES,
    compute_obs_size,
)


def parse_args():
    parser = argparse.ArgumentParser(description="Diagnose trained PPO checkpoints")
    parser.add_argument("--checkpoint", type=str, nargs="+", required=True,
                        help="One or more checkpoint .pt files")
    parser.add_argument("--megamek-dir", type=str, default="../megamek")
    parser.add_argument("--live", type=lambda x: bool(strtobool(x)), default=False,
                        nargs="?", const=True, help="Run a live episode for logit/value analysis")
    parser.add_argument("--port", type=int, default=9999)
    parser.add_argument("--config", type=str, default=None)
    parser.add_argument("--deterministic", type=lambda x: bool(strtobool(x)), default=False,
                        nargs="?", const=True)
    parser.add_argument("--cuda", type=lambda x: bool(strtobool(x)), default=True,
                        nargs="?", const=True)
    return parser.parse_args()


def load_checkpoint(path, device):
    """Load checkpoint dict and reconstruct agent."""
    ckpt = torch.load(path, map_location=device, weights_only=False)
    args = ckpt.get("args", {})

    # Reconstruct obs/action sizes from checkpoint args
    board_w = args.get("board_width", 16)
    board_h = args.get("board_height", 17)
    max_moves = args.get("max_legal_moves", 400)

    # Try to get sizes from the model weights directly
    actor_layers = [k for k in ckpt["model"].keys() if k.startswith("actor.")]
    critic_layers = [k for k in ckpt["model"].keys() if k.startswith("critic.")]

    # First actor layer input size = obs_size
    first_actor_weight = ckpt["model"]["actor.0.weight"]
    obs_size = first_actor_weight.shape[1]

    # Last actor layer output size = action_size
    last_actor_key = sorted([k for k in actor_layers if k.endswith(".weight")])[-1]
    action_size = ckpt["model"][last_actor_key].shape[0]

    agent = Agent(obs_size, action_size).to(device)
    agent.load_state_dict(ckpt["model"])
    agent.eval()

    return agent, ckpt, obs_size, action_size


def print_training_stats(ckpt, path):
    """Print training statistics from checkpoint."""
    print(f"  File: {os.path.basename(path)}")
    print(f"  global_step: {ckpt.get('global_step', '?')}")
    print(f"  update: {ckpt.get('update', '?')}")

    for key in ["total_games", "total_wins", "total_losses", "total_draws",
                "total_crashes", "total_early_terms"]:
        val = ckpt.get(key, "?")
        print(f"  {key}: {val}")

    total_games = ckpt.get("total_games", 0)
    total_wins = ckpt.get("total_wins", 0)
    if total_games > 0:
        print(f"  win_rate: {100 * total_wins / total_games:.1f}%")


def print_hyperparams(ckpt):
    """Print key hyperparameters."""
    args = ckpt.get("args", {})
    keys = ["learning_rate", "ent_coef", "gamma", "gae_lambda", "clip_coef",
            "clip_vloss", "vf_coef", "num_envs", "num_steps", "update_epochs",
            "minibatch_size", "target_kl", "max_grad_norm"]
    for k in keys:
        if k in args:
            print(f"  {k}: {args[k]}")


def print_weight_stats(agent):
    """Print per-layer weight statistics for actor and critic."""
    for network_name in ["actor", "critic"]:
        network = getattr(agent, network_name)
        print(f"\n  {network_name}:")
        print(f"  {'layer':<20s} {'shape':<20s} {'mean':>10s} {'std':>10s} {'min':>10s} {'max':>10s} {'flags'}")

        for name, module in network.named_modules():
            if isinstance(module, nn.Linear):
                w = module.weight.data
                flags = []
                if w.std() < 1e-6:
                    flags.append("DEAD")
                if w.abs().max() > 100:
                    flags.append("EXPLODING")

                flag_str = " ".join(flags) if flags else ""
                print(f"  {name + '.weight':<20s} {str(list(w.shape)):<20s} "
                      f"{w.mean():>10.6f} {w.std():>10.6f} {w.min():>10.4f} {w.max():>10.4f} {flag_str}")

                if module.bias is not None:
                    b = module.bias.data
                    b_std = b.std().item() if b.numel() > 1 else 0.0
                    print(f"  {name + '.bias':<20s} {str(list(b.shape)):<20s} "
                          f"{b.mean():>10.6f} {b_std:>10.6f} {b.min():>10.4f} {b.max():>10.4f}")


def run_live_episode(agent, obs_size, action_size, args, device):
    """Run one episode and collect detailed per-step diagnostics."""
    import gymnasium as gym
    from megamek_gym.config import MegaMekConfig

    cfg = MegaMekConfig.load(args.config) if args.config else MegaMekConfig()
    cfg.megamek_dir = os.path.abspath(args.megamek_dir)
    cfg.rl_port = args.port
    cfg.env_index = 0
    env = gym.make("MegaMekGym/MegaMek-v0", config=cfg)

    obs, info = env.reset()
    done = False

    steps = []
    step_num = 0

    while not done:
        obs_t = torch.tensor(obs, dtype=torch.float32).unsqueeze(0).to(device)
        mask = info["action_mask"]
        mask_t = torch.tensor(mask, dtype=torch.bool).unsqueeze(0).to(device)

        with torch.no_grad():
            raw_logits = agent.actor(obs_t).squeeze(0).cpu().numpy()
            value = agent.critic(obs_t).squeeze().item()

            # Masked logits and probabilities
            masked_logits = raw_logits.copy()
            masked_logits[~mask] = -1e8
            legal_logits = raw_logits[mask]

            # Softmax over legal moves only
            legal_shifted = legal_logits - legal_logits.max()
            legal_probs = np.exp(legal_shifted) / np.exp(legal_shifted).sum()
            entropy = -np.sum(legal_probs * np.log(legal_probs + 1e-10))

            # Select action
            if args.deterministic:
                action = int(np.argmax(masked_logits))
            else:
                full_probs = np.zeros_like(raw_logits)
                full_probs[mask] = legal_probs
                action = int(np.random.choice(len(full_probs), p=full_probs))

            n_legal = int(mask.sum())
            chosen_prob = legal_probs[np.where(np.where(mask)[0] == action)[0][0]] if action in np.where(mask)[0] else 0.0
            uniform_prob = 1.0 / n_legal if n_legal > 0 else 0.0

        step_data = {
            "step": step_num,
            "n_legal": n_legal,
            "value": value,
            "entropy": entropy,
            "max_entropy": np.log(n_legal) if n_legal > 1 else 0.0,
            "chosen_prob": chosen_prob,
            "uniform_prob": uniform_prob,
            "legal_logit_mean": float(legal_logits.mean()) if len(legal_logits) > 0 else 0.0,
            "legal_logit_std": float(legal_logits.std()) if len(legal_logits) > 1 else 0.0,
            "legal_logit_range": float(legal_logits.max() - legal_logits.min()) if len(legal_logits) > 1 else 0.0,
            "top_prob": float(legal_probs.max()) if len(legal_probs) > 0 else 0.0,
            "action": action,
            "obs": obs.copy(),
        }
        steps.append(step_data)

        obs, reward, terminated, truncated, info = env.step(action)
        step_data["reward"] = reward
        step_num += 1
        done = terminated or truncated

    outcome = OUTCOME_MAP.get(info.get("game_outcome", 0), "DRAW")
    env.close()

    return steps, outcome


def print_live_analysis(steps, outcome, obs_size):
    """Print analysis of a live episode."""
    print(f"\n  Episode: {len(steps)} steps, outcome={outcome}")

    # Value predictions
    values = [s["value"] for s in steps]
    print(f"\n  Value predictions:")
    print(f"    range: [{min(values):.4f}, {max(values):.4f}]")
    print(f"    mean: {np.mean(values):.4f}, std: {np.std(values):.4f}")
    if max(values) - min(values) < 0.01:
        print(f"    ** WARNING: Values nearly constant — critic not differentiating states **")

    # Per-step table
    print(f"\n  {'step':>4s} {'n_leg':>5s} {'value':>8s} {'entropy':>8s} {'max_ent':>8s} "
          f"{'top_p':>6s} {'chose_p':>7s} {'unif_p':>6s} {'logit_μ':>8s} {'logit_σ':>8s} {'reward':>7s}")
    print(f"  {'-'*4:>4s} {'-'*5:>5s} {'-'*8:>8s} {'-'*8:>8s} {'-'*8:>8s} "
          f"{'-'*6:>6s} {'-'*7:>7s} {'-'*6:>6s} {'-'*8:>8s} {'-'*8:>8s} {'-'*7:>7s}")

    for s in steps:
        print(f"  {s['step']:>4d} {s['n_legal']:>5d} {s['value']:>8.4f} {s['entropy']:>8.4f} {s['max_entropy']:>8.4f} "
              f"{s['top_prob']:>6.3f} {s['chosen_prob']:>7.4f} {s['uniform_prob']:>6.4f} "
              f"{s['legal_logit_mean']:>8.4f} {s['legal_logit_std']:>8.4f} {s.get('reward', 0):>7.3f}")

    # Entropy analysis
    entropies = [s["entropy"] for s in steps]
    max_entropies = [s["max_entropy"] for s in steps]
    entropy_ratios = [e / me if me > 0 else 0 for e, me in zip(entropies, max_entropies)]
    print(f"\n  Entropy ratio (actual/max):")
    print(f"    mean: {np.mean(entropy_ratios):.3f}, min: {min(entropy_ratios):.3f}, max: {max(entropy_ratios):.3f}")
    if np.mean(entropy_ratios) > 0.95:
        print(f"    Policy is near-uniform (ratio ~1.0) — hasn't learned preferences")
    elif np.mean(entropy_ratios) < 0.3:
        print(f"    ** WARNING: Policy is highly concentrated — possible entropy collapse **")

    # Action diversity
    # Extract destination from observation move features for chosen actions
    unique_actions = len(set(s["action"] for s in steps))
    print(f"\n  Action diversity: {unique_actions} unique actions / {len(steps)} steps")
    if unique_actions <= 2 and len(steps) > 5:
        print(f"    ** WARNING: Very low action diversity — agent may be stuck **")

    # Observation section statistics
    if len(steps) > 0:
        all_obs = np.array([s["obs"] for s in steps])
        total_size = all_obs.shape[1]

        # Figure out board size: total = board + 2*55 + 1 + max_moves*6
        # board_size = total - 111 - max_moves*6
        # We can infer from common sizes
        board_size = total_size - 2 * UNIT_FEATURES - GLOBAL_FEATURES
        max_moves_features = 0
        # Check if there are move features (board_size would be too large for any reasonable board)
        if board_size > 1000:
            # Has move features — figure out how many
            # board_size_actual + max_moves * 6 = board_size
            # Try common board sizes
            for bw, bh in [(16, 17)]:
                candidate_board = bw * bh
                remainder = board_size - candidate_board
                if remainder >= 0 and remainder % MOVE_FEATURES == 0:
                    max_moves_features = remainder // MOVE_FEATURES
                    board_size = candidate_board
                    break

        sections = [
            ("board", 0, board_size),
            ("rl_unit", board_size, UNIT_FEATURES),
            ("enemy_unit", board_size + UNIT_FEATURES, UNIT_FEATURES),
            ("global", board_size + 2 * UNIT_FEATURES, GLOBAL_FEATURES),
        ]
        if max_moves_features > 0:
            move_start = board_size + 2 * UNIT_FEATURES + GLOBAL_FEATURES
            sections.append(("move_features", move_start, max_moves_features * MOVE_FEATURES))

        print(f"\n  Observation section statistics (across {len(steps)} steps):")
        print(f"  {'section':<16s} {'size':>6s} {'mean':>8s} {'std':>8s} {'min':>8s} {'max':>8s} {'nonzero%':>8s}")
        for name, start, size in sections:
            section = all_obs[:, start:start + size]
            nonzero_pct = 100 * np.count_nonzero(section) / section.size
            print(f"  {name:<16s} {size:>6d} {section.mean():>8.4f} {section.std():>8.4f} "
                  f"{section.min():>8.4f} {section.max():>8.4f} {nonzero_pct:>7.1f}%")


def print_comparison_table(all_data):
    """Print side-by-side comparison of multiple checkpoints."""
    print(f"\n{'='*60}")
    print(f"  CHECKPOINT COMPARISON")
    print(f"{'='*60}")

    # Training stats comparison
    headers = ["metric"] + [os.path.basename(d["path"]) for d in all_data]
    col_width = max(16, max(len(h) for h in headers) + 2)

    print(f"\n  {'metric':<20s}", end="")
    for d in all_data:
        name = os.path.basename(d["path"])
        print(f" {name:>{col_width}s}", end="")
    print()

    for key in ["global_step", "update", "total_games", "total_wins", "total_losses",
                "total_draws", "total_crashes", "total_early_terms"]:
        print(f"  {key:<20s}", end="")
        for d in all_data:
            val = d["ckpt"].get(key, "?")
            print(f" {str(val):>{col_width}s}", end="")
        print()

    # Weight stats comparison
    print(f"\n  Weight norms:")
    print(f"  {'layer':<24s}", end="")
    for d in all_data:
        name = os.path.basename(d["path"])
        print(f" {name:>{col_width}s}", end="")
    print()

    for network_name in ["actor", "critic"]:
        for d in all_data:
            network = getattr(d["agent"], network_name)
            for name, module in network.named_modules():
                if isinstance(module, nn.Linear):
                    label = f"{network_name}.{name}.weight"
                    break
            break

        network0 = getattr(all_data[0]["agent"], network_name)
        for name, module in network0.named_modules():
            if isinstance(module, nn.Linear):
                label = f"{network_name}.{name}"
                print(f"  {label + '.w_norm':<24s}", end="")
                for d in all_data:
                    net = getattr(d["agent"], network_name)
                    for n, m in net.named_modules():
                        if n == name and isinstance(m, nn.Linear):
                            norm = m.weight.data.norm().item()
                            print(f" {norm:>{col_width}.4f}", end="")
                            break
                print()


if __name__ == "__main__":
    args = parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")

    all_data = []
    for path in args.checkpoint:
        agent, ckpt, obs_size, action_size = load_checkpoint(path, device)
        all_data.append({"path": path, "agent": agent, "ckpt": ckpt,
                         "obs_size": obs_size, "action_size": action_size})

    # Static analysis for each checkpoint
    for data in all_data:
        print(f"\n{'='*60}")
        print(f"  CHECKPOINT: {data['path']}")
        print(f"{'='*60}")

        print(f"\n--- Training Stats ---")
        print_training_stats(data["ckpt"], data["path"])

        print(f"\n--- Hyperparameters ---")
        print_hyperparams(data["ckpt"])

        print(f"\n--- Weight Statistics ---")
        print_weight_stats(data["agent"])

    # Comparison table if multiple checkpoints
    if len(all_data) > 1:
        print_comparison_table(all_data)

    # Live analysis (only for last checkpoint)
    if args.live:
        print(f"\n{'='*60}")
        print(f"  LIVE EPISODE ANALYSIS")
        print(f"  Checkpoint: {all_data[-1]['path']}")
        print(f"{'='*60}")

        agent = all_data[-1]["agent"]
        obs_size = all_data[-1]["obs_size"]
        action_size = all_data[-1]["action_size"]

        steps, outcome = run_live_episode(agent, obs_size, action_size, args, device)
        print_live_analysis(steps, outcome, obs_size)
