"""Diagnose whether critic features can predict returns.

Collects rollout data from live game environments, computes GAE returns,
then fits sklearn models (LinearRegression, RandomForest) to measure how
much variance in returns the 111 critic features explain.

Usage:
    # Collect data and analyze (random policy)
    poetry run python diagnose_features.py --megamek-dir ../megamek --num-steps 512 --num-envs 2

    # With a trained checkpoint (better signal)
    poetry run python diagnose_features.py --megamek-dir ../megamek --checkpoint runs/.../latest.pt

    # Analyze previously saved data
    poetry run python diagnose_features.py --load feature_data.npz
"""

import argparse
import dataclasses
import os
import signal
import sys
import time

import numpy as np
import torch
import gymnasium as gym

from megamek_gym.agent import Agent, HierarchicalAgent
from megamek_gym.config import MegaMekConfig
from megamek_gym.observation import OBS_SIZE


# Human-readable names for the 111 critic features
LOCATION_NAMES = ["HD", "CT", "RT", "LT", "RA", "LA", "RL", "LL"]

def _build_feature_names():
    names = []
    for prefix in ("rl", "enemy"):
        names += [f"{prefix}_x", f"{prefix}_y"]
        names += [f"{prefix}_facing_{i}" for i in range(6)]
        names += [f"{prefix}_mp_walk", f"{prefix}_mp_run", f"{prefix}_mp_jump"]
        names += [f"{prefix}_heat"]
        names += [f"{prefix}_prone", f"{prefix}_destroyed", f"{prefix}_deployed", f"{prefix}_retreated"]
        for loc in LOCATION_NAMES:
            names += [f"{prefix}_{loc}_armor", f"{prefix}_{loc}_internal",
                      f"{prefix}_{loc}_rear", f"{prefix}_{loc}_destroyed"]
        names += [f"{prefix}_wpn{i}_destroyed" for i in range(7)]
        names += [f"{prefix}_terrain_cover", f"{prefix}_elevation"]
    names += ["rl_moves_first"]
    names += [
        "hex_distance_to_enemy",
        "rl_range_quality_current",
        "enemy_range_quality_current",
        "has_los_current",
        "relative_elevation",
        "round_number",
    ]
    return names

FEATURE_NAMES = _build_feature_names()
assert len(FEATURE_NAMES) == OBS_SIZE, f"Expected {OBS_SIZE} names, got {len(FEATURE_NAMES)}"


def make_env(env_index, cfg):
    megamek_dir = str(os.path.abspath(cfg.megamek_dir)) if cfg.megamek_dir else None
    stagger_delay = cfg.stagger_delay
    backend = cfg.backend
    def thunk():
        if backend == "sim":
            from megamek_gym.sim.env import MegaMekSimEnv
            env_cfg = dataclasses.replace(cfg, env_index=env_index)
            env = MegaMekSimEnv(config=env_cfg)
        else:
            delay = env_index * stagger_delay
            if delay > 0:
                time.sleep(delay)
            env_cfg = dataclasses.replace(cfg, env_index=env_index, megamek_dir=megamek_dir)
            env = gym.make("MegaMekGym/MegaMek-v0", config=env_cfg)
        env = gym.wrappers.NormalizeObservation(env)
        env = gym.wrappers.RecordEpisodeStatistics(env)
        return env
    return thunk


def collect_rollout_data(cfg, checkpoint_path, num_steps, num_envs, gamma=0.99, gae_lambda=0.95):
    """Collect observations and compute GAE returns from live envs."""
    cfg = dataclasses.replace(cfg, num_envs=num_envs, stagger_delay=3)
    device = torch.device("cpu")

    if cfg.backend != "sim" and cfg.megamek_dir:
        from megamek_gym.java_process import JavaProcess
        JavaProcess.warmup_classpath(cfg.megamek_dir)

    envs = gym.vector.AsyncVectorEnv(
        [make_env(i, cfg) for i in range(num_envs)],
        autoreset_mode="SameStep",
    )

    def _shutdown(signum, frame):
        print(f"\nCaught signal {signum}, closing envs...")
        envs.close()
        sys.exit(1)
    signal.signal(signal.SIGINT, _shutdown)
    signal.signal(signal.SIGTERM, _shutdown)

    try:
        obs_size = envs.single_observation_space.shape[0]
        hierarchical = cfg.action_space_type == "hierarchical"
        critic_obs_size = OBS_SIZE

        if hierarchical:
            agent = HierarchicalAgent(obs_size, cfg.max_destinations,
                                      hidden_size=cfg.hidden_size, critic_obs_size=critic_obs_size)
        else:
            agent = Agent(obs_size, envs.single_action_space.n,
                          hidden_size=cfg.hidden_size, critic_obs_size=critic_obs_size)

        if checkpoint_path:
            ckpt = torch.load(checkpoint_path, map_location="cpu")
            agent.load_state_dict(ckpt["model"])
            print(f"Loaded checkpoint: {checkpoint_path}")

        agent.eval()

        # Buffers
        obs_buf = torch.zeros((num_steps, num_envs, obs_size))
        rewards_buf = torch.zeros((num_steps, num_envs))
        dones_buf = torch.zeros((num_steps, num_envs))
        values_buf = torch.zeros((num_steps, num_envs))

        next_obs, info = envs.reset()
        next_obs = torch.Tensor(next_obs)
        next_done = torch.zeros(num_envs)
        if hierarchical:
            next_dest_mask = torch.tensor(np.array(info["action_mask"]["dest_mask"]))
            next_facing_mask = torch.tensor(np.array(info["action_mask"]["facing_mask"]))
        else:
            next_mask = torch.tensor(np.array(info["action_mask"]))

        episodes = 0
        print(f"Collecting {num_steps} steps across {num_envs} envs...")
        for step in range(num_steps):
            if step % 64 == 0:
                print(f"  step {step}/{num_steps} ({episodes} episodes completed)")

            obs_buf[step] = next_obs
            dones_buf[step] = next_done

            with torch.no_grad():
                if hierarchical:
                    action, _, _, value = agent.get_action_and_value(
                        next_obs, next_dest_mask, next_facing_mask)
                else:
                    action, _, _, value = agent.get_action_and_value(next_obs, next_mask)
                values_buf[step] = value.flatten()

            next_obs, reward, terminated, truncated, info = envs.step(action.cpu().numpy())
            done = np.logical_or(terminated, truncated)
            rewards_buf[step] = torch.tensor(reward)
            next_obs = torch.Tensor(next_obs)
            next_done = torch.Tensor(done)

            if hierarchical:
                next_dest_mask = torch.tensor(np.array(info["action_mask"]["dest_mask"]))
                next_facing_mask = torch.tensor(np.array(info["action_mask"]["facing_mask"]))
            else:
                next_mask = torch.tensor(np.array(info["action_mask"]))

            episodes += int(done.sum())

        # GAE
        with torch.no_grad():
            next_value = agent.get_value(next_obs).reshape(1, -1)
            advantages = torch.zeros_like(rewards_buf)
            lastgaelam = 0
            for t in reversed(range(num_steps)):
                if t == num_steps - 1:
                    nextnonterminal = 1.0 - next_done
                    nextvalues = next_value
                else:
                    nextnonterminal = 1.0 - dones_buf[t + 1]
                    nextvalues = values_buf[t + 1]
                delta = rewards_buf[t] + gamma * nextvalues * nextnonterminal - values_buf[t]
                advantages[t] = lastgaelam = delta + gamma * gae_lambda * nextnonterminal * lastgaelam
            returns = advantages + values_buf

        # Flatten: (steps, envs, ...) -> (steps*envs, ...)
        features = obs_buf[:, :, :OBS_SIZE].reshape(-1, OBS_SIZE).numpy()
        returns_flat = returns.reshape(-1).numpy()
        rewards_flat = rewards_buf.reshape(-1).numpy()
        values_flat = values_buf.reshape(-1).numpy()

        print(f"Collected {features.shape[0]} samples ({episodes} episodes)")
        return features, returns_flat, rewards_flat, values_flat

    finally:
        envs.close()


def analyze(features, returns, rewards, values, output_path=None):
    """Fit sklearn models and report R² scores."""
    try:
        from sklearn.linear_model import LinearRegression
        from sklearn.ensemble import RandomForestRegressor
        from sklearn.model_selection import cross_val_score
    except ImportError:
        print("ERROR: scikit-learn not installed. Install with: pip install scikit-learn")
        sys.exit(1)

    n_samples, n_features = features.shape
    print(f"\n{'='*60}")
    print(f"  Feature Predictiveness Diagnostic")
    print(f"  Samples: {n_samples} | Features: {n_features}")
    print(f"  Returns — mean: {returns.mean():.3f}, std: {returns.std():.3f}")
    print(f"  Rewards — mean: {rewards.mean():.4f}, std: {rewards.std():.4f}")
    if values is not None:
        print(f"  Values  — mean: {values.mean():.3f}, std: {values.std():.3f}")
    print(f"{'='*60}\n")

    # Check for degenerate data
    if returns.std() < 1e-8:
        print("WARNING: Returns have near-zero variance. Cannot fit meaningful models.")
        print("This likely means all episodes had identical outcomes (e.g., all losses).")
        return

    # Filter out constant features
    feature_stds = features.std(axis=0)
    varying_mask = feature_stds > 1e-8
    n_varying = varying_mask.sum()
    print(f"Features with variance: {n_varying}/{n_features}")
    if n_varying == 0:
        print("ERROR: All features are constant. Cannot fit models.")
        return

    # Linear Regression
    print("\n--- Linear Regression ---")
    lr = LinearRegression()
    lr.fit(features, returns)
    lr_r2_train = lr.score(features, returns)
    lr_cv = cross_val_score(lr, features, returns, cv=5, scoring="r2")
    print(f"  Train R²: {lr_r2_train:.4f}")
    print(f"  CV R²:    {lr_cv.mean():.4f} ± {lr_cv.std():.4f}")

    # Top features by |coefficient|
    coef_abs = np.abs(lr.coef_)
    top_idx = np.argsort(coef_abs)[::-1][:15]
    print(f"\n  Top 15 features by |coefficient|:")
    for i, idx in enumerate(top_idx):
        print(f"    {i+1:2d}. {FEATURE_NAMES[idx]:30s}  coef={lr.coef_[idx]:+.4f}  (std={feature_stds[idx]:.3f})")

    # Random Forest
    print("\n--- Random Forest (100 trees) ---")
    rf = RandomForestRegressor(n_estimators=100, max_depth=10, n_jobs=-1, random_state=42)
    rf.fit(features, returns)
    rf_r2_train = rf.score(features, returns)
    rf_cv = cross_val_score(rf, features, returns, cv=5, scoring="r2", n_jobs=-1)
    print(f"  Train R²: {rf_r2_train:.4f}")
    print(f"  CV R²:    {rf_cv.mean():.4f} ± {rf_cv.std():.4f}")

    # Feature importances
    imp = rf.feature_importances_
    top_idx = np.argsort(imp)[::-1][:15]
    print(f"\n  Top 15 features by importance:")
    for i, idx in enumerate(top_idx):
        print(f"    {i+1:2d}. {FEATURE_NAMES[idx]:30s}  importance={imp[idx]:.4f}")

    # Interpretation
    best_cv = rf_cv.mean()
    print(f"\n{'='*60}")
    print(f"  VERDICT (based on RF CV R² = {best_cv:.4f}):")
    if best_cv < 0.15:
        print("  >> Features explain almost nothing. The critic features are")
        print("     severely lacking — consider adding more state information")
        print("     (e.g., round number, distance to enemy, terrain at current hex).")
    elif best_cv < 0.30:
        print("  >> Features explain very little. This is the bottleneck —")
        print("     adding more informative features will help more than")
        print("     increasing network capacity.")
    elif best_cv < 0.60:
        print("  >> Features have moderate predictive power. Both better features")
        print("     AND more network capacity could help. Try adding terrain/distance")
        print("     features to the critic, and also try increasing hidden size.")
    else:
        print("  >> Features are quite predictive! The neural net likely just needs")
        print("     more capacity or training time. Try increasing hidden_size,")
        print("     adding layers, or training longer.")
    print(f"{'='*60}")

    return {"lr_train_r2": lr_r2_train, "lr_cv_r2": lr_cv.mean(),
            "rf_train_r2": rf_r2_train, "rf_cv_r2": rf_cv.mean()}


def main():
    parser = argparse.ArgumentParser(description="Diagnose feature predictiveness for value function")
    parser.add_argument("--megamek-dir", type=str, default="../megamek")
    parser.add_argument("--config", type=str, default=None, help="Config YAML path")
    parser.add_argument("--checkpoint", type=str, default=None, help="Trained checkpoint .pt")
    parser.add_argument("--num-steps", type=int, default=512, help="Steps per env to collect")
    parser.add_argument("--num-envs", type=int, default=2, help="Number of parallel envs")
    parser.add_argument("--output", type=str, default="feature_data.npz", help="Save collected data to .npz")
    parser.add_argument("--load", type=str, default=None, help="Load previously saved .npz instead of collecting")
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--gae-lambda", type=float, default=0.95)
    args = parser.parse_args()

    if args.load:
        print(f"Loading data from {args.load}")
        data = np.load(args.load)
        features = data["features"]
        returns = data["returns"]
        rewards = data["rewards"]
        values = data.get("values", None)
        analyze(features, returns, rewards, values)
        return

    # Load config
    if args.checkpoint and not args.config:
        ckpt = torch.load(args.checkpoint, map_location="cpu")
        if "config" in ckpt:
            import dataclasses as dc
            valid_fields = {f.name for f in dc.fields(MegaMekConfig)}
            cfg = MegaMekConfig(**{k: v for k, v in ckpt["config"].items() if k in valid_fields})
        else:
            cfg = MegaMekConfig()
    elif args.config:
        cfg = MegaMekConfig.load(args.config)
    else:
        cfg = MegaMekConfig()

    cfg.megamek_dir = args.megamek_dir

    features, returns, rewards, values = collect_rollout_data(
        cfg, args.checkpoint, args.num_steps, args.num_envs, args.gamma, args.gae_lambda)

    # Save
    np.savez(args.output, features=features, returns=returns, rewards=rewards, values=values)
    print(f"Saved data to {args.output}")

    analyze(features, returns, rewards, values)


if __name__ == "__main__":
    main()
