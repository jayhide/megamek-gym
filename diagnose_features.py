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

from megamek_gym.agent import Agent, HierarchicalAgent, SpatialHierarchicalAgent
from megamek_gym.config import MegaMekConfig
from megamek_gym.observation import OBS_SIZE


# Human-readable names for critic features
LOCATION_NAMES = ["HD", "CT", "RT", "LT", "RA", "LA", "RL", "LL"]

# Names for the 14 CNN board channels (global-avg-pooled to scalar features)
CNN_CHANNEL_NAMES = [
    "cnn_elevation", "cnn_woods_cover",
    "cnn_self_location", "cnn_enemy_location",
    "cnn_self_facing", "cnn_enemy_facing",
    "cnn_reachable", "cnn_mp_used",
    "cnn_dist_to_enemy", "cnn_elev_advantage",
    "cnn_has_los", "cnn_enemy_rq",
    "cnn_best_rl_rq", "cnn_terrain_cover",
]

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
        names += [f"{prefix}_engine_hits", f"{prefix}_gyro_hits", f"{prefix}_sensor_hits"]
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
        spatial = cfg.use_cnn
        hierarchical = cfg.action_space_type == "hierarchical"
        critic_obs_size = OBS_SIZE

        if spatial:
            agent = SpatialHierarchicalAgent(
                board_h=cfg.resolved_board_height, board_w=cfg.resolved_board_width,
                hidden_size=cfg.hidden_size)
        elif hierarchical:
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
        if spatial or hierarchical:
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
                if spatial or hierarchical:
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

            if spatial or hierarchical:
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
        scalar_features = obs_buf[:, :, :OBS_SIZE].reshape(-1, OBS_SIZE).numpy()
        returns_flat = returns.reshape(-1).numpy()
        rewards_flat = rewards_buf.reshape(-1).numpy()
        values_flat = values_buf.reshape(-1).numpy()

        board_channels = None
        if spatial:
            from megamek_gym.observation import BOARD_CHANNELS
            bh, bw = cfg.resolved_board_height, cfg.resolved_board_width
            board_channels = obs_buf[:, :, OBS_SIZE:].reshape(
                -1, BOARD_CHANNELS, bh, bw).numpy()

        print(f"Collected {scalar_features.shape[0]} samples ({episodes} episodes)")
        return scalar_features, returns_flat, rewards_flat, values_flat, board_channels

    finally:
        envs.close()


def analyze(features, returns, rewards, values, output_path=None, feature_names=None):
    """Fit sklearn models and report R² scores."""
    if feature_names is None:
        feature_names = FEATURE_NAMES
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
        print(f"    {i+1:2d}. {feature_names[idx]:30s}  coef={lr.coef_[idx]:+.4f}  (std={feature_stds[idx]:.3f})")

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
        print(f"    {i+1:2d}. {feature_names[idx]:30s}  importance={imp[idx]:.4f}")

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


def analyze_cnn_features(scalar_features, board_channels, returns,
                         train_steps=300, lr=1e-3, batch_size=512):
    """Train a CNN probe to predict returns, then evaluate learned features with sklearn.

    1. Train SpatialEncoder + linear head end-to-end on (board_channels, scalar) → returns
    2. Freeze the CNN, extract pooled features for all samples
    3. Run RF/LR on (scalar + learned CNN features) to measure R²
    """
    from megamek_gym.agent import SpatialEncoder
    from megamek_gym.observation import BOARD_CHANNELS

    try:
        from sklearn.linear_model import LinearRegression
        from sklearn.ensemble import RandomForestRegressor
        from sklearn.model_selection import cross_val_score
    except ImportError:
        print("ERROR: scikit-learn not installed.")
        return

    n_samples = scalar_features.shape[0]
    _, n_channels, bh, bw = board_channels.shape

    print(f"\n{'='*60}")
    print(f"  CNN Feature Probe")
    print(f"  Training CNN + regression head for {train_steps} steps")
    print(f"  Board: {n_channels}ch × {bh}×{bw} | Scalar: {scalar_features.shape[1]}")
    print(f"{'='*60}\n")

    # Convert to tensors
    board_t = torch.tensor(board_channels, dtype=torch.float32)
    scalar_t = torch.tensor(scalar_features, dtype=torch.float32)
    returns_t = torch.tensor(returns, dtype=torch.float32)

    # Normalize returns for stable training
    ret_mean, ret_std = returns_t.mean(), returns_t.std()
    returns_norm = (returns_t - ret_mean) / (ret_std + 1e-8)

    # Build probe: SpatialEncoder → pool → concat scalar → linear → value
    encoder = SpatialEncoder(n_channels, hidden_channels=32)
    pool = torch.nn.AdaptiveAvgPool2d((1, 1))
    head = torch.nn.Sequential(
        torch.nn.Linear(32 + scalar_features.shape[1], 128),
        torch.nn.ReLU(),
        torch.nn.Linear(128, 1),
    )

    optimizer = torch.optim.Adam(
        list(encoder.parameters()) + list(head.parameters()), lr=lr)

    # Training loop
    encoder.train()
    head.train()
    indices = np.arange(n_samples)
    for step in range(train_steps):
        batch_idx = np.random.choice(indices, size=batch_size, replace=False)
        b_board = board_t[batch_idx]
        b_scalar = scalar_t[batch_idx]
        b_ret = returns_norm[batch_idx]

        feat_maps = encoder(b_board)
        pooled = pool(feat_maps).squeeze(-1).squeeze(-1)
        combined = torch.cat([pooled, b_scalar], dim=-1)
        pred = head(combined).squeeze(-1)

        loss = torch.nn.functional.mse_loss(pred, b_ret)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if step % 50 == 0 or step == train_steps - 1:
            print(f"  step {step:4d}/{train_steps}  loss={loss.item():.4f}")

    # Extract learned features for all samples
    print("\nExtracting CNN features for all samples...")
    encoder.eval()
    cnn_features_list = []
    with torch.no_grad():
        for i in range(0, n_samples, 2048):
            batch = board_t[i:i+2048]
            feat_maps = encoder(batch)
            pooled = pool(feat_maps).squeeze(-1).squeeze(-1)
            cnn_features_list.append(pooled.numpy())
    cnn_features = np.concatenate(cnn_features_list, axis=0)

    # Combine scalar + learned CNN features
    combined_features = np.concatenate([scalar_features, cnn_features], axis=1)
    cnn_feature_names = [f"cnn_learned_{i}" for i in range(32)]
    all_names = FEATURE_NAMES + cnn_feature_names

    print(f"\nCombined features: {combined_features.shape[1]} "
          f"({scalar_features.shape[1]} scalar + {cnn_features.shape[1]} CNN)")

    # Analyze scalar-only baseline
    print(f"\n--- Scalar features only (baseline) ---")
    rf_base = RandomForestRegressor(n_estimators=100, max_depth=10, n_jobs=-1, random_state=42)
    rf_base_cv = cross_val_score(rf_base, scalar_features, returns, cv=5, scoring="r2", n_jobs=-1)
    print(f"  RF CV R²: {rf_base_cv.mean():.4f} ± {rf_base_cv.std():.4f}")

    # Analyze CNN features only
    print(f"\n--- Learned CNN features only ---")
    rf_cnn = RandomForestRegressor(n_estimators=100, max_depth=10, n_jobs=-1, random_state=42)
    rf_cnn_cv = cross_val_score(rf_cnn, cnn_features, returns, cv=5, scoring="r2", n_jobs=-1)
    print(f"  RF CV R²: {rf_cnn_cv.mean():.4f} ± {rf_cnn_cv.std():.4f}")

    # Analyze combined
    print(f"\n--- Scalar + learned CNN features ---")
    lr_model = LinearRegression()
    lr_cv = cross_val_score(lr_model, combined_features, returns, cv=5, scoring="r2")
    print(f"  LR CV R²: {lr_cv.mean():.4f} ± {lr_cv.std():.4f}")

    rf = RandomForestRegressor(n_estimators=100, max_depth=10, n_jobs=-1, random_state=42)
    rf.fit(combined_features, returns)
    rf_cv = cross_val_score(rf, combined_features, returns, cv=5, scoring="r2", n_jobs=-1)
    print(f"  RF CV R²: {rf_cv.mean():.4f} ± {rf_cv.std():.4f}")

    # Feature importances (top 20 to see CNN features)
    imp = rf.feature_importances_
    top_idx = np.argsort(imp)[::-1][:20]
    print(f"\n  Top 20 features by RF importance:")
    for i, idx in enumerate(top_idx):
        print(f"    {i+1:2d}. {all_names[idx]:30s}  importance={imp[idx]:.4f}")

    # Summary
    delta = rf_cv.mean() - rf_base_cv.mean()
    print(f"\n{'='*60}")
    print(f"  CNN FEATURE PROBE RESULTS")
    print(f"  Scalar only RF CV R²:         {rf_base_cv.mean():.4f}")
    print(f"  Learned CNN only RF CV R²:    {rf_cnn_cv.mean():.4f}")
    print(f"  Scalar + CNN RF CV R²:        {rf_cv.mean():.4f}")
    print(f"  Delta from CNN features:      {delta:+.4f}")
    print(f"{'='*60}")


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
        data = np.load(args.load, allow_pickle=True)
        features = data["features"]
        returns = data["returns"]
        rewards = data["rewards"]
        values = data.get("values", None)
        board_channels = data["board_channels"] if "board_channels" in data else None
        analyze(features, returns, rewards, values)
        if board_channels is not None:
            analyze_cnn_features(features, board_channels, returns)
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

    features, returns, rewards, values, board_channels = collect_rollout_data(
        cfg, args.checkpoint, args.num_steps, args.num_envs, args.gamma, args.gae_lambda)

    # Save
    save_kwargs = dict(features=features, returns=returns, rewards=rewards, values=values)
    if board_channels is not None:
        save_kwargs["board_channels"] = board_channels
    np.savez(args.output, **save_kwargs)
    print(f"Saved data to {args.output}")

    analyze(features, returns, rewards, values)
    if board_channels is not None:
        analyze_cnn_features(features, board_channels, returns)


if __name__ == "__main__":
    main()
