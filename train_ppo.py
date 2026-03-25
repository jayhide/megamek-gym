import argparse
import dataclasses
from distutils.util import strtobool
import logging
import os
from pathlib import Path
import signal
import sys
import time
import random

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical
import gymnasium as gym
from torch.utils.tensorboard import SummaryWriter

from megamek_gym.agent import Agent
from megamek_gym.config import MegaMekConfig


def fmt_time(seconds):
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    return f"{h:02d}:{m:02d}:{s:02d}"


def make_env(env_index, cfg):
    # Resolve megamek_dir to absolute path before entering the subprocess,
    # since AsyncVectorEnv may change the working directory.
    megamek_dir = str(os.path.abspath(cfg.megamek_dir))
    stagger_delay = cfg.stagger_delay
    def thunk():
        delay = env_index * stagger_delay
        if delay > 0:
            time.sleep(delay)
        env_cfg = dataclasses.replace(cfg, env_index=env_index, megamek_dir=megamek_dir)
        env = gym.make("MegaMekGym/MegaMek-v0", config=env_cfg)
        env = gym.wrappers.NormalizeObservation(env)
        env = gym.wrappers.RecordEpisodeStatistics(env)
        return env
    return thunk


# Mapping from CLI arg names to config field names.
# Only entries where the names differ need to be listed;
# matching names are handled automatically.
_CLI_TO_CONFIG_RENAMES = {
    "port_base": "rl_port",
    "megamek_dir": "megamek_dir",
}

# CLI args that are session-specific and NOT backed by config.
_SESSION_ONLY = {"config", "resume", "track"}


def parse_args():
    parser = argparse.ArgumentParser(description="PPO training for MegaMek")

    # Session-only args (never saved to config)
    parser.add_argument("--config", type=str, default=None)
    parser.add_argument("--resume", type=str, default=None)
    parser.add_argument("--track", type=lambda x: bool(strtobool(x)), default=False, nargs="?", const=True)

    # All config-backed args: default=None so we detect CLI overrides
    parser.add_argument("--megamek-dir", type=str, default=None)
    parser.add_argument("--port-base", type=int, default=None)
    parser.add_argument("--exp-name", type=str, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--cuda", type=lambda x: bool(strtobool(x)), default=None, nargs="?", const=True)
    parser.add_argument("--num-envs", type=int, default=None)
    parser.add_argument("--stagger-delay", type=float, default=None,
                        help="Seconds between each env startup (0 to disable)")
    parser.add_argument("--total-timesteps", type=int, default=None)
    parser.add_argument("--num-steps", type=int, default=None)
    parser.add_argument("--num-minibatches", type=int, default=None)
    parser.add_argument("--update-epochs", type=int, default=None)
    parser.add_argument("--learning-rate", type=float, default=None)
    parser.add_argument("--anneal-lr", type=lambda x: bool(strtobool(x)), default=None, nargs="?", const=True)
    parser.add_argument("--gamma", type=float, default=None)
    parser.add_argument("--gae-lambda", type=float, default=None)
    parser.add_argument("--clip-coef", type=float, default=None)
    parser.add_argument("--clip-vloss", type=lambda x: bool(strtobool(x)), default=None, nargs="?", const=True)
    parser.add_argument("--ent-coef", type=float, default=None)
    parser.add_argument("--vf-coef", type=float, default=None)
    parser.add_argument("--max-grad-norm", type=float, default=None)
    parser.add_argument("--target-kl", type=float, default=None)
    parser.add_argument("--hidden-size", type=int, default=None)
    parser.add_argument("--save-interval", type=int, default=None)

    args = parser.parse_args()

    # Collect CLI overrides (non-None, non-session values) for re-application
    cli_overrides = {}
    for cli_key, cli_val in vars(args).items():
        if cli_key in _SESSION_ONLY or cli_val is None:
            continue
        config_key = _CLI_TO_CONFIG_RENAMES.get(cli_key, cli_key)
        cli_overrides[config_key] = cli_val

    # Load base config
    cfg = MegaMekConfig.load(args.config) if args.config else MegaMekConfig()

    # Apply CLI overrides
    for config_key, cli_val in cli_overrides.items():
        if hasattr(cfg, config_key):
            setattr(cfg, config_key, cli_val)

    return cfg, args, cli_overrides


_shutting_down = False

if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
        datefmt="%H:%M:%S",
    )
    cfg, args, cli_overrides = parse_args()

    # Load checkpoint early if resuming (needed for config and run_name)
    checkpoint = None
    if args.resume:
        checkpoint = torch.load(args.resume, map_location="cpu")
        # Use checkpoint config as base if no --config was explicitly provided
        if not args.config:
            cfg = MegaMekConfig(**checkpoint["config"])
            # Re-apply CLI overrides on top of checkpoint config
            for config_key, cli_val in cli_overrides.items():
                if hasattr(cfg, config_key):
                    setattr(cfg, config_key, cli_val)

    batch_size = cfg.num_envs * cfg.num_steps
    minibatch_size = batch_size // cfg.num_minibatches

    random.seed(cfg.seed)
    np.random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)
    torch.backends.cudnn.deterministic = cfg.cuda

    device = torch.device("cuda" if torch.cuda.is_available() and cfg.cuda else "cpu")

    # Reuse original run directory when resuming
    if checkpoint is not None:
        run_name = checkpoint.get("run_name", Path(args.resume).parent.parent.name)
    else:
        run_name = f"{cfg.exp_name}__{cfg.seed}__{int(time.time())}"

    writer = SummaryWriter(f"runs/{run_name}")
    writer.add_text("hyperparameters", "|param|value|\n|-|-|\n" + "\n".join(
        [f"|{key}|{value}|" for key, value in dataclasses.asdict(cfg).items()]
    ))

    # Save resolved config for reproducibility
    os.makedirs(f"runs/{run_name}", exist_ok=True)
    cfg.save(f"runs/{run_name}/config.yaml")

    # Resolve classpath once in the main process before spawning workers.
    # Workers read from the cached file, avoiding concurrent Gradle races.
    from megamek_gym.java_process import JavaProcess
    JavaProcess.warmup_classpath(cfg.megamek_dir)

    envs = gym.vector.AsyncVectorEnv(
      [make_env(i, cfg) for i in range(cfg.num_envs)],
      autoreset_mode="SameStep",
    )

    # Signal handler to ensure cleanup on Ctrl+C / external kill
    def _shutdown_handler(signum, frame):
        global _shutting_down
        if _shutting_down:
            return  # Avoid re-entrancy
        _shutting_down = True
        print(f"\nCaught signal {signum}, shutting down envs...")
        envs.close()
        writer.close()
        sys.exit(1)

    signal.signal(signal.SIGINT, _shutdown_handler)
    signal.signal(signal.SIGTERM, _shutdown_handler)

    try:  # try/finally to guarantee envs.close() on any exception

        agent = Agent(envs.single_observation_space.shape[0], envs.single_action_space.n, hidden_size=cfg.hidden_size).to(device)
        optimizer = optim.Adam(agent.parameters(), lr=cfg.learning_rate, eps=1e-5)

        if checkpoint is not None:
            agent.load_state_dict(checkpoint["model"])
            optimizer.load_state_dict(checkpoint["optimizer"])
            # Move optimizer state to correct device
            for state in optimizer.state.values():
                for k, v in state.items():
                    if isinstance(v, torch.Tensor):
                        state[k] = v.to(device)
            global_step = checkpoint["global_step"]
            start_update = checkpoint["update"] + 1
            total_games = checkpoint.get("total_games", 0)
            total_wins = checkpoint.get("total_wins", 0)
            total_losses = checkpoint.get("total_losses", 0)
            total_draws = checkpoint.get("total_draws", 0)
            total_crashes = checkpoint.get("total_crashes", 0)
            total_early_terms = checkpoint.get("total_early_terms", 0)
            total_moves_truncated_steps = checkpoint.get("total_moves_truncated_steps", 0)
            total_moves_truncated_count = checkpoint.get("total_moves_truncated_count", 0)
            print(f"Resumed from {args.resume} at update {start_update}, global_step {global_step}")
            print(f"  Restored stats: {total_games} games (W:{total_wins} L:{total_losses} D:{total_draws} C:{total_crashes} E:{total_early_terms})")
        else:
            start_update = 1

        # Initialize Rollout
        obs_buf = torch.zeros((cfg.num_steps, cfg.num_envs) + envs.single_observation_space.shape).to(device)
        actions_buf = torch.zeros((cfg.num_steps, cfg.num_envs)).to(device)
        logprobs_buf = torch.zeros((cfg.num_steps, cfg.num_envs)).to(device)
        rewards_buf = torch.zeros((cfg.num_steps, cfg.num_envs)).to(device)
        dones_buf = torch.zeros((cfg.num_steps, cfg.num_envs)).to(device)
        values_buf = torch.zeros((cfg.num_steps, cfg.num_envs)).to(device)
        masks_buf = torch.zeros((cfg.num_steps, cfg.num_envs, envs.single_action_space.n), dtype=torch.bool).to(device)

        # Start
        start_time = time.time()
        resume_step_offset = checkpoint["global_step"] if checkpoint is not None else 0
        if checkpoint is None:
            global_step = 0

        next_obs, info = envs.reset()

        # Restore observation normalization running stats after envs are alive
        if checkpoint is not None and "obs_rms" in checkpoint:
            from gymnasium.wrappers.utils import RunningMeanStd
            restored_rms = []
            for state in checkpoint["obs_rms"]:
                mean = np.array(state["mean"])
                rms = RunningMeanStd(shape=mean.shape)
                rms.mean = mean
                rms.var = np.array(state["var"])
                rms.count = state["count"]
                restored_rms.append(rms)
            envs.set_attr("obs_rms", restored_rms)
            print(f"  Restored obs normalization stats (count={checkpoint['obs_rms'][0]['count']:.0f})")

        next_obs = torch.Tensor(next_obs).to(device)
        next_done = torch.zeros(cfg.num_envs).to(device)
        next_mask = torch.tensor(np.array(info["action_mask"])).to(device)
        num_updates = cfg.total_timesteps // batch_size

        print(f"\n{'='*60}")
        print(f"  PPO Training — {cfg.exp_name}")
        print(f"  Device: {device} | Envs: {cfg.num_envs} | Stagger: {cfg.stagger_delay}s")
        n_params = sum(p.numel() for p in agent.parameters())
        print(f"  Obs: {envs.single_observation_space.shape[0]} | Actions: {envs.single_action_space.n} | Hidden: {cfg.hidden_size} | Params: {n_params:,}")
        print(f"  Timesteps: {cfg.total_timesteps:,} | Updates: {num_updates}")
        print(f"  Batch: {batch_size} | Minibatch: {minibatch_size}")
        print(f"  LR: {cfg.learning_rate} | Ent: {cfg.ent_coef} | Gamma: {cfg.gamma}")
        print(f"  Config: runs/{run_name}/config.yaml")
        print(f"  Java logs:")
        for i in range(cfg.num_envs):
            port = cfg.rl_port + i
            print(f"    env {i}: {cfg.megamek_dir}/rl_java_{port}.log")
        print(f"{'='*60}\n")

        recent_returns = []
        recent_wins, recent_losses, recent_draws, recent_rounds = [], [], [], []
        recent_lengths = []
        if checkpoint is None:
            total_games, total_wins, total_losses, total_draws, total_crashes, total_early_terms = 0, 0, 0, 0, 0, 0
            total_moves_truncated_steps = 0  # steps where legal moves exceeded max_legal_moves
            total_moves_truncated_count = 0  # total number of moves dropped across all steps
        rollout_n_legal = []

        for update in range(start_update, num_updates + 1):

            if cfg.anneal_lr:
                frac = 1.0 - (update - 1) / num_updates
                optimizer.param_groups[0]["lr"] = frac * cfg.learning_rate

            t_rollout_start = time.time()
            episodes_this_rollout = 0

            for step in range(0, cfg.num_steps):
                if step % 16 == 0:
                    print(f"  rollout {update}: step {step}/{cfg.num_steps}", flush=True)
                global_step += cfg.num_envs
                obs_buf[step] = next_obs
                dones_buf[step] = next_done
                masks_buf[step] = next_mask

                with torch.no_grad():
                    action, logprob, _, value = agent.get_action_and_value(next_obs, next_mask)
                    values_buf[step] = value.flatten()

                actions_buf[step] = action
                logprobs_buf[step] = logprob

                next_obs, reward, terminated, truncated, info = envs.step(action.cpu().numpy())
                done = np.logical_or(terminated, truncated)
                rewards_buf[step] = torch.tensor(reward).to(device)
                next_obs = torch.Tensor(next_obs).to(device)
                next_done = torch.Tensor(done).to(device)
                next_mask = torch.tensor(np.array(info["action_mask"])).to(device)
                rollout_n_legal.extend(info["n_legal_moves"])

                # Track move truncation
                trunc_counts = info.get("moves_truncated", np.zeros(cfg.num_envs))
                for tc in trunc_counts:
                    if tc > 0:
                        total_moves_truncated_steps += 1
                        total_moves_truncated_count += int(tc)

                # Log episode completions (SameStep autoreset mode)
                if "final_info" in info and "_final_info" in info:
                    final_info = info["final_info"]
                    final_mask = info["_final_info"]
                    for i in range(cfg.num_envs):
                        if not final_mask[i]:
                            continue
                        # Extract per-env episode stats
                        ep_data = final_info.get("episode", {})
                        ep_return = float(ep_data.get("r", [0])[i]) if "r" in ep_data else 0.0
                        ep_len = int(ep_data.get("l", [0])[i]) if "l" in ep_data else 0
                        outcome = int(final_info.get("game_outcome", np.zeros(cfg.num_envs))[i])
                        rounds = int(final_info.get("game_rounds", np.zeros(cfg.num_envs))[i])

                        recent_returns.append(ep_return)
                        recent_lengths.append(ep_len)
                        recent_rounds.append(rounds)

                        crashed = int(final_info.get("java_crash", np.zeros(cfg.num_envs))[i]) == 1
                        early_termed = int(final_info.get("early_termination", np.zeros(cfg.num_envs))[i]) == 1

                        episodes_this_rollout += 1
                        total_games += 1
                        if early_termed:
                            total_early_terms += 1
                        if crashed:
                            total_crashes += 1
                            outcome_str = "CRASH"
                        elif outcome == 1:
                            total_wins += 1
                            recent_wins.append(1)
                            outcome_str = "WIN"
                        elif outcome == -1:
                            total_losses += 1
                            recent_losses.append(1)
                            outcome_str = "LOSS"
                        else:
                            total_draws += 1
                            recent_draws.append(1)
                            outcome_str = "DRAW"

                        print(f"  episode done: return={ep_return:.2f}, len={ep_len}, rounds={rounds}, outcome={outcome_str}")
                        writer.add_scalar("charts/episodic_return", ep_return, global_step)
                        writer.add_scalar("charts/episodic_length", ep_len, global_step)
                        writer.add_scalar("charts/game_outcome", outcome, global_step)
                        writer.add_scalar("charts/game_rounds", rounds, global_step)
                        if total_games > 0:
                            writer.add_scalar("charts/win_rate", total_wins / total_games, global_step)
                        if crashed:
                            writer.add_scalar("charts/java_crashes", total_crashes, global_step)
                        if early_termed:
                            writer.add_scalar("charts/early_terminations", total_early_terms, global_step)

            t_rollout_end = time.time()

            # GAE
            with torch.no_grad():
                next_value = agent.get_value(next_obs).reshape(1, -1)
                advantages = torch.zeros_like(rewards_buf).to(device)
                lastgaelam = 0
                for t in reversed(range(cfg.num_steps)):
                    if t == cfg.num_steps - 1:
                        nextnonterminal = 1.0 - next_done
                        nextvalues = next_value
                    else:
                        nextnonterminal = 1.0 - dones_buf[t + 1]
                        nextvalues = values_buf[t + 1]
                    delta = rewards_buf[t] + cfg.gamma * nextvalues * nextnonterminal - values_buf[t]
                    advantages[t] = lastgaelam = delta + cfg.gamma * cfg.gae_lambda * nextnonterminal * lastgaelam
                returns = advantages + values_buf

            b_obs = obs_buf.reshape((-1,) + envs.single_observation_space.shape)
            b_logprobs = logprobs_buf.reshape(-1)
            b_actions = actions_buf.reshape(-1)
            b_advantages = advantages.reshape(-1)
            b_returns = returns.reshape(-1)
            b_values = values_buf.reshape(-1)
            b_masks = masks_buf.reshape((-1, envs.single_action_space.n))

            # Training
            b_inds = np.arange(batch_size)
            clipfracs = []
            for epoch in range(cfg.update_epochs):
                np.random.shuffle(b_inds)
                for start in range(0, batch_size, minibatch_size):
                    end = start + minibatch_size
                    mb_inds = b_inds[start:end]
                    _, newlogprob, entropy, newvalue = agent.get_action_and_value(b_obs[mb_inds], b_masks[mb_inds], b_actions.long()[mb_inds])
                    logratio = newlogprob - b_logprobs[mb_inds]
                    ratio = logratio.exp()

                    with torch.no_grad():
                        old_approx_kl = (-logratio).mean()
                        approx_kl = ((ratio - 1) - logratio).mean()
                        clipfracs += [((ratio - 1.0).abs() > cfg.clip_coef).float().mean().item()]

                    mb_advantages = b_advantages[mb_inds]
                    mb_advantages = (mb_advantages - mb_advantages.mean()) / (mb_advantages.std() + 1e-8)

                    # Policy Loss
                    pg_loss1 = -mb_advantages * ratio
                    pg_loss2 = -mb_advantages * torch.clamp(ratio, 1 - cfg.clip_coef, 1 + cfg.clip_coef)
                    pg_loss = torch.max(pg_loss1, pg_loss2).mean()

                    # Value Loss
                    newvalue = newvalue.view(-1)
                    if cfg.clip_vloss:
                        v_loss_unclipped = (newvalue - b_returns[mb_inds]) ** 2
                        v_clipped = b_values[mb_inds] + torch.clamp(
                            newvalue - b_values[mb_inds],
                            -cfg.clip_coef,
                            cfg.clip_coef,
                        )
                        v_loss_clipped = (v_clipped - b_returns[mb_inds]) ** 2
                        v_loss_max = torch.max(v_loss_unclipped, v_loss_clipped)
                        v_loss = 0.5 * v_loss_max.mean()
                    else:
                        v_loss = 0.5 * ((newvalue - b_returns[mb_inds]) ** 2).mean()

                    # Entropy Bonus
                    entropy_loss = entropy.mean()

                    # Combined Loss
                    loss = pg_loss - cfg.ent_coef * entropy_loss + v_loss * cfg.vf_coef

                    optimizer.zero_grad()
                    loss.backward()
                    nn.utils.clip_grad_norm_(agent.parameters(), cfg.max_grad_norm)
                    optimizer.step()

                if cfg.target_kl is not None and approx_kl > cfg.target_kl:
                    break

            t_train_end = time.time()

            y_pred, y_true = b_values.cpu().numpy(), b_returns.cpu().numpy()
            var_y = np.var(y_true)
            explained_var = np.nan if var_y == 0 else 1 - np.var(y_true - y_pred) / var_y

            rollout_s = t_rollout_end - t_rollout_start
            train_s = t_train_end - t_rollout_end

            writer.add_scalar("charts/reward_mean", rewards_buf.mean().item(), global_step)
            writer.add_scalar("charts/reward_std", rewards_buf.std().item(), global_step)
            writer.add_scalar("charts/value_mean", values_buf.mean().item(), global_step)
            writer.add_scalar("charts/value_std", values_buf.std().item(), global_step)
            writer.add_scalar("timing/rollout_seconds", rollout_s, global_step)
            writer.add_scalar("timing/train_seconds", train_s, global_step)
            writer.add_scalar("timing/episodes_per_rollout", episodes_this_rollout, global_step)
            writer.add_scalar("charts/learning_rate", optimizer.param_groups[0]["lr"], global_step)
            session_steps = global_step - resume_step_offset
            writer.add_scalar("charts/SPS", int(session_steps / (time.time() - start_time)), global_step)
            writer.add_scalar("losses/value_loss", v_loss.item(), global_step)
            writer.add_scalar("losses/policy_loss", pg_loss.item(), global_step)
            writer.add_scalar("losses/entropy", entropy_loss.item(), global_step)
            writer.add_scalar("losses/approx_kl", approx_kl.item(), global_step)
            writer.add_scalar("losses/clipfrac", np.mean(clipfracs), global_step)
            writer.add_scalar("losses/explained_variance", explained_var, global_step)

            legal_mean = int(np.mean(rollout_n_legal)) if rollout_n_legal else 0
            legal_max = int(np.max(rollout_n_legal)) if rollout_n_legal else 0
            if rollout_n_legal:
                writer.add_scalar("charts/n_legal_moves_mean", np.mean(rollout_n_legal), global_step)
                writer.add_scalar("charts/n_legal_moves_max", np.max(rollout_n_legal), global_step)
                writer.add_scalar("charts/n_legal_moves_min", np.min(rollout_n_legal), global_step)
            if total_moves_truncated_steps > 0:
                writer.add_scalar("charts/moves_truncated_steps", total_moves_truncated_steps, global_step)
                writer.add_scalar("charts/moves_truncated_count", total_moves_truncated_count, global_step)
            rollout_n_legal.clear()

            elapsed = time.time() - start_time
            sps = int(session_steps / elapsed)
            updates_done = update - start_update + 1
            eta_seconds = elapsed / updates_done * (num_updates - update)
            pct = 100.0 * update / num_updates
            print(
                f"[update {update}/{num_updates} | {pct:.1f}% | ETA {fmt_time(eta_seconds)}]"
                f" SPS={sps}"
                f" | pg={pg_loss.item():.4f} vf={v_loss.item():.4f} ent={entropy_loss.item():.3f}"
                f" | kl={approx_kl.item():.4f} clip={np.mean(clipfracs):.3f}"
                f" | ev={explained_var:.4f}"
                f" | legal={legal_mean}/{legal_max}"
                f" | rollout={rollout_s:.1f}s train={train_s:.1f}s episodes={episodes_this_rollout}"
            )

            # Checkpointing
            if update % cfg.save_interval == 0:
                os.makedirs(f"runs/{run_name}/checkpoints", exist_ok=True)
                save_checkpoint = {
                    "model": agent.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "global_step": global_step,
                    "update": update,
                    "config": dataclasses.asdict(cfg),
                    "run_name": run_name,
                    "obs_rms": [{"mean": rms.mean.tolist(), "var": rms.var.tolist(),
                                 "count": float(rms.count)}
                                for rms in envs.get_attr("obs_rms")],
                    "total_games": total_games,
                    "total_wins": total_wins,
                    "total_losses": total_losses,
                    "total_draws": total_draws,
                    "total_crashes": total_crashes,
                    "total_early_terms": total_early_terms,
                    "total_moves_truncated_steps": total_moves_truncated_steps,
                    "total_moves_truncated_count": total_moves_truncated_count,
                }
                torch.save(save_checkpoint, f"runs/{run_name}/checkpoints/step_{global_step}.pt")
                torch.save(save_checkpoint, f"runs/{run_name}/checkpoints/latest.pt")

                elapsed = time.time() - start_time
                summary_start = max(start_update, update - cfg.save_interval + 1)
                n_recent = len(recent_returns)
                n_recent_w = len(recent_wins)
                n_recent_l = len(recent_losses)
                n_recent_d = len(recent_draws)
                recent_wr = n_recent_w / n_recent * 100 if n_recent > 0 else 0
                cum_wr = total_wins / total_games * 100 if total_games > 0 else 0

                print(f"\n--- Summary (updates {summary_start}-{update}) ---")
                if recent_returns:
                    print(f"  Episodes: {n_recent} (W:{n_recent_w} L:{n_recent_l} D:{n_recent_d} C:{total_crashes} E:{total_early_terms} — {recent_wr:.1f}% win rate)")
                    print(f"  Mean return: {np.mean(recent_returns):.2f} | Mean length: {np.mean(recent_lengths):.0f} | Mean rounds: {np.mean(recent_rounds):.1f}")
                else:
                    print(f"  Episodes: 0")
                print(f"  Cumulative: {total_games} games (W:{total_wins} L:{total_losses} D:{total_draws} C:{total_crashes} E:{total_early_terms} — {cum_wr:.1f}%)")
                print(f"  Explained variance:  {explained_var:.4f}")
                print(f"  Learning rate:       {optimizer.param_groups[0]['lr']:.2e}")
                print(f"  Elapsed:             {fmt_time(elapsed)}")
                print()
                recent_returns.clear()
                recent_wins.clear()
                recent_losses.clear()
                recent_draws.clear()
                recent_rounds.clear()
                recent_lengths.clear()

        elapsed = time.time() - start_time
        session_steps = global_step - resume_step_offset
        sps = int(session_steps / elapsed) if elapsed > 0 else 0
        final_wr = total_wins / total_games * 100 if total_games > 0 else 0
        print(f"\nTraining complete. {global_step:,} steps in {fmt_time(elapsed)}. Final SPS: {sps}.")
        print(f"Total games: {total_games} (W:{total_wins} L:{total_losses} D:{total_draws} C:{total_crashes} E:{total_early_terms} — {final_wr:.1f}% win rate)")
        if total_moves_truncated_steps > 0:
            total_steps = global_step - resume_step_offset
            trunc_pct = total_moves_truncated_steps / max(total_steps, 1) * 100
            print(f"\n  WARNING: Legal moves exceeded max_legal_moves ({cfg.max_legal_moves}) on {total_moves_truncated_steps} steps ({trunc_pct:.2f}%).")
            print(f"           {total_moves_truncated_count} total moves were dropped (invisible to agent).")
            print(f"           Consider increasing max_legal_moves in your config.")
        print(f"\n{'='*60}")
        print(f"  Run directory:  runs/{run_name}")
        print(f"  Latest checkpoint: runs/{run_name}/checkpoints/latest.pt")
        print(f"\n  Resume training:")
        print(f"    poetry run python train_ppo.py --resume runs/{run_name}/checkpoints/latest.pt")
        print(f"\n  Evaluate:")
        print(f"    poetry run python eval.py --checkpoint runs/{run_name}/checkpoints/latest.pt --num-episodes 10")
        print(f"\n  TensorBoard:")
        print(f"    tensorboard --logdir runs/{run_name}")
        print(f"{'='*60}")

    finally:
        writer.close()
        envs.close()
