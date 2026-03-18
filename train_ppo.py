import argparse
from distutils.util import strtobool
import os
import time
import random

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical
import gymnasium as gym
from torch.utils.tensorboard import SummaryWriter

from megamek_gym.config import MegaMekConfig


def fmt_time(seconds):
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    return f"{h:02d}:{m:02d}:{s:02d}"


def make_env(env_index, args):
    # Resolve megamek_dir to absolute path before entering the subprocess,
    # since AsyncVectorEnv may change the working directory.
    megamek_dir = str(os.path.abspath(args.megamek_dir))
    def thunk():
        cfg = MegaMekConfig.load(args.config) if args.config else MegaMekConfig()
        cfg.env_index = env_index
        cfg.megamek_dir = megamek_dir
        cfg.rl_port = args.port_base
        env = gym.make("MegaMekGym/MegaMek-v0", config=cfg)
        env = gym.wrappers.RecordEpisodeStatistics(env)
        return env
    return thunk


class Agent(nn.Module):
    def __init__(self, obs_size, action_size):
        super().__init__()
        self.critic = nn.Sequential(
            nn.Linear(obs_size, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 1),
        )
        self.actor = nn.Sequential(
            nn.Linear(obs_size, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, action_size),
        )

    def get_value(self, obs):
        return self.critic(obs)

    def get_action_and_value(self, obs, action_mask, action=None):
        logits = self.actor(obs)
        invalid_mask = ~action_mask
        logits = logits.masked_fill(invalid_mask, -1e8)

        probs = Categorical(logits=logits)
        if action is None:
            action = probs.sample()
        return action, probs.log_prob(action), probs.entropy(), self.get_value(obs)


def parse_args():
    parser = argparse.ArgumentParser(description="PPO training for MegaMek")    

    # General
    parser.add_argument("--exp-name", type=str, default="megamek-ppo")
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--cuda", type=lambda x: bool(strtobool(x)), default=True, nargs="?", const=True)
    parser.add_argument("--track", type=lambda x: bool(strtobool(x)), default=False, nargs="?", const=True)

    # MegaMek environment
    parser.add_argument("--config", type=str, default=None)
    parser.add_argument("--megamek-dir", type=str, default="../megamek")
    parser.add_argument("--num-envs", type=int, default=4)
    parser.add_argument("--port-base", type=int, default=9999)

    # PPO core
    parser.add_argument("--total-timesteps", type=int, default=500_000)
    parser.add_argument("--num-steps", type=int, default=128)
    parser.add_argument("--num-minibatches", type=int, default=4)
    parser.add_argument("--update-epochs", type=int, default=4)

    # PPO hyperparameters
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--anneal-lr", type=lambda x: bool(strtobool(x)), default=True, nargs="?", const=True)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--gae-lambda", type=float, default=0.95)
    parser.add_argument("--clip-coef", type=float, default=0.2)
    parser.add_argument("--clip-vloss", type=lambda x: bool(strtobool(x)), default=True, nargs="?", const=True)
    parser.add_argument("--ent-coef", type=float, default=0.01)
    parser.add_argument("--vf-coef", type=float, default=0.5)
    parser.add_argument("--max-grad-norm", type=float, default=0.5)
    parser.add_argument("--target-kl", type=float, default=None)

    # Checkpointing
    parser.add_argument("--save-interval", type=int, default=50)
    parser.add_argument("--resume", type=str, default=None)

    args = parser.parse_args()
    args.batch_size = args.num_envs * args.num_steps
    args.minibatch_size = args.batch_size // args.num_minibatches
    return args

if __name__ == "__main__":
    args = parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = args.cuda

    device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")

    run_name = f"{args.exp_name}__{args.seed}__{int(time.time())}"
    writer = SummaryWriter(f"runs/{run_name}")
    writer.add_text("hyperparameters", "|param|value|\n|-|-|\n" + "\n".join(
        [f"|{key}|{value}|" for key, value in vars(args).items()]
    ))

    envs = gym.vector.AsyncVectorEnv(
      [make_env(i, args) for i in range(args.num_envs)]
    )

    agent = Agent(envs.single_observation_space.shape[0], envs.single_action_space.n).to(device)
    optimizer = optim.Adam(agent.parameters(), lr=args.learning_rate, eps=1e-5)

    if args.resume:
        checkpoint = torch.load(args.resume, map_location=device)
        agent.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        global_step = checkpoint["global_step"]
        start_update = checkpoint["update"] + 1
        print(f"Resumed from {args.resume} at update {start_update}, global_step {global_step}")
    else:
        start_update = 1

    # Initialize Rollout
    obs_buf = torch.zeros((args.num_steps, args.num_envs) + envs.single_observation_space.shape).to(device)
    actions_buf = torch.zeros((args.num_steps, args.num_envs)).to(device)
    logprobs_buf = torch.zeros((args.num_steps, args.num_envs)).to(device)
    rewards_buf = torch.zeros((args.num_steps, args.num_envs)).to(device)
    dones_buf = torch.zeros((args.num_steps, args.num_envs)).to(device)
    values_buf = torch.zeros((args.num_steps, args.num_envs)).to(device)
    masks_buf = torch.zeros((args.num_steps, args.num_envs, envs.single_action_space.n), dtype=torch.bool).to(device)

    # Start
    start_time = time.time()
    if not args.resume:
        global_step = 0

    next_obs, info = envs.reset()
    next_obs = torch.Tensor(next_obs).to(device)
    next_done = torch.zeros(args.num_envs).to(device)
    next_mask = torch.tensor(np.array(info["action_mask"])).to(device)
    num_updates = args.total_timesteps // args.batch_size

    print(f"\n{'='*60}")
    print(f"  PPO Training — {args.exp_name}")
    print(f"  Device: {device} | Envs: {args.num_envs}")
    print(f"  Obs: {envs.single_observation_space.shape[0]} | Actions: {envs.single_action_space.n}")
    print(f"  Timesteps: {args.total_timesteps:,} | Updates: {num_updates}")
    print(f"  Batch: {args.batch_size} | Minibatch: {args.minibatch_size}")
    print(f"  LR: {args.learning_rate} | Ent: {args.ent_coef} | Gamma: {args.gamma}")
    print(f"{'='*60}\n")

    recent_returns = []

    for update in range(start_update, num_updates + 1):

        if args.anneal_lr:
            frac = 1.0 - (update - 1) / num_updates
            optimizer.param_groups[0]["lr"] = frac * args.learning_rate

        for step in range(0, args.num_steps):
            global_step += args.num_envs
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

            # Log episode completions
            if "final_info" in info:
                for i, fi in enumerate(info["final_info"]):
                    if fi is not None and "episode" in fi:
                        ep_return = fi["episode"]["r"]
                        recent_returns.append(ep_return)
                        print(f"  episode done: return={ep_return:.2f}, len={fi['episode']['l']}")
                        writer.add_scalar("charts/episodic_return", ep_return, global_step)
                        writer.add_scalar("charts/episodic_length", fi["episode"]["l"], global_step)

        # GAE
        with torch.no_grad():
            next_value = agent.get_value(next_obs).reshape(1, -1)
            advantages = torch.zeros_like(rewards_buf).to(device)
            lastgaelam = 0
            for t in reversed(range(args.num_steps)):
                if t == args.num_steps - 1:
                    nextnonterminal = 1.0 - next_done
                    nextvalues = next_value
                else:
                    nextnonterminal = 1.0 - dones_buf[t + 1]
                    nextvalues = values_buf[t + 1]
                delta = rewards_buf[t] + args.gamma * nextvalues * nextnonterminal - values_buf[t]
                advantages[t] = lastgaelam = delta + args.gamma * args.gae_lambda * nextnonterminal * lastgaelam
            returns = advantages + values_buf

        b_obs = obs_buf.reshape((-1,) + envs.single_observation_space.shape)
        b_logprobs = logprobs_buf.reshape(-1)
        b_actions = actions_buf.reshape(-1)
        b_advantages = advantages.reshape(-1)
        b_returns = returns.reshape(-1)
        b_values = values_buf.reshape(-1)
        b_masks = masks_buf.reshape((-1, envs.single_action_space.n))

        # Training
        b_inds = np.arange(args.batch_size)
        clipfracs = []
        for epoch in range(args.update_epochs):
            np.random.shuffle(b_inds)
            for start in range(0, args.batch_size, args.minibatch_size):
                end = start + args.minibatch_size
                mb_inds = b_inds[start:end]
                _, newlogprob, entropy, newvalue = agent.get_action_and_value(b_obs[mb_inds], b_masks[mb_inds], b_actions.long()[mb_inds])
                logratio = newlogprob - b_logprobs[mb_inds]
                ratio = logratio.exp()

                with torch.no_grad():
                    old_approx_kl = (-logratio).mean()
                    approx_kl = ((ratio - 1) - logratio).mean()
                    clipfracs += [((ratio - 1.0).abs() > args.clip_coef).float().mean().item()]

                mb_advantages = b_advantages[mb_inds]
                mb_advantages = (mb_advantages - mb_advantages.mean()) / (mb_advantages.std() + 1e-8)

                # Policy Loss
                pg_loss1 = -mb_advantages * ratio
                pg_loss2 = -mb_advantages * torch.clamp(ratio, 1 - args.clip_coef, 1 + args.clip_coef)
                pg_loss = torch.max(pg_loss1, pg_loss2).mean()

                # Value Loss
                newvalue = newvalue.view(-1)
                if args.clip_vloss:
                    v_loss_unclipped = (newvalue - b_returns[mb_inds]) ** 2
                    v_clipped = b_values[mb_inds] + torch.clamp(
                        newvalue - b_values[mb_inds],
                        -args.clip_coef,
                        args.clip_coef,
                    )
                    v_loss_clipped = (v_clipped - b_returns[mb_inds]) ** 2
                    v_loss_max = torch.max(v_loss_unclipped, v_loss_clipped)
                    v_loss = 0.5 * v_loss_max.mean()
                else:
                    v_loss = 0.5 * ((newvalue - b_returns[mb_inds]) ** 2).mean()

                # Entropy Bonus
                entropy_loss = entropy.mean()
                
                # Combined Loss
                loss = pg_loss - args.ent_coef * entropy_loss + v_loss * args.vf_coef

                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(agent.parameters(), args.max_grad_norm)
                optimizer.step()

            if args.target_kl is not None and approx_kl > args.target_kl:
                break

        y_pred, y_true = b_values.cpu().numpy(), b_returns.cpu().numpy()
        var_y = np.var(y_true)
        explained_var = np.nan if var_y == 0 else 1 - np.var(y_true - y_pred) / var_y

        writer.add_scalar("charts/learning_rate", optimizer.param_groups[0]["lr"], global_step)
        writer.add_scalar("charts/SPS", int(global_step / (time.time() - start_time)), global_step)
        writer.add_scalar("losses/value_loss", v_loss.item(), global_step)
        writer.add_scalar("losses/policy_loss", pg_loss.item(), global_step)
        writer.add_scalar("losses/entropy", entropy_loss.item(), global_step)
        writer.add_scalar("losses/approx_kl", approx_kl.item(), global_step)
        writer.add_scalar("losses/clipfrac", np.mean(clipfracs), global_step)
        writer.add_scalar("losses/explained_variance", explained_var, global_step)

        elapsed = time.time() - start_time
        sps = int(global_step / elapsed)
        updates_done = update - start_update + 1
        eta_seconds = elapsed / updates_done * (num_updates - update)
        pct = 100.0 * update / num_updates
        print(
            f"[update {update}/{num_updates} | {pct:.1f}% | ETA {fmt_time(eta_seconds)}]"
            f" SPS={sps}"
            f" | pg={pg_loss.item():.4f} vf={v_loss.item():.4f} ent={entropy_loss.item():.3f}"
            f" | kl={approx_kl.item():.4f} clip={np.mean(clipfracs):.3f}"
        )

        # Checkpointing
        if update % args.save_interval == 0:
            os.makedirs(f"runs/{run_name}/checkpoints", exist_ok=True)
            checkpoint = {
                "model": agent.state_dict(),
                "optimizer": optimizer.state_dict(),
                "global_step": global_step,
                "update": update,
                "args": vars(args),
            }
            torch.save(checkpoint, f"runs/{run_name}/checkpoints/step_{global_step}.pt")
            torch.save(checkpoint, f"runs/{run_name}/checkpoints/latest.pt")

            elapsed = time.time() - start_time
            summary_start = max(start_update, update - args.save_interval + 1)
            print(f"\n--- Summary (updates {summary_start}-{update}) ---")
            if recent_returns:
                print(f"  Mean episodic return: {np.mean(recent_returns):.2f} ({len(recent_returns)} episodes)")
            else:
                print(f"  Mean episodic return: N/A (0 episodes)")
            print(f"  Explained variance:  {explained_var:.4f}")
            print(f"  Learning rate:       {optimizer.param_groups[0]['lr']:.2e}")
            print(f"  Elapsed:             {fmt_time(elapsed)}")
            print()
            recent_returns.clear()

    elapsed = time.time() - start_time
    sps = int(global_step / elapsed) if elapsed > 0 else 0
    print(f"\nTraining complete. {global_step:,} steps in {fmt_time(elapsed)}. Final SPS: {sps}.")

    writer.close()
    envs.close()
