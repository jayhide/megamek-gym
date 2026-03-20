"""Run games synchronously with autosaves enabled for visual inspection in MegaMek UI.

Uses the exact same config as training, but enables round saves and collects them
into per-game directories so you can load them in the MegaMek GUI.

Usage:
    # Random actions
    poetry run python inspect_games.py --config configs/default.yaml --num-games 3

    # Trained policy
    poetry run python inspect_games.py --config configs/default.yaml --num-games 5 \
        --checkpoint runs/megamek-ppo__1__*/checkpoints/latest.pt --deterministic
"""

import argparse
from distutils.util import strtobool
from pathlib import Path
import re
import shutil
import random

import numpy as np
import gymnasium as gym

from megamek_gym.agent import load_agent, select_action, OUTCOME_MAP
from megamek_gym.config import MegaMekConfig
from megamek_gym.reward import CompositeReward


def parse_args():
    parser = argparse.ArgumentParser(description="Run MegaMek games with saves for inspection")

    parser.add_argument("--config", type=str, default=None, help="Training config YAML")
    parser.add_argument("--megamek-dir", type=str, default="../megamek")
    parser.add_argument("--port", type=int, default=9999)
    parser.add_argument("--num-games", type=int, default=5)
    parser.add_argument("--output-dir", type=str, default="inspected_games")
    parser.add_argument("--checkpoint", type=str, default=None, help="Path to .pt checkpoint for trained policy")
    parser.add_argument("--deterministic", type=lambda x: bool(strtobool(x)),
                        default=False, nargs="?", const=True,
                        help="Use greedy action selection (only with --checkpoint)")
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--transcript", action="store_true",
                        help="Print round-by-round transcript after each game")
    parser.add_argument("--verbose", "-v", action="store_true",
                        help="Show damage details in transcript (requires --transcript)")

    return parser.parse_args()


def clear_saves(megamek_dir, port):
    """Delete all Round-*.sav.gz and autosave_*.sav.gz files from the JVM savegames dir."""
    savegames_dir = Path(megamek_dir) / "megamek" / f"run_{port}" / "savegames"
    if not savegames_dir.is_dir():
        return
    for f in savegames_dir.glob("Round-*.sav.gz"):
        f.unlink()
    for f in savegames_dir.glob("autosave_*.sav.gz"):
        f.unlink()


def collect_saves(megamek_dir, port, archive_dir):
    """Move saves from JVM dir to archive, and copy to MegaMek's main savegames dir for Replay."""
    savegames_dir = Path(megamek_dir) / "megamek" / f"run_{port}" / "savegames"
    replay_dir = Path(megamek_dir) / "megamek" / "savegames"
    if not savegames_dir.is_dir():
        return 0

    archive_dir.mkdir(parents=True, exist_ok=True)
    # Clear old saves from previous runs
    for f in archive_dir.glob("Round-*.sav.gz"):
        f.unlink()
    for f in archive_dir.glob("autosave_*.sav.gz"):
        f.unlink()
    replay_dir.mkdir(parents=True, exist_ok=True)

    # Clear old saves from the main replay dir
    for f in replay_dir.glob("Round-*.sav.gz"):
        f.unlink()
    for f in replay_dir.glob("autosave_*.sav.gz"):
        f.unlink()

    count = 0
    for f in sorted(savegames_dir.glob("Round-*.sav.gz")):
        shutil.copy2(str(f), replay_dir / f.name)
        shutil.move(str(f), archive_dir / f.name)
        count += 1
    # Also collect autosave (end-of-game state with final round's combat reports)
    for f in savegames_dir.glob("autosave_*.sav.gz"):
        shutil.copy2(str(f), replay_dir / f.name)
        shutil.move(str(f), archive_dir / f.name)
        count += 1
    return count


def print_game_transcript(game_dir, step_log, outcome="UNKNOWN", verbose=False):
    """Print round-by-round transcript with RL annotations."""
    from transcript import (
        parse_save, get_round_reports, diff_units,
        decode_combat_reports, decode_damage_reports,
        determine_end_condition,
        armor_summary, bold, cyan, green, yellow, red,
    )

    save_files = sorted(
        game_dir.glob("Round-*.sav.gz"),
        key=lambda p: int(re.match(r"Round-(\d+)-", p.name).group(1)),
    )
    if not save_files:
        return

    # Group step_log by round
    steps_by_round = {}
    for s in step_log:
        steps_by_round.setdefault(s["round"], []).append(s)

    saves = [parse_save(sf) for sf in save_files]

    # Header
    if saves and saves[0]["units"]:
        unit_names = [u["name"] for u in saves[0]["units"]]
        print(bold(f"\n=== Transcript: {' vs '.join(unit_names)} ==="))

    for i, save in enumerate(saves):
        round_num = save["round"]
        prev_save = saves[i - 1] if i > 0 else None

        print(bold(f"\n--- Round {round_num} ---"))

        # Movement diff
        if prev_save:
            move_lines = diff_units(prev_save["units"], save["units"])
            if move_lines:
                print(f"  {cyan('Movement:')}")
                for line in move_lines:
                    print(line)

        # Combat reports (only new ones from this round)
        round_reports = get_round_reports(prev_save, save)
        combat_lines = decode_combat_reports(round_reports)
        if combat_lines:
            print(f"\n  {cyan('Weapons fire:')}")
            for line in combat_lines:
                print(line)

        # Damage details (verbose mode)
        if verbose:
            damage_lines = decode_damage_reports(round_reports)
            if damage_lines:
                print(f"\n  {cyan('Damage:')}")
                for line in damage_lines:
                    print(line)

        # Unit status summary
        if save["units"]:
            print()
            for unit in save["units"]:
                print(armor_summary(unit))

        # RL annotations for this round
        round_steps = steps_by_round.get(round_num, [])
        if round_steps:
            total_reward = sum(s["reward"] for s in round_steps)
            reward_color = green if total_reward > 0 else (red if total_reward < 0 else yellow)
            print(f"\n  {cyan('RL Agent:')}")
            for s in round_steps:
                r = s["reward"]
                r_str = f"{r:+.3f}"
                # MegaMek calls the end-of-game phase "VICTORY" regardless of who won;
                # rename to avoid confusion
                phase = "GAME_END" if s["phase"] == "VICTORY" else s["phase"]
                print(f"    {phase}: action {s['action']}/{s['n_legal_moves']} legal"
                      f"  ->  reward {r_str}")
                # Show per-component reward breakdown
                details = s.get("reward_details", [])
                nonzero = [(name, raw, weighted) for name, raw, weighted in details
                           if abs(weighted) > 1e-6]
                if nonzero:
                    parts = []
                    for name, raw, weighted in nonzero:
                        # Shorten class names for readability
                        short = (name.replace("Reward", "")
                                 .replace("DamageDelta", "Damage")
                                 .replace("LocationDestruction", "LocDestroy")
                                 .replace("RangeAdvantage", "Range")
                                 .replace("WinLoss", "WinLoss"))
                        parts.append(f"{short}={weighted:+.3f}")
                    print(f"           ({', '.join(parts)})")
            cum = round_steps[-1]["cumulative_return"]
            print(f"    Cumulative return: {reward_color(f'{cum:+.3f}')}")

    # Check for autosave (end-of-game state with final round's combat reports)
    autosave_files = sorted(game_dir.glob("autosave_*.sav.gz"))
    if autosave_files and saves:
        final_save = parse_save(autosave_files[-1])
        last_round_save = saves[-1]

        # Extract combat reports from the final round (diff against last Round save)
        final_reports = get_round_reports(last_round_save, final_save)

        if final_reports:
            print(bold(f"\n--- Final attacks (Round {last_round_save['round']}) ---"))

            combat_lines = decode_combat_reports(final_reports)
            if combat_lines:
                print(f"\n  {cyan('Weapons fire:')}")
                for line in combat_lines:
                    print(line)

            if verbose:
                damage_lines = decode_damage_reports(final_reports)
                if damage_lines:
                    print(f"\n  {cyan('Damage:')}")
                    for line in damage_lines:
                        print(line)

            # Show final unit states from autosave
            if final_save["units"]:
                print()
                for unit in final_save["units"]:
                    print(armor_summary(unit))

    # Game outcome footer
    outcome_color = green if outcome == "WIN" else (red if outcome == "LOSS" else yellow)
    final_units = []
    if autosave_files and saves:
        final_units = final_save["units"]
    elif saves:
        final_units = saves[-1]["units"]

    end_condition = determine_end_condition(final_units, outcome) if final_units else ""
    summary = outcome
    if end_condition:
        summary += f" — {end_condition}"
    print(bold(f"\n=== {outcome_color(summary)} ==="))

    if step_log:
        cum = step_log[-1]["cumulative_return"]
        reward_color = green if cum > 0 else (red if cum < 0 else yellow)
        print(f"Final return: {reward_color(f'{cum:+.3f}')}")
    print()


def main():
    args = parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)

    # Load config (same as training) and enable saves
    cfg = MegaMekConfig.load(args.config) if args.config else MegaMekConfig()
    cfg.megamek_dir = args.megamek_dir
    cfg.rl_port = args.port
    cfg.env_index = 0
    cfg.max_rotating_round_saves = 100
    cfg.save_budget_mb = 10000  # Don't delete saves during run

    output_dir = Path(args.output_dir)

    env = gym.make("MegaMekGym/MegaMek-v0", config=cfg)

    # Load trained policy if provided (after env creation so dimensions are known)
    agent = None
    device = None
    if args.checkpoint:
        obs_size = env.observation_space.shape[0]
        action_size = env.action_space.n
        agent, checkpoint, device = load_agent(args.checkpoint, obs_size, action_size)
        print(f"Loaded checkpoint: {args.checkpoint}")
        print(f"  global_step={checkpoint.get('global_step', '?')}, deterministic={args.deterministic}")

    print(f"Running {args.num_games} games, saves → {output_dir}/")
    print(f"  config: {args.config or 'defaults'}")
    print(f"  policy: {'checkpoint' if agent else 'random'}")
    print()

    results = []

    # Clear stale saves from previous runs before starting
    clear_saves(args.megamek_dir, args.port)

    for ep in range(1, args.num_games + 1):
        obs, info = env.reset()
        done = False
        episode_return = 0.0
        steps = 0
        step_log = []

        while not done:
            if agent is not None:
                action = select_action(agent, obs, info["action_mask"], device, args.deterministic)
            else:
                action = np.random.randint(0, info.get("n_legal_moves", 1))

            obs, reward, terminated, truncated, info = env.step(action)
            episode_return += reward
            steps += 1
            done = terminated or truncated

            # Capture per-component reward breakdown
            reward_fn = env.unwrapped.reward_fn
            reward_details = []
            if isinstance(reward_fn, CompositeReward):
                reward_details = [
                    (name, raw, weighted)
                    for name, raw, weighted in reward_fn.last_details
                ]

            step_log.append({
                "round": info.get("round", 0),
                "phase": info.get("phase", ""),
                "action": action,
                "n_legal_moves": info.get("n_legal_moves", 0),
                "reward": reward,
                "reward_details": reward_details,
                "cumulative_return": episode_return,
            })

        outcome = OUTCOME_MAP.get(info.get("game_outcome", 0), "UNKNOWN")
        game_rounds = info.get("game_rounds", "?")

        # Collect saves before next reset overwrites them
        game_dir = output_dir / f"game_{ep:03d}_{outcome}"
        n_saves = collect_saves(args.megamek_dir, args.port, game_dir)

        results.append((outcome, game_rounds, steps, episode_return, n_saves, game_dir))
        print(f"Game {ep:3d}: {outcome:4s}  rounds={game_rounds}  steps={steps:4d}  "
              f"reward={episode_return:7.2f}  saves={n_saves}")

        if args.transcript and n_saves > 0:
            print_game_transcript(game_dir, step_log, outcome=outcome, verbose=args.verbose)

    env.close()

    # Summary
    wins = sum(1 for r in results if r[0] == "WIN")
    losses = sum(1 for r in results if r[0] == "LOSS")
    draws = sum(1 for r in results if r[0] == "DRAW")
    n = len(results)

    replay_dir = Path(args.megamek_dir) / "megamek" / "savegames"

    print()
    print(f"=== Summary ({n} games) ===")
    print(f"Record: {wins}W / {losses}L / {draws}D")
    print(f"Last game ready for Replay in: {replay_dir}")
    print(f"All games archived in: {output_dir}/")
    for outcome, rounds, steps, ret, saves, gdir in results:
        print(f"  {gdir}/  ({saves} saves)")
    if n > 1:
        print(f"  To replay an older game, copy its saves to {replay_dir}/")

    # Write summary file
    output_dir.mkdir(parents=True, exist_ok=True)
    with open(output_dir / "summary.txt", "w") as f:
        f.write(f"Config: {args.config or 'defaults'}\n")
        f.write(f"Checkpoint: {args.checkpoint or 'random'}\n")
        f.write(f"Deterministic: {args.deterministic}\n")
        f.write(f"Record: {wins}W / {losses}L / {draws}D\n\n")
        for i, (outcome, rounds, steps, ret, saves, gdir) in enumerate(results, 1):
            f.write(f"Game {i:03d}: {outcome:4s}  rounds={rounds}  steps={steps}  "
                    f"reward={ret:.2f}  saves={saves}  dir={gdir}\n")


if __name__ == "__main__":
    main()
