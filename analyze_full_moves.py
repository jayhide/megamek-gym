#!/usr/bin/env python3
"""Measure hidden walking paths in RL legal moves.

Plays N games with random actions and perf_log=true, then parses the Java log
for [rl-moves] diagnostic lines to compare the current (longest-only) path set
against the full Pareto frontier.

Requires the Java-side diagnostic logging (gated behind perfLog) that calls
getAllComputedPathsUnordered() alongside getLongestComputedPaths() and logs:
    [rl-moves] turn #N: current=X full=Y delta=+Z walk_hidden=W
    [rl-moves] game_summary: turns=N avg_current=X avg_full=Y ...

Usage:
    poetry run python analyze_full_moves.py --megamek-dir ../megamek --games 5
"""

import argparse
import re
import time
from pathlib import Path

import gymnasium
import numpy as np

import megamek_gym  # noqa: F401
from megamek_gym.config import MegaMekConfig

# Regex for per-turn [rl-moves] lines
TURN_RE = re.compile(
    r"\[rl-moves\] turn #(\d+): current=(\d+) full=(\d+) delta=\+(\d+) walk_hidden=(\d+)"
)
SUMMARY_RE = re.compile(
    r"\[rl-moves\] game_summary: turns=(\d+) avg_current=(\d+) avg_full=(\d+) "
    r"avg_delta=\+(\d+) avg_walk_hidden=(\d+)"
)


def parse_log(log_path: Path) -> list[dict]:
    """Parse [rl-moves] turn lines from the Java log file."""
    turns = []
    if not log_path.exists():
        return turns
    with open(log_path) as f:
        for line in f:
            m = TURN_RE.search(line)
            if m:
                turns.append({
                    "turn": int(m.group(1)),
                    "current": int(m.group(2)),
                    "full": int(m.group(3)),
                    "delta": int(m.group(4)),
                    "walk_hidden": int(m.group(5)),
                })
    return turns


def print_summary(turns: list[dict], games: int) -> None:
    """Print aggregate statistics."""
    if not turns:
        print("\nNo [rl-moves] data found in Java log!")
        return

    # Split into mobile turns (current > 3) and immobile turns
    mobile = [t for t in turns if t["current"] > 3]
    immobile = [t for t in turns if t["current"] <= 3]

    print(f"\n{'=' * 70}")
    print(f"RESULTS: {len(turns)} movement turns across {games} games")
    print(f"  Mobile turns: {len(mobile)}, Immobile turns: {len(immobile)}")
    print(f"{'=' * 70}")

    if not mobile:
        print("\nNo mobile turns to analyze!")
        return

    currents = [t["current"] for t in mobile]
    fulls = [t["full"] for t in mobile]
    deltas = [t["delta"] for t in mobile]
    walk_hiddens = [t["walk_hidden"] for t in mobile]

    print(f"\n  Current legal moves (longest-only):")
    print(f"    min={min(currents)}, max={max(currents)}, "
          f"mean={np.mean(currents):.0f}, median={np.median(currents):.0f}")

    print(f"\n  Full legal moves (all Pareto frontier paths):")
    print(f"    min={min(fulls)}, max={max(fulls)}, "
          f"mean={np.mean(fulls):.0f}, median={np.median(fulls):.0f}")

    print(f"\n  Delta (full - current):")
    print(f"    min={min(deltas)}, max={max(deltas)}, "
          f"mean={np.mean(deltas):.0f}, median={np.median(deltas):.0f}")

    total_current = sum(currents)
    total_full = sum(fulls)
    total_delta = sum(deltas)
    total_walk_hidden = sum(walk_hiddens)
    pct_hidden = 100 * total_delta / total_current if total_current > 0 else 0

    print(f"\n  Aggregate:")
    print(f"    Total current moves:  {total_current}")
    print(f"    Total full moves:     {total_full}")
    print(f"    Total hidden moves:   {total_delta} ({pct_hidden:.1f}% more in full set)")
    print(f"    Total walk hidden:    {total_walk_hidden} "
          f"({100 * total_walk_hidden / total_delta:.0f}% of hidden are walk-speed)"
          if total_delta > 0 else "")

    print(f"\n  Walk-speed paths hidden per turn:")
    print(f"    min={min(walk_hiddens)}, max={max(walk_hiddens)}, "
          f"mean={np.mean(walk_hiddens):.0f}, median={np.median(walk_hiddens):.0f}")

    # Distribution of delta as % of current
    pct_deltas = [100 * t["delta"] / t["current"] for t in mobile if t["current"] > 0]
    if pct_deltas:
        print(f"\n  Delta as % of current (per-turn):")
        print(f"    min={min(pct_deltas):.0f}%, max={max(pct_deltas):.0f}%, "
              f"mean={np.mean(pct_deltas):.0f}%, median={np.median(pct_deltas):.0f}%")

    # Would the full set exceed 400 cap?
    exceeds_400_current = sum(1 for c in currents if c > 400)
    exceeds_400_full = sum(1 for f in fulls if f > 400)
    print(f"\n  Exceeds 400-move cap:")
    print(f"    Current set: {exceeds_400_current}/{len(mobile)} turns")
    print(f"    Full set:    {exceeds_400_full}/{len(mobile)} turns")


def main():
    parser = argparse.ArgumentParser(
        description="Measure hidden walking paths in RL legal moves"
    )
    parser.add_argument("--megamek-dir", type=str, default="../megamek")
    parser.add_argument("--config", type=str, default=None,
                        help="YAML config file (default: sanity_check.yaml settings)")
    parser.add_argument("--port", type=int, default=None)
    parser.add_argument("--games", type=int, default=5, help="Number of games to play")
    parser.add_argument("--max-steps", type=int, default=50, help="Max steps per game")
    args = parser.parse_args()

    if args.config:
        config = MegaMekConfig.load(args.config)
    else:
        config = MegaMekConfig()
        config.rl_unit = "Commando COM-2D"
        config.opponent_unit = "Commando COM-2D"
        config.max_game_rounds = 40
        config.firing_strategy = "naive"
        config.max_rotating_round_saves = 0

    config.megamek_dir = args.megamek_dir
    config.perf_log = True  # Required for [rl-moves] diagnostics
    if args.port is not None:
        config.rl_port = args.port

    port = config.rl_port
    log_path = Path(config.megamek_dir).resolve() / f"rl_java_{port}.log"

    print(f"Unit: {config.rl_unit} vs {config.opponent_unit}")
    print(f"Board: {config.board}")
    print(f"Games: {args.games}, max steps/game: {args.max_steps}")
    print(f"Port: {port}, perf_log: {config.perf_log}")
    print(f"Java log: {log_path}")

    env = gymnasium.make("MegaMekGym/MegaMek-v0", config=config)

    total_steps = 0
    games_played = 0
    t0 = time.monotonic()

    for game_idx in range(args.games):
        print(f"\n--- Game {game_idx + 1}/{args.games} ---")

        obs, info = env.reset()
        n_legal = info.get("n_legal_moves", 0)

        step = 0
        while step < args.max_steps:
            if n_legal > 0:
                action = np.random.randint(0, n_legal)
            else:
                action = 0

            obs, reward, terminated, truncated, info = env.step(action)
            step += 1
            n_legal = info.get("n_legal_moves", 0)

            if step <= 2 or step % 10 == 0:
                print(f"  step {step}: n_legal={n_legal}, reward={reward:.3f}")

            if terminated or truncated:
                outcome = "terminated" if terminated else "truncated"
                print(f"  Game ended ({outcome}) at step {step}")
                break

        total_steps += step
        games_played += 1

    elapsed = time.monotonic() - t0
    print(f"\nPlayed {games_played} games, {total_steps} total steps in {elapsed:.1f}s")

    env.close()

    # Give Java a moment to flush logs
    time.sleep(1)

    # Parse and summarize
    turns = parse_log(log_path)
    print_summary(turns, games_played)

    # Also print raw game summaries from the log
    if log_path.exists():
        print(f"\n{'=' * 70}")
        print("Raw game summaries from Java log:")
        with open(log_path) as f:
            for line in f:
                if "[rl-moves] game_summary" in line:
                    # Extract just the [rl-moves] part
                    idx = line.index("[rl-moves]")
                    print(f"  {line[idx:].strip()}")


if __name__ == "__main__":
    main()
