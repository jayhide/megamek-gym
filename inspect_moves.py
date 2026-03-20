#!/usr/bin/env python3
"""Inspect legal moves from a real game to understand move set composition.

Plays one episode with random actions, collecting legal move data at each step.
Prints analysis of move counts, duplicates, facing distribution, etc.
"""

import argparse
import json
import time
from collections import Counter, defaultdict

import gymnasium
import numpy as np

import megamek_gym  # noqa: F401
from megamek_gym.config import MegaMekConfig


def analyze_moves(moves: list[dict], step_label: str):
    """Analyze a set of legal moves and print summary."""
    n = len(moves)
    if n == 0:
        print(f"  {step_label}: 0 moves")
        return

    # Unique destinations (ignoring facing)
    dest_hexes = set()
    dest_facing_combos = set()
    jumping_moves = []
    walking_moves = []
    prone_moves = []
    stand_still = []

    for m in moves:
        dx, dy = m.get("dest_x", -1), m.get("dest_y", -1)
        facing = m.get("facing", 0)
        mp = m.get("mp_used", 0)
        jumping = m.get("jumping", False)
        prone = m.get("prone", False)

        dest_hexes.add((dx, dy))
        dest_facing_combos.add((dx, dy, facing, jumping))

        if jumping:
            jumping_moves.append(m)
        else:
            walking_moves.append(m)
        if prone:
            prone_moves.append(m)
        if mp == 0:
            stand_still.append(m)

    # Check for true duplicates (same dest+facing+jumping+prone)
    sigs = Counter()
    for m in moves:
        sig = (m.get("dest_x"), m.get("dest_y"), m.get("facing"),
               m.get("mp_used"), m.get("jumping", False), m.get("prone", False))
        sigs[sig] += 1
    dupes = {k: v for k, v in sigs.items() if v > 1}

    # Facing distribution
    facing_counts = Counter(m.get("facing", 0) for m in moves)

    # MP distribution
    mp_values = [m.get("mp_used", 0) for m in moves]

    # Moves per hex
    hex_move_counts = Counter((m.get("dest_x"), m.get("dest_y")) for m in moves)
    avg_moves_per_hex = n / len(dest_hexes) if dest_hexes else 0

    print(f"\n  === {step_label}: {n} total moves ===")
    print(f"  Unique destination hexes: {len(dest_hexes)}")
    print(f"  Unique (hex, facing, jump) combos: {len(dest_facing_combos)}")
    print(f"  Avg moves per hex: {avg_moves_per_hex:.1f}")
    print(f"  Walking: {len(walking_moves)}, Jumping: {len(jumping_moves)}, "
          f"Prone: {len(prone_moves)}, Stand-still (mp=0): {len(stand_still)}")
    print(f"  MP range: {min(mp_values)}-{max(mp_values)}, "
          f"median={sorted(mp_values)[len(mp_values)//2]}")
    print(f"  Facing distribution: {dict(sorted(facing_counts.items()))}")

    if dupes:
        print(f"  EXACT DUPLICATES: {len(dupes)} signatures appear >1 time")
        for sig, count in sorted(dupes.items(), key=lambda x: -x[1])[:5]:
            print(f"    {sig} appears {count}x")

    # Show hexes with highest move counts (most facing/path variants)
    top_hexes = hex_move_counts.most_common(5)
    print(f"  Top hexes by move count: {top_hexes}")

    # Show stand-still moves in detail
    if stand_still:
        print(f"  Stand-still moves ({len(stand_still)}):")
        for m in stand_still:
            print(f"    dest=({m.get('dest_x')},{m.get('dest_y')}) "
                  f"facing={m.get('facing')} jump={m.get('jumping', False)} "
                  f"prone={m.get('prone', False)}")

    # Check: do all hexes have all 6 facings for walking?
    walk_hex_facings = defaultdict(set)
    for m in walking_moves:
        walk_hex_facings[(m.get("dest_x"), m.get("dest_y"))].add(m.get("facing"))
    full_facing_hexes = sum(1 for fs in walk_hex_facings.values() if len(fs) == 6)
    partial_facing_hexes = sum(1 for fs in walk_hex_facings.values() if len(fs) < 6)
    print(f"  Walk hexes with all 6 facings: {full_facing_hexes}, "
          f"with <6 facings: {partial_facing_hexes}")

    if jumping_moves:
        jump_hex_facings = defaultdict(set)
        for m in jumping_moves:
            jump_hex_facings[(m.get("dest_x"), m.get("dest_y"))].add(m.get("facing"))
        full_j = sum(1 for fs in jump_hex_facings.values() if len(fs) == 6)
        partial_j = sum(1 for fs in jump_hex_facings.values() if len(fs) < 6)
        print(f"  Jump hexes with all 6 facings: {full_j}, with <6: {partial_j}")

    # Walk vs jump overlap
    walk_hexes = set((m.get("dest_x"), m.get("dest_y")) for m in walking_moves)
    jump_hexes = set((m.get("dest_x"), m.get("dest_y")) for m in jumping_moves)
    if jump_hexes:
        overlap = walk_hexes & jump_hexes
        jump_only = jump_hexes - walk_hexes
        print(f"  Hexes reachable by walk AND jump: {len(overlap)}, "
              f"jump-only: {len(jump_only)}")


def main():
    parser = argparse.ArgumentParser(description="Inspect MegaMek legal moves")
    parser.add_argument("--megamek-dir", type=str, default="../megamek")
    parser.add_argument("--port", type=int, default=9999)
    parser.add_argument("--config", type=str, default=None)
    parser.add_argument("--max-steps", type=int, default=30,
                        help="Max steps to collect before stopping")
    parser.add_argument("--dump-json", action="store_true",
                        help="Dump raw legal_moves JSON for first observation")
    args = parser.parse_args()

    if args.config:
        config = MegaMekConfig.load(args.config)
    else:
        config = MegaMekConfig()

    config.megamek_dir = args.megamek_dir
    config.rl_port = args.port

    print(f"Unit: {config.rl_unit} vs {config.opponent_unit}")
    print(f"Board: {config.board}")

    env = gymnasium.make("MegaMekGym/MegaMek-v0", config=config)

    obs, info = env.reset()
    n_legal = info.get("n_legal_moves", 0)

    # Access raw legal moves from the env's internal state
    raw_obs = env.unwrapped._last_raw_obs
    legal_moves = raw_obs.get("legal_moves", [])

    if args.dump_json:
        print("\n=== RAW LEGAL MOVES (first obs) ===")
        print(json.dumps(legal_moves, indent=2))

    step = 0
    all_move_counts = []
    all_collected_moves = []

    analyze_moves(legal_moves, f"Step 0 (reset) r={raw_obs.get('round','?')} {raw_obs.get('phase','?')}")
    all_move_counts.append(len(legal_moves))
    all_collected_moves.append(legal_moves)

    while step < args.max_steps:
        if n_legal > 0:
            action = np.random.randint(0, n_legal)
        else:
            action = 0

        obs, reward, terminated, truncated, info = env.step(action)
        step += 1
        n_legal = info.get("n_legal_moves", 0)

        if terminated or truncated:
            print(f"\n  Game ended at step {step}")
            break

        raw_obs = env.unwrapped._last_raw_obs
        legal_moves = raw_obs.get("legal_moves", [])
        all_move_counts.append(len(legal_moves))
        all_collected_moves.append(legal_moves)

        analyze_moves(legal_moves,
                      f"Step {step} r={raw_obs.get('round','?')} {raw_obs.get('phase','?')}")

    # Summary
    print(f"\n{'='*70}")
    print(f"SUMMARY over {len(all_move_counts)} observations:")
    print(f"  Move counts: {all_move_counts}")
    print(f"  Min: {min(all_move_counts)}, Max: {max(all_move_counts)}, "
          f"Mean: {np.mean(all_move_counts):.0f}, Median: {np.median(all_move_counts):.0f}")

    # Duplicate analysis across all collected steps
    print(f"\n{'='*70}")
    print("DUPLICATE ANALYSIS (moves with identical features visible to agent):")
    print("  Signature = (dest_x, dest_y, facing, mp_used, jumping, prone)")
    total_moves = 0
    total_unique = 0
    total_dupes = 0
    dupe_counts_per_step = []
    for i, moves in enumerate(all_collected_moves):
        sigs = Counter()
        for m in moves:
            sig = (m.get("dest_x"), m.get("dest_y"), m.get("facing"),
                   m.get("mp_used"), m.get("jumping", False), m.get("prone", False))
            sigs[sig] += 1
        n_total = len(moves)
        n_unique = len(sigs)
        n_dupe_moves = n_total - n_unique  # extra copies beyond the first
        total_moves += n_total
        total_unique += n_unique
        total_dupes += n_dupe_moves
        dupe_counts_per_step.append(n_dupe_moves)

    print(f"  Total moves across all steps: {total_moves}")
    print(f"  Total unique signatures:      {total_unique}")
    print(f"  Total duplicate moves:        {total_dupes} "
          f"({100*total_dupes/total_moves:.1f}% of all moves)")
    print(f"  Duplicates per step: {dupe_counts_per_step}")
    if dupe_counts_per_step:
        print(f"  Avg dupes/step: {np.mean(dupe_counts_per_step):.1f}, "
              f"max: {max(dupe_counts_per_step)}")

    env.close()


if __name__ == "__main__":
    main()
