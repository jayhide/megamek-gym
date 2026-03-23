#!/usr/bin/env python3
"""Consolidated smoke test for the MegaMek RL training bridge.

Covers all critical integration paths between Python (megamek-gym) and Java (megamek).
Auto-starts Java via the Gymnasium env — no manual two-terminal setup needed.

Tests:
  1. Basic Episode      — reset, random moves, game ends, obs shape correct
  2. Truncation         — low round limit triggers truncated=True
  3. Termination        — natural game end triggers terminated=True
  4. Persistent Reset   — 3 episodes on same JVM without cold restart
  5. Cross-Validation   — Python tactical features match Java calculations

Usage:
    poetry run python smoke_test_all.py [--megamek-dir ../megamek] [--port 9999] [--verbose]

Runs in ~2-3 minutes. Exit code 0 if all tests pass, 1 if any fail.
"""

import argparse
import re
import signal
import sys
import time
import traceback
from pathlib import Path

import gymnasium  # noqa: F401
import numpy as np

import megamek_gym  # noqa: F401 — registers MegaMekGym/MegaMek-v0
from megamek_gym.config import MegaMekConfig
from megamek_gym.observation import (
    compute_obs_size, format_observation,
    MOVE_FEATURES, MOVE_FEATURE_NAMES, UNIT_FEATURES, GLOBAL_FEATURES,
)
from tests.test_cross_validation import validate_observation


# ---------------------------------------------------------------------------
# Timeout mechanism
# ---------------------------------------------------------------------------

class TestTimeout(Exception):
    pass


def _alarm_handler(signum, frame):
    raise TestTimeout("Test timed out (120s)")


signal.signal(signal.SIGALRM, _alarm_handler)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

DEFAULT_CONFIG_PATH = Path(__file__).parent / "configs" / "default.yaml"


def base_config(megamek_dir, port, **overrides):
    """Load default.yaml and apply per-test overrides."""
    config = MegaMekConfig.load(str(DEFAULT_CONFIG_PATH))
    config.megamek_dir = megamek_dir
    config.rl_port = port
    for k, v in overrides.items():
        setattr(config, k, v)
    return config


def read_peak_memory(megamek_dir, port):
    """Read peak JVM memory from rl_java_{port}.log [rl-mem] lines."""
    log_path = Path(megamek_dir) / f"rl_java_{port}.log"
    if not log_path.exists():
        return None
    peak_used = 0
    max_heap = 0
    for line in log_path.read_text().splitlines():
        m = re.search(r"\[rl-mem\].*used=(\d+)MB.*max=(\d+)MB", line)
        if m:
            used = int(m.group(1))
            heap = int(m.group(2))
            peak_used = max(peak_used, used)
            max_heap = max(max_heap, heap)
    if peak_used > 0:
        return peak_used, max_heap
    return None


def run_episode(env, max_steps=500, action_fn=None, verbose=False):
    """Run one episode. Returns (steps, terminated, truncated, final_info, obs_shape)."""
    obs, info = env.reset()
    obs_shape = obs.shape
    n_legal = info.get("n_legal_moves", 0)

    if verbose:
        print(f"    Reset complete. Obs shape: {obs_shape}, legal moves: {n_legal}")
        inner = env.unwrapped
        if inner._last_raw_obs:
            print(format_observation(inner._last_raw_obs, inner._rl_owner_id))

    step = 0
    while True:
        if action_fn is not None:
            action = action_fn(info)
        else:
            n_legal = info.get("n_legal_moves", 0)
            action = np.random.randint(0, max(n_legal, 1))

        obs, reward, terminated, truncated, info = env.step(action)
        step += 1

        if verbose:
            print(
                f"    Step {step:3d} r={info.get('round', '?'):>2} "
                f"{info.get('phase', '?'):<14s} "
                f"reward={reward:+.3f} moves={info.get('n_legal_moves', 0):>4d}"
            )

        if terminated or truncated:
            if verbose:
                inner = env.unwrapped
                if inner._last_raw_obs:
                    print(format_observation(inner._last_raw_obs, inner._rl_owner_id))
            return step, terminated, truncated, info, obs_shape

        if step >= max_steps:
            raise RuntimeError(f"Episode did not end after {max_steps} steps")


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_basic_episode(megamek_dir, port, verbose):
    """Basic episode: reset, random moves, game ends, obs shape correct."""
    config = base_config(megamek_dir, port,
                         max_game_rounds=50, perf_log=True)

    env = gymnasium.make("MegaMekGym/MegaMek-v0", config=config)
    try:
        steps, terminated, truncated, info, obs_shape = run_episode(
            env, verbose=verbose
        )

        bw = config.resolved_board_width
        bh = config.resolved_board_height
        expected_size = compute_obs_size(bw, bh, config.max_legal_moves)
        if obs_shape != (expected_size,):
            return False, f"obs shape {obs_shape} != expected ({expected_size},)"

        if not (terminated or truncated):
            return False, "Episode did not terminate or truncate"

        detail = (
            f"{steps} steps, terminated={terminated}, truncated={truncated}, "
            f"obs_shape={obs_shape}"
        )
        return True, detail
    finally:
        env.close()


def test_truncation(megamek_dir, port, verbose):
    """Truncation: low round limit -> truncated=True, terminated=False."""
    config = base_config(megamek_dir, port,
                         max_game_rounds=3, perf_log=True)

    env = gymnasium.make("MegaMekGym/MegaMek-v0", config=config)
    try:
        steps, terminated, truncated, info, _ = run_episode(
            env, action_fn=lambda _info: 0, verbose=verbose
        )

        if not truncated:
            return False, f"Expected truncated=True, got truncated={truncated}"
        if terminated:
            return False, (
                f"Expected terminated=False, got terminated={terminated} "
                "(unit destroyed before round limit — try increasing max_game_rounds)"
            )

        detail = (
            f"{steps} steps, truncated={truncated}, "
            f"round={info.get('round', '?')}"
        )
        return True, detail
    finally:
        env.close()


def test_termination(megamek_dir, port, verbose):
    """Termination: natural game end -> terminated=True, truncated=False."""
    config = base_config(megamek_dir, port,
                         rl_unit="Locust LCT-1V",
                         opponent_unit="Trebuchet TBT-5S",
                         max_game_rounds=0,
                         perf_log=True)

    env = gymnasium.make("MegaMekGym/MegaMek-v0", config=config)
    try:
        steps, terminated, truncated, info, _ = run_episode(
            env, max_steps=1000, verbose=verbose
        )

        if not terminated:
            return False, f"Expected terminated=True, got terminated={terminated}"
        if truncated:
            return False, f"Expected truncated=False, got truncated={truncated}"

        outcome = info.get("game_outcome", "?")
        detail = (
            f"{steps} steps, terminated={terminated}, "
            f"game_outcome={outcome}"
        )
        return True, detail
    finally:
        env.close()


def test_cross_validation(megamek_dir, port, verbose):
    """Cross-validation: compare Python tactical features against Java."""
    config = base_config(megamek_dir, port,
                         max_game_rounds=50, perf_log=True)

    env = gymnasium.make("MegaMekGym/MegaMek-v0", config=config)
    try:
        obs, info = env.reset()
        inner = env.unwrapped
        all_mismatches = []
        total_moves_checked = 0
        step = 0

        while True:
            raw_obs = inner._last_raw_obs
            if raw_obs and raw_obs.get("legal_moves"):
                mismatches = validate_observation(raw_obs, inner._rl_owner_id)
                total_moves_checked += len(raw_obs["legal_moves"])
                if mismatches:
                    all_mismatches.extend(mismatches)
                    if verbose:
                        for m in mismatches:
                            print(f"    MISMATCH: {m}")

            n_legal = info.get("n_legal_moves", 0)
            action = np.random.randint(0, max(n_legal, 1))
            obs, reward, terminated, truncated, info = env.step(action)
            step += 1

            if terminated or truncated:
                break
            if step >= 500:
                raise RuntimeError("Episode did not end after 500 steps")

        if all_mismatches:
            return False, (
                f"{len(all_mismatches)} mismatches in {total_moves_checked} moves "
                f"({step} steps). First: {all_mismatches[0]}"
            )

        return True, (
            f"{step} steps, {total_moves_checked} moves checked, "
            f"0 mismatches"
        )
    finally:
        env.close()


def test_persistent_reset(megamek_dir, port, verbose):
    """Persistent reset: 3 episodes on same JVM without cold restart."""
    config = base_config(megamek_dir, port,
                         max_game_rounds=3, perf_log=True)

    num_episodes = 3
    env = gymnasium.make("MegaMekGym/MegaMek-v0", config=config)
    try:
        episode_results = []
        for ep in range(num_episodes):
            steps, terminated, truncated, info, _ = run_episode(
                env, action_fn=lambda _info: 0, verbose=verbose
            )
            label = "cold start" if ep == 0 else "persistent reset"
            if verbose:
                print(f"    Episode {ep+1}: {steps} steps, truncated={truncated} ({label})")
            episode_results.append((steps, terminated, truncated))

        for i, (steps, terminated, truncated) in enumerate(episode_results):
            if not truncated:
                return False, f"Episode {i+1}: expected truncated=True, got {truncated}"
            if terminated:
                return False, f"Episode {i+1}: expected terminated=False, got {terminated}"

        summaries = [f"ep{i+1}={s} steps" for i, (s, _, _) in enumerate(episode_results)]
        return True, f"{num_episodes} episodes completed ({', '.join(summaries)})"
    finally:
        env.close()


def test_auto_wake_pilot(megamek_dir, port, verbose):
    """Auto-wake pilot: force unconscious on turn 1, verify auto-wake restores movement."""
    config = base_config(megamek_dir, port,
                         max_game_rounds=50,
                         auto_wake_pilot=True,
                         force_unconscious_on_turn=1)

    env = gymnasium.make("MegaMekGym/MegaMek-v0", config=config)
    try:
        obs, info = env.reset()
        saw_auto_wake = False
        step = 0

        while True:
            n_legal = info.get("n_legal_moves", 0)
            auto_wake_count = info.get("auto_wake_count", 0)

            if auto_wake_count > 0 and not saw_auto_wake:
                saw_auto_wake = True
                if verbose:
                    print(f"    Step {step}: auto-wake fired (count={auto_wake_count}), "
                          f"n_legal_moves={n_legal}")
                # After auto-wake, the unit should have full movement (> 1 legal move)
                if n_legal <= 1:
                    return False, (
                        f"Step {step}: auto-wake fired but n_legal_moves={n_legal} "
                        "(expected > 1 after waking pilot)"
                    )

            action = np.random.randint(0, max(n_legal, 1))
            obs, reward, terminated, truncated, info = env.step(action)
            step += 1

            if verbose and step <= 5:
                print(
                    f"    Step {step:3d} r={info.get('round', '?'):>2} "
                    f"reward={reward:+.3f} moves={info.get('n_legal_moves', 0):>4d} "
                    f"auto_wake={info.get('auto_wake_count', 0)}"
                )

            if terminated or truncated:
                break
            if step >= 500:
                return False, "Episode did not end after 500 steps"

        if not saw_auto_wake:
            return False, (
                "force_unconscious_on_turn=1 but auto_wake_count never > 0 "
                "(auto-wake did not fire)"
            )

        return True, (
            f"{step} steps, auto-wake verified "
            f"(pilot forced unconscious on turn 1, woken with full movement)"
        )
    finally:
        env.close()


def test_fixed_deployment(megamek_dir, port, verbose):
    """Fixed deployment: units deploy at exact coords, consistent across episodes."""
    config = base_config(megamek_dir, port,
                         max_game_rounds=3,
                         rl_fixed_coords=(8, 2),
                         opponent_fixed_coords=(8, 14))

    board_w = config.resolved_board_width
    board_h = config.resolved_board_height
    board_size = board_w * board_h

    num_episodes = 2
    env = gymnasium.make("MegaMekGym/MegaMek-v0", config=config)
    try:
        initial_obs_list = []
        for ep in range(num_episodes):
            obs, info = env.reset()
            initial_obs_list.append(obs.copy())
            n_legal = info.get("n_legal_moves", 0)
            if verbose:
                print(f"    Episode {ep+1}: reset ok, n_legal_moves={n_legal}")

            # Play out the episode (stand still)
            steps = 0
            while True:
                obs, reward, terminated, truncated, info = env.step(0)
                steps += 1
                if terminated or truncated:
                    break
                if steps >= 500:
                    return False, f"Episode {ep+1} did not end after 500 steps"

            if verbose:
                print(f"    Episode {ep+1}: {steps} steps, "
                      f"terminated={terminated}, truncated={truncated}")

        # Verify RL unit position from the observation. The 55 RL unit features
        # start at index board_size. Position features (first 2: x/W, y/H)
        # should match the fixed coords in both episodes.
        rl_state_start = board_size
        for ep in range(num_episodes):
            rl_x_norm = initial_obs_list[ep][rl_state_start]
            rl_y_norm = initial_obs_list[ep][rl_state_start + 1]
            actual_x = round(rl_x_norm * board_w)
            actual_y = round(rl_y_norm * board_h)
            if verbose:
                print(f"    Episode {ep+1}: RL unit at ({actual_x}, {actual_y}) "
                      f"(expected (8, 2))")
            if (actual_x, actual_y) != (8, 2):
                return False, (
                    f"Episode {ep+1}: RL unit at ({actual_x}, {actual_y}), "
                    f"expected (8, 2)"
                )

        return True, (
            f"{num_episodes} episodes, RL unit deployed at (8, 2) in both"
        )
    finally:
        env.close()


def test_board_consistency(megamek_dir, port, verbose):
    """Board consistency: board loaded from file is identical across resets.

    Catches the bug where a path format mismatch in scanForBoards caused the
    server to discard the configured board name as "unavailable" and silently
    fall back to a randomly-generated board every game.
    """
    config = base_config(megamek_dir, port,
                         max_game_rounds=3,
                         rl_fixed_coords=(8, 2),
                         opponent_fixed_coords=(8, 14))

    num_episodes = 3
    env = gymnasium.make("MegaMekGym/MegaMek-v0", config=config)
    try:
        boards = []         # list of {(x,y): elevation} dicts
        legal_counts = []   # n_legal_moves at first step

        for ep in range(num_episodes):
            obs, info = env.reset()
            raw_obs = env.unwrapped._last_raw_obs
            n_legal = info.get("n_legal_moves", 0)
            legal_counts.append(n_legal)

            # Extract board elevations
            board_hexes = raw_obs.get("board", {}).get("hexes", [])
            elev_map = {(h["x"], h["y"]): h.get("elevation", 0) for h in board_hexes}
            boards.append(elev_map)

            if verbose:
                print(f"    Episode {ep+1}: n_legal={n_legal}, "
                      f"board hexes={len(elev_map)}, "
                      f"elev range=[{min(elev_map.values())}, {max(elev_map.values())}]")

            # Play out to completion
            steps = 0
            while True:
                obs, reward, terminated, truncated, info = env.step(0)
                steps += 1
                if terminated or truncated or steps >= 500:
                    break

        # --- Check 1: all boards are identical across resets ---
        for ep in range(1, num_episodes):
            if boards[ep] != boards[0]:
                diffs = []
                for coord in boards[0]:
                    if boards[ep].get(coord) != boards[0][coord]:
                        diffs.append(
                            f"({coord[0]},{coord[1]}): "
                            f"ep1={boards[0][coord]} vs ep{ep+1}={boards[ep].get(coord)}"
                        )
                return False, (
                    f"Board changed between episode 1 and {ep+1}! "
                    f"{len(diffs)} hex(es) differ. First: {diffs[0]}"
                )

        # --- Check 2: board has expected size ---
        expected_hexes = config.resolved_board_width * config.resolved_board_height
        if len(boards[0]) != expected_hexes:
            return False, (
                f"Board has {len(boards[0])} hexes, expected {expected_hexes}"
            )

        # --- Check 3: board matches known properties of the configured board ---
        # The configured board should have the RL start hex (8,2) at a
        # reasonable elevation.  A randomly-generated board would likely differ.
        # We check that (8,2) is NOT deep water or extreme elevation.
        rl_elev = boards[0].get((8, 2), None)
        if rl_elev is None:
            return False, "RL start hex (8,2) not found on board"
        if abs(rl_elev) > 2:
            return False, (
                f"RL start hex (8,2) has elevation {rl_elev} — "
                f"expected near 0 for configured map (board may be randomly generated)"
            )

        # --- Check 4: RL unit has reasonable legal moves (not stuck) ---
        for ep, n in enumerate(legal_counts):
            if n < 50:
                return False, (
                    f"Episode {ep+1}: only {n} legal moves at start "
                    f"(expected 100+ on correct board — board may be wrong)"
                )

        detail = (
            f"{num_episodes} episodes, board identical across resets, "
            f"{len(boards[0])} hexes, legal moves: "
            + "/".join(str(n) for n in legal_counts)
        )
        return True, detail
    finally:
        env.close()


def test_pilot_stats(megamek_dir, port, verbose):
    """Pilot stats: both units in mirror matchup have identical gunnery/piloting."""
    config = base_config(megamek_dir, port,
                         max_game_rounds=3)

    env = gymnasium.make("MegaMekGym/MegaMek-v0", config=config)
    try:
        obs, info = env.reset()
        raw_obs = env.unwrapped._last_raw_obs
        units = raw_obs.get("units", [])

        if len(units) != 2:
            return False, f"Expected 2 units, got {len(units)}"

        for i, unit in enumerate(units):
            if "gunnery" not in unit or "piloting" not in unit:
                return False, (
                    f"Unit {i} ({unit.get('chassis', '?')} {unit.get('model', '?')}) "
                    f"missing gunnery/piloting fields (keys: {list(unit.keys())})"
                )

        u0, u1 = units[0], units[1]
        g0, g1 = u0["gunnery"], u1["gunnery"]
        p0, p1 = u0["piloting"], u1["piloting"]

        if verbose:
            print(f"    Unit 0: {u0['chassis']} {u0['model']} — gunnery={g0}, piloting={p0}")
            print(f"    Unit 1: {u1['chassis']} {u1['model']} — gunnery={g1}, piloting={p1}")

        if g0 != g1:
            return False, f"Gunnery mismatch: unit 0 has {g0}, unit 1 has {g1}"
        if p0 != p1:
            return False, f"Piloting mismatch: unit 0 has {p0}, unit 1 has {p1}"

        # Play out the episode so the JVM exits cleanly
        step = 0
        while True:
            obs, reward, terminated, truncated, info = env.step(0)
            step += 1
            if terminated or truncated or step >= 500:
                break

        return True, f"Both units: gunnery={g0}, piloting={p0}"
    finally:
        env.close()


def test_feature_distributions(megamek_dir, port, verbose):
    """Feature distributions: per-move features must vary between legal moves."""
    config = base_config(megamek_dir, port, max_game_rounds=50)
    env = gymnasium.make("MegaMekGym/MegaMek-v0", config=config)

    board_w = config.resolved_board_width
    board_h = config.resolved_board_height
    board_size = board_w * board_h
    move_offset = board_size + 2 * UNIT_FEATURES + GLOBAL_FEATURES

    constant_count = [0] * MOVE_FEATURES
    total_multi_move_steps = 0

    try:
        for ep in range(2):
            obs, info = env.reset()
            # Collect from reset obs
            n = info.get("n_legal_moves", 0)
            if n >= 2:
                move_block = obs[move_offset : move_offset + n * MOVE_FEATURES]
                move_matrix = move_block.reshape(n, MOVE_FEATURES)
                for f in range(MOVE_FEATURES):
                    col = move_matrix[:, f]
                    if np.all(col == col[0]):
                        constant_count[f] += 1
                total_multi_move_steps += 1

            step = 0
            while True:
                n_legal = info.get("n_legal_moves", 0)
                action = np.random.randint(0, max(n_legal, 1))
                obs, reward, terminated, truncated, info = env.step(action)
                step += 1

                if terminated or truncated:
                    break

                n = info.get("n_legal_moves", 0)
                if n >= 2:
                    move_block = obs[move_offset : move_offset + n * MOVE_FEATURES]
                    move_matrix = move_block.reshape(n, MOVE_FEATURES)
                    for f in range(MOVE_FEATURES):
                        col = move_matrix[:, f]
                        if np.all(col == col[0]):
                            constant_count[f] += 1
                    total_multi_move_steps += 1

                if step >= 500:
                    break
    finally:
        env.close()

    if total_multi_move_steps < 5:
        return True, f"Insufficient data ({total_multi_move_steps} multi-move steps)"

    # Report
    dead_features = []
    print(f"    Per-Move Feature Distributions ({total_multi_move_steps} multi-move steps):")
    print(f"    {'Feature':<22s} {'Constant%':>9s}  Status")
    for f in range(MOVE_FEATURES):
        pct = constant_count[f] / total_multi_move_steps * 100
        name = MOVE_FEATURE_NAMES[f]
        if pct == 100.0:
            status = "DEAD WEIGHT"
            dead_features.append(name)
        elif pct > 80.0:
            status = "LOW VARIANCE"
        else:
            status = "OK"
        print(f"    {name:<22s} {pct:8.1f}%  {status}")

    if dead_features:
        return False, f"Dead weight features (100% constant within-step): {', '.join(dead_features)}"
    return True, f"All {MOVE_FEATURES} per-move features show within-step variance"


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

TESTS = [
    ("Basic Episode", test_basic_episode, 0),
    ("Truncation", test_truncation, 1),
    ("Termination", test_termination, 2),
    ("Persistent Reset", test_persistent_reset, 3),
    ("Cross-Validation", test_cross_validation, 4),
    ("Auto-Wake Pilot", test_auto_wake_pilot, 5),
    ("Fixed Deployment", test_fixed_deployment, 6),
    ("Board Consistency", test_board_consistency, 7),
    ("Pilot Stats", test_pilot_stats, 8),
    ("Feature Distributions", test_feature_distributions, 9),
]


def main():
    parser = argparse.ArgumentParser(
        description="Consolidated MegaMek RL smoke test"
    )
    parser.add_argument("--megamek-dir", type=str, default="../megamek")
    parser.add_argument("--port", type=int, default=9999,
                        help="Base port (each test offsets by 0-3)")
    parser.add_argument("--verbose", action="store_true",
                        help="Print per-step details")
    args = parser.parse_args()

    print("=" * 60)
    print("MegaMek RL Bridge — Consolidated Smoke Test")
    print("=" * 60)
    print(f"  megamek-dir: {args.megamek_dir}")
    print(f"  base port:   {args.port}")
    print()

    results = []
    t_total = time.monotonic()

    for i, (name, test_fn, port_offset) in enumerate(TESTS, 1):
        port = args.port + port_offset
        print(f"[{i}/{len(TESTS)}] {name} (port {port}) ...")

        t0 = time.monotonic()
        signal.alarm(120)
        try:
            passed, detail = test_fn(args.megamek_dir, port, args.verbose)
        except TestTimeout:
            passed, detail = False, "Timed out after 120s"
        except Exception:
            passed, detail = False, traceback.format_exc().strip().split("\n")[-1]
        finally:
            signal.alarm(0)

        elapsed = time.monotonic() - t0
        status = "PASS" if passed else "FAIL"
        mem = read_peak_memory(args.megamek_dir, port)
        mem_str = f" | JVM peak {mem[0]}MB/{mem[1]}MB" if mem else ""
        print(f"  [{status}] {detail} ({elapsed:.1f}s{mem_str})")
        print()
        results.append((name, passed))

    elapsed_total = time.monotonic() - t_total
    num_passed = sum(1 for _, p in results if p)
    print("=" * 60)
    print(f"Results: {num_passed}/{len(results)} passed ({elapsed_total:.1f}s)")
    for name, passed in results:
        print(f"  {'PASS' if passed else 'FAIL'}: {name}")
    print("=" * 60)

    sys.exit(0 if num_passed == len(results) else 1)


if __name__ == "__main__":
    main()
