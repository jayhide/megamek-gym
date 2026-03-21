#!/usr/bin/env python3
"""Consolidated smoke test for the MegaMek RL training bridge.

Covers all critical integration paths between Python (megamek-gym) and Java (megamek).
Auto-starts Java via the Gymnasium env — no manual two-terminal setup needed.

Tests:
  1. Basic Episode    — reset, random moves, game ends, obs shape correct
  2. Truncation       — low round limit triggers truncated=True
  3. Termination      — natural game end triggers terminated=True
  4. Persistent Reset — 3 episodes on same JVM without cold restart

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
from megamek_gym.observation import compute_obs_size


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
            return step, terminated, truncated, info, obs_shape

        if step >= max_steps:
            raise RuntimeError(f"Episode did not end after {max_steps} steps")


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_basic_episode(megamek_dir, port, verbose):
    """Basic episode: reset, random moves, game ends, obs shape correct."""
    config = MegaMekConfig(
        megamek_dir=megamek_dir,
        rl_port=port,
        max_game_rounds=50,
        firing_strategy="naive",
        max_rotating_round_saves=0,
        perf_log=True,
    )

    env = gymnasium.make("MegaMekGym/MegaMek-v0", config=config)
    try:
        steps, terminated, truncated, info, obs_shape = run_episode(
            env, verbose=verbose
        )

        expected_size = compute_obs_size(16, 17, config.max_legal_moves)
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
    config = MegaMekConfig(
        megamek_dir=megamek_dir,
        rl_port=port,
        max_game_rounds=3,
        firing_strategy="naive",
        max_rotating_round_saves=0,
        perf_log=True,
    )

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
    config = MegaMekConfig(
        megamek_dir=megamek_dir,
        rl_port=port,
        rl_unit="Locust LCT-1V",
        opponent_unit="Commando COM-2D",
        max_game_rounds=0,
        java_timeout_minutes=5,
        firing_strategy="naive",
        max_rotating_round_saves=0,
        perf_log=True,
    )

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


def test_persistent_reset(megamek_dir, port, verbose):
    """Persistent reset: 3 episodes on same JVM without cold restart."""
    config = MegaMekConfig(
        megamek_dir=megamek_dir,
        rl_port=port,
        max_game_rounds=3,
        firing_strategy="naive",
        max_rotating_round_saves=0,
        perf_log=True,
    )

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


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

TESTS = [
    ("Basic Episode", test_basic_episode, 0),
    ("Truncation", test_truncation, 1),
    ("Termination", test_termination, 2),
    ("Persistent Reset", test_persistent_reset, 3),
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
