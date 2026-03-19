#!/usr/bin/env python3
"""Diagnose the multi-env reset hang.

Test 1: Single-process reset (no multiprocessing)
  - Create env, step until game ends, call reset(), observe.
  - If this hangs, the issue is in _cleanup()/stop() itself.

Test 2: Multiprocessing reset (simulates AsyncVectorEnv worker)
  - Run test 1 inside multiprocessing.Process.
  - If test 1 passes but test 2 hangs, the issue is multiprocessing-specific.

Test 3: Two envs with AsyncVectorEnv (reproduces the original bug)
  - Run 2 envs, step until one game ends and auto-resets.

Usage:
  poetry run python diagnose_reset.py --megamek-dir ../megamek --test 1
  poetry run python diagnose_reset.py --megamek-dir ../megamek --test 2
  poetry run python diagnose_reset.py --megamek-dir ../megamek --test 3
"""

import argparse
import logging
import multiprocessing
import time

import gymnasium
import numpy as np

import megamek_gym  # noqa: F401 — registers MegaMekGym/MegaMek-v0
from megamek_gym.config import MegaMekConfig

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(name)s %(levelname)s %(message)s",
)
logger = logging.getLogger(__name__)


def run_single_env_reset(megamek_dir: str, port: int):
    """Test 1: single-process game + reset."""
    config = MegaMekConfig(megamek_dir=megamek_dir, rl_port=port)
    env = gymnasium.make("MegaMekGym/MegaMek-v0", config=config)

    logger.info("=== Test: initial reset ===")
    obs, info = env.reset()
    logger.info("Initial reset done. Legal moves: %d", info.get("n_legal_moves", len(info.get("legal_moves", []))))

    step = 0
    while True:
        n_legal = info.get("n_legal_moves", len(info.get("legal_moves", [])))
        action = np.random.randint(0, max(n_legal, 1))
        obs, reward, terminated, truncated, info = env.step(action)
        step += 1
        if step % 50 == 0:
            logger.info("Step %d, round=%s, phase=%s", step, info.get("round"), info.get("phase"))
        if terminated or truncated:
            logger.info("Game ended at step %d (terminated=%s, truncated=%s)", step, terminated, truncated)
            break

    logger.info("=== Test: calling reset() after game end ===")
    t0 = time.monotonic()
    obs, info = env.reset()
    elapsed = time.monotonic() - t0
    logger.info("Second reset completed in %.1fs! Legal moves: %d", elapsed, info.get("n_legal_moves", len(info.get("legal_moves", []))))

    env.close()
    logger.info("=== Test 1 PASSED ===")


def test_1(args):
    """Single-process reset test."""
    logger.info(">>> TEST 1: Single-process reset <<<")
    run_single_env_reset(args.megamek_dir, args.port)


def test_2(args):
    """Multiprocessing reset test — run test 1 inside a child process."""
    logger.info(">>> TEST 2: Multiprocessing reset <<<")
    timeout = 300  # 5 minutes

    proc = multiprocessing.Process(
        target=run_single_env_reset,
        args=(args.megamek_dir, args.port),
    )
    proc.start()
    logger.info("Child process started (pid=%d), joining with %ds timeout...", proc.pid, timeout)
    proc.join(timeout=timeout)

    if proc.is_alive():
        logger.error("Child process STILL ALIVE after %ds — hang confirmed in multiprocessing!", timeout)
        proc.terminate()
        proc.join(timeout=10)
        if proc.is_alive():
            proc.kill()
        logger.error("=== Test 2 FAILED (hang in multiprocessing) ===")
    elif proc.exitcode != 0:
        logger.error("Child process exited with code %d", proc.exitcode)
        logger.error("=== Test 2 FAILED (child crashed) ===")
    else:
        logger.info("=== Test 2 PASSED ===")


def test_3(args):
    """AsyncVectorEnv reset test — 2 envs, step through auto-resets like train_ppo.py."""
    logger.info(">>> TEST 3: AsyncVectorEnv with 2 envs (continues after auto-reset) <<<")

    def make_env(idx):
        def _init():
            config = MegaMekConfig(
                megamek_dir=args.megamek_dir,
                rl_port=args.port,
                env_index=idx,
            )
            return gymnasium.make("MegaMekGym/MegaMek-v0", config=config)
        return _init

    envs = gymnasium.vector.AsyncVectorEnv([make_env(0), make_env(1)])
    logger.info("AsyncVectorEnv created with 2 envs")

    obs, info = envs.reset()
    logger.info("Initial reset done for both envs")

    max_steps = 500
    resets_seen = 0
    target_resets = 2  # Continue until we've seen at least 2 auto-resets
    for step in range(1, max_steps + 1):
        # Pick random actions for each env using their action masks
        actions = []
        for i in range(2):
            n_legal = int(info["n_legal_moves"][i])
            actions.append(np.random.randint(0, max(n_legal, 1)))
        actions = np.array(actions)

        obs, reward, terminated, truncated, info = envs.step(actions)

        if step % 50 == 0:
            logger.info("Step %d (resets_seen=%d)", step, resets_seen)

        if any(terminated) or any(truncated):
            which = [i for i in range(2) if terminated[i] or truncated[i]]
            resets_seen += len(which)
            logger.info(
                "Step %d: env(s) %s finished — auto-reset #%d triggered!",
                step, which, resets_seen,
            )
            logger.info("Step %d: auto-reset completed successfully!", step)
            if resets_seen >= target_resets:
                logger.info("Saw %d auto-resets, test complete!", resets_seen)
                break
            # Continue stepping — this is what train_ppo.py does
    else:
        if resets_seen > 0:
            logger.info("Reached %d steps with %d resets (not enough for target=%d but no hang)",
                        max_steps, resets_seen, target_resets)
        else:
            logger.warning("Reached %d steps without any game ending (increase max_steps?)", max_steps)

    envs.close()
    logger.info("=== Test 3 PASSED ===")


def main():
    parser = argparse.ArgumentParser(description="Diagnose multi-env reset hang")
    parser.add_argument("--megamek-dir", type=str, default="../megamek")
    parser.add_argument("--port", type=int, default=9999)
    parser.add_argument("--test", type=int, choices=[1, 2, 3], required=True,
                        help="Which test to run (1=single, 2=multiprocessing, 3=AsyncVectorEnv)")
    args = parser.parse_args()

    {1: test_1, 2: test_2, 3: test_3}[args.test](args)


if __name__ == "__main__":
    main()
