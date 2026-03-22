"""Measure total memory usage across different numbers of parallel RL environments.

Usage:
    poetry run python mem_benchmark.py --megamek-dir ../megamek --env-counts 1,2,4,8
"""

from __future__ import annotations

import argparse
import dataclasses
import os
import re
import signal
import sys
import time

import numpy as np

from megamek_gym.config import MegaMekConfig
from megamek_gym.env import MegaMekEnv


def read_rss_mb(pid: int) -> float | None:
    """Read VmRSS from /proc/{pid}/status. Returns MB or None if unavailable."""
    try:
        with open(f"/proc/{pid}/status") as f:
            for line in f:
                if line.startswith("VmRSS:"):
                    return int(line.split()[1]) / 1024.0
    except (FileNotFoundError, ProcessLookupError, PermissionError):
        return None
    return None


def read_jvm_heap(megamek_dir: str, port: int) -> tuple[int, int] | None:
    """Read peak JVM heap from [rl-mem] lines in rl_java_{port}.log."""
    from pathlib import Path

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


def run_episode(env: MegaMekEnv) -> int:
    """Run one episode with random actions. Returns step count."""
    obs, info = env.reset()
    steps = 0
    while True:
        n_legal = info.get("n_legal_moves", 1)
        action = np.random.randint(0, max(n_legal, 1))
        obs, reward, terminated, truncated, info = env.step(action)
        steps += 1
        if terminated or truncated:
            return steps


def benchmark_n_envs(
    n: int,
    base_config: MegaMekConfig,
    episodes_per_env: int,
    stagger_delay: float,
    port_base: int,
    verbose: bool,
) -> dict:
    """Start N envs, run episodes, measure memory, tear down."""
    envs: list[MegaMekEnv] = []
    print(f"\n--- {n} environment(s) ---")

    # Start all envs with stagger
    for i in range(n):
        cfg = dataclasses.replace(base_config, env_index=i, rl_port=port_base)
        env = MegaMekEnv(config=cfg)
        port = port_base + i
        print(f"  Starting env {i} on port {port}...", end=" ", flush=True)
        try:
            obs, info = env.reset()
            print("ok")
        except Exception as e:
            print(f"FAILED: {e}")
            env.close()
            continue
        envs.append(env)
        if i < n - 1 and stagger_delay > 0:
            time.sleep(stagger_delay)

    if not envs:
        print("  No environments started successfully.")
        return None

    # Run episodes on each env (first reset already counted as episode 1)
    for ep in range(1, episodes_per_env):
        for i, env in enumerate(envs):
            if verbose:
                print(f"  Env {i}: episode {ep + 1}/{episodes_per_env}...", end=" ", flush=True)
            steps = run_episode(env)
            if verbose:
                print(f"{steps} steps")

    # Measure memory while all JVMs alive
    jvm_rss_list = []
    jvm_heap_used_list = []
    jvm_heap_max = 0
    for i, env in enumerate(envs):
        pid = env._java._process.pid
        rss = read_rss_mb(pid)
        jvm_rss_list.append(rss or 0)

        port = port_base + i
        heap = read_jvm_heap(base_config.megamek_dir, port)
        if heap:
            jvm_heap_used_list.append(heap[0])
            jvm_heap_max = max(jvm_heap_max, heap[1])

    python_rss = read_rss_mb(os.getpid()) or 0

    # Tear down
    for env in envs:
        env.close()

    per_jvm_rss = sum(jvm_rss_list) / len(jvm_rss_list) if jvm_rss_list else 0
    total_jvm_rss = sum(jvm_rss_list)
    avg_heap_used = sum(jvm_heap_used_list) / len(jvm_heap_used_list) if jvm_heap_used_list else 0

    result = {
        "n": len(envs),
        "per_jvm_rss": per_jvm_rss,
        "total_jvm_rss": total_jvm_rss,
        "python_rss": python_rss,
        "total_rss": total_jvm_rss + python_rss,
        "jvm_heap_used": avg_heap_used,
        "jvm_heap_max": jvm_heap_max,
    }

    print(
        f"  Per-JVM RSS: {per_jvm_rss:.0f} MB | "
        f"Total JVM: {total_jvm_rss:.0f} MB | "
        f"Python: {python_rss:.0f} MB | "
        f"Total: {total_jvm_rss + python_rss:.0f} MB | "
        f"Heap used: {avg_heap_used:.0f} MB"
    )
    return result


def main():
    parser = argparse.ArgumentParser(description="Measure memory usage across N parallel RL environments")
    parser.add_argument("--megamek-dir", default="../megamek")
    parser.add_argument("--env-counts", default="1,2,4,8", help="Comma-separated env counts to test")
    parser.add_argument("--episodes-per-env", type=int, default=3)
    parser.add_argument("--port-base", type=int, default=9999)
    parser.add_argument("--stagger-delay", type=float, default=3.0)
    parser.add_argument("--max-game-rounds", type=int, default=10)
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    env_counts = [int(x.strip()) for x in args.env_counts.split(",")]
    megamek_dir = os.path.abspath(args.megamek_dir)

    # Clean shutdown on Ctrl-C
    signal.signal(signal.SIGINT, lambda *_: (print("\nInterrupted."), sys.exit(1)))

    print("=" * 72)
    print("  MegaMek Memory Benchmark")
    print("=" * 72)
    print(f"Episodes per env: {args.episodes_per_env} | "
          f"Max rounds: {args.max_game_rounds} | Force GC: on | Mem log: 1")

    base_config = MegaMekConfig(
        megamek_dir=megamek_dir,
        max_game_rounds=args.max_game_rounds,
        firing_strategy="naive",
        max_rotating_round_saves=0,
        force_gc=True,
        mem_log=1,
    )

    results = []
    for i, n in enumerate(env_counts):
        # Use different port ranges per iteration to avoid TIME_WAIT conflicts
        port_base = args.port_base + i * 100
        result = benchmark_n_envs(
            n, base_config, args.episodes_per_env, args.stagger_delay, port_base, args.verbose,
        )
        if result:
            results.append(result)
        if i < len(env_counts) - 1:
            time.sleep(2)  # let ports clear

    if not results:
        print("\nNo results collected.")
        return

    # Summary table
    print("\n" + "=" * 72)
    print("  Summary")
    print("=" * 72)
    header = f"{'Envs':>4}  {'Per-JVM RSS':>12}  {'Total JVM RSS':>14}  {'Python RSS':>11}  {'Total RSS':>10}  {'JVM Heap Used':>14}  {'JVM Heap Max':>13}"
    print(header)
    print("-" * len(header))
    for r in results:
        print(
            f"{r['n']:>4}  "
            f"{r['per_jvm_rss']:>9.0f} MB  "
            f"{r['total_jvm_rss']:>11.0f} MB  "
            f"{r['python_rss']:>8.0f} MB  "
            f"{r['total_rss']:>7.0f} MB  "
            f"{r['jvm_heap_used']:>11.0f} MB  "
            f"{r['jvm_heap_max']:>10.0f} MB"
        )
    print()


if __name__ == "__main__":
    main()
