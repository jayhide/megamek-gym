"""Performance test: measure startup and step timing for multiple MegaMek envs.

Diagnoses timeout issues when running parallel environments by measuring
where time is spent (Java start, socket connection, first observation)
and testing mitigation strategies like staggered startup.
"""

import argparse
import dataclasses
import os
import signal
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np

import megamek_gym  # noqa: F401 — registers the gymnasium env
from megamek_gym.config import MegaMekConfig
from megamek_gym.env import MegaMekEnv

_shutdown = threading.Event()


def run_single_env(env_index, config, num_steps, start_delay=0.0):
    """Launch one env, reset, run steps, close. Returns timing results dict."""
    if start_delay > 0:
        # Check shutdown flag periodically during delay
        deadline = time.monotonic() + start_delay
        while time.monotonic() < deadline:
            if _shutdown.is_set():
                return {"env_index": env_index, "error": "Shutdown before start"}
            time.sleep(min(0.5, deadline - time.monotonic()))

    result = {
        "env_index": env_index,
        "port": config.rl_port + env_index,
        "start_delay": start_delay,
        "reset_timing": {},
        "step_times": [],
        "error": None,
    }

    cfg = dataclasses.replace(config, env_index=env_index)
    env = MegaMekEnv(config=cfg)

    try:
        obs, info = env.reset()
        result["reset_timing"] = info.get("reset_timing", {})
        result["legal_moves_at_reset"] = len(info.get("legal_moves", []))

        for _ in range(num_steps):
            if _shutdown.is_set():
                break
            legal = info.get("legal_moves", [])
            action = np.random.randint(0, max(len(legal), 1))

            t_step = time.monotonic()
            obs, reward, terminated, truncated, info = env.step(action)
            result["step_times"].append(time.monotonic() - t_step)

            if terminated or truncated:
                break
    except Exception as e:
        result["error"] = str(e)
        # Grab partial timing from env (available even if reset failed mid-way)
        if hasattr(env, "_reset_timing") and env._reset_timing:
            result["reset_timing"] = env._reset_timing
    finally:
        env.close()

    return result


def print_results(results, mode, wall_time, config):
    """Print formatted timing table and diagnostics."""
    timeout_limit = config.connection_retries * config.connection_retry_delay

    print(f"\n{'='*72}")
    print(f"  MegaMek Performance Test")
    print(f"  Mode: {mode} | Envs: {len(results)} | Port base: {config.rl_port}")
    print(f"  Timeout limit: {timeout_limit:.0f}s ({config.connection_retries} retries x {config.connection_retry_delay}s)")
    print(f"{'='*72}")

    # Per-env table
    print(f"\n--- Per-Environment Results ---")
    header = f"{'Env':>3}  {'Port':>5}  {'Java Start':>10}  {'Connect':>9}  {'First Obs':>9}  {'Total Reset':>11}  {'Avg Step':>8}  {'Status'}"
    print(header)
    print("-" * len(header))

    ok_results = []
    for r in results:
        idx = r["env_index"]
        port = r["port"]
        timing = r.get("reset_timing", {})

        if r["error"]:
            timing = r.get("reset_timing", {})
            partial = ""
            if timing.get("java_start_s") is not None:
                partial += f"  java={timing['java_start_s']:.1f}s"
            if timing.get("connect_s") is not None:
                partial += f"  conn={timing['connect_s']:.1f}s"
            print(f"{idx:>3}  {port:>5}  ERROR: {r['error'][:50]}")
            if partial:
                print(f"{'':>3}  {'':>5}  (partial timing:{partial})")
            continue

        java_s = timing.get("java_start_s", 0)
        conn_s = timing.get("connect_s", 0)
        obs_s = timing.get("first_obs_s", 0)
        total_s = timing.get("total_reset_s", 0)
        avg_step = np.mean(r["step_times"]) if r["step_times"] else 0

        status = "OK"
        if total_s > timeout_limit * 0.8:
            status = "SLOW"

        print(
            f"{idx:>3}  {port:>5}"
            f"  {java_s:>9.1f}s"
            f"  {conn_s:>8.1f}s"
            f"  {obs_s:>8.1f}s"
            f"  {total_s:>10.1f}s"
            f"  {avg_step:>7.1f}s"
            f"  {status}"
        )
        ok_results.append(r)

    # Aggregate
    if ok_results:
        resets = [r["reset_timing"]["total_reset_s"] for r in ok_results]
        connects = [r["reset_timing"]["connect_s"] for r in ok_results]
        java_starts = [r["reset_timing"]["java_start_s"] for r in ok_results]
        first_obs = [r["reset_timing"]["first_obs_s"] for r in ok_results]
        all_steps = [s for r in ok_results for s in r["step_times"]]

        print(f"\n--- Aggregate ---")
        print(f"  Wall time:        {wall_time:.1f}s")
        print(f"  Slowest reset:    {max(resets):.1f}s  (env {ok_results[resets.index(max(resets))]['env_index']})")
        print(f"  Fastest reset:    {min(resets):.1f}s  (env {ok_results[resets.index(min(resets))]['env_index']})")
        print(f"  Mean reset:       {np.mean(resets):.1f}s")
        if all_steps:
            print(f"  Mean step time:   {np.mean(all_steps):.2f}s")
        print(f"  Mean java start:  {np.mean(java_starts):.1f}s")
        print(f"  Mean connect:     {np.mean(connects):.1f}s")
        print(f"  Mean first obs:   {np.mean(first_obs):.1f}s")

        # Identify bottleneck phase
        phases = {
            "Java start": np.mean(java_starts),
            "Connection": np.mean(connects),
            "First obs": np.mean(first_obs),
        }
        bottleneck = max(phases, key=phases.get)
        print(f"  Bottleneck:       {bottleneck} (mean {phases[bottleneck]:.1f}s, max {max(connects if bottleneck == 'Connection' else java_starts if bottleneck == 'Java start' else first_obs):.1f}s)")

    # Diagnosis
    errors = [r for r in results if r["error"]]
    slow = [r for r in ok_results if r["reset_timing"]["total_reset_s"] > timeout_limit * 0.8]

    if errors or slow:
        print(f"\n--- Diagnosis ---")
        for r in errors:
            print(f"  ERROR: Env {r['env_index']} (port {r['port']}): {r['error']}")
        for r in slow:
            pct = r["reset_timing"]["total_reset_s"] / timeout_limit * 100
            print(f"  WARNING: Env {r['env_index']} reset took {r['reset_timing']['total_reset_s']:.1f}s ({pct:.0f}% of {timeout_limit:.0f}s timeout)")
        if mode == "parallel" and (errors or slow):
            print(f"  TIP: Try --mode staggered --stagger-delay 10 to reduce contention")
        if any(r.get("reset_timing", {}).get("java_start_s", 0) > 30 for r in ok_results):
            print(f"  TIP: Java start is slow. Ensure Gradle daemon is running (./gradlew --status)")
    elif ok_results:
        print(f"\n  All {len(ok_results)} envs started OK within timeout limits.")

    print()


def main():
    parser = argparse.ArgumentParser(
        description="Performance test for MegaMek multi-env startup"
    )
    parser.add_argument("--megamek-dir", type=str, default="../megamek")
    parser.add_argument("--config", type=str, default=None)
    parser.add_argument("--port-base", type=int, default=9999)
    parser.add_argument("--num-envs", type=int, default=4)
    parser.add_argument("--num-steps", type=int, default=5)
    parser.add_argument(
        "--mode",
        choices=["parallel", "sequential", "staggered"],
        default="parallel",
    )
    parser.add_argument("--stagger-delay", type=float, default=10.0)
    args = parser.parse_args()

    config = MegaMekConfig.load(args.config) if args.config else MegaMekConfig()
    config.megamek_dir = str(os.path.abspath(args.megamek_dir))
    config.rl_port = args.port_base

    # Signal handler for clean shutdown
    def _shutdown_handler(signum, frame):
        if _shutdown.is_set():
            return
        _shutdown.set()
        print(f"\nCaught signal {signum}, shutting down...")

    signal.signal(signal.SIGINT, _shutdown_handler)
    signal.signal(signal.SIGTERM, _shutdown_handler)

    print(f"Starting perf test: {args.num_envs} envs, mode={args.mode}, "
          f"ports {args.port_base}-{args.port_base + args.num_envs - 1}")

    if args.mode == "sequential":
        results = []
        t0 = time.monotonic()
        for i in range(args.num_envs):
            if _shutdown.is_set():
                break
            print(f"  Launching env {i}...")
            r = run_single_env(i, config, args.num_steps)
            total = r.get("reset_timing", {}).get("total_reset_s")
            total_str = f"{total:.1f}s" if total is not None else "N/A"
            status = "OK" if not r["error"] else f"ERROR: {r['error'][:50]}"
            print(f"  Env {i} done: {total_str} reset — {status}")
            results.append(r)
        wall_time = time.monotonic() - t0

    elif args.mode == "parallel":
        t0 = time.monotonic()
        results = [None] * args.num_envs
        with ThreadPoolExecutor(max_workers=args.num_envs) as pool:
            futures = {
                pool.submit(run_single_env, i, config, args.num_steps): i
                for i in range(args.num_envs)
            }
            for f in as_completed(futures):
                idx = futures[f]
                results[idx] = f.result()
                total = results[idx].get("reset_timing", {}).get("total_reset_s")
                total_str = f"{total:.1f}s" if total is not None else "N/A"
                status = "OK" if not results[idx]["error"] else f"ERROR"
                print(f"  Env {idx} done: {total_str} reset — {status}")
        wall_time = time.monotonic() - t0

    elif args.mode == "staggered":
        t0 = time.monotonic()
        results = [None] * args.num_envs
        with ThreadPoolExecutor(max_workers=args.num_envs) as pool:
            futures = {
                pool.submit(
                    run_single_env, i, config, args.num_steps,
                    start_delay=i * args.stagger_delay,
                ): i
                for i in range(args.num_envs)
            }
            for f in as_completed(futures):
                idx = futures[f]
                results[idx] = f.result()
                total = results[idx].get("reset_timing", {}).get("total_reset_s")
                total_str = f"{total:.1f}s" if total is not None else "N/A"
                status = "OK" if not results[idx]["error"] else f"ERROR"
                print(f"  Env {idx} done: {total_str} reset (delay={idx * args.stagger_delay:.0f}s) — {status}")
        wall_time = time.monotonic() - t0

    print_results(results, args.mode, wall_time, config)


if __name__ == "__main__":
    main()
