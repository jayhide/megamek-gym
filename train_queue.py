"""Sequential training queue runner.

Runs multiple train_ppo.py sessions one after another, with error recovery
so a single bad run doesn't kill the whole batch.

Usage:
    poetry run python train_queue.py sweep.yaml
    poetry run python train_queue.py sweep.yaml --dry-run
    poetry run python train_queue.py --resume runs/queue__*/queue_state.yaml

Queue YAML format:
    runs:
      - name: "lr-high"              # optional display label
        config: configs/lr_high.yaml  # required: complete training config
      - config: configs/lr_low.yaml   # name defaults to config stem
"""

import argparse
import os
import shutil
import signal
import subprocess
import sys
import time
from pathlib import Path

import yaml


def fmt_time(seconds):
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    return f"{h:02d}:{m:02d}:{s:02d}"


def parse_args():
    parser = argparse.ArgumentParser(description="Run a queue of training sessions")
    parser.add_argument("queue_file", type=str, nargs="?", default=None,
                        help="Path to queue YAML file")
    parser.add_argument("--dry-run", action="store_true",
                        help="Print commands without executing")
    parser.add_argument("--resume", type=str, default=None,
                        help="Path to queue_state.yaml from an interrupted queue")
    return parser.parse_args()


def load_queue(path):
    with open(path) as f:
        queue = yaml.safe_load(f)

    if not isinstance(queue, dict) or "runs" not in queue:
        print(f"Error: {path} must contain a 'runs' list")
        sys.exit(1)

    runs = queue["runs"]
    if not isinstance(runs, list) or len(runs) == 0:
        print(f"Error: {path} 'runs' must be a non-empty list")
        sys.exit(1)

    # Validate and normalize
    for i, run in enumerate(runs):
        if "config" not in run:
            print(f"Error: run {i} missing 'config' field")
            sys.exit(1)
        if not os.path.exists(run["config"]):
            print(f"Error: config not found: {run['config']}")
            sys.exit(1)
        if "name" not in run:
            run["name"] = Path(run["config"]).stem

    return runs


def load_state(path):
    with open(path) as f:
        return yaml.safe_load(f)


def save_state(path, state):
    with open(path, "w") as f:
        yaml.dump(state, f, default_flow_style=False, sort_keys=False)


def find_run_dir(exp_name, start_time):
    """Find the run directory created by train_ppo.py after start_time."""
    runs_dir = Path("runs")
    if not runs_dir.exists():
        return None
    candidates = []
    for d in runs_dir.iterdir():
        if d.is_dir() and d.name.startswith(exp_name + "__"):
            if d.stat().st_mtime >= start_time:
                candidates.append(d)
    if candidates:
        return str(max(candidates, key=lambda d: d.stat().st_mtime))
    return None


def get_exp_name(config_path):
    """Read exp_name from a config YAML."""
    try:
        with open(config_path) as f:
            cfg = yaml.safe_load(f)
        return cfg.get("exp_name", Path(config_path).stem)
    except Exception:
        return Path(config_path).stem


_stop_after_current = False


def main():
    global _stop_after_current
    args = parse_args()

    if args.resume:
        state = load_state(args.resume)
        queue_dir = Path(args.resume).parent
        runs = load_queue(state["queue_file"])
        completed = state.get("completed", [])
        print(f"Resuming queue from {args.resume} ({len(completed)}/{len(runs)} completed)")
    else:
        if not args.queue_file:
            print("Error: queue_file is required (or use --resume)")
            sys.exit(1)
        runs = load_queue(args.queue_file)
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        queue_dir = Path(f"runs/queue__{timestamp}")
        queue_dir.mkdir(parents=True, exist_ok=True)
        shutil.copy2(args.queue_file, queue_dir / "queue.yaml")
        state = {
            "queue_file": args.queue_file,
            "started_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "completed": [],
        }
        completed = []

    state_path = queue_dir / "queue_state.yaml"
    total_runs = len(runs)
    start_idx = len(completed)

    print(f"\n=== Training Queue: {total_runs} runs ===\n")

    if args.dry_run:
        for i, run in enumerate(runs):
            status = "DONE" if i < start_idx else "PENDING"
            cmd = ["poetry", "run", "python", "train_ppo.py", "--config", run["config"]]
            print(f"[{i+1}/{total_runs}] {run['name']} ({status})")
            print(f"  {' '.join(cmd)}")
        return

    # Signal handling: first Ctrl+C stops after current run, second kills
    def handle_signal(sig, frame):
        global _stop_after_current
        if _stop_after_current:
            print("\nForce quit.")
            sys.exit(1)
        _stop_after_current = True
        print("\nInterrupted -- will stop after current run finishes.")
        print("Press Ctrl+C again to force quit.")

    signal.signal(signal.SIGINT, handle_signal)
    signal.signal(signal.SIGTERM, handle_signal)

    queue_start = time.time()
    results = list(completed)  # carry forward completed results

    for i in range(start_idx, total_runs):
        if _stop_after_current:
            print(f"\nStopping queue (interrupted before run {i+1}).")
            break

        run = runs[i]
        cmd = ["poetry", "run", "python", "train_ppo.py", "--config", run["config"]]
        exp_name = get_exp_name(run["config"])

        print(f"[{i+1}/{total_runs}] {run['name']} ({run['config']})")
        print(f"  Started: {time.strftime('%Y-%m-%d %H:%M:%S')}")

        run_start = time.time()
        try:
            result = subprocess.run(cmd)
            exit_code = result.returncode
        except Exception as e:
            print(f"  Error launching: {e}")
            exit_code = -1

        duration = time.time() - run_start
        success = exit_code == 0
        run_dir = find_run_dir(exp_name, run_start) if success else None

        status_str = "SUCCESS" if success else f"FAILED (exit code {exit_code})"
        print(f"  {status_str} in {fmt_time(duration)}")
        if run_dir:
            print(f"  Run dir: {run_dir}")
        print()

        entry = {
            "index": i,
            "name": run["name"],
            "config": run["config"],
            "status": "success" if success else "failed",
            "exit_code": exit_code,
            "duration_seconds": round(duration, 1),
            "run_dir": run_dir,
        }
        results.append(entry)
        state["completed"] = results
        save_state(state_path, state)

    # Summary
    total_time = time.time() - queue_start
    succeeded = sum(1 for r in results if r["status"] == "success")
    failed = sum(1 for r in results if r["status"] == "failed")

    print("=== Queue Summary ===")
    name_width = max(len(r["name"]) for r in results) if results else 10
    print(f"  {'Run':<{name_width}}  Status    Duration   Config")
    for r in results:
        status = "SUCCESS" if r["status"] == "success" else "FAILED"
        dur = fmt_time(r["duration_seconds"])
        print(f"  {r['name']:<{name_width}}  {status:<9} {dur}  {r['config']}")

    print()
    completed_count = len(results)
    skipped = total_runs - completed_count
    parts = [f"{succeeded}/{total_runs} succeeded"]
    if failed:
        parts.append(f"{failed} failed")
    if skipped:
        parts.append(f"{skipped} skipped")
    print(f"  {', '.join(parts)}")
    print(f"  Total wall time: {fmt_time(total_time)}")
    print(f"  State: {state_path}")


if __name__ == "__main__":
    main()
