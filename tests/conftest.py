"""Shared pytest fixtures for megamek-gym tests.

CLI options (--megamek-dir, --port, etc.) are registered via the
pytest11 entry point in megamek_gym/pytest_plugin.py.
"""

from __future__ import annotations

import logging
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

import pytest

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Trace task builder
# ---------------------------------------------------------------------------

def _build_trace_tasks(megamek_dir, base_port, max_rounds, random_actions):
    """Build dict of name -> callable for all trace collections."""
    from tests.sim_validation.collector import collect_game_trace, random_action_fn
    from tests.sim_validation.dual_collector import (
        collect_dual_game_trace,
        FIRING_WAYPOINTS_A, FIRING_WAYPOINTS_B,
    )

    tasks = {}

    # 1. java_trace (single-bot, via MegaMekEnv)
    action_fn = random_action_fn if random_actions else None
    tasks["java_trace"] = lambda: collect_game_trace(
        megamek_dir=megamek_dir,
        port=base_port,
        max_rounds=max_rounds,
        action_fn=action_fn,
    )

    # 2. walk_patrol_trace
    tasks["walk_patrol_trace"] = lambda: collect_dual_game_trace(
        megamek_dir=megamek_dir, port=base_port + 1,
        max_rounds=15, movement_mode="walk",
    )

    # 3. run_patrol_trace
    tasks["run_patrol_trace"] = lambda: collect_dual_game_trace(
        megamek_dir=megamek_dir, port=base_port + 2,
        max_rounds=15, movement_mode="run",
    )

    # 4. prone_patrol_trace
    tasks["prone_patrol_trace"] = lambda: collect_dual_game_trace(
        megamek_dir=megamek_dir, port=base_port + 3,
        max_rounds=15, movement_mode="walk",
        initial_state={"entities": [{"owner": 0, "prone": True}]},
    )

    # 5. firing_patrol_trace
    tasks["firing_patrol_trace"] = lambda: collect_dual_game_trace(
        megamek_dir=megamek_dir, port=base_port + 4,
        max_rounds=15, movement_mode="walk",
        waypoints_a=FIRING_WAYPOINTS_A,
        waypoints_b=FIRING_WAYPOINTS_B,
    )

    # 6. damaged_patrol_trace
    tasks["damaged_patrol_trace"] = lambda: collect_dual_game_trace(
        megamek_dir=megamek_dir, port=base_port + 5,
        max_rounds=20, movement_mode="walk",
        waypoints_a=FIRING_WAYPOINTS_A,
        waypoints_b=FIRING_WAYPOINTS_B,
        firing_strategy="naive",
        initial_state={"entities": [
            {"owner": 0,
             "armor": {"RT": 0, "LT": 0, "RA": 0, "LA": 0, "RL": 0, "LL": 0},
             "internal": {"RT": 1, "LT": 1, "RA": 1, "LA": 1, "RL": 1, "LL": 1}},
            {"owner": 1,
             "armor": {"RT": 0, "LT": 0, "RA": 0, "LA": 0, "RL": 0, "LL": 0},
             "internal": {"RT": 1, "LT": 1, "RA": 1, "LA": 1, "RL": 1, "LL": 1}},
        ]},
    )

    # 7. heated_patrol_trace
    tasks["heated_patrol_trace"] = lambda: collect_dual_game_trace(
        megamek_dir=megamek_dir, port=base_port + 6,
        max_rounds=15, movement_mode="walk",
        firing_strategy="naive",
        initial_state={"entities": [{"owner": 0, "heat": 15}]},
    )

    # 8. weapons_damaged_patrol_trace
    tasks["weapons_damaged_patrol_trace"] = lambda: collect_dual_game_trace(
        megamek_dir=megamek_dir, port=base_port + 7,
        max_rounds=15, movement_mode="walk",
        waypoints_a=FIRING_WAYPOINTS_A,
        waypoints_b=FIRING_WAYPOINTS_B,
        firing_strategy="naive",
        initial_state={"entities": [{"owner": 0, "weapons_destroyed": [0, 2]}]},
    )

    # 9. gyro_destroyed_trace
    tasks["gyro_destroyed_trace"] = lambda: collect_dual_game_trace(
        megamek_dir=megamek_dir, port=base_port + 8,
        max_rounds=20, movement_mode="walk",
        waypoints_a=FIRING_WAYPOINTS_A,
        waypoints_b=FIRING_WAYPOINTS_B,
        firing_strategy="naive",
        initial_state={"entities": [
            {"owner": 0,
             "crit_state": {"gyro_hits": 2},
             "armor": {"RT": 2, "LT": 2, "RA": 0, "LA": 0, "RL": 3, "LL": 3}},
            {"owner": 1,
             "crit_state": {"gyro_hits": 2},
             "armor": {"RT": 2, "LT": 2, "RA": 0, "LA": 0, "RL": 3, "LL": 3}},
        ]},
    )

    # 10. hip_damaged_trace
    tasks["hip_damaged_trace"] = lambda: collect_dual_game_trace(
        megamek_dir=megamek_dir, port=base_port + 9,
        max_rounds=15, movement_mode="walk",
        initial_state={"entities": [
            {"owner": 0,
             "crit_state": {"right_leg": {"hip_hits": 1}}},
        ]},
    )

    # 11. shutdown_patrol_trace
    tasks["shutdown_patrol_trace"] = lambda: collect_dual_game_trace(
        megamek_dir=megamek_dir, port=base_port + 10,
        max_rounds=15, movement_mode="walk",
        firing_strategy="none",
        initial_state={"entities": [
            {"owner": 0, "heat": 35},
        ]},
    )

    return tasks


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="session")
def megamek_dir(request):
    return request.config.getoption("--megamek-dir")


@pytest.fixture(scope="session")
def base_port(request):
    return request.config.getoption("--port")


@pytest.fixture(scope="session")
def _all_traces(request):
    """Collect all game traces in parallel.

    Launches up to --trace-workers JVM processes concurrently.
    Each trace uses a unique port so there are no conflicts.
    """
    from megamek_gym.java_process import JavaProcess

    megamek_dir = request.config.getoption("--megamek-dir")
    base_port = request.config.getoption("--port")
    max_rounds = request.config.getoption("--max-rounds")
    random_actions = request.config.getoption("--random-actions")
    workers = request.config.getoption("--trace-workers")

    # Pre-warm classpath to avoid concurrent Gradle invocations
    t_cp = time.monotonic()
    JavaProcess.warmup_classpath(megamek_dir)
    cp_secs = time.monotonic() - t_cp

    tasks = _build_trace_tasks(megamek_dir, base_port, max_rounds, random_actions)
    timings = {}  # name -> seconds

    def _timed(name, fn):
        t0 = time.monotonic()
        result = fn()
        timings[name] = time.monotonic() - t0
        return result

    t_total = time.monotonic()

    if workers <= 1:
        # Sequential fallback
        results = {}
        for name, fn in tasks.items():
            logger.info("Collecting trace: %s", name)
            try:
                results[name] = _timed(name, fn)
            except Exception as e:
                logger.error("Trace %s failed: %s", name, e)
                results[name] = e
    else:
        logger.info("Collecting %d traces with %d parallel workers", len(tasks), workers)
        results = {}
        with ThreadPoolExecutor(max_workers=workers) as pool:
            future_to_name = {
                pool.submit(_timed, name, fn): name
                for name, fn in tasks.items()
            }
            for future in as_completed(future_to_name):
                name = future_to_name[future]
                try:
                    results[name] = future.result()
                    logger.info("Trace %s completed", name)
                except Exception as e:
                    logger.error("Trace %s failed: %s", name, e)
                    results[name] = e

    wall_secs = time.monotonic() - t_total

    # Print timing summary
    print(f"\n{'='*60}")
    print(f"Trace collection summary ({workers} worker(s))")
    print(f"{'='*60}")
    print(f"  Classpath warmup: {cp_secs:.1f}s")
    for name in tasks:
        t = timings.get(name)
        status = f"{t:.1f}s" if t is not None else "FAILED"
        print(f"  {name}: {status}")
    seq_total = sum(timings.values())
    print(f"  {'─'*40}")
    print(f"  Sequential total: {seq_total:.1f}s")
    print(f"  Wall time:        {wall_secs:.1f}s")
    if seq_total > 0:
        print(f"  Speedup:          {seq_total / wall_secs:.1f}x")
    print(f"{'='*60}\n")

    return results


def _get_trace(_all_traces, name):
    """Extract a trace from the master dict, re-raising stored exceptions."""
    result = _all_traces[name]
    if isinstance(result, Exception):
        raise result
    return result


@pytest.fixture(scope="session")
def java_trace(_all_traces):
    return _get_trace(_all_traces, "java_trace")


@pytest.fixture(scope="session")
def walk_patrol_trace(_all_traces):
    return _get_trace(_all_traces, "walk_patrol_trace")


@pytest.fixture(scope="session")
def run_patrol_trace(_all_traces):
    return _get_trace(_all_traces, "run_patrol_trace")


@pytest.fixture(scope="session")
def prone_patrol_trace(_all_traces):
    return _get_trace(_all_traces, "prone_patrol_trace")


@pytest.fixture(scope="session")
def firing_patrol_trace(_all_traces):
    return _get_trace(_all_traces, "firing_patrol_trace")


@pytest.fixture(scope="session")
def damaged_patrol_trace(_all_traces):
    return _get_trace(_all_traces, "damaged_patrol_trace")


@pytest.fixture(scope="session")
def heated_patrol_trace(_all_traces):
    return _get_trace(_all_traces, "heated_patrol_trace")


@pytest.fixture(scope="session")
def weapons_damaged_patrol_trace(_all_traces):
    return _get_trace(_all_traces, "weapons_damaged_patrol_trace")


@pytest.fixture(scope="session")
def gyro_destroyed_trace(_all_traces):
    return _get_trace(_all_traces, "gyro_destroyed_trace")


@pytest.fixture(scope="session")
def hip_damaged_trace(_all_traces):
    return _get_trace(_all_traces, "hip_damaged_trace")


@pytest.fixture(scope="session")
def shutdown_patrol_trace(_all_traces):
    return _get_trace(_all_traces, "shutdown_patrol_trace")
