"""Shared pytest configuration for megamek-gym tests.

Adds --megamek-dir and --port CLI options for integration/validation tests,
and provides session-scoped fixtures for Java game traces.
"""

from __future__ import annotations

import pytest


# ---------------------------------------------------------------------------
# CLI options
# ---------------------------------------------------------------------------

def pytest_addoption(parser):
    parser.addoption(
        "--megamek-dir",
        default="../megamek",
        help="Path to MegaMek checkout (for integration/validation tests)",
    )
    parser.addoption(
        "--port",
        default=9999,
        type=int,
        help="Base TCP port for Java bridge (for integration/validation tests)",
    )
    parser.addoption(
        "--max-rounds",
        default=10,
        type=int,
        help="Max game rounds for Java trace collection (validation tests)",
    )
    parser.addoption(
        "--random-actions",
        action="store_true",
        default=False,
        help="Use random actions in Java trace (better for damage/heat validation)",
    )


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
def java_trace(request):
    """Collect a single Java game trace, shared across all validation tests.

    Only created when validation tests actually run (lazy).
    """
    from tests.sim_validation.collector import collect_game_trace, random_action_fn

    megamek_dir = request.config.getoption("--megamek-dir")
    port = request.config.getoption("--port")
    max_rounds = request.config.getoption("--max-rounds")
    random_actions = request.config.getoption("--random-actions")

    action_fn = random_action_fn if random_actions else None
    trace = collect_game_trace(
        megamek_dir=megamek_dir,
        port=port,
        max_rounds=max_rounds,
        action_fn=action_fn,
    )
    return trace
