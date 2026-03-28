"""Shared pytest fixtures for megamek-gym tests.

CLI options (--megamek-dir, --port, etc.) are registered via the
pytest11 entry point in megamek_gym/pytest_plugin.py.
"""

from __future__ import annotations

import pytest


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
