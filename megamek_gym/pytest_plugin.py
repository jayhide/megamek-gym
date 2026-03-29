"""pytest plugin for megamek-gym CLI options.

Registered as a pytest11 entry point so that --megamek-dir and related
options are known *before* rootdir/conftest discovery.  This prevents
pytest from treating the --megamek-dir value as a test path argument.
"""

from __future__ import annotations


def pytest_configure(config):
    config.addinivalue_line(
        "markers",
        "integration: smoke tests requiring a live JVM (deselected by default)",
    )
    config.addinivalue_line(
        "markers",
        "validation: sim-vs-Java cross-validation requiring a live JVM (deselected by default)",
    )


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
    parser.addoption(
        "--trace-workers",
        default=11,
        type=int,
        help="Parallel workers for trace collection (1 = sequential)",
    )
