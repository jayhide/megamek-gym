"""Validate sim-only prone hexes against Java's ground-truth pathfinders.

Uses the ``validate_hexes`` bridge message to ask Java whether each sim-only
hex is reachable via *any* of its pathfinders (LongestPathFinder + ShortestPathFinder).
This is the definitive test: if Java can't reach a hex with any algorithm,
it is genuinely unreachable under MegaMek's movement rules and Python should
not be generating it.

Requires the ``validate_hexes`` handler in ``RLBotClient.java``.
"""

from __future__ import annotations

import json
import random

import gymnasium
import pytest

from megamek_gym.config import MegaMekConfig
from tests.sim_validation.reconstruct import extract_unit_state
from tests.sim_validation.test_legal_moves import validate_legal_moves


def _connect_and_play_until_prone(megamek_dir: str, port: int, max_attempts: int = 30):
    """Play games until we find a prone step with sim-only hexes.

    Returns (env, raw_obs, sim_only_hexes, walk_mp) or None.
    """
    for attempt in range(max_attempts):
        config = MegaMekConfig(
            megamek_dir=megamek_dir,
            rl_port=port,
            rl_unit="Trebuchet TBT-5S",
            opponent_unit="Trebuchet TBT-5S",
            board="Map Set 6/16x17 Woodland",
            max_game_rounds=50,
            max_rotating_round_saves=0,
            auto_wake_pilot=True,
            firing_strategy="naive",
            rl_fixed_coords=(14, 1),
            opponent_fixed_coords=(1, 15),
        )

        env = gymnasium.make("MegaMekGym/MegaMek-v0", config=config)
        inner = env.unwrapped
        obs, info = env.reset()

        for step in range(200):
            raw = inner._last_raw_obs
            if raw.get("terminated") or raw.get("truncated"):
                break
            if not raw.get("legal_moves"):
                break

            java_unit = extract_unit_state(raw, inner._rl_owner_id)
            if java_unit and java_unit.get("prone", False):
                result = validate_legal_moves(
                    raw, inner._rl_owner_id, step_idx=step, algorithm="deque"
                )
                if result.sim_only_hexes:
                    return env, inner, raw, result

            # Random action
            legal = raw.get("legal_moves", [])
            action = random.randint(0, len(legal) - 1) if legal else 0
            obs, reward, term, trunc, info = env.step(action)

        env.close()

    return None


def _send_validate_hexes(inner, hexes: list[tuple[int, int]]) -> list[dict]:
    """Send validate_hexes message to Java and get results."""
    hex_list = [{"x": x, "y": y} for x, y in hexes]
    msg = json.dumps({"type": "validate_hexes", "hexes": hex_list})

    # Send via the env's socket
    inner._sock.sendall((msg + "\n").encode("utf-8"))

    # Read response via the buffered reader
    response_line = inner._reader.readline()
    if not response_line:
        raise ConnectionError("Java closed connection during validate_hexes")

    result = json.loads(response_line)
    return result.get("results", [])


@pytest.mark.validation
class TestProneHexValidity:
    """Validate sim-only prone hexes are legal in Java."""

    def test_sim_only_hexes_are_legal_in_java(self, megamek_dir, base_port):
        """For each sim-only hex, ask Java (LPF + SPF) if the hex is reachable.

        This is the ground-truth test for the prone move gap. If Java's
        pathfinders can reach the hex, Python's extra move is valid. If not,
        Python has a pathfinding bug generating unreachable destinations.

        Currently EXPECTED TO FAIL: Python's has_just_stood pathfinding
        has an unidentified bug producing unreachable run-speed hexes.
        """
        if megamek_dir is None:
            pytest.skip("--megamek-dir not provided")

        port = base_port or 9999

        found = _connect_and_play_until_prone(megamek_dir, port)
        if found is None:
            pytest.skip("No prone steps with sim-only hexes found in 30 games")

        env, inner, raw, result = found

        try:
            sim_only = sorted(result.sim_only_hexes)
            print(f"\nProne at {result.unit_pos}, walk_mp={result._walk_mp}")
            print(f"Sim-only hexes to validate: {len(sim_only)}")

            # Send validate_hexes to Java
            validation = _send_validate_hexes(inner, sim_only)

            assert len(validation) == len(sim_only), (
                f"Expected {len(sim_only)} results, got {len(validation)}"
            )

            legal_count = 0
            reachable_count = 0
            unreachable = []
            illegal = []

            for hex_coord, v in zip(sim_only, validation):
                x, y = hex_coord
                assert v["x"] == x and v["y"] == y, f"Hex mismatch: expected ({x},{y}), got ({v['x']},{v['y']})"

                if v["reachable"]:
                    reachable_count += 1
                if v["legal"]:
                    legal_count += 1

                if not v["reachable"]:
                    unreachable.append(f"({x},{y})")
                elif not v["legal"]:
                    illegal.append(f"({x},{y}) mp={v['mp_used']} type={v['move_type']}")

                print(f"  ({x},{y}): reachable={v['reachable']} legal={v['legal']} "
                      f"mp={v['mp_used']} hm={v['hexes_moved']} type={v['move_type']}")

            print(f"\nResults: {legal_count}/{len(sim_only)} legal, "
                  f"{reachable_count}/{len(sim_only)} reachable")

            if unreachable:
                print(f"Unreachable by ShortestPathFinder: {unreachable}")
            if illegal:
                print(f"Reachable but illegal: {illegal}")

            # The key assertion: all sim-only hexes should be legal
            assert legal_count == len(sim_only), (
                f"{len(sim_only) - legal_count} sim-only hexes are NOT legal in Java! "
                f"Unreachable: {unreachable}, Illegal: {illegal}"
            )

        finally:
            # Send an actual action so Java doesn't hang
            try:
                action_msg = json.dumps({"type": "action", "move_index": 0})
                inner._sock.sendall((action_msg + "\n").encode("utf-8"))
            except Exception:
                pass
            env.close()
