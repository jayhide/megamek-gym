"""Pytest wrappers for smoke_test_all.py integration tests.

Each test launches a JVM via the Gymnasium env, so these are slow (~10-20s each).
Deselected by default; run with:

    poetry run pytest -m integration --megamek-dir ../megamek
"""

from __future__ import annotations

import pytest

from smoke_test_all import (
    base_config,
    run_episode,
    _random_action,
)

import gymnasium
import numpy as np

import megamek_gym  # noqa: F401 — registers env
from megamek_gym.config import MegaMekConfig
from megamek_gym.observation import (
    compute_obs_size,
    compute_obs_size_hierarchical,
    MOVE_FEATURES,
    MOVE_FEATURE_NAMES,
    UNIT_FEATURES,
    GLOBAL_FEATURES,
)
from tests.test_cross_validation import validate_observation


# Each test uses a unique port offset to avoid collisions when running in parallel.

@pytest.mark.integration
class TestBasicEpisode:
    def test_basic_episode(self, megamek_dir, base_port):
        port = base_port + 0
        config = base_config(megamek_dir, port, max_game_rounds=50, perf_log=True)
        env = gymnasium.make("MegaMekGym/MegaMek-v0", config=config)
        try:
            steps, terminated, truncated, info, obs_shape = run_episode(env)

            bw = config.resolved_board_width
            bh = config.resolved_board_height
            expected_size = compute_obs_size(bw, bh, config.max_legal_moves)
            assert obs_shape == (expected_size,), (
                f"obs shape {obs_shape} != expected ({expected_size},)"
            )
            assert terminated or truncated, "Episode did not terminate or truncate"
        finally:
            env.close()


@pytest.mark.integration
class TestTruncation:
    def test_truncation(self, megamek_dir, base_port):
        port = base_port + 1
        config = base_config(megamek_dir, port, max_game_rounds=3, perf_log=True)
        env = gymnasium.make("MegaMekGym/MegaMek-v0", config=config)
        try:
            steps, terminated, truncated, info, _ = run_episode(
                env, action_fn=lambda _info: 0,
            )
            assert truncated, f"Expected truncated=True, got {truncated}"
            assert not terminated, (
                f"Expected terminated=False, got {terminated} "
                "(unit destroyed before round limit — try increasing max_game_rounds)"
            )
        finally:
            env.close()


@pytest.mark.integration
class TestTermination:
    def test_termination(self, megamek_dir, base_port):
        port = base_port + 2
        config = base_config(
            megamek_dir, port,
            rl_unit="Locust LCT-1V",
            opponent_unit="Trebuchet TBT-5S",
            max_game_rounds=0,
            perf_log=True,
        )
        env = gymnasium.make("MegaMekGym/MegaMek-v0", config=config)
        try:
            steps, terminated, truncated, info, _ = run_episode(env, max_steps=1000)
            assert terminated, f"Expected terminated=True, got {terminated}"
            assert not truncated, f"Expected truncated=False, got {truncated}"
        finally:
            env.close()


@pytest.mark.integration
class TestPersistentReset:
    def test_persistent_reset(self, megamek_dir, base_port):
        port = base_port + 3
        config = base_config(megamek_dir, port, max_game_rounds=3, perf_log=True)
        num_episodes = 3
        env = gymnasium.make("MegaMekGym/MegaMek-v0", config=config)
        try:
            for ep in range(num_episodes):
                steps, terminated, truncated, info, _ = run_episode(
                    env, action_fn=lambda _info: 0,
                )
                assert truncated, f"Episode {ep+1}: expected truncated=True"
                assert not terminated, f"Episode {ep+1}: expected terminated=False"
        finally:
            env.close()


@pytest.mark.integration
class TestCrossValidation:
    def test_cross_validation(self, megamek_dir, base_port):
        port = base_port + 4
        config = base_config(megamek_dir, port, max_game_rounds=50, perf_log=True)
        env = gymnasium.make("MegaMekGym/MegaMek-v0", config=config)
        try:
            obs, info = env.reset()
            inner = env.unwrapped
            all_mismatches = []
            total_moves_checked = 0

            step = 0
            while True:
                raw_obs = inner._last_raw_obs
                if raw_obs and raw_obs.get("legal_moves"):
                    mismatches = validate_observation(raw_obs, inner._rl_owner_id)
                    total_moves_checked += len(raw_obs["legal_moves"])
                    all_mismatches.extend(mismatches)

                action = _random_action(info, inner._hierarchical)
                obs, reward, terminated, truncated, info = env.step(action)
                step += 1
                if terminated or truncated or step >= 500:
                    break

            assert not all_mismatches, (
                f"{len(all_mismatches)} mismatches in {total_moves_checked} moves. "
                f"First: {all_mismatches[0]}"
            )
        finally:
            env.close()


@pytest.mark.integration
class TestAutoWakePilot:
    def test_auto_wake_pilot(self, megamek_dir, base_port):
        port = base_port + 5
        config = base_config(
            megamek_dir, port,
            max_game_rounds=50,
            auto_wake_pilot=True,
            force_unconscious_on_turn=1,
        )
        env = gymnasium.make("MegaMekGym/MegaMek-v0", config=config)
        try:
            obs, info = env.reset()
            saw_auto_wake = False
            step = 0

            while True:
                n_legal = info.get("n_legal_moves", 0)
                auto_wake_count = info.get("auto_wake_count", 0)

                if auto_wake_count > 0 and not saw_auto_wake:
                    saw_auto_wake = True
                    assert n_legal > 1, (
                        f"auto-wake fired but n_legal_moves={n_legal} (expected > 1)"
                    )

                action = _random_action(info, env.unwrapped._hierarchical)
                obs, reward, terminated, truncated, info = env.step(action)
                step += 1
                if terminated or truncated or step >= 500:
                    break

            assert saw_auto_wake, (
                "force_unconscious_on_turn=1 but auto_wake_count never > 0"
            )
        finally:
            env.close()


@pytest.mark.integration
class TestFixedDeployment:
    def test_fixed_deployment(self, megamek_dir, base_port):
        port = base_port + 6
        config = base_config(
            megamek_dir, port,
            max_game_rounds=3,
            rl_fixed_coords=(8, 2),
            opponent_fixed_coords=(8, 14),
        )
        board_w = config.resolved_board_width
        board_h = config.resolved_board_height
        board_size = board_w * board_h

        num_episodes = 2
        env = gymnasium.make("MegaMekGym/MegaMek-v0", config=config)
        try:
            for ep in range(num_episodes):
                obs, info = env.reset()

                # Check RL unit position
                rl_state_start = board_size
                rl_x_norm = obs[rl_state_start]
                rl_y_norm = obs[rl_state_start + 1]
                actual_x = round(rl_x_norm * board_w)
                actual_y = round(rl_y_norm * board_h)
                assert (actual_x, actual_y) == (8, 2), (
                    f"Episode {ep+1}: RL unit at ({actual_x}, {actual_y}), expected (8, 2)"
                )

                # Play out episode
                step = 0
                while True:
                    obs, reward, terminated, truncated, info = env.step(0)
                    step += 1
                    if terminated or truncated or step >= 500:
                        break
        finally:
            env.close()


@pytest.mark.integration
class TestBoardConsistency:
    def test_board_consistency(self, megamek_dir, base_port):
        port = base_port + 7
        config = base_config(
            megamek_dir, port,
            max_game_rounds=3,
            rl_fixed_coords=(8, 2),
            opponent_fixed_coords=(8, 14),
        )
        num_episodes = 3
        env = gymnasium.make("MegaMekGym/MegaMek-v0", config=config)
        try:
            boards = []
            legal_counts = []

            for ep in range(num_episodes):
                obs, info = env.reset()
                raw_obs = env.unwrapped._last_raw_obs
                n_legal = info.get("n_legal_moves", 0)
                legal_counts.append(n_legal)

                board_hexes = raw_obs.get("board", {}).get("hexes", [])
                elev_map = {(h["x"], h["y"]): h.get("elevation", 0) for h in board_hexes}
                boards.append(elev_map)

                # Play out
                step = 0
                while True:
                    obs, reward, terminated, truncated, info = env.step(0)
                    step += 1
                    if terminated or truncated or step >= 500:
                        break

            # Boards identical across resets
            for ep in range(1, num_episodes):
                assert boards[ep] == boards[0], (
                    f"Board changed between episode 1 and {ep+1}"
                )

            # Correct size
            expected_hexes = config.resolved_board_width * config.resolved_board_height
            assert len(boards[0]) == expected_hexes

            # RL start hex reasonable
            rl_elev = boards[0].get((8, 2))
            assert rl_elev is not None, "RL start hex (8,2) not found"
            assert abs(rl_elev) <= 2, (
                f"RL start hex (8,2) elevation {rl_elev} — board may be randomly generated"
            )

            # Enough legal moves
            for ep, n in enumerate(legal_counts):
                assert n >= 50, (
                    f"Episode {ep+1}: only {n} legal moves (expected 100+ on correct board)"
                )
        finally:
            env.close()


@pytest.mark.integration
class TestPilotStats:
    def test_pilot_stats(self, megamek_dir, base_port):
        port = base_port + 8
        config = base_config(megamek_dir, port, max_game_rounds=3)
        env = gymnasium.make("MegaMekGym/MegaMek-v0", config=config)
        try:
            obs, info = env.reset()
            raw_obs = env.unwrapped._last_raw_obs
            units = raw_obs.get("units", [])

            assert len(units) == 2, f"Expected 2 units, got {len(units)}"
            u0, u1 = units[0], units[1]
            assert u0["gunnery"] == u1["gunnery"], (
                f"Gunnery mismatch: {u0['gunnery']} vs {u1['gunnery']}"
            )
            assert u0["piloting"] == u1["piloting"], (
                f"Piloting mismatch: {u0['piloting']} vs {u1['piloting']}"
            )

            # Play out
            step = 0
            while True:
                obs, reward, terminated, truncated, info = env.step(0)
                step += 1
                if terminated or truncated or step >= 500:
                    break
        finally:
            env.close()


@pytest.mark.integration
class TestHierarchicalActionSpace:
    def test_hierarchical_action_space(self, megamek_dir, base_port):
        port = base_port + 10
        config = base_config(
            megamek_dir, port,
            action_space_type="hierarchical",
            max_destinations=100,
            max_game_rounds=10,
        )
        env = gymnasium.make("MegaMekGym/MegaMek-v0", config=config)
        try:
            obs, info = env.reset()

            bw = config.resolved_board_width
            bh = config.resolved_board_height
            expected_size = compute_obs_size_hierarchical(bw, bh, config.max_destinations)
            assert obs.shape == (expected_size,), (
                f"obs shape {obs.shape} != expected ({expected_size},)"
            )

            mask = info["action_mask"]
            assert isinstance(mask, dict), f"Expected dict, got {type(mask).__name__}"
            assert "dest_mask" in mask and "facing_mask" in mask
            assert mask["dest_mask"].shape == (config.max_destinations,)
            assert mask["facing_mask"].shape == (config.max_destinations, 6)
            assert int(mask["dest_mask"].sum()) > 0, "No valid destinations after reset"

            # Run a few steps
            step = 0
            while True:
                action = _random_action(info, hierarchical=True)
                obs, reward, terminated, truncated, info = env.step(action)
                step += 1
                if terminated or truncated or step >= 200:
                    break
        finally:
            env.close()


@pytest.mark.integration
class TestFeatureDistributions:
    def test_feature_distributions(self, megamek_dir, base_port):
        port = base_port + 9
        config = base_config(megamek_dir, port, max_game_rounds=50)
        env = gymnasium.make("MegaMekGym/MegaMek-v0", config=config)

        board_w = config.resolved_board_width
        board_h = config.resolved_board_height
        board_size = board_w * board_h
        max_moves = config.max_legal_moves
        move_offset = board_size + 2 * UNIT_FEATURES + GLOBAL_FEATURES

        constant_count = [0] * MOVE_FEATURES
        total_multi_move_steps = 0

        try:
            for ep in range(2):
                obs, info = env.reset()
                n = min(info.get("n_legal_moves", 0), max_moves)
                if n >= 2:
                    move_block = obs[move_offset: move_offset + n * MOVE_FEATURES]
                    move_matrix = move_block.reshape(n, MOVE_FEATURES)
                    for f in range(MOVE_FEATURES):
                        if np.all(move_matrix[:, f] == move_matrix[0, f]):
                            constant_count[f] += 1
                    total_multi_move_steps += 1

                step = 0
                while True:
                    n_legal = info.get("n_legal_moves", 0)
                    action = np.random.randint(0, max(n_legal, 1))
                    obs, reward, terminated, truncated, info = env.step(action)
                    step += 1
                    if terminated or truncated or step >= 500:
                        break

                    n = min(info.get("n_legal_moves", 0), max_moves)
                    if n >= 2:
                        move_block = obs[move_offset: move_offset + n * MOVE_FEATURES]
                        move_matrix = move_block.reshape(n, MOVE_FEATURES)
                        for f in range(MOVE_FEATURES):
                            if np.all(move_matrix[:, f] == move_matrix[0, f]):
                                constant_count[f] += 1
                        total_multi_move_steps += 1
        finally:
            env.close()

        if total_multi_move_steps < 5:
            pytest.skip(f"Insufficient data ({total_multi_move_steps} multi-move steps)")

        dead_features = []
        for f in range(MOVE_FEATURES):
            pct = constant_count[f] / total_multi_move_steps * 100
            if pct == 100.0:
                dead_features.append(MOVE_FEATURE_NAMES[f])

        assert not dead_features, (
            f"Dead weight features (100% constant): {', '.join(dead_features)}"
        )
