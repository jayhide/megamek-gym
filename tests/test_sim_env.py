"""Smoke tests for MegaMekSimEnv (pure-Python simulator gymnasium env).

No JVM needed — these run fast as unit tests.
"""

import numpy as np
import pytest

from megamek_gym.config import MegaMekConfig
from megamek_gym.sim.env import MegaMekSimEnv


@pytest.fixture
def env():
    e = MegaMekSimEnv()
    yield e
    e.close()


@pytest.fixture
def env_from_config():
    cfg = MegaMekConfig(
        backend="sim",
        action_space_type="hierarchical",
        rl_unit="Trebuchet TBT-5S",
        opponent_unit="Trebuchet TBT-5S",
        rl_fixed_coords=(8, 1),
        opponent_fixed_coords=(8, 16),
        max_game_rounds=40,
        max_destinations=125,
    )
    e = MegaMekSimEnv(config=cfg)
    yield e
    e.close()


class TestReset:
    def test_reset_returns_obs_and_info(self, env):
        obs, info = env.reset(seed=42)
        assert isinstance(obs, np.ndarray)
        assert obs.dtype == np.float32
        assert obs.shape == env.observation_space.shape

    def test_reset_info_keys(self, env):
        _, info = env.reset(seed=42)
        expected_keys = {
            "action_mask", "round", "phase", "n_legal_moves",
            "moves_truncated", "game_outcome", "game_rounds",
            "java_crash", "early_termination", "auto_wake_count",
        }
        assert expected_keys.issubset(info.keys())

    def test_reset_action_mask_structure(self, env):
        _, info = env.reset(seed=42)
        mask = info["action_mask"]
        assert "dest_mask" in mask
        assert "facing_mask" in mask
        assert mask["dest_mask"].shape == (env.max_destinations,)
        assert mask["facing_mask"].shape == (env.max_destinations, 6)
        assert mask["dest_mask"].any(), "At least one destination should be valid"

    def test_reset_deterministic_with_seed(self, env):
        obs1, _ = env.reset(seed=123)
        obs2, _ = env.reset(seed=123)
        np.testing.assert_array_equal(obs1, obs2)

    def test_reset_from_config(self, env_from_config):
        obs, info = env_from_config.reset(seed=42)
        assert isinstance(obs, np.ndarray)
        assert info["n_legal_moves"] > 0


class TestStep:
    def test_step_returns_five_tuple(self, env):
        env.reset(seed=42)
        mask = env.action_masks()
        action = _sample_action(mask)
        result = env.step(action)
        assert len(result) == 5
        obs, reward, terminated, truncated, info = result
        assert isinstance(obs, np.ndarray)
        assert isinstance(reward, float)
        assert isinstance(terminated, bool)
        assert isinstance(truncated, bool)
        assert isinstance(info, dict)

    def test_step_obs_shape(self, env):
        env.reset(seed=42)
        action = _sample_action(env.action_masks())
        obs, *_ = env.step(action)
        assert obs.shape == env.observation_space.shape

    def test_step_info_keys(self, env):
        env.reset(seed=42)
        action = _sample_action(env.action_masks())
        _, _, _, _, info = env.step(action)
        expected_keys = {
            "action_mask", "round", "phase", "n_legal_moves",
            "moves_truncated", "game_outcome", "game_rounds",
            "java_crash", "early_termination", "auto_wake_count",
        }
        assert expected_keys.issubset(info.keys())
        assert info["java_crash"] == 0
        assert info["early_termination"] == 0

    def test_invalid_action_does_not_crash(self, env):
        env.reset(seed=42)
        # Completely invalid action — should fall back gracefully
        obs, reward, terminated, truncated, info = env.step(np.array([999, 5]))
        assert isinstance(obs, np.ndarray)


class TestFullEpisode:
    def test_episode_completes(self, env):
        """Play a full episode and verify it terminates."""
        obs, info = env.reset(seed=42)
        total_steps = 0
        max_steps = 500  # Safety limit
        while total_steps < max_steps:
            action = _sample_action(env.action_masks())
            obs, reward, terminated, truncated, info = env.step(action)
            total_steps += 1
            if terminated or truncated:
                break
        assert terminated or truncated, f"Episode did not finish in {max_steps} steps"
        assert info["game_outcome"] in (1, -1, 0)
        assert info["game_rounds"] > 0

    def test_multiple_episodes(self, env):
        """Play 3 episodes back-to-back to verify reset works."""
        for ep in range(3):
            obs, info = env.reset(seed=ep)
            assert info["n_legal_moves"] > 0
            done = False
            steps = 0
            while not done and steps < 500:
                action = _sample_action(env.action_masks())
                obs, reward, terminated, truncated, info = env.step(action)
                done = terminated or truncated
                steps += 1
            assert done, f"Episode {ep} did not finish"

    def test_rewards_are_finite(self, env):
        """All rewards in an episode should be finite."""
        env.reset(seed=42)
        rewards = []
        for _ in range(500):
            action = _sample_action(env.action_masks())
            _, reward, terminated, truncated, _ = env.step(action)
            rewards.append(reward)
            if terminated or truncated:
                break
        assert all(np.isfinite(r) for r in rewards)


class TestConfigValidation:
    def test_sim_requires_hierarchical(self):
        with pytest.raises(ValueError, match="hierarchical"):
            MegaMekConfig(backend="sim", action_space_type="flat")

    def test_sim_with_hierarchical_ok(self):
        cfg = MegaMekConfig(backend="sim", action_space_type="hierarchical")
        assert cfg.backend == "sim"


def _sample_action(mask):
    """Sample a valid hierarchical action from action masks."""
    dest_mask = mask["dest_mask"]
    valid_dests = np.where(dest_mask)[0]
    if len(valid_dests) == 0:
        return np.array([0, 0])
    dest = np.random.choice(valid_dests)
    facing_mask = mask["facing_mask"][dest]
    valid_facings = np.where(facing_mask)[0]
    if len(valid_facings) == 0:
        return np.array([dest, 0])
    facing = np.random.choice(valid_facings)
    return np.array([dest, facing])
