"""Tests for early termination detection (prone + leg destroyed)."""

import pytest

from megamek_gym.env import MegaMekEnv


def _loc(name, armor, armor_max, internal, internal_max):
    return {"location": name, "armor": armor, "armor_max": armor_max,
            "internal": internal, "internal_max": internal_max,
            "rear_armor": 0, "rear_armor_max": 0}


# Standard mech locations for testing
def _full_armor(ll_internal=5, rl_internal=5, prone=False):
    """Build a unit dict with full mech locations."""
    return {
        "owner": 0,
        "prone": prone,
        "armor": [
            _loc("HD", 9, 9, 3, 3),
            _loc("CT", 16, 16, 11, 11),
            _loc("LT", 12, 12, 8, 8),
            _loc("RT", 12, 12, 8, 8),
            _loc("LA", 8, 8, 4, 4),
            _loc("RA", 8, 8, 4, 4),
            _loc("LL", 10, 10, ll_internal, 5),
            _loc("RL", 10, 10, rl_internal, 5),
        ],
    }


def _make_obs(unit):
    return {"units": [unit]}


class TestCheckEarlyTermination:
    """Test MegaMekEnv._check_early_termination logic."""

    @pytest.fixture
    def env(self):
        """Create a minimal env instance (no Java needed)."""
        env = MegaMekEnv.__new__(MegaMekEnv)
        env._rl_owner_id = 0
        # Prevent __del__ warnings from _cleanup
        env._reader = None
        env._sock = None
        env._java = None
        env.config = None
        return env

    def test_healthy_unit(self, env):
        obs = _make_obs(_full_armor(prone=False))
        assert env._check_early_termination(obs) is False

    def test_prone_but_legs_intact(self, env):
        obs = _make_obs(_full_armor(prone=True, ll_internal=5, rl_internal=5))
        assert env._check_early_termination(obs) is False

    def test_leg_destroyed_but_not_prone(self, env):
        obs = _make_obs(_full_armor(prone=False, ll_internal=0))
        assert env._check_early_termination(obs) is False

    def test_prone_with_left_leg_destroyed(self, env):
        obs = _make_obs(_full_armor(prone=True, ll_internal=0))
        assert env._check_early_termination(obs) is True

    def test_prone_with_right_leg_destroyed(self, env):
        obs = _make_obs(_full_armor(prone=True, rl_internal=0))
        assert env._check_early_termination(obs) is True

    def test_prone_with_both_legs_destroyed(self, env):
        obs = _make_obs(_full_armor(prone=True, ll_internal=0, rl_internal=0))
        assert env._check_early_termination(obs) is True

    def test_empty_units(self, env):
        obs = {"units": []}
        assert env._check_early_termination(obs) is False

    def test_prone_with_leg_armor_destroyed_negative(self, env):
        """Java sends internal=-3 (ARMOR_DESTROYED) for destroyed locations."""
        obs = _make_obs(_full_armor(prone=True, ll_internal=-3))
        assert env._check_early_termination(obs) is True

    def test_prone_with_both_legs_armor_destroyed_negative(self, env):
        obs = _make_obs(_full_armor(prone=True, ll_internal=-3, rl_internal=-3))
        assert env._check_early_termination(obs) is True

    def test_no_leg_locations(self, env):
        """Unit with no leg locations in armor array (shouldn't happen, but be safe)."""
        unit = {
            "owner": 0,
            "prone": True,
            "armor": [_loc("CT", 16, 16, 11, 11)],
        }
        obs = _make_obs(unit)
        assert env._check_early_termination(obs) is False
