"""Tests for early termination detection (prone + leg destroyed)."""

import pytest

from megamek_gym.env import MegaMekEnv


def _loc(name, armor, armor_max, internal, internal_max):
    return {"location": name, "armor": armor, "armor_max": armor_max,
            "internal": internal, "internal_max": internal_max,
            "rear_armor": 0, "rear_armor_max": 0}


def _unit(owner=0, ll_internal=5, rl_internal=5, prone=False):
    """Build a unit dict with full mech locations."""
    return {
        "owner": owner,
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


# Backward compat alias used by existing tests
def _full_armor(ll_internal=5, rl_internal=5, prone=False):
    return _unit(owner=0, ll_internal=ll_internal, rl_internal=rl_internal, prone=prone)


def _make_obs(*units):
    return {"units": list(units)}


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

    # --- RL unit checks (LOSS) ---

    def test_healthy_unit(self, env):
        obs = _make_obs(_full_armor(prone=False))
        assert env._check_early_termination(obs) is None

    def test_prone_but_legs_intact(self, env):
        obs = _make_obs(_full_armor(prone=True, ll_internal=5, rl_internal=5))
        assert env._check_early_termination(obs) is None

    def test_leg_destroyed_but_not_prone(self, env):
        obs = _make_obs(_full_armor(prone=False, ll_internal=0))
        assert env._check_early_termination(obs) is None

    def test_prone_with_left_leg_destroyed(self, env):
        obs = _make_obs(_full_armor(prone=True, ll_internal=0))
        assert env._check_early_termination(obs) == "LOSS"

    def test_prone_with_right_leg_destroyed(self, env):
        obs = _make_obs(_full_armor(prone=True, rl_internal=0))
        assert env._check_early_termination(obs) == "LOSS"

    def test_prone_with_both_legs_destroyed(self, env):
        obs = _make_obs(_full_armor(prone=True, ll_internal=0, rl_internal=0))
        assert env._check_early_termination(obs) == "LOSS"

    def test_empty_units(self, env):
        obs = {"units": []}
        assert env._check_early_termination(obs) is None

    def test_prone_with_leg_armor_destroyed_negative(self, env):
        """Java sends internal=-3 (ARMOR_DESTROYED) for destroyed locations."""
        obs = _make_obs(_full_armor(prone=True, ll_internal=-3))
        assert env._check_early_termination(obs) == "LOSS"

    def test_prone_with_both_legs_armor_destroyed_negative(self, env):
        obs = _make_obs(_full_armor(prone=True, ll_internal=-3, rl_internal=-3))
        assert env._check_early_termination(obs) == "LOSS"

    def test_no_leg_locations(self, env):
        """Unit with no leg locations in armor array (shouldn't happen, but be safe)."""
        unit = {
            "owner": 0,
            "prone": True,
            "armor": [_loc("CT", 16, 16, 11, 11)],
        }
        obs = _make_obs(unit)
        assert env._check_early_termination(obs) is None

    # --- Enemy unit checks (WIN) ---

    def test_enemy_healthy(self, env):
        obs = _make_obs(_unit(owner=0), _unit(owner=1))
        assert env._check_early_termination(obs) is None

    def test_enemy_prone_with_left_leg_destroyed(self, env):
        obs = _make_obs(_unit(owner=0), _unit(owner=1, prone=True, ll_internal=0))
        assert env._check_early_termination(obs) == "WIN"

    def test_enemy_prone_with_right_leg_destroyed(self, env):
        obs = _make_obs(_unit(owner=0), _unit(owner=1, prone=True, rl_internal=0))
        assert env._check_early_termination(obs) == "WIN"

    def test_enemy_prone_but_legs_intact(self, env):
        obs = _make_obs(_unit(owner=0), _unit(owner=1, prone=True))
        assert env._check_early_termination(obs) is None

    def test_enemy_leg_destroyed_but_not_prone(self, env):
        obs = _make_obs(_unit(owner=0), _unit(owner=1, prone=False, ll_internal=0))
        assert env._check_early_termination(obs) is None

    def test_both_crippled_rl_checked_first(self, env):
        """When both units are crippled, RL is checked first -> LOSS."""
        obs = _make_obs(
            _unit(owner=0, prone=True, ll_internal=0),
            _unit(owner=1, prone=True, rl_internal=0),
        )
        assert env._check_early_termination(obs) == "LOSS"
