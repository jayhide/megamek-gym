"""Tests for reward functions."""

import pytest

from megamek_gym.reward import CompositeReward, DamageDeltaReward, WinLossReward


def _make_obs(rl_armor=20, rl_internal=10, enemy_armor=15, enemy_internal=8,
              rl_destroyed=False, enemy_destroyed=False,
              rl_retreated=False, enemy_retreated=False, terminated=False):
    return {
        "terminated": terminated,
        "units": [
            {
                "id": 1,
                "owner": 0,
                "destroyed": rl_destroyed,
                "retreated": rl_retreated,
                "armor": [
                    {"armor": rl_armor, "armor_max": 30,
                     "internal": rl_internal, "internal_max": 15,
                     "rear_armor": 0, "rear_armor_max": 0},
                ],
            },
            {
                "id": 2,
                "owner": 1,
                "destroyed": enemy_destroyed,
                "retreated": enemy_retreated,
                "armor": [
                    {"armor": enemy_armor, "armor_max": 20,
                     "internal": enemy_internal, "internal_max": 10,
                     "rear_armor": 0, "rear_armor_max": 0},
                ],
            },
        ],
    }


class TestDamageDeltaReward:
    def test_first_obs_returns_zero(self):
        r = DamageDeltaReward()
        r.reset()
        r.set_rl_owner(0)
        obs = _make_obs()
        assert r.compute({}, obs, False) == 0.0

    def test_damage_to_enemy(self):
        r = DamageDeltaReward(normalizer=100.0)
        r.reset()
        r.set_rl_owner(0)
        obs1 = _make_obs(enemy_armor=15)
        r.compute({}, obs1, False)  # baseline

        obs2 = _make_obs(enemy_armor=10)  # enemy lost 5 armor
        reward = r.compute(obs1, obs2, False)
        assert reward == pytest.approx(5 / 100)

    def test_damage_to_self(self):
        r = DamageDeltaReward(normalizer=100.0)
        r.reset()
        r.set_rl_owner(0)
        obs1 = _make_obs()
        r.compute({}, obs1, False)

        obs2 = _make_obs(rl_armor=15)  # RL lost 5 armor
        reward = r.compute(obs1, obs2, False)
        assert reward == pytest.approx(-5 / 100)

    def test_terminal_empty_units(self):
        r = DamageDeltaReward()
        r.reset()
        r.set_rl_owner(0)
        obs1 = _make_obs()
        r.compute({}, obs1, False)

        terminal = {"terminated": True, "units": []}
        assert r.compute(obs1, terminal, True) == 0.0


class TestWinLossReward:
    def test_no_reward_before_terminal(self):
        r = WinLossReward(scale=10.0)
        r.reset()
        r.set_rl_owner(0)
        obs = _make_obs()
        assert r.compute({}, obs, False) == 0.0

    def test_win(self):
        r = WinLossReward(scale=10.0)
        r.reset()
        r.set_rl_owner(0)
        obs = _make_obs(enemy_destroyed=True)
        assert r.compute({}, obs, True) == 10.0

    def test_loss(self):
        r = WinLossReward(scale=10.0)
        r.reset()
        r.set_rl_owner(0)
        obs = _make_obs(rl_destroyed=True)
        assert r.compute({}, obs, True) == -10.0

    def test_draw(self):
        r = WinLossReward()
        r.reset()
        r.set_rl_owner(0)
        obs = _make_obs(rl_destroyed=True, enemy_destroyed=True)
        assert r.compute({}, obs, True) == 0.0

    def test_terminal_empty_uses_prev(self):
        r = WinLossReward(scale=1.0)
        r.reset()
        r.set_rl_owner(0)
        prev = _make_obs(enemy_destroyed=True)
        terminal = {"terminated": True, "units": []}
        assert r.compute(prev, terminal, True) == 1.0

    def test_game_outcome_win(self):
        r = WinLossReward(scale=10.0)
        r.reset()
        r.set_rl_owner(0)
        terminal = {"terminated": True, "units": [], "game_outcome": "WIN"}
        assert r.compute({}, terminal, True) == 10.0

    def test_game_outcome_loss(self):
        r = WinLossReward(scale=10.0)
        r.reset()
        r.set_rl_owner(0)
        terminal = {"terminated": True, "units": [], "game_outcome": "LOSS"}
        assert r.compute({}, terminal, True) == -10.0

    def test_game_outcome_draw(self):
        r = WinLossReward(scale=10.0)
        r.reset()
        r.set_rl_owner(0)
        terminal = {"terminated": True, "units": [], "game_outcome": "DRAW"}
        assert r.compute({}, terminal, True) == 0.0

    def test_enemy_retreated_is_win(self):
        r = WinLossReward(scale=10.0)
        r.reset()
        r.set_rl_owner(0)
        obs = _make_obs(enemy_retreated=True)
        assert r.compute({}, obs, True) == 10.0

    def test_own_retreated_is_loss(self):
        r = WinLossReward(scale=10.0)
        r.reset()
        r.set_rl_owner(0)
        obs = _make_obs(rl_retreated=True)
        assert r.compute({}, obs, True) == -10.0

    def test_both_retreated_is_draw(self):
        r = WinLossReward()
        r.reset()
        r.set_rl_owner(0)
        obs = _make_obs(rl_retreated=True, enemy_retreated=True)
        assert r.compute({}, obs, True) == 0.0


class TestCompositeReward:
    def test_default_components(self):
        r = CompositeReward()
        r.reset()
        r.set_rl_owner(0)
        obs = _make_obs()
        # First call should be 0 (baseline)
        assert r.compute({}, obs, False) == 0.0

    def test_weighted_sum(self):
        dd = DamageDeltaReward(normalizer=100.0)
        wl = WinLossReward(scale=1.0)
        r = CompositeReward([(dd, 2.0), (wl, 5.0)])
        r.reset()
        r.set_rl_owner(0)

        obs1 = _make_obs(enemy_armor=15)
        r.compute({}, obs1, False)

        # Enemy lost 5 armor, terminated with enemy destroyed
        obs2 = _make_obs(enemy_armor=10, enemy_destroyed=True)
        reward = r.compute(obs1, obs2, True)
        # dd: 2.0 * 5/100 = 0.1, wl: 5.0 * 1.0 = 5.0
        assert reward == pytest.approx(5.1)
