"""Tests for reward functions."""

import pytest

from megamek_gym.reward import (
    CompositeReward, CoverReward, DamageDeltaReward, LocationDestructionReward,
    RangeAdvantageReward, WinLossReward,
    _cover_value, _hex_distance, _range_quality,
)


def _make_obs(rl_armor=20, rl_internal=10, enemy_armor=15, enemy_internal=8,
              rl_destroyed=False, enemy_destroyed=False,
              rl_retreated=False, enemy_retreated=False, terminated=False,
              rl_locations=None, enemy_locations=None):
    """Build a test observation.

    *rl_locations* / *enemy_locations* override the default single-location
    armor list with a full list of location dicts (for LocationDestructionReward
    tests).  When ``None``, a single "CT" location is generated from the scalar
    armor/internal args.
    """
    if rl_locations is None:
        rl_locations = [
            {"location": "CT", "armor": rl_armor, "armor_max": 30,
             "internal": rl_internal, "internal_max": 15,
             "rear_armor": 0, "rear_armor_max": 0},
        ]
    if enemy_locations is None:
        enemy_locations = [
            {"location": "CT", "armor": enemy_armor, "armor_max": 20,
             "internal": enemy_internal, "internal_max": 10,
             "rear_armor": 0, "rear_armor_max": 0},
        ]
    return {
        "terminated": terminated,
        "units": [
            {
                "id": 1,
                "owner": 0,
                "destroyed": rl_destroyed,
                "retreated": rl_retreated,
                "armor": rl_locations,
            },
            {
                "id": 2,
                "owner": 1,
                "destroyed": enemy_destroyed,
                "retreated": enemy_retreated,
                "armor": enemy_locations,
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

    def test_internal_damage_weighted_higher(self):
        """Internal structure damage is worth 2x armor damage by default."""
        r = DamageDeltaReward(normalizer=100.0)  # default internal_multiplier=2.0
        r.reset()
        r.set_rl_owner(0)
        obs1 = _make_obs(enemy_armor=15, enemy_internal=8)
        r.compute({}, obs1, False)

        # Enemy loses 3 internal structure (no armor change)
        obs2 = _make_obs(enemy_armor=15, enemy_internal=5)
        reward = r.compute(obs1, obs2, False)
        # 3 IS * 2x multiplier = 6 weighted HP lost
        assert reward == pytest.approx(6 / 100)

    def test_internal_multiplier_custom(self):
        r = DamageDeltaReward(normalizer=100.0, internal_multiplier=3.0)
        r.reset()
        r.set_rl_owner(0)
        obs1 = _make_obs(enemy_internal=8)
        r.compute({}, obs1, False)

        obs2 = _make_obs(enemy_internal=6)  # lost 2 IS
        reward = r.compute(obs1, obs2, False)
        assert reward == pytest.approx(6 / 100)  # 2 * 3.0 = 6

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


def _loc(name, armor, armor_max, internal, internal_max, rear_armor=0, rear_armor_max=0):
    """Shorthand for building a location dict."""
    return {"location": name, "armor": armor, "armor_max": armor_max,
            "internal": internal, "internal_max": internal_max,
            "rear_armor": rear_armor, "rear_armor_max": rear_armor_max}


class TestLocationDestructionReward:
    def test_first_obs_returns_zero(self):
        r = LocationDestructionReward()
        r.reset()
        r.set_rl_owner(0)
        obs = _make_obs()
        assert r.compute({}, obs, False) == 0.0

    def test_enemy_ct_destroyed(self):
        r = LocationDestructionReward()
        r.reset()
        r.set_rl_owner(0)
        obs1 = _make_obs(enemy_internal=5)
        r.compute({}, obs1, False)

        obs2 = _make_obs(enemy_internal=0)  # CT destroyed
        reward = r.compute(obs1, obs2, False)
        assert reward == pytest.approx(1.0)  # CT weight = 1.0

    def test_own_ct_destroyed(self):
        r = LocationDestructionReward()
        r.reset()
        r.set_rl_owner(0)
        obs1 = _make_obs(rl_internal=5)
        r.compute({}, obs1, False)

        obs2 = _make_obs(rl_internal=0)  # own CT destroyed
        reward = r.compute(obs1, obs2, False)
        assert reward == pytest.approx(-1.0)

    def test_arm_weighted_less_than_ct(self):
        r = LocationDestructionReward()
        r.reset()
        r.set_rl_owner(0)
        enemy_locs1 = [_loc("RA", 0, 10, 3, 5)]
        obs1 = _make_obs(enemy_locations=enemy_locs1)
        r.compute({}, obs1, False)

        enemy_locs2 = [_loc("RA", 0, 10, 0, 5)]
        obs2 = _make_obs(enemy_locations=enemy_locs2)
        reward = r.compute(obs1, obs2, False)
        assert reward == pytest.approx(0.2)  # RA weight = 0.2

    def test_no_reward_if_internal_decreases_but_not_zero(self):
        r = LocationDestructionReward()
        r.reset()
        r.set_rl_owner(0)
        obs1 = _make_obs(enemy_internal=8)
        r.compute({}, obs1, False)

        obs2 = _make_obs(enemy_internal=3)  # damaged but not destroyed
        reward = r.compute(obs1, obs2, False)
        assert reward == 0.0

    def test_terminal_empty_units(self):
        r = LocationDestructionReward()
        r.reset()
        r.set_rl_owner(0)
        obs1 = _make_obs()
        r.compute({}, obs1, False)

        terminal = {"terminated": True, "units": []}
        assert r.compute(obs1, terminal, True) == 0.0

    def test_multiple_locations_destroyed(self):
        r = LocationDestructionReward()
        r.reset()
        r.set_rl_owner(0)
        enemy_locs1 = [_loc("LA", 0, 5, 3, 5), _loc("RA", 0, 5, 2, 5)]
        obs1 = _make_obs(enemy_locations=enemy_locs1)
        r.compute({}, obs1, False)

        enemy_locs2 = [_loc("LA", 0, 5, 0, 5), _loc("RA", 0, 5, 0, 5)]
        obs2 = _make_obs(enemy_locations=enemy_locs2)
        reward = r.compute(obs1, obs2, False)
        assert reward == pytest.approx(0.4)  # LA(0.2) + RA(0.2)


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


# --- Hex distance helper ---

class TestHexDistance:
    def test_same_hex(self):
        assert _hex_distance(3, 4, 3, 4) == 0

    def test_adjacent(self):
        # Even column (0): neighbors of (0,0) include (1,0), (0,1), (1,-1) etc.
        assert _hex_distance(0, 0, 1, 0) == 1
        assert _hex_distance(0, 0, 0, 1) == 1

    def test_known_distance(self):
        # (0,0) to (2,2): cube(0,0)=(0,0,0), cube(2,2)=(2,1,-3) → dist=3
        assert _hex_distance(0, 0, 2, 2) == 3

    def test_symmetric(self):
        assert _hex_distance(1, 3, 5, 7) == _hex_distance(5, 7, 1, 3)

    def test_odd_column_offset(self):
        # Odd column (1): (1,0) to (2,0) should be 1
        assert _hex_distance(1, 0, 2, 0) == 1


# --- Range quality helper ---

def _weapon(damage, short, medium, long, min_range=0, destroyed=False):
    return {"damage": damage, "short_range": short, "medium_range": medium,
            "long_range": long, "min_range": min_range, "destroyed": destroyed}


class TestRangeQuality:
    def test_short_range(self):
        unit = {"weapons": [_weapon(5, 3, 6, 9)]}
        assert _range_quality(unit, 2) == pytest.approx(1.0)

    def test_medium_range(self):
        unit = {"weapons": [_weapon(5, 3, 6, 9)]}
        assert _range_quality(unit, 5) == pytest.approx(0.5)

    def test_long_range(self):
        unit = {"weapons": [_weapon(5, 3, 6, 9)]}
        assert _range_quality(unit, 8) == pytest.approx(0.0)

    def test_out_of_range(self):
        unit = {"weapons": [_weapon(5, 3, 6, 9)]}
        assert _range_quality(unit, 15) == pytest.approx(-0.5)

    def test_below_min_range(self):
        unit = {"weapons": [_weapon(10, 6, 12, 18, min_range=3)]}
        assert _range_quality(unit, 1) == pytest.approx(-0.5)

    def test_damage_weighted_average(self):
        # Weapon A: damage=10, short=3 → at dist 2: short → 1.0
        # Weapon B: damage=5,  short=1, medium=2, long=3 → at dist 2: medium → 0.5
        # Weighted avg: (10*1.0 + 5*0.5) / 15 = 12.5/15
        unit = {"weapons": [
            _weapon(10, 3, 6, 9),   # distance 2 → short → 1.0
            _weapon(5, 1, 2, 3),    # distance 2 → medium → 0.5
        ]}
        assert _range_quality(unit, 2) == pytest.approx(12.5 / 15)

    def test_all_weapons_destroyed(self):
        unit = {"weapons": [_weapon(5, 3, 6, 9, destroyed=True)]}
        assert _range_quality(unit, 2) == 0.0

    def test_no_weapons(self):
        unit = {"weapons": []}
        assert _range_quality(unit, 5) == 0.0

    def test_zero_damage_weapon_ignored(self):
        unit = {"weapons": [_weapon(0, 3, 6, 9), _weapon(5, 3, 6, 9)]}
        assert _range_quality(unit, 2) == pytest.approx(1.0)


# --- RangeAdvantageReward ---

def _make_range_obs(rl_x=5, rl_y=5, enemy_x=8, enemy_y=5,
                    rl_weapons=None, enemy_weapons=None,
                    rl_destroyed=False, enemy_destroyed=False,
                    terminated=False):
    """Build an observation for range advantage tests."""
    if rl_weapons is None:
        rl_weapons = [_weapon(5, 3, 6, 9)]
    if enemy_weapons is None:
        enemy_weapons = [_weapon(5, 3, 6, 9)]
    obs = {
        "terminated": terminated,
        "units": [
            {"id": 1, "owner": 0, "x": rl_x, "y": rl_y,
             "destroyed": rl_destroyed, "weapons": rl_weapons,
             "armor": [{"location": "CT", "armor": 20, "armor_max": 30,
                        "internal": 10, "internal_max": 15,
                        "rear_armor": 0, "rear_armor_max": 0}]},
            {"id": 2, "owner": 1, "x": enemy_x, "y": enemy_y,
             "destroyed": enemy_destroyed, "weapons": enemy_weapons,
             "armor": [{"location": "CT", "armor": 15, "armor_max": 20,
                        "internal": 8, "internal_max": 10,
                        "rear_armor": 0, "rear_armor_max": 0}]},
        ],
    }
    return obs


class TestRangeAdvantageReward:
    def test_rl_at_good_range(self):
        """RL has short-range weapons, enemy has long-range; close distance favors RL."""
        r = RangeAdvantageReward()
        r.reset()
        r.set_rl_owner(0)
        # hex distance (0,0)→(2,0) = 2
        # RL weapon short=3: dist 2 ≤ 3 → 1.0
        # Enemy weapon short=1, medium=1, long=15: dist 2 > medium(1), ≤ long(15) → 0.0
        obs = _make_range_obs(
            rl_x=0, rl_y=0, enemy_x=2, enemy_y=0,
            rl_weapons=[_weapon(5, 3, 6, 9)],      # dist 2 → short → 1.0
            enemy_weapons=[_weapon(5, 1, 1, 15)],   # dist 2 → long → 0.0
        )
        reward = r.compute({}, obs, False)
        assert reward == pytest.approx(1.0)  # 1.0 - 0.0

    def test_rl_at_bad_range(self):
        """Enemy at better range → negative reward."""
        r = RangeAdvantageReward()
        r.reset()
        r.set_rl_owner(0)
        obs = _make_range_obs(
            rl_x=0, rl_y=0, enemy_x=2, enemy_y=0,
            rl_weapons=[_weapon(5, 1, 1, 15)],   # dist 2 → long → 0.0
            enemy_weapons=[_weapon(5, 3, 6, 9)],  # dist 2 → short → 1.0
        )
        reward = r.compute({}, obs, False)
        assert reward == pytest.approx(-1.0)

    def test_undeployed_rl_unit(self):
        r = RangeAdvantageReward()
        r.reset()
        r.set_rl_owner(0)
        obs = _make_range_obs(rl_x=-1, rl_y=-1)
        assert r.compute({}, obs, False) == 0.0

    def test_undeployed_enemy_unit(self):
        r = RangeAdvantageReward()
        r.reset()
        r.set_rl_owner(0)
        obs = _make_range_obs(enemy_x=-1, enemy_y=-1)
        assert r.compute({}, obs, False) == 0.0

    def test_terminal_empty_units(self):
        r = RangeAdvantageReward()
        r.reset()
        r.set_rl_owner(0)
        terminal = {"terminated": True, "units": []}
        assert r.compute({}, terminal, True) == 0.0

    def test_destroyed_unit_skipped(self):
        r = RangeAdvantageReward()
        r.reset()
        r.set_rl_owner(0)
        obs = _make_range_obs(rl_destroyed=True)
        assert r.compute({}, obs, False) == 0.0

    def test_scale(self):
        r = RangeAdvantageReward(scale=2.0)
        r.reset()
        r.set_rl_owner(0)
        obs = _make_range_obs(
            rl_x=0, rl_y=0, enemy_x=2, enemy_y=0,
            rl_weapons=[_weapon(5, 3, 6, 9)],      # dist 2 → short → 1.0
            enemy_weapons=[_weapon(5, 1, 1, 15)],   # dist 2 → long → 0.0
        )
        reward = r.compute({}, obs, False)
        assert reward == pytest.approx(2.0)


# --- Cover value helper ---

class TestCoverValue:
    def test_light_woods(self):
        hexes = [{"x": 3, "y": 4, "terrain": "Level: 1  Features: Light Woods; "}]
        assert _cover_value(hexes, 3, 4) == 1.0

    def test_heavy_woods(self):
        hexes = [{"x": 3, "y": 4, "terrain": "Level: 1  Features: Heavy Woods; "}]
        assert _cover_value(hexes, 3, 4) == 2.0

    def test_clear(self):
        hexes = [{"x": 3, "y": 4, "terrain": "Level: 0  Features: ; "}]
        assert _cover_value(hexes, 3, 4) == 0.0

    def test_rough_no_cover(self):
        hexes = [{"x": 3, "y": 4, "terrain": "Level: 1  Features: Rough; "}]
        assert _cover_value(hexes, 3, 4) == 0.0

    def test_hex_not_found(self):
        hexes = [{"x": 0, "y": 0, "terrain": "Level: 1  Features: Light Woods; "}]
        assert _cover_value(hexes, 5, 5) == 0.0


# --- CoverReward ---

def _make_cover_obs(rl_x=5, rl_y=5, enemy_x=8, enemy_y=5,
                    rl_terrain="Level: 0  Features: ; ",
                    enemy_terrain="Level: 0  Features: ; ",
                    rl_destroyed=False, enemy_destroyed=False,
                    include_board=True):
    """Build an observation for cover reward tests."""
    obs = {
        "terminated": False,
        "units": [
            {"id": 1, "owner": 0, "x": rl_x, "y": rl_y,
             "destroyed": rl_destroyed,
             "armor": [{"location": "CT", "armor": 20, "armor_max": 30,
                        "internal": 10, "internal_max": 15,
                        "rear_armor": 0, "rear_armor_max": 0}]},
            {"id": 2, "owner": 1, "x": enemy_x, "y": enemy_y,
             "destroyed": enemy_destroyed,
             "armor": [{"location": "CT", "armor": 15, "armor_max": 20,
                        "internal": 8, "internal_max": 10,
                        "rear_armor": 0, "rear_armor_max": 0}]},
        ],
    }
    if include_board:
        obs["board"] = {
            "hexes": [
                {"x": rl_x, "y": rl_y, "terrain": rl_terrain},
                {"x": enemy_x, "y": enemy_y, "terrain": enemy_terrain},
            ],
        }
    return obs


class TestCoverReward:
    def test_rl_in_light_woods(self):
        r = CoverReward()
        r.reset()
        r.set_rl_owner(0)
        obs = _make_cover_obs(
            rl_terrain="Level: 1  Features: Light Woods; ",
        )
        assert r.compute({}, obs, False) == pytest.approx(1.0)

    def test_rl_in_heavy_woods(self):
        r = CoverReward()
        r.reset()
        r.set_rl_owner(0)
        obs = _make_cover_obs(
            rl_terrain="Level: 1  Features: Heavy Woods; ",
        )
        assert r.compute({}, obs, False) == pytest.approx(2.0)

    def test_rl_in_open(self):
        r = CoverReward()
        r.reset()
        r.set_rl_owner(0)
        obs = _make_cover_obs(
            rl_terrain="Level: 0  Features: ; ",
        )
        assert r.compute({}, obs, False) == pytest.approx(0.0)

    def test_enemy_cover_ignored(self):
        """Enemy being in woods should not affect the reward."""
        r = CoverReward()
        r.reset()
        r.set_rl_owner(0)
        obs = _make_cover_obs(
            rl_terrain="Level: 1  Features: Light Woods; ",
            enemy_terrain="Level: 1  Features: Heavy Woods; ",
        )
        # Only RL cover matters
        assert r.compute({}, obs, False) == pytest.approx(1.0)

    def test_destroyed_unit_returns_zero(self):
        r = CoverReward()
        r.reset()
        r.set_rl_owner(0)
        obs = _make_cover_obs(
            rl_terrain="Level: 1  Features: Light Woods; ",
            rl_destroyed=True,
        )
        assert r.compute({}, obs, False) == 0.0

    def test_undeployed_unit_returns_zero(self):
        r = CoverReward()
        r.reset()
        r.set_rl_owner(0)
        obs = _make_cover_obs(rl_x=-1, rl_y=-1)
        assert r.compute({}, obs, False) == 0.0

    def test_empty_units_returns_zero(self):
        r = CoverReward()
        r.reset()
        r.set_rl_owner(0)
        obs = {"terminated": True, "units": []}
        assert r.compute({}, obs, True) == 0.0

    def test_scale(self):
        r = CoverReward(scale=3.0)
        r.reset()
        r.set_rl_owner(0)
        obs = _make_cover_obs(
            rl_terrain="Level: 1  Features: Heavy Woods; ",
        )
        assert r.compute({}, obs, False) == pytest.approx(6.0)

    def test_no_board_hexes(self):
        r = CoverReward()
        r.reset()
        r.set_rl_owner(0)
        obs = _make_cover_obs(include_board=False)
        assert r.compute({}, obs, False) == 0.0
