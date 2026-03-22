"""Tests for reward functions."""

import pytest

from megamek_gym.reward import (
    CompositeReward, CoverReward, DamageDeltaReward, LocationDestructionReward,
    PronePenaltyReward, RangeAdvantageReward, WinLossReward,
    cover_value, _get_prone_status, hex_bearing, hex_distance,
    in_firing_arc, range_quality,
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

    def test_destroyed_location_negative_internal_clamped(self):
        """Java sends internal=-3 for destroyed locations; should be treated as 0."""
        r = DamageDeltaReward(normalizer=100.0)
        r.reset()
        r.set_rl_owner(0)
        # Enemy starts with internal=5
        obs1 = _make_obs(enemy_internal=5)
        r.compute({}, obs1, False)

        # Enemy location destroyed: internal=-3 (ARMOR_DESTROYED)
        obs2 = _make_obs(enemy_internal=-3)
        reward = r.compute(obs1, obs2, False)
        # Damage = 5 internal lost * 2x multiplier = 10
        assert reward == pytest.approx(10 / 100)

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
        assert hex_distance(3, 4, 3, 4) == 0

    def test_adjacent(self):
        # Even column (0): neighbors of (0,0) include (1,0), (0,1), (1,-1) etc.
        assert hex_distance(0, 0, 1, 0) == 1
        assert hex_distance(0, 0, 0, 1) == 1

    def test_known_distance(self):
        # (0,0) to (2,2): cube(0,0)=(0,0,0), cube(2,2)=(2,1,-3) → dist=3
        assert hex_distance(0, 0, 2, 2) == 3

    def test_symmetric(self):
        assert hex_distance(1, 3, 5, 7) == hex_distance(5, 7, 1, 3)

    def test_odd_column_offset(self):
        # Odd column (1): (1,0) to (2,0) should be 1
        assert hex_distance(1, 0, 2, 0) == 1


# --- Hex bearing helper ---

class TestHexBearing:
    def test_same_hex(self):
        assert hex_bearing(3, 3, 3, 3) == 0.0

    def test_due_north(self):
        # (5, 5) to (5, 3) — straight up (decreasing y = north)
        bearing = hex_bearing(5, 5, 5, 3)
        assert bearing == pytest.approx(0.0, abs=1.0)

    def test_due_south(self):
        bearing = hex_bearing(5, 3, 5, 5)
        assert bearing == pytest.approx(180.0, abs=1.0)

    def test_northeast(self):
        # Moving right and up in hex grid — bearing should be roughly 0-90°
        bearing = hex_bearing(4, 4, 5, 3)
        assert 0 < bearing < 90

    def test_southeast(self):
        bearing = hex_bearing(4, 4, 5, 5)
        assert 90 < bearing < 180

    def test_symmetry(self):
        """Bearing from A→B and B→A should differ by ~180°."""
        b1 = hex_bearing(2, 3, 5, 1)
        b2 = hex_bearing(5, 1, 2, 3)
        diff = abs(b1 - b2)
        if diff > 180:
            diff = 360 - diff
        assert diff == pytest.approx(180.0, abs=1.0)


# --- Firing arc helper ---

class TestInFiringArc:
    def test_forward_weapon_facing_north_target_north(self):
        """CT weapon, facing 0 (north), target due north → in arc."""
        assert in_firing_arc(0, 1, 0.0) is True

    def test_forward_weapon_facing_north_target_east(self):
        """CT weapon, facing 0 (north), target 90° east → outside forward arc (±60°)."""
        assert in_firing_arc(0, 1, 90.0) is False

    def test_forward_weapon_facing_north_target_ne(self):
        """CT weapon, facing 0 (north), target 60° NE → exactly at boundary."""
        assert in_firing_arc(0, 1, 60.0) is True

    def test_forward_weapon_facing_north_target_behind(self):
        """CT weapon, facing 0 (north), target 180° south → behind, out of arc."""
        assert in_firing_arc(0, 1, 180.0) is False

    def test_right_arm_extends_right(self):
        """RA (loc 4), facing 0 (north), target 120° → at RA boundary (120° right)."""
        # Forward arc covers ±60°. RA extends to 120° clockwise from facing.
        assert in_firing_arc(0, 4, 120.0) is True

    def test_right_arm_limit(self):
        """RA (loc 4), facing 0 (north), target 160° → outside even RA arc (>120°)."""
        assert in_firing_arc(0, 4, 160.0) is False

    def test_left_arm_extends_left(self):
        """LA (loc 5), facing 0 (north), target 240° → at LA boundary (120° left)."""
        assert in_firing_arc(0, 5, 240.0) is True

    def test_left_arm_limit(self):
        """LA (loc 5), facing 0 (north), target 200° → outside even LA arc (>120° left)."""
        assert in_firing_arc(0, 5, 200.0) is False

    def test_facing_3_south(self):
        """Facing 3 (south = 180°), CT weapon, target due south → in arc."""
        assert in_firing_arc(3, 1, 180.0) is True

    def test_facing_3_north_behind(self):
        """Facing 3 (south = 180°), CT weapon, target due north → behind."""
        assert in_firing_arc(3, 1, 0.0) is False

    def test_head_same_as_forward(self):
        """HD (loc 0) uses forward arc."""
        assert in_firing_arc(0, 0, 45.0) is True
        assert in_firing_arc(0, 0, 180.0) is False

    def test_leg_same_as_forward(self):
        """Legs (loc 6, 7) use forward arc."""
        assert in_firing_arc(0, 6, 45.0) is True
        assert in_firing_arc(0, 7, 180.0) is False


# --- Range quality helper ---

def _weapon(damage, short, medium, long, min_range=0, destroyed=False, location=1):
    return {"damage": damage, "short_range": short, "medium_range": medium,
            "long_range": long, "min_range": min_range, "destroyed": destroyed,
            "location": location}


class TestRangeQuality:
    def test_short_range(self):
        unit = {"weapons": [_weapon(5, 3, 6, 9)]}
        assert range_quality(unit, 2) == pytest.approx(1.0)

    def test_medium_range(self):
        unit = {"weapons": [_weapon(5, 3, 6, 9)]}
        assert range_quality(unit, 5) == pytest.approx(0.5)

    def test_long_range(self):
        unit = {"weapons": [_weapon(5, 3, 6, 9)]}
        assert range_quality(unit, 8) == pytest.approx(0.0)

    def test_out_of_range(self):
        unit = {"weapons": [_weapon(5, 3, 6, 9)]}
        assert range_quality(unit, 15) == pytest.approx(-0.5)

    def test_below_min_range(self):
        unit = {"weapons": [_weapon(10, 6, 12, 18, min_range=3)]}
        assert range_quality(unit, 1) == pytest.approx(-0.5)

    def test_damage_weighted_average(self):
        # Weapon A: damage=10, short=3 → at dist 2: short → 1.0
        # Weapon B: damage=5,  short=1, medium=2, long=3 → at dist 2: medium → 0.5
        # Weighted avg: (10*1.0 + 5*0.5) / 15 = 12.5/15
        unit = {"weapons": [
            _weapon(10, 3, 6, 9),   # distance 2 → short → 1.0
            _weapon(5, 1, 2, 3),    # distance 2 → medium → 0.5
        ]}
        assert range_quality(unit, 2) == pytest.approx(12.5 / 15)

    def test_all_weapons_destroyed(self):
        unit = {"weapons": [_weapon(5, 3, 6, 9, destroyed=True)]}
        assert range_quality(unit, 2) == 0.0

    def test_no_weapons(self):
        unit = {"weapons": []}
        assert range_quality(unit, 5) == 0.0

    def test_zero_damage_weapon_ignored(self):
        unit = {"weapons": [_weapon(0, 3, 6, 9), _weapon(5, 3, 6, 9)]}
        assert range_quality(unit, 2) == pytest.approx(1.0)


# --- Range quality with facing ---

class TestRangeQualityWithFacing:
    def test_weapon_in_arc_unchanged(self):
        """CT weapon facing toward target — same score as without facing."""
        unit = {"weapons": [_weapon(5, 3, 6, 9, location=1)]}  # CT
        # Without facing
        score_no_facing = range_quality(unit, 2)
        # With facing 0 (north), target due north at (5, 3) from (5, 5)
        score_facing = range_quality(unit, 2, target_x=5, target_y=3,
                                       unit_x=5, unit_y=5, unit_facing=0)
        assert score_no_facing == pytest.approx(1.0)
        assert score_facing == pytest.approx(1.0)

    def test_weapon_out_of_arc_penalized(self):
        """CT weapon facing away from target — score halved."""
        unit = {"weapons": [_weapon(5, 3, 6, 9, location=1)]}  # CT
        # Facing 3 (south = 180°), target is north at (5, 3) from (5, 5)
        score = range_quality(unit, 2, target_x=5, target_y=3,
                                unit_x=5, unit_y=5, unit_facing=3)
        # Normal short-range score is 1.0, halved to 0.5
        assert score == pytest.approx(0.5)

    def test_mixed_weapons_in_and_out_of_arc(self):
        """One weapon in arc, one out — damage-weighted average."""
        # Both at short range (score 1.0 base), but one in CT (in arc) and one in CT facing away
        # Let's use: facing 0 (north), target east at bearing ~90° (boundary)
        # CT weapon at 90° is at the boundary (in arc), but target at 150° would be out
        # Use: facing 0, target due south (180°)
        # CT weapon: out of arc → 1.0 * 0.5 = 0.5
        # RA weapon (loc 4): also out since 180° > 150° → 1.0 * 0.5 = 0.5
        # LA weapon (loc 5): also out since 180° → left_diff = (0-180)%360 = 180 > 150 → out
        # Better test: facing 0, target at 120° bearing
        # CT (loc 1): 120° > 90° → out of arc → score * 0.5
        # RA (loc 4): 120° ≤ 150° → in arc → full score
        unit = {"weapons": [
            _weapon(5, 3, 6, 9, location=1),   # CT — out of forward arc at 120°
            _weapon(5, 3, 6, 9, location=4),   # RA — in extended right arc at 120°
        ]}
        # We need actual hex coords that produce ~120° bearing
        # (0,0) to (2,2): bearing should be roughly south-east
        # Let's just test with explicit bearing by using _range_quality directly
        # Facing 0, and we pick coords where bearing ≈ 120°
        # For a clean test, use facing=0 and south (180°) target
        # CT: out → 1.0 * 0.5 = 0.5, weighted by damage 5
        # RA (loc 4): 180° > 150° → also out → 1.0 * 0.5 = 0.5
        # That's not interesting. Let me use facing=1 (60°) and bearing ~120° target
        # Forward arc: 60° ± 90° = [-30°, 150°] → 330°-150°. Bearing 180° is outside.
        # RA extends right to 60°+150° = 210°. Bearing 180° < 210° → in arc.
        unit_facing1 = {"weapons": [
            _weapon(10, 3, 6, 9, location=1),  # CT — out of forward arc at 180°
            _weapon(5, 3, 6, 9, location=4),   # RA — in extended arc at 180°
        ]}
        score = range_quality(unit_facing1, 2, target_x=5, target_y=7,
                                unit_x=5, unit_y=5, unit_facing=1)
        # CT: short range = 1.0 * 0.5 (out of arc) = 0.5, weight 10
        # RA: short range = 1.0 (in arc), weight 5
        # Weighted avg: (10*0.5 + 5*1.0) / 15 = 10/15 = 0.6667
        assert score == pytest.approx(10.0 / 15.0, abs=0.05)

    def test_backward_compat_no_facing(self):
        """Without facing args, behaves identically to old version."""
        unit = {"weapons": [_weapon(5, 3, 6, 9, location=1)]}
        assert range_quality(unit, 2) == pytest.approx(1.0)


# --- RangeAdvantageReward ---

def _make_range_obs(rl_x=5, rl_y=5, enemy_x=8, enemy_y=5,
                    rl_weapons=None, enemy_weapons=None,
                    rl_destroyed=False, enemy_destroyed=False,
                    rl_facing=None, enemy_facing=None,
                    terminated=False):
    """Build an observation for range advantage tests."""
    if rl_weapons is None:
        rl_weapons = [_weapon(5, 3, 6, 9)]
    if enemy_weapons is None:
        enemy_weapons = [_weapon(5, 3, 6, 9)]
    rl_unit = {"id": 1, "owner": 0, "x": rl_x, "y": rl_y,
               "destroyed": rl_destroyed, "weapons": rl_weapons,
               "armor": [{"location": "CT", "armor": 20, "armor_max": 30,
                          "internal": 10, "internal_max": 15,
                          "rear_armor": 0, "rear_armor_max": 0}]}
    enemy_unit = {"id": 2, "owner": 1, "x": enemy_x, "y": enemy_y,
                  "destroyed": enemy_destroyed, "weapons": enemy_weapons,
                  "armor": [{"location": "CT", "armor": 15, "armor_max": 20,
                             "internal": 8, "internal_max": 10,
                             "rear_armor": 0, "rear_armor_max": 0}]}
    if rl_facing is not None:
        rl_unit["facing"] = rl_facing
    if enemy_facing is not None:
        enemy_unit["facing"] = enemy_facing
    obs = {
        "terminated": terminated,
        "units": [rl_unit, enemy_unit],
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
        # 0.3*0.0 + 0.7*(0.0 - 1.0) = -0.7
        assert reward == pytest.approx(-0.7)

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

    def test_mirror_matchup_facing_breaks_tie(self):
        """Mirror matchup: same weapons, same distance. Facing toward target wins."""
        r = RangeAdvantageReward()
        r.reset()
        r.set_rl_owner(0)
        weapons = [_weapon(5, 3, 6, 9, location=1)]  # CT weapon
        # RL at (5, 5) facing 0 (north), enemy at (5, 3) facing 3 (south=toward RL)
        # RL facing north, enemy is north → RL faces toward enemy → in arc
        # Enemy facing south, RL is south of enemy → enemy faces toward RL → in arc
        # Both face each other → should still cancel (both in arc)
        # For a real tie-break: RL faces AWAY from enemy
        # RL facing 3 (south), enemy at (5, 3) is north → RL faces away
        # Enemy facing 3 (south), RL at (5, 5) is south → enemy faces toward RL
        obs = _make_range_obs(
            rl_x=5, rl_y=5, enemy_x=5, enemy_y=3,
            rl_weapons=weapons, enemy_weapons=weapons,
            rl_facing=3, enemy_facing=3,  # both face south
        )
        reward = r.compute({}, obs, False)
        # RL faces south, enemy is north → RL weapons out of arc → score * 0.5 = 0.5
        # Enemy faces south, RL is south → enemy weapons in arc → score = 1.0
        # 0.3*0.5 + 0.7*(0.5 - 1.0) = 0.15 - 0.35 = -0.2
        assert reward == pytest.approx(-0.2)

    def test_both_facing_each_other_absolute_bonus(self):
        """Mirror matchup with both facing each other → absolute term gives positive reward."""
        r = RangeAdvantageReward()
        r.reset()
        r.set_rl_owner(0)
        weapons = [_weapon(5, 3, 6, 9, location=1)]  # CT weapon
        # RL at (5,5) facing 0 (north), enemy at (5,3) facing 3 (south)
        # Both face each other → both in arc → rl_quality=1.0, enemy_quality=1.0
        obs = _make_range_obs(
            rl_x=5, rl_y=5, enemy_x=5, enemy_y=3,
            rl_weapons=weapons, enemy_weapons=weapons,
            rl_facing=0, enemy_facing=3,
        )
        reward = r.compute({}, obs, False)
        # 0.3*1.0 + 0.7*(1.0 - 1.0) = 0.3
        assert reward == pytest.approx(0.3)

    def test_no_facing_field_gives_absolute_bonus(self):
        """Without facing, mirror matchup gets absolute bonus (no arc penalty applied)."""
        r = RangeAdvantageReward()
        r.reset()
        r.set_rl_owner(0)
        # Same weapons, same distance → differential=0, but absolute term adds rl_quality
        obs = _make_range_obs(
            rl_x=0, rl_y=0, enemy_x=2, enemy_y=0,
            rl_weapons=[_weapon(5, 3, 6, 9)],
            enemy_weapons=[_weapon(5, 3, 6, 9)],
        )
        reward = r.compute({}, obs, False)
        # 0.3*1.0 + 0.7*0.0 = 0.3
        assert reward == pytest.approx(0.3)

    def test_mirror_out_of_range_penalty(self):
        """Mirror matchup out of range → mild negative from absolute term."""
        r = RangeAdvantageReward()
        r.reset()
        r.set_rl_owner(0)
        # Both out of range: short=3, dist=15 → quality=-0.5 each
        obs = _make_range_obs(
            rl_x=0, rl_y=0, enemy_x=15, enemy_y=0,
            rl_weapons=[_weapon(5, 3, 6, 9)],
            enemy_weapons=[_weapon(5, 3, 6, 9)],
        )
        reward = r.compute({}, obs, False)
        # 0.3*(-0.5) + 0.7*0.0 = -0.15
        assert reward == pytest.approx(-0.15)

    def test_mirror_in_range_positive(self):
        """Mirror matchup in range → positive from absolute term."""
        r = RangeAdvantageReward()
        r.reset()
        r.set_rl_owner(0)
        # Both in short range: dist=2, short=3 → quality=1.0 each
        obs = _make_range_obs(
            rl_x=0, rl_y=0, enemy_x=2, enemy_y=0,
            rl_weapons=[_weapon(5, 3, 6, 9)],
            enemy_weapons=[_weapon(5, 3, 6, 9)],
        )
        reward = r.compute({}, obs, False)
        # 0.3*1.0 + 0.7*0.0 = 0.3
        assert reward == pytest.approx(0.3)

    def test_absolute_weight_zero_is_pure_differential(self):
        """absolute_weight=0 gives old pure-differential behavior."""
        r = RangeAdvantageReward(absolute_weight=0.0)
        r.reset()
        r.set_rl_owner(0)
        obs = _make_range_obs(
            rl_x=0, rl_y=0, enemy_x=2, enemy_y=0,
            rl_weapons=[_weapon(5, 3, 6, 9)],
            enemy_weapons=[_weapon(5, 3, 6, 9)],
        )
        reward = r.compute({}, obs, False)
        assert reward == pytest.approx(0.0)

    def test_absolute_weight_one_is_pure_absolute(self):
        """absolute_weight=1.0 gives pure RL range quality."""
        r = RangeAdvantageReward(absolute_weight=1.0)
        r.reset()
        r.set_rl_owner(0)
        obs = _make_range_obs(
            rl_x=0, rl_y=0, enemy_x=2, enemy_y=0,
            rl_weapons=[_weapon(5, 3, 6, 9)],      # dist 2 → short → 1.0
            enemy_weapons=[_weapon(5, 1, 1, 15)],   # dist 2 → long → 0.0
        )
        reward = r.compute({}, obs, False)
        # Pure absolute: 1.0 * 1.0 = 1.0 (enemy quality ignored)
        assert reward == pytest.approx(1.0)


# --- Cover value helper ---

class TestCoverValue:
    def test_light_woods(self):
        hexes = [{"x": 3, "y": 4, "terrain": "Level: 1  Features: Light Woods; "}]
        assert cover_value(hexes, 3, 4) == 1.0

    def test_heavy_woods(self):
        hexes = [{"x": 3, "y": 4, "terrain": "Level: 1  Features: Heavy Woods; "}]
        assert cover_value(hexes, 3, 4) == 2.0

    def test_clear(self):
        hexes = [{"x": 3, "y": 4, "terrain": "Level: 0  Features: ; "}]
        assert cover_value(hexes, 3, 4) == 0.0

    def test_rough_no_cover(self):
        hexes = [{"x": 3, "y": 4, "terrain": "Level: 1  Features: Rough; "}]
        assert cover_value(hexes, 3, 4) == 0.0

    def test_hex_not_found(self):
        hexes = [{"x": 0, "y": 0, "terrain": "Level: 1  Features: Light Woods; "}]
        assert cover_value(hexes, 5, 5) == 0.0


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


# --- Prone status helper ---

class TestGetProneStatus:
    def test_not_prone(self):
        units = [{"owner": 0, "destroyed": False, "prone": False}]
        assert _get_prone_status(units, 0) is False

    def test_prone(self):
        units = [{"owner": 0, "destroyed": False, "prone": True}]
        assert _get_prone_status(units, 0) is True

    def test_destroyed_unit_skipped(self):
        units = [{"owner": 0, "destroyed": True, "prone": True}]
        assert _get_prone_status(units, 0) is None

    def test_no_matching_owner(self):
        units = [{"owner": 1, "destroyed": False, "prone": True}]
        assert _get_prone_status(units, 0) is None

    def test_missing_prone_field(self):
        units = [{"owner": 0, "destroyed": False}]
        assert _get_prone_status(units, 0) is False


# --- PronePenaltyReward ---

def _make_prone_obs(rl_prone=False, enemy_prone=False,
                    rl_destroyed=False, enemy_destroyed=False):
    """Build an observation for prone penalty tests."""
    return {
        "terminated": False,
        "units": [
            {"id": 1, "owner": 0, "prone": rl_prone,
             "destroyed": rl_destroyed,
             "armor": [{"location": "CT", "armor": 20, "armor_max": 30,
                        "internal": 10, "internal_max": 15,
                        "rear_armor": 0, "rear_armor_max": 0}]},
            {"id": 2, "owner": 1, "prone": enemy_prone,
             "destroyed": enemy_destroyed,
             "armor": [{"location": "CT", "armor": 15, "armor_max": 20,
                        "internal": 8, "internal_max": 10,
                        "rear_armor": 0, "rear_armor_max": 0}]},
        ],
    }


class TestPronePenaltyReward:
    def test_first_obs_returns_zero(self):
        r = PronePenaltyReward()
        r.reset()
        r.set_rl_owner(0)
        obs = _make_prone_obs(rl_prone=False)
        assert r.compute({}, obs, False) == 0.0

    def test_not_prone_to_prone_penalty(self):
        r = PronePenaltyReward()
        r.reset()
        r.set_rl_owner(0)
        obs1 = _make_prone_obs(rl_prone=False)
        r.compute({}, obs1, False)  # baseline

        obs2 = _make_prone_obs(rl_prone=True)
        reward = r.compute(obs1, obs2, False)
        assert reward == pytest.approx(-1.0)

    def test_prone_to_not_prone_no_reward(self):
        r = PronePenaltyReward()
        r.reset()
        r.set_rl_owner(0)
        obs1 = _make_prone_obs(rl_prone=True)
        r.compute({}, obs1, False)  # baseline

        obs2 = _make_prone_obs(rl_prone=False)
        reward = r.compute(obs1, obs2, False)
        assert reward == 0.0

    def test_prone_to_prone_no_reward(self):
        r = PronePenaltyReward()
        r.reset()
        r.set_rl_owner(0)
        obs1 = _make_prone_obs(rl_prone=True)
        r.compute({}, obs1, False)

        obs2 = _make_prone_obs(rl_prone=True)
        assert r.compute(obs1, obs2, False) == 0.0

    def test_not_prone_to_not_prone_no_reward(self):
        r = PronePenaltyReward()
        r.reset()
        r.set_rl_owner(0)
        obs1 = _make_prone_obs(rl_prone=False)
        r.compute({}, obs1, False)

        obs2 = _make_prone_obs(rl_prone=False)
        assert r.compute(obs1, obs2, False) == 0.0

    def test_terminal_empty_units(self):
        r = PronePenaltyReward()
        r.reset()
        r.set_rl_owner(0)
        obs1 = _make_prone_obs(rl_prone=False)
        r.compute({}, obs1, False)

        terminal = {"terminated": True, "units": []}
        assert r.compute(obs1, terminal, True) == 0.0

    def test_scale(self):
        r = PronePenaltyReward(scale=2.0)
        r.reset()
        r.set_rl_owner(0)
        obs1 = _make_prone_obs(rl_prone=False)
        r.compute({}, obs1, False)

        obs2 = _make_prone_obs(rl_prone=True)
        assert r.compute(obs1, obs2, False) == pytest.approx(-2.0)

    def test_destroyed_unit_returns_zero(self):
        r = PronePenaltyReward()
        r.reset()
        r.set_rl_owner(0)
        obs1 = _make_prone_obs(rl_prone=False)
        r.compute({}, obs1, False)

        obs2 = _make_prone_obs(rl_prone=True, rl_destroyed=True)
        assert r.compute(obs1, obs2, False) == 0.0
