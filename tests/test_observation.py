"""Tests for observation flattening."""

import numpy as np
import pytest

from megamek_gym.observation import (
    BOARD_SIZE,
    GLOBAL_FEATURES,
    MOVE_FEATURES,
    OBS_SIZE,
    UNIT_FEATURES,
    compute_obs_size,
    flatten_observation,
    identify_rl_owner,
)


def _make_unit(owner, unit_id=1, x=5, y=7, facing=2, **kwargs):
    return {
        "id": unit_id,
        "owner": owner,
        "chassis": "Test",
        "model": "T-1",
        "x": x,
        "y": y,
        "facing": facing,
        "mp_walk": 6,
        "mp_run": 9,
        "mp_jump": 4,
        "heat": 5,
        "prone": False,
        "destroyed": False,
        "deployed": True,
        "armor": [
            {
                "location": "CT",
                "armor": 20,
                "armor_max": 30,
                "internal": 10,
                "internal_max": 15,
                "rear_armor": 8,
                "rear_armor_max": 10,
            },
            {
                "location": "LT",
                "armor": 15,
                "armor_max": 20,
                "internal": 8,
                "internal_max": 10,
                "rear_armor": 5,
                "rear_armor_max": 8,
            },
        ],
        "weapons": [
            {"name": "Medium Laser", "location": 0, "destroyed": False, "damage": 5,
             "min_range": 0, "short_range": 3, "medium_range": 6, "long_range": 9},
            {"name": "SRM 4", "location": 1, "destroyed": True, "damage": 8,
             "min_range": 0, "short_range": 3, "medium_range": 6, "long_range": 9},
        ],
        **kwargs,
    }


def _make_obs(rl_owner=0, enemy_owner=1, hexes=None):
    if hexes is None:
        hexes = [{"x": 0, "y": 0, "elevation": 3}]
    return {
        "type": "observation",
        "round": 1,
        "phase": "MOVEMENT",
        "active_entity_id": 1,
        "reward": 0.0,
        "terminated": False,
        "truncated": False,
        "board": {"width": 16, "height": 17, "hexes": hexes},
        "units": [
            _make_unit(rl_owner, unit_id=1, x=5, y=7),
            _make_unit(enemy_owner, unit_id=2, x=10, y=12, facing=4),
        ],
        "legal_moves": [
            {"index": 0, "dest_x": 5, "dest_y": 6, "facing": 2, "mp_used": 1, "jumping": False, "prone": False, "has_los": True},
            {"index": 1, "dest_x": 6, "dest_y": 7, "facing": 3, "mp_used": 2, "jumping": True, "prone": False, "has_los": False},
        ],
    }


class TestFlattenObservation:
    def test_output_shape(self):
        obs = _make_obs()
        flat = flatten_observation(obs, rl_owner_id=0)
        assert flat.shape == (OBS_SIZE,)
        assert flat.dtype == np.float32

    def test_board_elevation(self):
        hexes = [{"x": 3, "y": 2, "elevation": 5}]
        obs = _make_obs(hexes=hexes)
        flat = flatten_observation(obs, rl_owner_id=0)
        idx = 2 * 16 + 3  # row 2, col 3
        assert flat[idx] == pytest.approx(0.5)  # 5/10

    def test_unit_position(self):
        obs = _make_obs()
        flat = flatten_observation(obs, rl_owner_id=0)
        # RL unit at x=5, y=7
        offset = BOARD_SIZE
        assert flat[offset] == pytest.approx(5 / 16)
        assert flat[offset + 1] == pytest.approx(7 / 17)

    def test_facing_one_hot(self):
        obs = _make_obs()
        flat = flatten_observation(obs, rl_owner_id=0)
        offset = BOARD_SIZE + 2  # after position
        # RL unit facing=2
        expected = [0, 0, 1, 0, 0, 0]
        np.testing.assert_array_equal(flat[offset : offset + 6], expected)

    def test_movement_points(self):
        obs = _make_obs()
        flat = flatten_observation(obs, rl_owner_id=0)
        offset = BOARD_SIZE + 8  # pos(2) + facing(6)
        assert flat[offset] == pytest.approx(6 / 20)  # walk
        assert flat[offset + 1] == pytest.approx(9 / 20)  # run
        assert flat[offset + 2] == pytest.approx(4 / 20)  # jump

    def test_heat(self):
        obs = _make_obs()
        flat = flatten_observation(obs, rl_owner_id=0)
        offset = BOARD_SIZE + 11  # pos + facing + mp
        assert flat[offset] == pytest.approx(5 / 30)

    def test_status_flags(self):
        obs = _make_obs()
        flat = flatten_observation(obs, rl_owner_id=0)
        offset = BOARD_SIZE + 12
        assert flat[offset] == 0.0  # not prone
        assert flat[offset + 1] == 0.0  # not destroyed
        assert flat[offset + 2] == 1.0  # deployed
        assert flat[offset + 3] == 0.0  # not retreated

    def test_armor_ratios(self):
        obs = _make_obs()
        flat = flatten_observation(obs, rl_owner_id=0)
        offset = BOARD_SIZE + 16  # pos(2) + facing(6) + mp(3) + heat(1) + status(4)
        # CT: armor 20/30, internal 10/15, rear 8/10, not destroyed
        assert flat[offset] == pytest.approx(20 / 30)
        assert flat[offset + 1] == pytest.approx(10 / 15)
        assert flat[offset + 2] == pytest.approx(8 / 10)
        assert flat[offset + 3] == 0.0  # not location-destroyed

    def test_destroyed_location_negative_internal(self):
        """Java sends internal=-3 (ARMOR_DESTROYED) for destroyed locations."""
        obs = _make_obs()
        # Set CT to destroyed: armor=-3, internal=-3
        obs["units"][0]["armor"][0]["armor"] = -3
        obs["units"][0]["armor"][0]["internal"] = -3
        flat = flatten_observation(obs, rl_owner_id=0)
        offset = BOARD_SIZE + 16  # armor section for RL unit
        # internal ratio should be clamped to 0.0, not negative
        assert flat[offset + 1] == pytest.approx(0.0)
        # location destroyed flag should be 1.0
        assert flat[offset + 3] == 1.0

    def test_weapon_destroyed_flags(self):
        obs = _make_obs()
        flat = flatten_observation(obs, rl_owner_id=0)
        offset = BOARD_SIZE + 16 + 32  # after armor (8 locs × 4)
        assert flat[offset] == 0.0  # Medium Laser not destroyed
        assert flat[offset + 1] == 1.0  # SRM 4 destroyed
        assert flat[offset + 2] == 0.0  # padded

    def test_enemy_unit_offset(self):
        obs = _make_obs()
        flat = flatten_observation(obs, rl_owner_id=0)
        enemy_offset = BOARD_SIZE + UNIT_FEATURES
        # Enemy at x=10, y=12
        assert flat[enemy_offset] == pytest.approx(10 / 16)
        assert flat[enemy_offset + 1] == pytest.approx(12 / 17)

    def test_rl_moves_first_false(self):
        obs = _make_obs()
        flat = flatten_observation(obs, rl_owner_id=0)
        global_offset = BOARD_SIZE + 2 * UNIT_FEATURES
        assert flat[global_offset] == 0.0

    def test_rl_moves_first_true(self):
        obs = _make_obs()
        obs["rl_moves_first"] = True
        flat = flatten_observation(obs, rl_owner_id=0)
        global_offset = BOARD_SIZE + 2 * UNIT_FEATURES
        assert flat[global_offset] == 1.0

    def test_terminal_empty_obs(self):
        obs = {
            "type": "observation",
            "active_entity_id": -1,
            "terminated": True,
            "board": {"width": 0, "height": 0, "hexes": []},
            "units": [],
            "legal_moves": [],
        }
        flat = flatten_observation(obs, rl_owner_id=0)
        np.testing.assert_array_equal(flat, np.zeros(OBS_SIZE, dtype=np.float32))

    def test_values_in_range(self):
        obs = _make_obs()
        flat = flatten_observation(obs, rl_owner_id=0)
        assert flat.min() >= -1.0
        assert flat.max() <= 1.0


class TestMoveFeatures:
    """Tests for legal move feature embedding in the observation."""

    MAX_MOVES = 10  # small value for tests

    def _flat_with_moves(self, obs=None, max_legal_moves=None):
        if obs is None:
            obs = _make_obs()
        if max_legal_moves is None:
            max_legal_moves = self.MAX_MOVES
        return flatten_observation(
            obs, rl_owner_id=0,
            legal_moves=obs["legal_moves"],
            max_legal_moves=max_legal_moves,
        )

    def test_obs_size_with_moves(self):
        size = compute_obs_size(16, 17, max_legal_moves=1000)
        assert size == 16 * 17 + 2 * UNIT_FEATURES + GLOBAL_FEATURES + 1000 * MOVE_FEATURES

    def test_backward_compat_no_moves(self):
        """Without max_legal_moves, obs size is unchanged."""
        size = compute_obs_size(16, 17)
        assert size == OBS_SIZE

    def test_output_shape(self):
        flat = self._flat_with_moves()
        expected = compute_obs_size(16, 17, self.MAX_MOVES)
        assert flat.shape == (expected,)

    def test_kinematic_feature_values(self):
        flat = self._flat_with_moves()
        move_offset = BOARD_SIZE + 2 * UNIT_FEATURES + GLOBAL_FEATURES

        # Move 0: dest_x=5, dest_y=6, facing=2, mp_used=1, prone=False
        assert flat[move_offset] == pytest.approx(5 / 16)
        assert flat[move_offset + 1] == pytest.approx(6 / 17)
        assert flat[move_offset + 2] == pytest.approx(2 / 5.0)
        assert flat[move_offset + 3] == pytest.approx(1 / 20.0)
        assert flat[move_offset + 4] == 0.0  # not prone

        # Move 1: dest_x=6, dest_y=7, facing=3, mp_used=2, prone=False
        m1 = move_offset + MOVE_FEATURES
        assert flat[m1] == pytest.approx(6 / 16)
        assert flat[m1 + 1] == pytest.approx(7 / 17)
        assert flat[m1 + 2] == pytest.approx(3 / 5.0)
        assert flat[m1 + 3] == pytest.approx(2 / 20.0)
        assert flat[m1 + 4] == 0.0  # not prone

    def test_padding_zeros(self):
        """Unused move slots should be all zeros."""
        flat = self._flat_with_moves()
        move_offset = BOARD_SIZE + 2 * UNIT_FEATURES + GLOBAL_FEATURES
        # Moves 2..9 should be zeros (only 2 legal moves)
        pad_start = move_offset + 2 * MOVE_FEATURES
        pad_end = move_offset + self.MAX_MOVES * MOVE_FEATURES
        np.testing.assert_array_equal(flat[pad_start:pad_end], 0.0)

    def test_no_legal_moves_all_zeros(self):
        """When legal_moves is empty, all move feature slots are zero."""
        obs = _make_obs()
        obs["legal_moves"] = []
        flat = flatten_observation(
            obs, rl_owner_id=0,
            legal_moves=[],
            max_legal_moves=self.MAX_MOVES,
        )
        move_offset = BOARD_SIZE + 2 * UNIT_FEATURES + GLOBAL_FEATURES
        move_end = move_offset + self.MAX_MOVES * MOVE_FEATURES
        np.testing.assert_array_equal(flat[move_offset:move_end], 0.0)

    def test_deployment_move_zeros(self):
        """Deployment moves (no mp_used/prone) should default to 0."""
        obs = _make_obs()
        obs["legal_moves"] = [
            {"index": 0, "dest_x": 3, "dest_y": 4, "facing": 0},
        ]
        flat = flatten_observation(
            obs, rl_owner_id=0,
            legal_moves=obs["legal_moves"],
            max_legal_moves=self.MAX_MOVES,
        )
        move_offset = BOARD_SIZE + 2 * UNIT_FEATURES + GLOBAL_FEATURES
        assert flat[move_offset] == pytest.approx(3 / 16)
        assert flat[move_offset + 1] == pytest.approx(4 / 17)
        assert flat[move_offset + 2] == pytest.approx(0 / 5.0)
        assert flat[move_offset + 3] == 0.0  # mp_used defaults to 0
        assert flat[move_offset + 4] == 0.0  # prone defaults to 0

    def test_unit_features_unchanged(self):
        """Adding move features should not affect board or unit encoding."""
        obs = _make_obs()
        flat_without = flatten_observation(obs, rl_owner_id=0)
        flat_with = self._flat_with_moves(obs)
        # First OBS_SIZE elements should be identical
        np.testing.assert_array_equal(flat_with[:OBS_SIZE], flat_without[:OBS_SIZE])


class TestTacticalMoveFeatures:
    """Tests for pre-computed tactical features in the move embedding."""

    MAX_MOVES = 10

    def test_move_feature_count(self):
        assert MOVE_FEATURES == 11

    def test_dist_to_enemy(self):
        """Distance from move destination to enemy position."""
        from megamek_gym.reward import hex_distance
        obs = _make_obs()
        # Move 0 goes to (5, 6), enemy at (10, 12)
        flat = flatten_observation(
            obs, rl_owner_id=0,
            legal_moves=obs["legal_moves"],
            max_legal_moves=self.MAX_MOVES,
        )
        move_offset = BOARD_SIZE + 2 * UNIT_FEATURES + GLOBAL_FEATURES
        expected_dist = hex_distance(5, 6, 10, 12)
        max_dim = max(16, 17)
        assert flat[move_offset + 5] == pytest.approx(expected_dist / max_dim)

    def test_range_quality_feature(self):
        """RL weapon effectiveness from hypothetical move position."""
        from megamek_gym.reward import hex_distance, range_quality
        # Place enemy close enough to be in Medium Laser range (short=3)
        obs = _make_obs()
        obs["units"][1]["x"] = 7
        obs["units"][1]["y"] = 7
        # Move 0 goes to (5, 6) — close to enemy at (7, 7)
        flat = flatten_observation(
            obs, rl_owner_id=0,
            legal_moves=obs["legal_moves"],
            max_legal_moves=self.MAX_MOVES,
        )
        move_offset = BOARD_SIZE + 2 * UNIT_FEATURES + GLOBAL_FEATURES
        dist = hex_distance(5, 6, 7, 7)
        expected = range_quality(
            obs["units"][0], dist,
            target_x=7, target_y=7,
            unit_x=5, unit_y=6, unit_facing=2,
        )
        assert flat[move_offset + 6] == pytest.approx(expected)
        assert expected != 0.0  # sanity: should be a real score

    def test_enemy_range_quality_feature(self):
        """Enemy weapon effectiveness at move distance."""
        from megamek_gym.reward import hex_distance, range_quality
        obs = _make_obs()
        obs["units"][1]["x"] = 7
        obs["units"][1]["y"] = 7
        flat = flatten_observation(
            obs, rl_owner_id=0,
            legal_moves=obs["legal_moves"],
            max_legal_moves=self.MAX_MOVES,
        )
        move_offset = BOARD_SIZE + 2 * UNIT_FEATURES + GLOBAL_FEATURES
        dist = hex_distance(5, 6, 7, 7)
        expected = range_quality(
            obs["units"][1], dist,
            target_x=5, target_y=6,
            unit_x=7, unit_y=7, unit_facing=4,
        )
        assert flat[move_offset + 7] == pytest.approx(expected)
        assert expected != 0.0

    def test_terrain_cover_feature(self):
        """Cover value at move destination."""
        hexes = [
            {"x": 5, "y": 6, "elevation": 0, "terrain": "Light Woods"},
            {"x": 6, "y": 7, "elevation": 0},
        ]
        obs = _make_obs(hexes=hexes)
        flat = flatten_observation(
            obs, rl_owner_id=0,
            legal_moves=obs["legal_moves"],
            max_legal_moves=self.MAX_MOVES,
        )
        move_offset = BOARD_SIZE + 2 * UNIT_FEATURES + GLOBAL_FEATURES
        # Move 0 dest (5,6) is Light Woods → cover_value=1.0, normalized /2.0 = 0.5
        assert flat[move_offset + 8] == pytest.approx(0.5)
        # Move 1 dest (6,7) has no terrain → 0.0
        m1 = move_offset + MOVE_FEATURES
        assert flat[m1 + 8] == pytest.approx(0.0)

    def test_elevation_diff_feature(self):
        """Elevation difference between move dest and enemy position."""
        hexes = [
            {"x": 5, "y": 6, "elevation": 3},
            {"x": 10, "y": 12, "elevation": 1},  # enemy hex
        ]
        obs = _make_obs(hexes=hexes)
        flat = flatten_observation(
            obs, rl_owner_id=0,
            legal_moves=obs["legal_moves"],
            max_legal_moves=self.MAX_MOVES,
        )
        move_offset = BOARD_SIZE + 2 * UNIT_FEATURES + GLOBAL_FEATURES
        # Move 0 dest (5,6) elev=3, enemy (10,12) elev=1 → diff=2, /10 = 0.2
        assert flat[move_offset + 9] == pytest.approx(0.2)

    def test_has_los_feature(self):
        """LOS boolean feature from Java-side precomputed lookup."""
        obs = _make_obs()
        flat = flatten_observation(
            obs, rl_owner_id=0,
            legal_moves=obs["legal_moves"],
            max_legal_moves=self.MAX_MOVES,
        )
        move_offset = BOARD_SIZE + 2 * UNIT_FEATURES + GLOBAL_FEATURES
        # Move 0 has_los=True → 1.0
        assert flat[move_offset + 10] == pytest.approx(1.0)
        # Move 1 has_los=False → 0.0
        m1 = move_offset + MOVE_FEATURES
        assert flat[m1 + 10] == pytest.approx(0.0)

    def test_has_los_defaults_false(self):
        """Missing has_los field defaults to 0.0."""
        obs = _make_obs()
        obs["legal_moves"] = [
            {"index": 0, "dest_x": 5, "dest_y": 6, "facing": 2, "mp_used": 1,
             "jumping": False, "prone": False},  # no has_los field
        ]
        flat = flatten_observation(
            obs, rl_owner_id=0,
            legal_moves=obs["legal_moves"],
            max_legal_moves=self.MAX_MOVES,
        )
        move_offset = BOARD_SIZE + 2 * UNIT_FEATURES + GLOBAL_FEATURES
        assert flat[move_offset + 10] == pytest.approx(0.0)

    def test_tactical_features_no_enemy(self):
        """When enemy is missing, tactical features default to 0."""
        obs = _make_obs()
        obs["units"] = [obs["units"][0]]  # remove enemy
        flat = flatten_observation(
            obs, rl_owner_id=0,
            legal_moves=obs["legal_moves"],
            max_legal_moves=self.MAX_MOVES,
        )
        move_offset = BOARD_SIZE + 2 * UNIT_FEATURES + GLOBAL_FEATURES
        # All 6 tactical features should be 0.0 (dist, range_quality, enemy_range_quality, cover, elev_diff, has_los)
        for feat_idx in range(5, 11):
            assert flat[move_offset + feat_idx] == 0.0

    def test_tactical_features_padding_zeros(self):
        """Unused move slots should have zero tactical features."""
        obs = _make_obs()
        flat = flatten_observation(
            obs, rl_owner_id=0,
            legal_moves=obs["legal_moves"],
            max_legal_moves=self.MAX_MOVES,
        )
        move_offset = BOARD_SIZE + 2 * UNIT_FEATURES + GLOBAL_FEATURES
        # Slots 2..9 should be all zeros
        pad_start = move_offset + 2 * MOVE_FEATURES
        pad_end = move_offset + self.MAX_MOVES * MOVE_FEATURES
        np.testing.assert_array_equal(flat[pad_start:pad_end], 0.0)


class TestIdentifyRlOwner:
    def test_finds_owner(self):
        obs = _make_obs()
        assert identify_rl_owner(obs, active_entity_id=1) == 0

    def test_raises_on_missing(self):
        obs = _make_obs()
        with pytest.raises(ValueError):
            identify_rl_owner(obs, active_entity_id=999)
