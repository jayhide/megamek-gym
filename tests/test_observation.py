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
            {"name": "Medium Laser", "location": 0, "destroyed": False, "damage": 5},
            {"name": "SRM 4", "location": 1, "destroyed": True, "damage": 8},
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
            {"index": 0, "dest_x": 5, "dest_y": 6, "facing": 2, "mp_used": 1, "jumping": False, "prone": False},
            {"index": 1, "dest_x": 6, "dest_y": 7, "facing": 3, "mp_used": 2, "jumping": True, "prone": False},
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

    def test_move_feature_values(self):
        flat = self._flat_with_moves()
        move_offset = BOARD_SIZE + 2 * UNIT_FEATURES + GLOBAL_FEATURES

        # Move 0: dest_x=5, dest_y=6, facing=2, mp_used=1, jumping=False, prone=False
        assert flat[move_offset] == pytest.approx(5 / 16)
        assert flat[move_offset + 1] == pytest.approx(6 / 17)
        assert flat[move_offset + 2] == pytest.approx(2 / 5.0)
        assert flat[move_offset + 3] == pytest.approx(1 / 20.0)
        assert flat[move_offset + 4] == 0.0  # not jumping
        assert flat[move_offset + 5] == 0.0  # not prone

        # Move 1: dest_x=6, dest_y=7, facing=3, mp_used=2, jumping=True, prone=False
        m1 = move_offset + MOVE_FEATURES
        assert flat[m1] == pytest.approx(6 / 16)
        assert flat[m1 + 1] == pytest.approx(7 / 17)
        assert flat[m1 + 2] == pytest.approx(3 / 5.0)
        assert flat[m1 + 3] == pytest.approx(2 / 20.0)
        assert flat[m1 + 4] == 1.0  # jumping
        assert flat[m1 + 5] == 0.0  # not prone

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
        """Deployment moves (no mp_used/jumping/prone) should default to 0."""
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
        assert flat[move_offset + 4] == 0.0  # jumping defaults to 0
        assert flat[move_offset + 5] == 0.0  # prone defaults to 0

    def test_unit_features_unchanged(self):
        """Adding move features should not affect board or unit encoding."""
        obs = _make_obs()
        flat_without = flatten_observation(obs, rl_owner_id=0)
        flat_with = self._flat_with_moves(obs)
        # First OBS_SIZE elements should be identical
        np.testing.assert_array_equal(flat_with[:OBS_SIZE], flat_without[:OBS_SIZE])


class TestIdentifyRlOwner:
    def test_finds_owner(self):
        obs = _make_obs()
        assert identify_rl_owner(obs, active_entity_id=1) == 0

    def test_raises_on_missing(self):
        obs = _make_obs()
        with pytest.raises(ValueError):
            identify_rl_owner(obs, active_entity_id=999)
