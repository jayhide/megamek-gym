"""Tests for observation flattening."""

import numpy as np
import pytest

from megamek_gym.observation import (
    BOARD_SIZE,
    OBS_SIZE,
    UNIT_FEATURES,
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
            {"index": 0, "dest_x": 5, "dest_y": 6, "facing": 2, "mp_used": 1},
            {"index": 1, "dest_x": 6, "dest_y": 7, "facing": 3, "mp_used": 2},
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


class TestIdentifyRlOwner:
    def test_finds_owner(self):
        obs = _make_obs()
        assert identify_rl_owner(obs, active_entity_id=1) == 0

    def test_raises_on_missing(self):
        obs = _make_obs()
        with pytest.raises(ValueError):
            identify_rl_owner(obs, active_entity_id=999)
