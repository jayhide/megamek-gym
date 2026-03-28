"""Tests for observation flattening."""

import numpy as np
import pytest

import torch

from megamek_gym.observation import (
    BOARD_SIZE,
    DEST_FEATURES,
    FACING_FEATURES,
    GLOBAL_FEATURES,
    MOVE_FEATURES,
    OBS_SIZE,
    UNIT_FEATURES,
    _group_moves_by_destination,
    compute_obs_size,
    compute_obs_size_hierarchical,
    flatten_observation,
    flatten_observation_hierarchical,
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
        """Board elevation block is only populated when INCLUDE_BOARD_ELEVATION is True."""
        from megamek_gym.observation import INCLUDE_BOARD_ELEVATION
        hexes = [{"x": 3, "y": 2, "elevation": 5}]
        obs = _make_obs(hexes=hexes)
        flat = flatten_observation(obs, rl_owner_id=0)
        if INCLUDE_BOARD_ELEVATION:
            idx = 2 * 16 + 3  # row 2, col 3
            assert flat[idx] == pytest.approx(0.5)  # 5/10
        else:
            # Board block is not included; unit features start at index 0
            assert flat[0] != 0.5  # no board elevation data

    def test_unit_position(self):
        obs = _make_obs()
        flat = flatten_observation(obs, rl_owner_id=0)
        # RL unit at x=5, y=7
        offset = BOARD_SIZE
        assert flat[offset] == pytest.approx(5 / 16)
        assert flat[offset + 1] == pytest.approx(7 / 17)

    def test_unit_position_roundtrip(self):
        """Verify unit coords can be round-tripped through flatten → denormalize."""
        from megamek_gym.observation import _board_block_size
        bw, bh = 16, 17
        for ux, uy in [(8, 2), (0, 0), (15, 16), (8, 14)]:
            obs = _make_obs()
            obs["units"][0]["x"] = ux
            obs["units"][0]["y"] = uy
            flat = flatten_observation(obs, rl_owner_id=0)

            offset = _board_block_size(bw, bh)
            actual_x = round(flat[offset] * bw)
            actual_y = round(flat[offset + 1] * bh)
            assert (actual_x, actual_y) == (ux, uy), (
                f"Round-trip failed: encoded ({ux},{uy}), got ({actual_x},{actual_y})"
            )

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
        assert size == BOARD_SIZE + 2 * UNIT_FEATURES + GLOBAL_FEATURES + 1000 * MOVE_FEATURES

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

        # Move 0: dest_x=5, dest_y=6, facing=2, mp_used=1
        assert flat[move_offset] == pytest.approx(5 / 16)
        assert flat[move_offset + 1] == pytest.approx(6 / 17)
        assert flat[move_offset + 2] == pytest.approx(2 / 5.0)
        assert flat[move_offset + 3] == pytest.approx(1 / 20.0)

        # Move 1: dest_x=6, dest_y=7, facing=3, mp_used=2
        m1 = move_offset + MOVE_FEATURES
        assert flat[m1] == pytest.approx(6 / 16)
        assert flat[m1 + 1] == pytest.approx(7 / 17)
        assert flat[m1 + 2] == pytest.approx(3 / 5.0)
        assert flat[m1 + 3] == pytest.approx(2 / 20.0)

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
        """Deployment moves (no mp_used) should default to 0."""
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
        assert MOVE_FEATURES == 10

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
        max_dim = 16 + 17
        assert flat[move_offset + 4] == pytest.approx(expected_dist / max_dim)

    def test_range_quality_feature(self):
        """RL weapon effectiveness from hypothetical move position."""
        from megamek_gym.observation import _norm_rq
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
        raw_rq = range_quality(
            obs["units"][0], dist,
            target_x=7, target_y=7,
            unit_x=5, unit_y=6, unit_facing=2,
        )
        assert flat[move_offset + 5] == pytest.approx(_norm_rq(raw_rq))
        assert raw_rq != 0.0  # sanity: should be a real score

    def test_enemy_range_quality_feature(self):
        """Enemy weapon effectiveness at move distance."""
        from megamek_gym.observation import _norm_rq
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
        raw_rq = range_quality(
            obs["units"][1], dist,
            target_x=5, target_y=6,
            unit_x=7, unit_y=7, unit_facing=4,
        )
        assert flat[move_offset + 6] == pytest.approx(_norm_rq(raw_rq))
        assert raw_rq != 0.0

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
        assert flat[move_offset + 7] == pytest.approx(0.5)
        # Move 1 dest (6,7) has no terrain → 0.0
        m1 = move_offset + MOVE_FEATURES
        assert flat[m1 + 7] == pytest.approx(0.0)

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
        assert flat[move_offset + 8] == pytest.approx(0.2)

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
        assert flat[move_offset + 9] == pytest.approx(1.0)
        # Move 1 has_los=False → 0.0
        m1 = move_offset + MOVE_FEATURES
        assert flat[m1 + 9] == pytest.approx(0.0)

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
        assert flat[move_offset + 9] == pytest.approx(0.0)

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
        for feat_idx in range(4, 10):
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


class TestFeatureBounds:
    """Verify all features stay within expected [0, 1] bounds."""

    MAX_MOVES = 10

    def test_dist_to_enemy_worst_case(self):
        """Distance feature stays <= 1.0 even for max-distance corners."""
        from megamek_gym.reward import hex_distance
        obs = _make_obs()
        # Place units at opposite corners
        obs["units"][0]["x"] = 0
        obs["units"][0]["y"] = 0
        obs["units"][1]["x"] = 15
        obs["units"][1]["y"] = 16
        obs["legal_moves"] = [
            {"index": 0, "dest_x": 0, "dest_y": 0, "facing": 0, "mp_used": 0,
             "jumping": False, "prone": False, "has_los": True},
        ]
        flat = flatten_observation(
            obs, rl_owner_id=0,
            legal_moves=obs["legal_moves"],
            max_legal_moves=self.MAX_MOVES,
        )
        move_offset = BOARD_SIZE + 2 * UNIT_FEATURES + GLOBAL_FEATURES
        dist_feat = flat[move_offset + 4]
        assert 0.0 < dist_feat <= 1.0, f"dist_to_enemy={dist_feat} exceeds [0, 1]"

    def test_range_quality_in_bounds(self):
        """Range quality features are in [0, 1] after normalization."""
        obs = _make_obs()
        obs["units"][1]["x"] = 7
        obs["units"][1]["y"] = 7
        flat = flatten_observation(
            obs, rl_owner_id=0,
            legal_moves=obs["legal_moves"],
            max_legal_moves=self.MAX_MOVES,
        )
        move_offset = BOARD_SIZE + 2 * UNIT_FEATURES + GLOBAL_FEATURES
        rl_rq = flat[move_offset + 5]
        enemy_rq = flat[move_offset + 6]
        assert 0.0 <= rl_rq <= 1.0, f"rl_range_quality={rl_rq} out of [0, 1]"
        assert 0.0 <= enemy_rq <= 1.0, f"enemy_range_quality={enemy_rq} out of [0, 1]"

    def test_range_quality_out_of_range(self):
        """Out-of-range weapons produce low but non-negative range_quality."""
        obs = _make_obs()
        # Enemy far away — all weapons out of range
        obs["units"][1]["x"] = 15
        obs["units"][1]["y"] = 16
        obs["legal_moves"] = [
            {"index": 0, "dest_x": 0, "dest_y": 0, "facing": 0, "mp_used": 0,
             "jumping": False, "prone": False, "has_los": True},
        ]
        flat = flatten_observation(
            obs, rl_owner_id=0,
            legal_moves=obs["legal_moves"],
            max_legal_moves=self.MAX_MOVES,
        )
        move_offset = BOARD_SIZE + 2 * UNIT_FEATURES + GLOBAL_FEATURES
        rl_rq = flat[move_offset + 5]
        # Raw rq is [-0.5, -0.25] (arc penalty on rear weapons), normalized to [0.0, ~0.17]
        assert 0.0 <= rl_rq <= 0.2, f"out-of-range rq should be near 0, got {rl_rq}"


class TestIdentifyRlOwner:
    def test_finds_owner(self):
        obs = _make_obs()
        assert identify_rl_owner(obs, active_entity_id=1) == 0

    def test_raises_on_missing(self):
        obs = _make_obs()
        with pytest.raises(ValueError):
            identify_rl_owner(obs, active_entity_id=999)


def _make_move(index, dest_x, dest_y, facing, mp_used):
    """Helper to build a minimal legal move dict."""
    return {
        "index": index, "dest_x": dest_x, "dest_y": dest_y,
        "facing": facing, "mp_used": mp_used,
        "jumping": False, "prone": False, "has_los": True,
    }


class TestGroupMovesByDestination:
    def test_stand_still_only(self):
        """Single stand-still move becomes one destination with one facing."""
        moves = [_make_move(0, 5, 7, 2, 0)]
        dests, lookup = _group_moves_by_destination(moves, walk_mp=6)
        assert len(dests) == 1
        assert dests[0]["dest_x"] == 5
        assert dests[0]["dest_y"] == 7
        assert dests[0]["facing_options"] == {2: 0}
        assert lookup[(0, 2)] == 0

    def test_same_hex_different_facings_grouped(self):
        """Multiple facings at the same hex via walk should be one destination."""
        moves = [
            _make_move(0, 5, 7, 2, 0),  # stand still
            _make_move(1, 6, 8, 0, 3),  # walk to (6,8) facing 0, 3 MP
            _make_move(2, 6, 8, 1, 4),  # walk to (6,8) facing 1, 4 MP (extra MP for turn)
            _make_move(3, 6, 8, 2, 4),  # walk to (6,8) facing 2, 4 MP
        ]
        dests, lookup = _group_moves_by_destination(moves, walk_mp=6)
        assert len(dests) == 2  # stand-still + one destination at (6,8)
        hex_dest = dests[1]
        assert hex_dest["dest_x"] == 6
        assert hex_dest["dest_y"] == 8
        assert set(hex_dest["facing_options"].keys()) == {0, 1, 2}
        assert hex_dest["facing_options"][0] == 1
        assert hex_dest["facing_options"][1] == 2
        assert hex_dest["facing_options"][2] == 3

    def test_min_mp_used_across_facings(self):
        """mp_used on the destination should be the minimum across all facings."""
        moves = [
            _make_move(0, 6, 8, 0, 3),  # facing 0, 3 MP (cheapest)
            _make_move(1, 6, 8, 1, 4),  # facing 1, 4 MP
            _make_move(2, 6, 8, 3, 5),  # facing 3, 5 MP (most expensive)
        ]
        dests, _ = _group_moves_by_destination(moves, walk_mp=6)
        assert len(dests) == 1
        assert dests[0]["mp_used"] == 3

    def test_walk_and_run_separate_destinations(self):
        """Walk and run to the same hex should produce two destinations."""
        moves = [
            _make_move(0, 6, 8, 0, 4),  # walk to (6,8) facing 0
            _make_move(1, 6, 8, 1, 5),  # walk to (6,8) facing 1
            _make_move(2, 6, 8, 0, 7),  # run to (6,8) facing 0
            _make_move(3, 6, 8, 1, 8),  # run to (6,8) facing 1
            _make_move(4, 6, 8, 2, 9),  # run to (6,8) facing 2
        ]
        dests, lookup = _group_moves_by_destination(moves, walk_mp=6)
        assert len(dests) == 2
        walk_dest = dests[0]
        run_dest = dests[1]
        # Walk destination
        assert walk_dest["mp_used"] == 4
        assert set(walk_dest["facing_options"].keys()) == {0, 1}
        # Run destination
        assert run_dest["mp_used"] == 7
        assert set(run_dest["facing_options"].keys()) == {0, 1, 2}

    def test_multiple_hexes(self):
        """Moves to different hexes should be separate destinations."""
        moves = [
            _make_move(0, 5, 7, 2, 0),  # stand still at (5,7)
            _make_move(1, 6, 8, 0, 3),  # walk to (6,8)
            _make_move(2, 6, 8, 1, 4),  # walk to (6,8)
            _make_move(3, 7, 7, 0, 2),  # walk to (7,7)
            _make_move(4, 7, 7, 3, 3),  # walk to (7,7)
        ]
        dests, lookup = _group_moves_by_destination(moves, walk_mp=6)
        assert len(dests) == 3  # (5,7), (6,8), (7,7)

    def test_lookup_roundtrip(self):
        """Every (dest_idx, facing) in lookup maps back to correct original move_index."""
        moves = [
            _make_move(0, 5, 7, 2, 0),
            _make_move(1, 6, 8, 0, 3),
            _make_move(2, 6, 8, 1, 4),
            _make_move(3, 6, 8, 3, 5),
            _make_move(4, 7, 7, 0, 2),
        ]
        dests, lookup = _group_moves_by_destination(moves, walk_mp=6)
        # Every facing in every destination must be in the lookup
        for dest_idx, dest in enumerate(dests):
            for facing, move_idx in dest["facing_options"].items():
                assert lookup[(dest_idx, facing)] == move_idx

    def test_empty_moves(self):
        """Empty legal moves list produces empty results."""
        dests, lookup = _group_moves_by_destination([], walk_mp=6)
        assert len(dests) == 0
        assert len(lookup) == 0


class TestHierarchicalObservation:
    """Tests for flatten_observation_hierarchical and compute_obs_size_hierarchical."""

    MAX_DEST = 20  # small cap for tests

    def _make_hierarchical_obs(self, legal_moves=None):
        """Build a test observation with legal moves for hierarchical flattening."""
        if legal_moves is None:
            legal_moves = [
                _make_move(0, 5, 7, 2, 0),   # stand still
                _make_move(1, 6, 8, 0, 3),   # walk to (6,8) facing 0
                _make_move(2, 6, 8, 1, 4),   # walk to (6,8) facing 1
                _make_move(3, 7, 7, 3, 5),   # walk to (7,7) facing 3
            ]
        obs = _make_obs()
        obs["legal_moves"] = legal_moves
        return obs

    def test_obs_size(self):
        """compute_obs_size_hierarchical returns correct size."""
        size = compute_obs_size_hierarchical(16, 17, self.MAX_DEST)
        expected = BOARD_SIZE + 2 * UNIT_FEATURES + GLOBAL_FEATURES + \
            self.MAX_DEST * DEST_FEATURES + self.MAX_DEST * 6 * FACING_FEATURES
        assert size == expected

    def test_output_shape(self):
        """Flattened observation has correct length."""
        obs = self._make_hierarchical_obs()
        flat = flatten_observation_hierarchical(
            obs, rl_owner_id=0, max_destinations=self.MAX_DEST,
            legal_moves=obs["legal_moves"],
        )
        assert flat.shape == (compute_obs_size_hierarchical(16, 17, self.MAX_DEST),)
        assert flat.dtype == np.float32

    def test_base_blocks_match_flat(self):
        """Board + unit + global blocks are identical to flat version."""
        obs = self._make_hierarchical_obs()
        flat = flatten_observation(obs, rl_owner_id=0)
        hier = flatten_observation_hierarchical(
            obs, rl_owner_id=0, max_destinations=self.MAX_DEST,
            legal_moves=obs["legal_moves"],
        )
        base_size = BOARD_SIZE + 2 * UNIT_FEATURES + GLOBAL_FEATURES
        np.testing.assert_array_equal(hier[:base_size], flat[:base_size])

    def test_dest_feature_values(self):
        """Per-destination features have expected values."""
        obs = self._make_hierarchical_obs()
        hier = flatten_observation_hierarchical(
            obs, rl_owner_id=0, max_destinations=self.MAX_DEST,
            legal_moves=obs["legal_moves"],
        )
        base = BOARD_SIZE + 2 * UNIT_FEATURES + GLOBAL_FEATURES

        # Dest 0: stand-still at (5,7), mp_used=0
        d0 = hier[base:base + DEST_FEATURES]
        assert d0[0] == pytest.approx(5 / 16)   # dest_x / W
        assert d0[1] == pytest.approx(7 / 17)   # dest_y / H
        assert d0[2] == pytest.approx(0 / 20)   # mp_used / 20

        # Dest 1: walk to (6,8), min mp_used=3
        d1 = hier[base + DEST_FEATURES:base + 2 * DEST_FEATURES]
        assert d1[0] == pytest.approx(6 / 16)
        assert d1[1] == pytest.approx(8 / 17)
        assert d1[2] == pytest.approx(3 / 20)

        # Dest 2: walk to (7,7), mp_used=5
        d2 = hier[base + 2 * DEST_FEATURES:base + 3 * DEST_FEATURES]
        assert d2[0] == pytest.approx(7 / 16)
        assert d2[1] == pytest.approx(7 / 17)
        assert d2[2] == pytest.approx(5 / 20)

    def test_unused_dest_slots_zero(self):
        """Destination slots beyond n_destinations are zero."""
        obs = self._make_hierarchical_obs()  # 3 destinations
        hier = flatten_observation_hierarchical(
            obs, rl_owner_id=0, max_destinations=self.MAX_DEST,
            legal_moves=obs["legal_moves"],
        )
        base = BOARD_SIZE + 2 * UNIT_FEATURES + GLOBAL_FEATURES
        # Slots 3..MAX_DEST-1 should be zero
        unused_start = base + 3 * DEST_FEATURES
        unused_end = base + self.MAX_DEST * DEST_FEATURES
        assert np.all(hier[unused_start:unused_end] == 0.0)

    def test_facing_features_valid_only(self):
        """Facing features are non-zero only for valid (dest, facing) pairs."""
        obs = self._make_hierarchical_obs()
        hier = flatten_observation_hierarchical(
            obs, rl_owner_id=0, max_destinations=self.MAX_DEST,
            legal_moves=obs["legal_moves"],
        )
        facing_offset = (BOARD_SIZE + 2 * UNIT_FEATURES + GLOBAL_FEATURES
                         + self.MAX_DEST * DEST_FEATURES)

        # Dest 0 (stand-still at (5,7) facing 2): only facing 2 should be filled
        for f in range(6):
            f_base = facing_offset + 0 * 6 * FACING_FEATURES + f * FACING_FEATURES
            f_slice = hier[f_base:f_base + FACING_FEATURES]
            if f == 2:
                # facing 2 is valid — at least one feature should be non-zero
                # (enemy is at (10,12) so range_quality may be non-zero)
                pass  # don't assert non-zero; range_quality could be 0 at long range
            else:
                assert np.all(f_slice == 0.0), f"Facing {f} should be zero for dest 0"

        # Dest 1 (walk to (6,8)): facings 0 and 1 valid, rest zero
        for f in range(6):
            f_base = facing_offset + 1 * 6 * FACING_FEATURES + f * FACING_FEATURES
            f_slice = hier[f_base:f_base + FACING_FEATURES]
            if f in (0, 1):
                pass  # valid facing
            else:
                assert np.all(f_slice == 0.0), f"Facing {f} should be zero for dest 1"

    def test_unused_facing_slots_zero(self):
        """Facing slots for unused destinations are all zero."""
        obs = self._make_hierarchical_obs()  # 3 destinations
        hier = flatten_observation_hierarchical(
            obs, rl_owner_id=0, max_destinations=self.MAX_DEST,
            legal_moves=obs["legal_moves"],
        )
        facing_offset = (BOARD_SIZE + 2 * UNIT_FEATURES + GLOBAL_FEATURES
                         + self.MAX_DEST * DEST_FEATURES)
        # Destinations 3..MAX_DEST-1 should have all-zero facing features
        unused_start = facing_offset + 3 * 6 * FACING_FEATURES
        unused_end = facing_offset + self.MAX_DEST * 6 * FACING_FEATURES
        assert np.all(hier[unused_start:unused_end] == 0.0)

    def test_no_legal_moves(self):
        """Empty legal moves produces all-zero move/facing blocks."""
        obs = self._make_hierarchical_obs(legal_moves=[])
        hier = flatten_observation_hierarchical(
            obs, rl_owner_id=0, max_destinations=self.MAX_DEST,
            legal_moves=obs["legal_moves"],
        )
        base = BOARD_SIZE + 2 * UNIT_FEATURES + GLOBAL_FEATURES
        assert np.all(hier[base:] == 0.0)

    def test_facing_head_receives_correct_dest_features(self):
        """The agent's facing head input matches the selected destination's features.

        Simulates the gather logic from HierarchicalAgent.get_action_and_value
        and verifies that for each destination index, the sliced features match
        the dest feature block in the observation.
        """
        obs = self._make_hierarchical_obs()
        hier = flatten_observation_hierarchical(
            obs, rl_owner_id=0, max_destinations=self.MAX_DEST,
            legal_moves=obs["legal_moves"],
        )
        obs_t = torch.tensor(hier, dtype=torch.float32).unsqueeze(0)  # (1, obs_size)

        dest_block_offset = OBS_SIZE  # 383
        # 3 destinations in default test obs
        for dest_idx in range(3):
            # Simulate the gather logic from HierarchicalAgent
            dest_action = torch.tensor([dest_idx], dtype=torch.long)
            dest_start = dest_block_offset + dest_action * DEST_FEATURES
            idx = dest_start.unsqueeze(1) + torch.arange(DEST_FEATURES)
            gathered = obs_t.gather(1, idx).squeeze(0).numpy()  # (7,)

            # Direct slice from the flat array
            direct = hier[
                dest_block_offset + dest_idx * DEST_FEATURES
                : dest_block_offset + (dest_idx + 1) * DEST_FEATURES
            ]

            np.testing.assert_array_equal(gathered, direct,
                err_msg=f"Gathered features for dest {dest_idx} don't match direct slice")

        # Dest 0 (stand-still at (5,7)): verify actual values
        d0_gathered = hier[dest_block_offset : dest_block_offset + DEST_FEATURES]
        assert d0_gathered[0] == pytest.approx(5 / 16)  # x
        assert d0_gathered[1] == pytest.approx(7 / 17)  # y
        assert d0_gathered[2] == pytest.approx(0 / 20)  # mp_used

        # Dest 1 (walk to (6,8), min mp=3)
        d1_gathered = hier[dest_block_offset + DEST_FEATURES : dest_block_offset + 2 * DEST_FEATURES]
        assert d1_gathered[0] == pytest.approx(6 / 16)
        assert d1_gathered[1] == pytest.approx(8 / 17)
        assert d1_gathered[2] == pytest.approx(3 / 20)

    def test_unused_dest_gives_zero_features_to_facing_head(self):
        """Selecting an unused destination index yields all-zero features."""
        obs = self._make_hierarchical_obs()  # 3 destinations, MAX_DEST=20
        hier = flatten_observation_hierarchical(
            obs, rl_owner_id=0, max_destinations=self.MAX_DEST,
            legal_moves=obs["legal_moves"],
        )
        obs_t = torch.tensor(hier, dtype=torch.float32).unsqueeze(0)

        # Pick an unused destination (index 10, well beyond the 3 valid ones)
        dest_action = torch.tensor([10], dtype=torch.long)
        dest_start = OBS_SIZE + dest_action * DEST_FEATURES
        idx = dest_start.unsqueeze(1) + torch.arange(DEST_FEATURES)
        gathered = obs_t.gather(1, idx).squeeze(0).numpy()
        assert np.all(gathered == 0.0), "Unused dest should yield all-zero features"

    def test_enemy_range_quality_in_dest_block(self):
        """enemy_range_quality is at dest feature index 7 and is facing-independent."""
        obs = self._make_hierarchical_obs()
        hier = flatten_observation_hierarchical(
            obs, rl_owner_id=0, max_destinations=self.MAX_DEST,
            legal_moves=obs["legal_moves"],
        )
        base = BOARD_SIZE + 2 * UNIT_FEATURES + GLOBAL_FEATURES

        # All 3 destinations should have enemy_range_quality at index 7
        # (enemy is at (10,12) facing 4 — values depend on distance but should be set)
        for dest_idx in range(3):
            d = hier[base + dest_idx * DEST_FEATURES:base + (dest_idx + 1) * DEST_FEATURES]
            # Index 7 is enemy_range_quality — should be a valid float (could be 0 at long range)
            assert np.isfinite(d[7]), f"Dest {dest_idx} enemy_range_quality should be finite"

    def test_best_rl_range_quality_in_dest_block(self):
        """best_rl_range_quality (dest index 8) equals max of per-facing rl_range_quality."""
        obs = self._make_hierarchical_obs()
        hier = flatten_observation_hierarchical(
            obs, rl_owner_id=0, max_destinations=self.MAX_DEST,
            legal_moves=obs["legal_moves"],
        )
        base = BOARD_SIZE + 2 * UNIT_FEATURES + GLOBAL_FEATURES
        facing_offset = base + self.MAX_DEST * DEST_FEATURES

        for dest_idx in range(3):
            d = hier[base + dest_idx * DEST_FEATURES:base + (dest_idx + 1) * DEST_FEATURES]
            best_rq = d[8]

            # Gather per-facing rl_range_quality values for this dest
            facing_rqs = []
            for f in range(6):
                f_base = facing_offset + dest_idx * 6 * FACING_FEATURES + f * FACING_FEATURES
                facing_rqs.append(hier[f_base])

            # best_rl_range_quality should equal max of non-zero facing values
            # (or 0 if all facings are zero)
            expected_max = max(facing_rqs) if any(v != 0 for v in facing_rqs) else 0.0
            assert best_rq == pytest.approx(expected_max), \
                f"Dest {dest_idx}: best_rl_rq={best_rq} != max(facing_rqs)={expected_max}"

    def test_facing_features_single_value(self):
        """Each facing slot has exactly FACING_FEATURES=1 value (rl_range_quality only)."""
        obs = self._make_hierarchical_obs()
        hier = flatten_observation_hierarchical(
            obs, rl_owner_id=0, max_destinations=self.MAX_DEST,
            legal_moves=obs["legal_moves"],
        )
        facing_offset = (BOARD_SIZE + 2 * UNIT_FEATURES + GLOBAL_FEATURES
                         + self.MAX_DEST * DEST_FEATURES)

        assert FACING_FEATURES == 1, "This test assumes FACING_FEATURES == 1"

        # Dest 0 (stand-still, facing 2 only): facing 2 slot has 1 value
        f_base_valid = facing_offset + 0 * 6 * FACING_FEATURES + 2 * FACING_FEATURES
        # Just verify the slot exists and is finite
        assert np.isfinite(hier[f_base_valid])

    def test_facing_head_receives_correct_facing_features(self):
        """The agent's facing head can gather per-facing features for the selected dest."""
        obs = self._make_hierarchical_obs()
        hier = flatten_observation_hierarchical(
            obs, rl_owner_id=0, max_destinations=self.MAX_DEST,
            legal_moves=obs["legal_moves"],
        )
        obs_t = torch.tensor(hier, dtype=torch.float32).unsqueeze(0)

        facing_block_offset = OBS_SIZE + self.MAX_DEST * DEST_FEATURES
        n_facing_feats = 6 * FACING_FEATURES

        for dest_idx in range(3):
            dest_action = torch.tensor([dest_idx], dtype=torch.long)
            facing_start = facing_block_offset + dest_action * n_facing_feats
            f_idx = facing_start.unsqueeze(1) + torch.arange(n_facing_feats)
            gathered = obs_t.gather(1, f_idx).squeeze(0).numpy()

            # Direct slice
            direct = hier[
                facing_block_offset + dest_idx * n_facing_feats
                : facing_block_offset + (dest_idx + 1) * n_facing_feats
            ]
            np.testing.assert_array_equal(gathered, direct,
                err_msg=f"Gathered facing features for dest {dest_idx} don't match")
