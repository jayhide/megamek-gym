"""Tests for MegaMekConfig."""

import tempfile
from pathlib import Path

import pytest

from megamek_gym.config import MegaMekConfig, parse_board_dimensions


class TestParseBoardDimensions:
    def test_standard_board(self):
        assert parse_board_dimensions("Map Set 6/16x17 BattleForce 2") == (16, 17)

    def test_simple_dimensions(self):
        assert parse_board_dimensions("32x32 Arena") == (32, 32)

    def test_no_dimensions(self):
        assert parse_board_dimensions("Random Board Name") is None

    def test_dimensions_only(self):
        assert parse_board_dimensions("20x24") == (20, 24)


class TestMegaMekConfig:
    def test_defaults(self):
        cfg = MegaMekConfig()
        assert cfg.rl_unit == "Commando COM-2D"
        assert cfg.resolved_board_width == 16
        assert cfg.resolved_board_height == 17

    def test_auto_derive_dimensions(self):
        cfg = MegaMekConfig(board="Map Set 2/32x34 Something")
        assert cfg.board_width is None
        assert cfg.resolved_board_width == 32
        assert cfg.resolved_board_height == 34

    def test_explicit_dimensions_override(self):
        cfg = MegaMekConfig(board="Map Set 6/16x17 BattleForce 2",
                            board_width=20, board_height=24)
        assert cfg.resolved_board_width == 20
        assert cfg.resolved_board_height == 24

    def test_unparseable_board_no_dims_raises(self):
        with pytest.raises(ValueError, match="Cannot derive board width"):
            MegaMekConfig(board="Custom Board Without Dims")

    def test_unparseable_board_with_explicit_dims(self):
        cfg = MegaMekConfig(board="Custom Board", board_width=10, board_height=12)
        assert cfg.resolved_board_width == 10
        assert cfg.resolved_board_height == 12

    def test_yaml_roundtrip(self):
        cfg = MegaMekConfig(
            rl_unit="Locust LCT-1V",
            opponent_unit="Hunchback HBK-4G",
            rl_port=10000,
        )
        with tempfile.NamedTemporaryFile(suffix=".yaml", mode="w", delete=False) as f:
            path = f.name
        cfg.save(path)
        loaded = MegaMekConfig.load(path)
        assert loaded.rl_unit == "Locust LCT-1V"
        assert loaded.opponent_unit == "Hunchback HBK-4G"
        assert loaded.rl_port == 10000
        assert loaded.board == cfg.board
        Path(path).unlink()

    def test_yaml_omits_none_board_dims(self):
        cfg = MegaMekConfig()
        with tempfile.NamedTemporaryFile(suffix=".yaml", mode="w", delete=False) as f:
            path = f.name
        cfg.save(path)
        content = Path(path).read_text()
        assert "board_width" not in content
        assert "board_height" not in content
        Path(path).unlink()

    def test_yaml_includes_explicit_board_dims(self):
        cfg = MegaMekConfig(board_width=20, board_height=24)
        with tempfile.NamedTemporaryFile(suffix=".yaml", mode="w", delete=False) as f:
            path = f.name
        cfg.save(path)
        content = Path(path).read_text()
        assert "board_width: 20" in content
        assert "board_height: 24" in content
        Path(path).unlink()

    def test_opponent_type_default(self):
        cfg = MegaMekConfig()
        assert cfg.opponent_type == "princess"

    def test_opponent_type_yaml_roundtrip(self):
        cfg = MegaMekConfig(opponent_type="rl")
        with tempfile.NamedTemporaryFile(suffix=".yaml", mode="w", delete=False) as f:
            path = f.name
        cfg.save(path)
        loaded = MegaMekConfig.load(path)
        assert loaded.opponent_type == "rl"
        Path(path).unlink()

    def test_fixed_coords_default_none(self):
        cfg = MegaMekConfig()
        assert cfg.rl_fixed_coords is None
        assert cfg.opponent_fixed_coords is None

    def test_fixed_coords_tuple(self):
        cfg = MegaMekConfig(rl_fixed_coords=(8, 2), opponent_fixed_coords=(8, 14))
        assert cfg.rl_fixed_coords == (8, 2)
        assert cfg.opponent_fixed_coords == (8, 14)

    def test_fixed_coords_list_converted_to_tuple(self):
        """YAML deserializes [8, 2] as a list; __post_init__ converts to tuple."""
        cfg = MegaMekConfig(rl_fixed_coords=[8, 2], opponent_fixed_coords=[8, 14])
        assert cfg.rl_fixed_coords == (8, 2)
        assert isinstance(cfg.rl_fixed_coords, tuple)
        assert cfg.opponent_fixed_coords == (8, 14)
        assert isinstance(cfg.opponent_fixed_coords, tuple)

    def test_fixed_coords_yaml_roundtrip(self):
        cfg = MegaMekConfig(rl_fixed_coords=(8, 2), opponent_fixed_coords=(8, 14))
        with tempfile.NamedTemporaryFile(suffix=".yaml", mode="w", delete=False) as f:
            path = f.name
        cfg.save(path)
        loaded = MegaMekConfig.load(path)
        assert loaded.rl_fixed_coords == (8, 2)
        assert loaded.opponent_fixed_coords == (8, 14)
        Path(path).unlink()

    def test_fixed_coords_none_omitted_from_yaml(self):
        cfg = MegaMekConfig()
        with tempfile.NamedTemporaryFile(suffix=".yaml", mode="w", delete=False) as f:
            path = f.name
        cfg.save(path)
        content = Path(path).read_text()
        assert "rl_fixed_coords" not in content
        assert "opponent_fixed_coords" not in content
        Path(path).unlink()
