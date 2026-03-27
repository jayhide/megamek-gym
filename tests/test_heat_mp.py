"""Tests for heat-based MP reduction in the Python sim."""

import pytest

from megamek_gym.sim.unit import UNIT_TEMPLATES, Unit, Location
from megamek_gym.sim.board import BOARD
from megamek_gym.sim.movement import enumerate_moves


def _make_unit(heat=0) -> Unit:
    """Create a TBT-5S unit with given heat level."""
    tmpl = UNIT_TEMPLATES["Trebuchet TBT-5S"]
    unit = Unit(template=tmpl, entity_id=0, owner=0)
    unit.x, unit.y, unit.facing = 8, 8, 0
    unit.heat = heat
    return unit


class TestHeatMPReduction:
    """Heat penalty: walk_mp -= heat // 5 (matches Java Entity.getHeatMPReduction)."""

    def test_no_heat(self):
        unit = _make_unit()
        assert unit.walk_mp == 5  # TBT-5S base

    def test_heat_4_no_penalty(self):
        unit = _make_unit(heat=4)
        assert unit.walk_mp == 5  # 4 // 5 = 0

    def test_heat_5_minus_1(self):
        unit = _make_unit(heat=5)
        assert unit.walk_mp == 4

    def test_heat_10_minus_2(self):
        unit = _make_unit(heat=10)
        assert unit.walk_mp == 3

    def test_heat_15_minus_3(self):
        unit = _make_unit(heat=15)
        assert unit.walk_mp == 2

    def test_heat_20_minus_4(self):
        unit = _make_unit(heat=20)
        assert unit.walk_mp == 1

    def test_heat_24_minus_4(self):
        unit = _make_unit(heat=24)
        assert unit.walk_mp == 1

    def test_heat_25_zeroes_mp(self):
        unit = _make_unit(heat=25)
        assert unit.walk_mp == 0  # 25 // 5 = 5 = full base MP

    def test_heat_30_clamped(self):
        unit = _make_unit(heat=30)
        assert unit.walk_mp == 0  # 30 // 5 = 6, clamped to 0

    def test_run_mp_derives_from_walk(self):
        unit = _make_unit(heat=5)
        assert unit.walk_mp == 4
        assert unit.run_mp == 6  # ceil(4 * 1.5) = 6

    def test_heat_plus_leg_damage_stacks(self):
        unit = _make_unit(heat=5)
        unit.leg_actuator_hits[0] = 1  # -1 from actuator + -1 from heat
        assert unit.walk_mp == 3

    def test_heat_plus_leg_damage_clamps_to_zero(self):
        unit = _make_unit(heat=20)
        unit.leg_actuator_hits[0] = 2
        # base 5 - 2 actuator - 4 heat = -1, clamped to 0
        assert unit.walk_mp == 0

    def test_shutdown_unit_no_moves(self):
        unit = _make_unit()
        unit.shutdown = True
        moves = enumerate_moves(unit, BOARD)
        assert moves == []
