"""Tests for heat-based MP reduction and startup/shutdown in the Python sim."""

import pytest

from megamek_gym.sim.unit import UNIT_TEMPLATES, Unit, Location
from megamek_gym.sim.board import BOARD
from megamek_gym.sim.heat import attempt_startup, check_overheat
from megamek_gym.sim.movement import enumerate_moves


class _FixedRng:
    """Mock RNG that returns predetermined randint values in sequence."""

    def __init__(self, values: list[int]):
        self._values = list(values)
        self._idx = 0

    def randint(self, a: int, b: int) -> int:
        val = self._values[self._idx]
        self._idx += 1
        return val


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

    def test_shutdown_unit_stand_still_only(self):
        """Shutdown unit gets exactly 1 legal move: stand still at current pos."""
        unit = _make_unit()
        unit.shutdown = True
        moves = enumerate_moves(unit, BOARD)
        assert len(moves) == 1
        m = moves[0]
        assert m["dest_x"] == unit.x
        assert m["dest_y"] == unit.y
        assert m["facing"] == unit.facing
        assert m["mp_used"] == 0


class TestStartup:
    """Tests for attempt_startup() matching Java HeatResolver lines 568-645."""

    def test_not_shutdown_returns_none(self):
        unit = _make_unit(heat=20)
        assert attempt_startup(unit) is None
        assert not unit.shutdown

    def test_auto_restart_below_14(self):
        unit = _make_unit(heat=10)
        unit.shutdown = True
        event = attempt_startup(unit)
        assert not unit.shutdown
        assert event["type"] == "startup"
        assert event["auto"] is True

    def test_no_startup_at_heat_30(self):
        """Heat >= 30: auto-shutdown threshold, cannot attempt startup."""
        unit = _make_unit(heat=30)
        unit.shutdown = True
        event = attempt_startup(unit)
        assert event is None
        assert unit.shutdown

    def test_startup_roll_success(self):
        """Heat 14, TN=4: roll of 2+2=4 succeeds."""
        unit = _make_unit(heat=14)
        unit.shutdown = True
        event = attempt_startup(unit, _FixedRng([2, 2]))  # 2d6 = 4
        assert event["type"] == "startup"
        assert not unit.shutdown
        assert event["tn"] == 4
        assert event["roll"] == 4

    def test_startup_roll_failure(self):
        """Heat 14, TN=4: roll of 1+2=3 fails."""
        unit = _make_unit(heat=14)
        unit.shutdown = True
        event = attempt_startup(unit, _FixedRng([1, 2]))  # 2d6 = 3
        assert event["type"] == "startup_failed"
        assert unit.shutdown
        assert event["tn"] == 4
        assert event["roll"] == 3

    def test_startup_tn_formula(self):
        """Verify TN = 4 + floor((heat-14)/4) * 2 for each heat bracket."""
        expected_tns = {14: 4, 17: 4, 18: 6, 21: 6, 22: 8, 25: 8, 26: 10, 29: 10}
        # Roll high enough to always succeed, so we can read the TN from the event
        high_roll = _FixedRng([6, 6])
        for heat, expected_tn in expected_tns.items():
            unit = _make_unit(heat=heat)
            unit.shutdown = True
            high_roll._idx = 0  # reuse
            event = attempt_startup(unit, high_roll)
            assert event["tn"] == expected_tn, f"heat={heat}: expected TN {expected_tn}, got {event['tn']}"

    def test_startup_clears_shutdown(self):
        """Successful startup clears the shutdown flag."""
        unit = _make_unit(heat=18)
        unit.shutdown = True
        # TN=6, roll 6 → success
        event = attempt_startup(unit, _FixedRng([3, 3]))
        assert event["type"] == "startup"
        assert not unit.shutdown

    def test_startup_failure_preserves_shutdown(self):
        """Failed startup leaves shutdown flag set."""
        unit = _make_unit(heat=18)
        unit.shutdown = True
        # TN=6, roll 5 → failure
        event = attempt_startup(unit, _FixedRng([3, 2]))
        assert event["type"] == "startup_failed"
        assert unit.shutdown
