"""Tests for engine and sensor damage rules in the Python sim."""

import random

import pytest

from megamek_gym.sim.board import BOARD
from megamek_gym.sim.firing import compute_to_hit, resolve_firing
from megamek_gym.sim.heat import apply_heat
from megamek_gym.sim.los import LosTable
from megamek_gym.sim.unit import UNIT_TEMPLATES, Unit


@pytest.fixture
def los_table():
    return LosTable(BOARD)


def _make_unit(entity_id=1, owner=0, x=8, y=8, facing=0):
    tmpl = UNIT_TEMPLATES["Trebuchet TBT-5S"]
    u = Unit(template=tmpl, entity_id=entity_id, owner=owner)
    u.x, u.y = x, y
    u.facing = facing
    return u


class TestEngineHeat:
    """Engine crits add +5 heat per damaged slot per turn."""

    def test_zero_engine_hits_no_extra_heat(self):
        u = _make_unit()
        u.engine_hits = 0
        u.heat = 0
        apply_heat(u, 0)
        assert u.heat == 0

    def test_one_engine_hit_adds_5_heat(self):
        u = _make_unit()
        u.engine_hits = 1
        u.heat = 0
        apply_heat(u, 0)
        assert u.heat == 5

    def test_two_engine_hits_add_10_heat(self):
        u = _make_unit()
        u.engine_hits = 2
        u.heat = 0
        apply_heat(u, 0)
        assert u.heat == 10

    def test_engine_heat_stacks_with_weapon_heat(self):
        u = _make_unit()
        u.engine_hits = 1
        u.heat = 0
        apply_heat(u, 3)  # 3 weapon heat + 5 engine = 8
        assert u.heat == 8

    def test_engine_heat_stacks_with_running(self):
        u = _make_unit()
        u.engine_hits = 1
        u.movement_type = "run"
        u.heat = 0
        apply_heat(u, 0)  # 0 weapon + 2 run + 5 engine = 7
        assert u.heat == 7


class TestSensorToHit:
    """Sensor damage adds +2 per hit to to-hit rolls."""

    def test_one_sensor_hit_plus_2(self, los_table):
        attacker = _make_unit(entity_id=1, x=8, y=8, facing=0)
        target = _make_unit(entity_id=2, owner=1, x=8, y=6, facing=3)
        weapon = attacker.template.weapons[0]  # Medium Laser

        attacker.sensor_hits = 0
        base_tn = compute_to_hit(attacker, target, weapon, BOARD, los_table)

        attacker.sensor_hits = 1
        hit_tn = compute_to_hit(attacker, target, weapon, BOARD, los_table)

        assert base_tn is not None
        assert hit_tn is not None
        assert hit_tn == base_tn + 2

    def test_two_sensor_hits_plus_4(self, los_table):
        attacker = _make_unit(entity_id=1, x=8, y=8, facing=0)
        target = _make_unit(entity_id=2, owner=1, x=8, y=6, facing=3)
        weapon = attacker.template.weapons[0]

        attacker.sensor_hits = 0
        base_tn = compute_to_hit(attacker, target, weapon, BOARD, los_table)

        attacker.sensor_hits = 2
        hit_tn = compute_to_hit(attacker, target, weapon, BOARD, los_table)

        assert base_tn is not None
        assert hit_tn is not None
        assert hit_tn == base_tn + 4


class TestSensorsDestroyedCannotFire:
    """2 sensor hits = sensors destroyed = cannot fire any weapons."""

    def test_sensors_destroyed_cannot_fire(self, los_table):
        attacker = _make_unit(entity_id=1, x=8, y=8, facing=0)
        target = _make_unit(entity_id=2, owner=1, x=8, y=6, facing=3)
        attacker.sensor_hits = 2

        result = resolve_firing(attacker, target, BOARD, los_table,
                                rng=random.Random(42))
        assert result["total_damage"] == 0
        assert result["heat_generated"] == 0
        assert result["hits"] == []

    def test_one_sensor_hit_can_still_fire(self, los_table):
        attacker = _make_unit(entity_id=1, x=8, y=8, facing=0)
        target = _make_unit(entity_id=2, owner=1, x=8, y=6, facing=3)
        attacker.sensor_hits = 1

        # With 1 sensor hit, firing should still be attempted.
        # Use a fixed seed — even if all shots miss, heat_generated > 0
        # proves weapons were fired (heat is generated on attempt).
        result = resolve_firing(attacker, target, BOARD, los_table,
                                rng=random.Random(42))
        assert result["heat_generated"] > 0
