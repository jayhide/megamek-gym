"""Debug script: compare Python princess scoring against Java debug log values."""
from __future__ import annotations

import re
import sys

from megamek_gym.reward import hex_bearing, hex_distance
from megamek_gym.sim.board import BOARD
from megamek_gym.sim.los import LosTable
from megamek_gym.sim.princess import (
    _approx_closest_range,
    _facing_diff,
    _is_in_forward_arc,
    _kick_damage,
    _max_damage_at_range,
    _sum_weapon_damage,
    score_move,
    _BRAVERY,
    _FALL_SHAME,
    _HYPER_AGGRESSION,
    _FACING_MOD_MULTIPLIER,
    _HERD_MENTALITY,
    _KICK_DAMAGE,
    _KICK_HIT_PROB,
    _QUICK_DAMAGE_DISCOUNT,
    _UNMOVED_DAMAGE_DISCOUNT,
)
from megamek_gym.sim.unit import TBT_5S, Unit


def parse_java_log(line: str) -> dict:
    """Parse a [rl-princess-debug] log line into a dict of values."""
    # Extract key=value pairs
    result = {}
    # Remove timestamp prefix
    m = re.search(r'\[rl-princess-debug\]\s*(.*)', line)
    if not m:
        return result
    rest = m.group(1)

    # Parse fields
    patterns = {
        'dest': r'dest=\((\d+),(\d+)\)',
        'facing': r'facing=(\d+)',
        'rank': r'rank=([-\d.]+)',
        'successProb': r'successProb=([\d.]+)',
        'fallMod': r'fallMod=([-\d.]+)',
        'myFiring': r'myFiring=([-\d.]+)',
        'myPhysical': r'myPhysical=([-\d.]+)',
        'damageExpTotal': r'damageExpTotal=([-\d.]+)',
        'dmgExpPath': r'dmgExpPath=([-\d.]+)',
        'braveryMod': r'braveryMod=([-\d.]+)',
        'closestEnemyDist': r'closestEnemyDist=([-\d.]+)',
        'aggressionMod': r'aggressionMod=([-\d.]+)',
        'facingDiff': r'facingDiff=([-\d.]+)',
        'facingMod': r'facingMod=([-\d.]+)',
        'herdingMod': r'herdingMod=([-\d.]+)',
        'movementMod': r'movementMod=([-\d.]+)',
        'friendsX': r'friendsX=([-\d.]+)',
        'friendsY': r'friendsY=([-\d.]+)',
        'friendsDist': r'friendsDist=([-\d.]+)',
        'evalPath': r'evalPath=(\w+)',
        'enemyMyDmg': r'enemyMyDmg=([-\d.]+)',
        'enemyDmgTaken': r'enemyDmgTaken=([-\d.]+)',
        'enemyPhysDmg': r'enemyPhysDmg=([-\d.]+)',
        'enemyPos': r'enemyPos=\((\d+),(\d+)\)',
        'enemyFacing': r'enemyFacing=(\d+)',
        'closestRange': r'closestRange=([-\d.]+)',
    }

    for key, pat in patterns.items():
        m2 = re.search(pat, rest)
        if m2:
            if key == 'dest':
                result['dest_x'] = int(m2.group(1))
                result['dest_y'] = int(m2.group(2))
            elif key == 'enemyPos':
                result['enemy_x'] = int(m2.group(1))
                result['enemy_y'] = int(m2.group(2))
            elif key == 'evalPath':
                result[key] = m2.group(1)
            else:
                result[key] = float(m2.group(1))
    return result


def compute_python_scores(java: dict, los_table: LosTable) -> dict:
    """Compute Python's scoring components for the same destination Java chose."""
    dest_x = java['dest_x']
    dest_y = java['dest_y']
    facing = int(java['facing'])
    ex = java['enemy_x']
    ey = java['enemy_y']
    enemy_facing = int(java['enemyFacing'])
    enemy_has_moved = java['evalPath'] == 'moved'

    # Build unit obs dicts from template
    unit = Unit(TBT_5S, owner=1, entity_id=1)
    unit.x, unit.y, unit.facing = dest_x, dest_y, facing
    unit_obs = unit.to_obs_dict()

    enemy = Unit(TBT_5S, owner=0, entity_id=0)
    enemy.x, enemy.y, enemy.facing = ex, ey, enemy_facing
    enemy_obs = enemy.to_obs_dict()

    board = BOARD
    dist_to_enemy = hex_distance(dest_x, dest_y, ex, ey)
    success_prob = java.get('successProb', 1.0)

    # Fall mod
    if success_prob <= 0.0:
        fall_mod = 1000.0
    else:
        fall_mod = (1.0 - success_prob) * _FALL_SHAME

    if enemy_has_moved:
        discount = _QUICK_DAMAGE_DISCOUNT
        my_firing_raw = _max_damage_at_range(unit_obs, dest_x, dest_y, facing, ex, ey, board, los_table)
        my_firing_damage = my_firing_raw * discount
        my_physical = _kick_damage() if dist_to_enemy <= 1 else 0.0
        max_damage_done = my_firing_damage + my_physical
        enemy_firing_raw = _max_damage_at_range(enemy_obs, ex, ey, enemy_facing, dest_x, dest_y, board, los_table)
        enemy_damage = enemy_firing_raw * discount
        expected_damage_taken = enemy_damage
        if dist_to_enemy <= 1:
            expected_damage_taken += _kick_damage() * _KICK_HIT_PROB
        approx_range = int(dist_to_enemy)
        in_arc = True  # N/A for moved path
    else:
        discount = _UNMOVED_DAMAGE_DISCOUNT
        enemy_run_mp = enemy_obs.get("mp_run", 0)
        approx_range = _approx_closest_range(dest_x, dest_y, ex, ey, enemy_run_mp)
        in_arc = _is_in_forward_arc(dest_x, dest_y, facing, ex, ey)
        my_firing_raw = _sum_weapon_damage(unit_obs, approx_range) if in_arc else 0.0
        my_firing_damage = my_firing_raw * discount
        max_damage_done = my_firing_damage
        my_physical = 0.0
        enemy_firing_raw = _sum_weapon_damage(enemy_obs, approx_range)
        enemy_damage = enemy_firing_raw * discount
        expected_damage_taken = enemy_damage

    bravery_mod = success_prob * max_damage_done * _BRAVERY - expected_damage_taken

    d = dist_to_enemy
    if d == 0:
        d = 2
    aggression_mod = d * _HYPER_AGGRESSION

    # Herding
    fx = java.get('friendsX', -1)
    fy = java.get('friendsY', -1)
    if fx >= 0 and fy >= 0:
        herding_mod = hex_distance(dest_x, dest_y, int(fx), int(fy)) * _HERD_MENTALITY
    else:
        herding_mod = 0.0

    facing_mod = _FACING_MOD_MULTIPLIER * _facing_diff(dest_x, dest_y, facing, ex, ey, unit_obs)

    utility = -fall_mod + bravery_mod - aggression_mod - herding_mod - facing_mod

    return {
        'dist_to_enemy': dist_to_enemy,
        'approx_range': approx_range,
        'in_arc': in_arc,
        'discount': discount,
        'my_firing_raw': my_firing_raw,
        'my_firing': my_firing_damage,
        'my_physical': my_physical,
        'max_damage_done': max_damage_done,
        'enemy_firing_raw': enemy_firing_raw,
        'enemy_damage': enemy_damage,
        'expected_damage_taken': expected_damage_taken,
        'fall_mod': fall_mod,
        'bravery_mod': bravery_mod,
        'aggression_mod': aggression_mod,
        'herding_mod': herding_mod,
        'facing_mod': facing_mod,
        'utility': utility,
    }


def main():
    log_path = sys.argv[1] if len(sys.argv) > 1 else "../megamek/rl_java_9999.log"

    # Read Java log
    with open(log_path) as f:
        lines = [l for l in f if 'rl-princess-debug' in l]

    if not lines:
        print(f"No [rl-princess-debug] lines in {log_path}")
        return

    los_table = LosTable(BOARD)

    for i, line in enumerate(lines):
        java = parse_java_log(line)
        if not java:
            continue

        py = compute_python_scores(java, los_table)

        print(f"\n{'='*70}")
        print(f"Round {i+1}: dest=({java['dest_x']},{java['dest_y']}) facing={int(java['facing'])} "
              f"enemy=({java['enemy_x']},{java['enemy_y']}) evalPath={java['evalPath']}")
        print(f"{'='*70}")

        # Component comparison
        fields = [
            ('closestRange (Java) vs approx_range (Py)',
             f"Java={int(java.get('closestRange', -1))}  Python={py['approx_range']}"),
            ('in_forward_arc', f"Python={py['in_arc']}"),
            ('dist_to_enemy', f"Java={java.get('closestEnemyDist', -1):.1f}  Python={py['dist_to_enemy']:.1f}"),
            ('myFiring', f"Java={java.get('myFiring', 0):.2f}  Python={py['my_firing']:.2f}  "
                         f"(raw: Java={java.get('myFiring', 0) / py['discount']:.1f}  "
                         f"Python={py['my_firing_raw']:.1f})"),
            ('myPhysical', f"Java={java.get('myPhysical', 0):.2f}  Python={py['my_physical']:.2f}"),
            ('damageExpTotal', f"Java={java.get('damageExpTotal', 0):.2f}  Python={py['expected_damage_taken']:.2f}"),
            ('braveryMod', f"Java={java.get('braveryMod', 0):.2f}  Python={py['bravery_mod']:.2f}"),
            ('aggressionMod', f"Java={java.get('aggressionMod', 0):.2f}  Python={py['aggression_mod']:.2f}"),
            ('herdingMod', f"Java={java.get('herdingMod', 0):.2f}  Python={py['herding_mod']:.2f}"),
            ('facingMod', f"Java={java.get('facingMod', 0):.2f}  Python={py['facing_mod']:.2f}"),
            ('utility', f"Java={java.get('rank', 0):.2f}  Python={py['utility']:.2f}"),
        ]

        for name, val in fields:
            # Highlight differences
            print(f"  {name}: {val}")


if __name__ == "__main__":
    main()
