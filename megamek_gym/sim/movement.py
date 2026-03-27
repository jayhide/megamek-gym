"""Movement enumeration via BFS, replicating MegaMek's LongestPathFinder.

Produces legal moves matching the Java bridge's JSON format so that
observation.py can consume them directly.

BFS state is (x, y, facing) — facing changes cost 1 MP each (standard
for ground mechs), matching Java's MoveStep TURN_LEFT/TURN_RIGHT costs.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

from megamek_gym.sim.board import Board, Terrain, WIDTH, HEIGHT, neighbor

if TYPE_CHECKING:
    from megamek_gym.sim.unit import Unit


def _hex_mp_cost(board: Board, from_x: int, from_y: int,
                 to_x: int, to_y: int) -> int:
    """Total MP cost to enter a hex from an adjacent hex.

    Returns -1 if the hex is impassable (water).
    """
    t = board.terrain(to_x, to_y)

    if t == Terrain.WATER:
        return -1

    cost = 1  # Base
    if t == Terrain.LIGHT_WOODS:
        cost += 1
    elif t == Terrain.HEAVY_WOODS:
        cost += 2
    elif t == Terrain.ROUGH:
        cost += 1

    diff = abs(board.elevation(to_x, to_y) - board.elevation(from_x, from_y))
    cost += diff
    return cost


def _psr_success_prob(target: int) -> float:
    """Probability of rolling >= target on 2d6."""
    if target <= 2:
        return 1.0
    if target > 12:
        return 0.0
    successes = 0
    for d1 in range(1, 7):
        for d2 in range(1, 7):
            if d1 + d2 >= target:
                successes += 1
    return successes / 36.0


# Pre-compute PSR probabilities
_PSR_PROBS = {t: _psr_success_prob(t) for t in range(2, 14)}


def enumerate_moves(unit: Unit, board: Board,
                    enemy: Unit | None = None,
                    max_moves: int = 500) -> list[dict]:
    """Enumerate legal moves for a unit."""
    if unit.destroyed or unit.shutdown or not unit.deployed:
        return []

    if unit.prone:
        moves = _enumerate_prone_moves(unit, board)
    else:
        moves = _enumerate_standing_moves(unit, board)

    if len(moves) > max_moves:
        moves = moves[:max_moves]

    for i, m in enumerate(moves):
        m["index"] = i

    return moves


def _is_better(new_mp: int, new_hm: int, new_psr: float,
               e_mp: int, e_hm: int, e_psr: float) -> bool:
    """Check if a new path is better than an existing one.

    Prefers less MP used, then more hexes moved, then higher PSR.
    Matches Java's MovePathMinMPMaxDistanceComparator.
    """
    if new_mp < e_mp:
        return True
    if new_mp == e_mp and new_hm > e_hm:
        return True
    if new_mp == e_mp and new_hm == e_hm and new_psr > e_psr:
        return True
    return False


def _facing_bfs(board: Board, start_x: int, start_y: int, start_facing: int,
                walk_mp: int, run_mp: int, piloting: int,
                free_turns: bool = False,
                ) -> tuple[dict, dict]:
    """Run facing-aware BFS to find reachable (x, y, facing) states.

    Returns (best_walk, best_run) dicts mapping
    (x, y, facing) -> (mp_used, hexes_moved, psr_prob).

    Transitions:
    - FORWARD: move to hex in current facing direction (terrain MP cost)
    - TURN_LEFT/RIGHT: same hex, facing ±1 (1 MP each, or 0 if free_turns)

    free_turns: True for units that just stood up from prone (hasJustStood).
    """
    turn_cost = 0 if free_turns else 1

    # State: (x, y, facing, mp_used, hexes_moved, psr_prob)
    best_walk: dict[tuple[int, int, int], tuple[int, int, float]] = {}
    best_run: dict[tuple[int, int, int], tuple[int, int, float]] = {}

    start_key = (start_x, start_y, start_facing)
    best_walk[start_key] = (0, 0, 1.0)

    queue: list[tuple[int, int, int, int, int, float]] = [
        (start_x, start_y, start_facing, 0, 0, 1.0)
    ]
    qi = 0

    while qi < len(queue):
        cx, cy, cf, mp, hm, psr = queue[qi]
        qi += 1

        # --- TURN transitions ---
        for new_facing in ((cf + 1) % 6, (cf + 5) % 6):
            new_mp = mp + turn_cost
            if new_mp > run_mp:
                continue

            key = (cx, cy, new_facing)
            is_run = new_mp > walk_mp
            best = best_run if is_run else best_walk
            existing = best.get(key)

            should_add = False
            if existing is None:
                should_add = True
            else:
                should_add = _is_better(new_mp, hm, psr, *existing)

            if should_add:
                best[key] = (new_mp, hm, psr)
                queue.append((cx, cy, new_facing, new_mp, hm, psr))

        # --- FORWARD transition (move in facing direction) ---
        nx, ny = neighbor(cx, cy, cf)
        if not (0 <= nx < WIDTH and 0 <= ny < HEIGHT):
            continue

        cost = _hex_mp_cost(board, cx, cy, nx, ny)
        if cost < 0:
            continue  # Impassable

        new_mp = mp + cost
        if new_mp > run_mp:
            continue

        # PSR checks
        new_psr = psr
        is_running = new_mp > walk_mp

        # Heavy woods + running = PSR
        if is_running and board.terrain(nx, ny) == Terrain.HEAVY_WOODS:
            new_psr *= _PSR_PROBS.get(piloting, 0.5)

        # Large elevation change = PSR
        elev_diff = abs(board.elevation(nx, ny) - board.elevation(cx, cy))
        if elev_diff >= 2:
            new_psr *= _PSR_PROBS.get(piloting, 0.5)

        if new_psr < 0.3:
            continue

        new_hm = hm + 1
        key = (nx, ny, cf)  # After FORWARD, facing stays the same
        is_run = new_mp > walk_mp
        best = best_run if is_run else best_walk
        existing = best.get(key)

        should_add = False
        if existing is None:
            should_add = True
        else:
            should_add = _is_better(new_mp, new_hm, new_psr, *existing)

        if should_add:
            best[key] = (new_mp, new_hm, new_psr)
            queue.append((nx, ny, cf, new_mp, new_hm, new_psr))

    return best_walk, best_run


def _enumerate_standing_moves(unit: Unit, board: Board) -> list[dict]:
    """Enumerate moves for a standing mech using facing-aware BFS."""
    walk_mp = unit.walk_mp
    run_mp = unit.run_mp

    if walk_mp <= 0:
        return _stand_still_moves(unit)

    best_walk, best_run = _facing_bfs(
        board, unit.x, unit.y, unit.facing,
        walk_mp, run_mp, unit.template.piloting,
    )

    # Build move dicts
    moves: list[dict] = []

    # Standing still at current position (all 6 facings, 0 MP)
    for f in range(6):
        moves.append({"dest_x": unit.x, "dest_y": unit.y, "facing": f, "mp_used": 0})

    # Collect all reachable (x, y, facing) with walk/run dedup
    # Group by (x, y, facing) to apply the run-only-if-more-hexes filter
    all_keys = set(best_walk.keys()) | set(best_run.keys())

    for key in sorted(all_keys):
        x, y, f = key
        if x == unit.x and y == unit.y:
            # Standing-still moves already added above; but facing changes
            # at the start hex with mp > 0 are valid moves too
            walk_data = best_walk.get(key)
            if walk_data and walk_data[0] > 0:
                moves.append({"dest_x": x, "dest_y": y, "facing": f, "mp_used": walk_data[0]})
            # Run data at start hex unlikely but handle it
            run_data = best_run.get(key)
            if run_data and run_data[0] > 0:
                walk_hm = walk_data[1] if walk_data else -1
                if run_data[1] > walk_hm:
                    moves.append({"dest_x": x, "dest_y": y, "facing": f, "mp_used": run_data[0]})
            continue

        walk_data = best_walk.get(key)
        run_data = best_run.get(key)

        # Skip run if it doesn't provide more hexes than walk
        if walk_data and run_data:
            if run_data[1] <= walk_data[1]:
                run_data = None

        if walk_data:
            moves.append({"dest_x": x, "dest_y": y, "facing": f, "mp_used": walk_data[0]})
        if run_data:
            moves.append({"dest_x": x, "dest_y": y, "facing": f, "mp_used": run_data[0]})

    return moves


def _enumerate_prone_moves(unit: Unit, board: Board) -> list[dict]:
    """Enumerate moves for a prone mech."""
    moves: list[dict] = []

    # Stay prone (stand still)
    for f in range(6):
        moves.append({"dest_x": unit.x, "dest_y": unit.y, "facing": f, "mp_used": 0})

    stand_cost = math.ceil(unit.walk_mp / 2)
    remaining_walk = unit.walk_mp - stand_cost

    if remaining_walk < 0:
        return moves

    if remaining_walk == 0:
        for f in range(6):
            moves.append({
                "dest_x": unit.x, "dest_y": unit.y, "facing": f,
                "mp_used": stand_cost,
            })
        return moves

    remaining_run = unit.run_mp - stand_cost

    best_walk, best_run = _facing_bfs(
        board, unit.x, unit.y, unit.facing,
        remaining_walk, max(remaining_walk, remaining_run),
        unit.template.piloting,
        free_turns=True,  # hasJustStood: turns are free after standing up
    )

    all_keys = set(best_walk.keys()) | set(best_run.keys())

    for key in sorted(all_keys):
        x, y, f = key
        walk_data = best_walk.get(key)
        run_data = best_run.get(key)

        if walk_data and run_data and run_data[1] <= walk_data[1]:
            run_data = None

        if walk_data:
            total_mp = walk_data[0] + stand_cost
            # Skip if this duplicates the stay-prone move (0 MP at start hex)
            if not (x == unit.x and y == unit.y and total_mp == 0):
                moves.append({
                    "dest_x": x, "dest_y": y, "facing": f,
                    "mp_used": total_mp,
                })
        if run_data:
            total_mp = run_data[0] + stand_cost
            moves.append({
                "dest_x": x, "dest_y": y, "facing": f,
                "mp_used": total_mp,
            })

    return moves


def _stand_still_moves(unit: Unit) -> list[dict]:
    return [
        {"dest_x": unit.x, "dest_y": unit.y, "facing": f, "mp_used": 0}
        for f in range(6)
    ]
