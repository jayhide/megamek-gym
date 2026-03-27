"""Movement enumeration via BFS, replicating MegaMek's LongestPathFinder.

Produces legal moves matching the Java bridge's JSON format so that
observation.py can consume them directly.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

from megamek_gym.sim.board import Board, Terrain, WIDTH, HEIGHT

if TYPE_CHECKING:
    from megamek_gym.sim.unit import Unit


# Pre-computed neighbor table for the entire board.
# _NEIGHBORS[y * WIDTH + x] = list of (nx, ny, cost_clear, cost_lw, cost_hw)
# We defer initialization until first use.
_NEIGHBORS: list[list[tuple[int, int]]] | None = None
_BOARD_REF: Board | None = None


def _init_neighbor_table(board: Board) -> None:
    """Pre-compute neighbor + MP cost data for the board."""
    global _NEIGHBORS, _BOARD_REF
    if _NEIGHBORS is not None and _BOARD_REF is board:
        return
    _BOARD_REF = board

    from megamek_gym.sim.board import neighbors_with_dir
    _NEIGHBORS = []
    for y in range(HEIGHT):
        for x in range(WIDTH):
            nbrs = []
            for nx, ny, _d in neighbors_with_dir(x, y):
                nbrs.append((nx, ny))
            _NEIGHBORS.append(nbrs)


def _hex_mp_cost_fast(board: Board, from_x: int, from_y: int,
                      to_x: int, to_y: int) -> int:
    """Total MP cost to enter a hex from an adjacent hex.

    Returns -1 if the hex is impassable (water).
    """
    t = board.terrain(to_x, to_y)

    # Water is impassable for ground mechs
    if t == Terrain.WATER:
        return -1

    cost = 1  # Base
    if t == Terrain.LIGHT_WOODS:
        cost += 1
    elif t == Terrain.HEAVY_WOODS:
        cost += 2
    elif t == Terrain.ROUGH:
        cost += 1

    diff = board.elevation(to_x, to_y) - board.elevation(from_x, from_y)
    if diff > 0:
        cost += diff
    elif diff < -1:
        cost += (-diff) - 1
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
    if unit.destroyed or not unit.deployed:
        return []

    _init_neighbor_table(board)

    if unit.prone:
        moves = _enumerate_prone_moves(unit, board)
    else:
        moves = _enumerate_standing_moves(unit, board)

    if len(moves) > max_moves:
        moves = moves[:max_moves]

    for i, m in enumerate(moves):
        m["index"] = i

    return moves


def _enumerate_standing_moves(unit: Unit, board: Board) -> list[dict]:
    """Enumerate moves for a standing mech using optimized BFS."""
    walk_mp = unit.walk_mp
    run_mp = unit.run_mp

    if walk_mp <= 0:
        return _stand_still_moves(unit)

    # BFS: find reachable hexes with (min_mp, max_hexes) per hex
    # For each hex, track best walk path and best run path separately
    # best_walk[idx] = (mp_used, hexes_moved, psr_prob) or None
    # best_run[idx] = (mp_used, hexes_moved, psr_prob) or None
    size = WIDTH * HEIGHT
    best_walk: list[tuple[int, int, float] | None] = [None] * size
    best_run: list[tuple[int, int, float] | None] = [None] * size

    start_idx = unit.y * WIDTH + unit.x
    piloting = unit.template.piloting

    # Queue entries: (x, y, mp_used, hexes_moved, psr_prob)
    queue: list[tuple[int, int, int, int, float]] = [
        (unit.x, unit.y, 0, 0, 1.0)
    ]
    best_walk[start_idx] = (0, 0, 1.0)
    qi = 0

    while qi < len(queue):
        cx, cy, mp, hm, psr = queue[qi]
        qi += 1

        c_idx = cy * WIDTH + cx
        nbrs = _NEIGHBORS[c_idx]

        for nx, ny in nbrs:
            cost = _hex_mp_cost_fast(board, cx, cy, nx, ny)
            if cost < 0:
                continue  # Impassable (water)

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
            n_idx = ny * WIDTH + nx

            # Update best walk or run path
            is_run = new_mp > walk_mp
            if is_run:
                existing = best_run[n_idx]
            else:
                existing = best_walk[n_idx]

            should_add = False
            if existing is None:
                should_add = True
            else:
                e_mp, e_hm, e_psr = existing
                # Prefer more hexes moved, then less MP used, then higher PSR
                if new_hm > e_hm:
                    should_add = True
                elif new_hm == e_hm and new_mp < e_mp:
                    should_add = True
                elif new_hm == e_hm and new_mp == e_mp and new_psr > e_psr:
                    should_add = True

            if should_add:
                if is_run:
                    best_run[n_idx] = (new_mp, new_hm, new_psr)
                else:
                    best_walk[n_idx] = (new_mp, new_hm, new_psr)
                queue.append((nx, ny, new_mp, new_hm, new_psr))

    # Build move dicts
    moves: list[dict] = []

    # Standing still at current position (all 6 facings)
    for f in range(6):
        moves.append({"dest_x": unit.x, "dest_y": unit.y, "facing": f, "mp_used": 0})

    for idx in range(size):
        if idx == start_idx:
            continue

        y = idx // WIDTH
        x = idx % WIDTH

        walk_data = best_walk[idx]
        run_data = best_run[idx]

        # Skip run if it doesn't provide more hexes than walk
        if walk_data and run_data:
            if run_data[1] <= walk_data[1]:
                run_data = None

        for data in (walk_data, run_data):
            if data is None:
                continue
            mp_used = data[0]
            for f in range(6):
                moves.append({
                    "dest_x": x, "dest_y": y, "facing": f, "mp_used": mp_used,
                })

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

    # Re-use standing enumeration with reduced MP
    # Create a temporary unit-like object with reduced MP
    # Actually, just run BFS with mp budget = remaining walk/run
    remaining_run = unit.run_mp - stand_cost

    size = WIDTH * HEIGHT
    best_walk: list[tuple[int, int, float] | None] = [None] * size
    best_run: list[tuple[int, int, float] | None] = [None] * size

    start_idx = unit.y * WIDTH + unit.x
    piloting = unit.template.piloting

    queue: list[tuple[int, int, int, int, float]] = [
        (unit.x, unit.y, 0, 0, 1.0)
    ]
    best_walk[start_idx] = (0, 0, 1.0)
    qi = 0

    max_mp = max(remaining_walk, remaining_run) if remaining_run > 0 else remaining_walk

    while qi < len(queue):
        cx, cy, mp, hm, psr = queue[qi]
        qi += 1

        c_idx = cy * WIDTH + cx
        nbrs = _NEIGHBORS[c_idx]

        for nx, ny in nbrs:
            cost = _hex_mp_cost_fast(board, cx, cy, nx, ny)
            if cost < 0:
                continue  # Impassable (water)

            new_mp = mp + cost

            if new_mp > max_mp:
                continue

            new_psr = psr
            is_running = new_mp > remaining_walk

            if is_running and board.terrain(nx, ny) == Terrain.HEAVY_WOODS:
                new_psr *= _PSR_PROBS.get(piloting, 0.5)

            elev_diff = abs(board.elevation(nx, ny) - board.elevation(cx, cy))
            if elev_diff >= 2:
                new_psr *= _PSR_PROBS.get(piloting, 0.5)

            if new_psr < 0.3:
                continue

            new_hm = hm + 1
            n_idx = ny * WIDTH + nx

            is_run = new_mp > remaining_walk
            if is_run:
                existing = best_run[n_idx]
            else:
                existing = best_walk[n_idx]

            should_add = False
            if existing is None:
                should_add = True
            else:
                e_mp, e_hm, e_psr = existing
                if new_hm > e_hm:
                    should_add = True
                elif new_hm == e_hm and new_mp < e_mp:
                    should_add = True
                elif new_hm == e_hm and new_mp == e_mp and new_psr > e_psr:
                    should_add = True

            if should_add:
                if is_run:
                    best_run[n_idx] = (new_mp, new_hm, new_psr)
                else:
                    best_walk[n_idx] = (new_mp, new_hm, new_psr)
                queue.append((nx, ny, new_mp, new_hm, new_psr))

    for idx in range(size):
        if idx == start_idx:
            continue

        y = idx // WIDTH
        x = idx % WIDTH

        walk_data = best_walk[idx]
        run_data = best_run[idx]

        if walk_data and run_data and run_data[1] <= walk_data[1]:
            run_data = None

        for data in (walk_data, run_data):
            if data is None:
                continue
            mp_used = data[0] + stand_cost
            for f in range(6):
                moves.append({
                    "dest_x": x, "dest_y": y, "facing": f, "mp_used": mp_used,
                })

    return moves


def _stand_still_moves(unit: Unit) -> list[dict]:
    return [
        {"dest_x": unit.x, "dest_y": unit.y, "facing": f, "mp_used": 0}
        for f in range(6)
    ]
