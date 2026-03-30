"""Diagnostic: compare has_just_stood BFS against brute-force enumeration.

The brute-force approach tries ALL possible step sequences up to the MP budget,
giving us ground truth for what's reachable. If has_just_stood BFS produces
hexes not found by brute force, those hexes are genuinely unreachable.
"""

from __future__ import annotations

import math
from megamek_gym.sim.board import BOARD, _NEIGHBOR_TABLE
from megamek_gym.sim.movement import (
    _facing_bfs_deque, _MP_COST_TABLE, _ELEV_DIFF_TABLE,
    _STEP_FORWARD, _PSR_PROBS,
    enumerate_moves,
)
from megamek_gym.sim.unit import Unit, UNIT_TEMPLATES


def brute_force_enumerate(board, sx, sy, sf, walk_mp, run_mp, piloting,
                          blocked_hex=None):
    """Brute-force DFS enumerate ALL reachable (x,y,facing,mp,hm) states.

    Models the exact Java behavior for prone GET_UP:
    1. Start at (sx, sy, sf) with mp=0, hasJustStood=True
    2. Free turns until first FORWARD step
    3. Normal turns (cost 1) after first FORWARD
    4. Turn constraints: no opposite turn, max 3 consecutive
    5. No backward movement (separate pass, not modeled here)

    Returns dict: (x, y, facing) -> (min_mp, max_hm, psr) for walk and run.
    """
    best_walk = {}  # (x,y,f) -> (mp, hm, psr) — best by (max hm, max psr)
    best_run = {}

    # State: (x, y, facing, mp, hm, psr, has_just_stood, last_turn_dir, consec_turns)
    # last_turn_dir: 0=none/forward, 1=right, 2=left
    # Use DFS with memoization on (x, y, facing, mp, hm, has_just_stood, last_turn_dir, consec)
    # to avoid exponential blowup

    # visited: set of (x, y, facing, mp, hm, js, ltd, consec) to avoid re-exploring
    visited = set()

    def update_best(x, y, f, mp, hm, psr):
        is_run = mp > walk_mp
        best = best_run if is_run else best_walk
        key = (x, y, f)
        existing = best.get(key)
        if existing is None:
            best[key] = (mp, hm, psr)
        elif hm > existing[1] or (hm == existing[1] and psr > existing[2]):
            best[key] = (mp, hm, psr)

    def dfs(x, y, f, mp, hm, psr, js, ltd, consec):
        state = (x, y, f, mp, hm, js, ltd, consec)
        if state in visited:
            return
        visited.add(state)

        # Record this state
        update_best(x, y, f, mp, hm, psr)

        # Generate TURN_RIGHT (facing + 1)
        # Can't turn right if: last was LEFT (opposite), or last was RIGHT and consec >= 3
        if ltd != 2 and not (ltd == 1 and consec >= 3):
            turn_cost = 0 if js else 1
            new_mp = mp + turn_cost
            if new_mp <= run_mp:
                new_consec = (consec + 1) if ltd == 1 else 1
                new_f = (f + 1) % 6
                dfs(x, y, new_f, new_mp, hm, psr, js, 1, new_consec)

        # Generate TURN_LEFT (facing - 1 = facing + 5)
        if ltd != 1 and not (ltd == 2 and consec >= 3):
            turn_cost = 0 if js else 1
            new_mp = mp + turn_cost
            if new_mp <= run_mp:
                new_consec = (consec + 1) if ltd == 2 else 1
                new_f = (f + 5) % 6
                dfs(x, y, new_f, new_mp, hm, psr, js, 2, new_consec)

        # Generate FORWARD (clears hasJustStood)
        nx, ny = _NEIGHBOR_TABLE[x][y][f]
        if nx >= 0:
            if blocked_hex is not None and (nx, ny) == blocked_hex:
                return
            cost = _MP_COST_TABLE[x][y][f]
            if cost >= 0:
                new_mp = mp + cost
                if new_mp <= run_mp:
                    new_psr = psr
                    if _ELEV_DIFF_TABLE[x][y][f] >= 2:
                        new_psr *= _PSR_PROBS.get(piloting, 0.0)
                    # FORWARD clears hasJustStood and resets turn state
                    dfs(nx, ny, f, new_mp, hm + 1, new_psr, False, 0, 0)

    # Start DFS from start position with hasJustStood=True
    dfs(sx, sy, sf, 0, 0, 1.0, True, 0, 0)

    return best_walk, best_run


def compare_algorithms(x, y, f, walk_mp, run_mp, piloting=5, blocked_hex=None):
    """Compare has_just_stood BFS against brute-force."""
    remaining_walk = walk_mp - 2  # stand cost
    remaining_run = run_mp - 2

    # Deque BFS with has_just_stood
    deque_w, deque_r = _facing_bfs_deque(
        BOARD, x, y, f,
        remaining_walk, remaining_run, piloting,
        backward=False, blocked_hex=blocked_hex,
        has_just_stood=True,
    )

    # Brute force (ground truth)
    bf_w, bf_r = brute_force_enumerate(
        BOARD, x, y, f, remaining_walk, remaining_run, piloting,
        blocked_hex=blocked_hex,
    )

    # Compare hex sets
    deque_walk_hexes = {(kx, ky) for (kx, ky, kf) in deque_w}
    deque_run_hexes = {(kx, ky) for (kx, ky, kf) in deque_r}
    bf_walk_hexes = {(kx, ky) for (kx, ky, kf) in bf_w}
    bf_run_hexes = {(kx, ky) for (kx, ky, kf) in bf_r}

    walk_only_deque = deque_walk_hexes - bf_walk_hexes
    walk_only_bf = bf_walk_hexes - deque_walk_hexes
    run_only_deque = deque_run_hexes - bf_run_hexes
    run_only_bf = bf_run_hexes - deque_run_hexes

    has_diff = walk_only_deque or walk_only_bf or run_only_deque or run_only_bf

    if has_diff:
        print(f"\n{'='*60}")
        print(f"MISMATCH at ({x},{y},f={f}) walk={walk_mp} run={run_mp}")
        print(f"  Deque: walk={len(deque_walk_hexes)} run={len(deque_run_hexes)}")
        print(f"  BruteForce: walk={len(bf_walk_hexes)} run={len(bf_run_hexes)}")

        if walk_only_deque:
            print(f"  Walk DEQUE-only: {sorted(walk_only_deque)}")
        if walk_only_bf:
            print(f"  Walk BF-only: {sorted(walk_only_bf)}")
        if run_only_deque:
            print(f"  Run DEQUE-only ({len(run_only_deque)}): {sorted(run_only_deque)[:20]}")
            # For first deque-only hex, show details
            for hx, hy in sorted(run_only_deque)[:3]:
                for (kx, ky, kf), (mp, hm, psr) in deque_r.items():
                    if kx == hx and ky == hy:
                        print(f"    ({kx},{ky},f={kf}) mp={mp} hm={hm} psr={psr:.3f}")
        if run_only_bf:
            print(f"  Run BF-only ({len(run_only_bf)}): {sorted(run_only_bf)[:20]}")
            for hx, hy in sorted(run_only_bf)[:3]:
                for (kx, ky, kf), (mp, hm, psr) in bf_r.items():
                    if kx == hx and ky == hy:
                        print(f"    ({kx},{ky},f={kf}) mp={mp} hm={hm} psr={psr:.3f}")

    return has_diff, len(run_only_deque), len(run_only_bf)


def main():
    template = UNIT_TEMPLATES["Trebuchet TBT-5S"]

    total_deque_extra = 0
    total_bf_extra = 0
    mismatches = 0

    # Test a grid of positions
    import sys
    positions = []
    for x in range(0, BOARD.width, 3):
        for y in range(0, BOARD.height, 3):
            for f in range(6):
                positions.append((x, y, f))

    print(f"Testing {len(positions)} positions...")

    for i, (x, y, f) in enumerate(positions):
        has_diff, d_extra, bf_extra = compare_algorithms(
            x, y, f,
            walk_mp=template.walk_mp, run_mp=template.run_mp,
            piloting=template.piloting,
        )
        if has_diff:
            mismatches += 1
            total_deque_extra += d_extra
            total_bf_extra += bf_extra
        if (i + 1) % 50 == 0:
            print(f"  ...tested {i+1}/{len(positions)}")

    print(f"\n{'='*60}")
    print(f"Results: {mismatches}/{len(positions)} positions with mismatches")
    print(f"  Deque-only hex instances: {total_deque_extra}")
    print(f"  BruteForce-only hex instances: {total_bf_extra}")


if __name__ == "__main__":
    main()
