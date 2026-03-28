"""Movement enumeration via BFS, replicating MegaMek's LongestPathFinder.

Produces legal moves matching the Java bridge's JSON format so that
observation.py can consume them directly.

Two algorithms are available (selected via the ``algorithm`` parameter):

- ``"deque"`` (default): Faithful port of Java's ``LongestPathFinder``
  deque-based relaxation.  Uses a priority queue sorted by (mp ASC, hexes
  DESC) and stores a Pareto-frontier of (mp, hexes) per ``(x, y, facing)``
  node, expanding from every accepted path.  Runs forward and backward
  passes separately (matching Java's two ``LongestPathFinder`` invocations)
  and merges results.  Higher fidelity but ~2-4x slower per call.

- ``"bfs"``: Multi-objective BFS with flat per-state tracking.
  Fast (~97% hex match vs Java).
"""

from __future__ import annotations

import heapq
import math
from typing import TYPE_CHECKING

from megamek_gym.sim.board import Board, Terrain, WIDTH, HEIGHT, neighbor


if TYPE_CHECKING:
    from megamek_gym.sim.unit import Unit


# ---------------------------------------------------------------------------
# Step-type constants for turn constraint tracking in the deque pathfinder
# ---------------------------------------------------------------------------
_STEP_NONE = 0
_STEP_FORWARD = 1
_STEP_TURN_LEFT = 2
_STEP_TURN_RIGHT = 3
_MAX_CONSEC_TURNS = 3

# "Just stood" turn types — free turns after GET_UP, with turn constraints.
# Matching Java's hasJustStood behavior: turns are free as long as no FORWARD
# step has been taken.  The last_step tracks the direction for turn constraints
# (no opposite turn, max 3 consecutive) while the "just stood" flag ensures
# the turn costs 0 MP.  Once a FORWARD step occurs, hasJustStood clears and
# normal turn costs apply.
_STEP_JS_TURN_LEFT = 4   # Free turn left (hasJustStood)
_STEP_JS_TURN_RIGHT = 5  # Free turn right (hasJustStood)


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


def _apply_move_legality_filter(moves: list[dict], unit: Unit) -> list[dict]:
    """Post-filter matching Java's MoveStep.compileIllegal() edge cases.

    Catches damaged-unit conditions that the pathfinder doesn't model:
    - Gyro destroyed (2 hits): prone can only turn one hex-side; standing
      has no non-tracked MP (stand-still only for biped Mechs).
    - Both arms destroyed + either leg destroyed: can't stand up from prone.
    """
    from megamek_gym.sim.unit import Location

    # Gyro destroyed: 2 crits = destroyed (MoveStep.java:2219-2249)
    if unit.gyro_hits >= 2:
        if unit.prone:
            # Can only change one hex-side (turn once) from current facing
            allowed_facings = {unit.facing, (unit.facing + 1) % 6,
                               (unit.facing + 5) % 6}
            return [m for m in moves
                    if m["dest_x"] == unit.x and m["dest_y"] == unit.y
                    and m["facing"] in allowed_facings]
        else:
            # Standing with destroyed gyro: no MP for biped Mechs
            return [m for m in moves if m["mp_used"] == 0]

    # Can't stand if both arms destroyed + either leg destroyed
    # (MoveStep.java:2251-2259)
    if unit.prone:
        la_dead = unit.loc_destroyed[Location.LA]
        ra_dead = unit.loc_destroyed[Location.RA]
        rl_dead = unit.loc_destroyed[Location.RL]
        ll_dead = unit.loc_destroyed[Location.LL]
        if la_dead and ra_dead and (rl_dead or ll_dead):
            # Can't stand — only stay-prone move is legal
            return [m for m in moves
                    if m["dest_x"] == unit.x and m["dest_y"] == unit.y
                    and m["hexes_moved"] == 0 and m["mp_used"] == 0]

    return moves


def enumerate_moves(unit: Unit, board: Board,
                    enemy: Unit | None = None,
                    algorithm: str = "deque") -> list[dict]:
    """Enumerate legal moves for a unit.

    *algorithm*: ``"deque"`` (default, Java LongestPathFinder port with
    Pareto-deque relaxation) or ``"bfs"`` (simpler multi-objective BFS).
    """
    if unit.destroyed or unit.shutdown or not unit.deployed:
        return []

    # Enemy hex is blocked during pathfinding AND filtered from output.
    # Java's MovePathLegalityFilter calls isMovementPossible() on every
    # neighbor, which blocks Mechs from entering enemy-occupied hexes
    # (MoveStep.java:3396-3401).  This prevents exploring THROUGH the
    # enemy hex, not just ending there.
    enemy_xy: tuple[int, int] | None = None
    if enemy is not None and not enemy.destroyed and enemy.deployed:
        enemy_xy = (enemy.x, enemy.y)

    if unit.prone:
        moves = _enumerate_prone_moves(unit, board, algorithm=algorithm,
                                       enemy_xy=enemy_xy)
    else:
        moves = _enumerate_standing_moves(unit, board, algorithm=algorithm,
                                          enemy_xy=enemy_xy)

    # compileIllegal() edge cases for damaged units (defense-in-depth)
    moves = _apply_move_legality_filter(moves, unit)

    # Stacking filter (defense-in-depth): remove any moves that end on
    # the enemy hex.  The pathfinder already blocks entering the enemy hex
    # during exploration (matching Java's isMovementPossible), but this
    # output filter catches edge cases like the start hex coinciding with
    # the enemy hex.
    if enemy_xy is not None:
        ex, ey = enemy_xy
        moves = [m for m in moves
                 if m["dest_x"] != ex or m["dest_y"] != ey]

    # PSR fall tolerance: reject paths with >70% cumulative fall risk,
    # matching Java's getMovePathSuccessProbability() < FALL_TOLERANCE (0.3).
    # Java skips the GET_UP PSR in this calculation; Python matches naturally
    # since stand-up PSR is never computed in the pathfinders.
    moves = [m for m in moves if m.get("success_probability", 1.0) >= 0.3]

    for i, m in enumerate(moves):
        m["index"] = i

    return moves


def _is_better_output(new_hm: int, new_psr: float,
                      e_hm: int, e_psr: float) -> bool:
    """Check if a new path is better for output selection.

    Matches Java's isBetterPath: prefers more hexes moved (better TMM
    defensive modifier), tiebroken by higher PSR success probability.
    """
    if new_hm != e_hm:
        return new_hm > e_hm
    return new_psr > e_psr


def _facing_bfs(board: Board, start_x: int, start_y: int, start_facing: int,
                walk_mp: int, run_mp: int, piloting: int,
                free_turns: bool = False,
                start_facings: list[int] | None = None,
                blocked_hex: tuple[int, int] | None = None,
                ) -> tuple[dict, dict]:
    """Run facing-aware BFS to find reachable (x, y, facing) states.

    Returns (best_walk, best_run) dicts mapping
    (x, y, facing) -> (mp_used, hexes_moved, psr_prob).

    Uses multi-objective exploration to match Java's LongestPathFinder, which
    keeps a deque of Pareto-optimal paths per state. A state is re-explored
    when EITHER a cheaper path (lower MP) or a longer path (more hexes) is
    found within each walk/run category. This ensures intermediate states are
    explored via both short routes (for reachability) and long routes (for TMM),
    allowing the BFS to find both walk-speed and run-speed variants that Java
    finds through its multi-path storage.

    Output dicts store the best path per Java's isBetterPath criterion: most
    hexes moved, tiebroken by highest PSR success probability.

    Transitions:
    - FORWARD: move to hex in current facing direction (terrain MP cost)
    - TURN_LEFT/RIGHT: same hex, facing ±1 (1 MP each, or 0 if free_turns)

    free_turns: True for units that just stood up from prone (hasJustStood).

    start_facings: Seed BFS from multiple facings at the start hex (all at
    mp=0).  Used for prone paths where GET_UP allows free initial turns:
    seeding all 6 facings in a single BFS is equivalent to free turns at
    the start hex only, without making ALL turns free.
    """
    turn_cost = 0 if free_turns else 1

    # Output dicts: track best path per Java's criterion (max hexes, then PSR)
    best_walk: dict[tuple[int, int, int], tuple[int, int, float]] = {}
    best_run: dict[tuple[int, int, int], tuple[int, int, float]] = {}

    # Multi-objective exploration: track (min_mp, max_hm) per state per
    # category. Re-explore when either objective improves. Bounded by
    # run_mp × states per category (each state explored at most run_mp times
    # for hexes improvements + once for each cheaper path).
    min_mp_walk: dict[tuple[int, int, int], int] = {}
    max_hm_walk: dict[tuple[int, int, int], int] = {}
    min_mp_run: dict[tuple[int, int, int], int] = {}
    max_hm_run: dict[tuple[int, int, int], int] = {}

    facings = start_facings if start_facings is not None else [start_facing]
    queue: list[tuple[int, int, int, int, int, float]] = []
    for sf in facings:
        key = (start_x, start_y, sf)
        best_walk[key] = (0, 0, 1.0)
        min_mp_walk[key] = 0
        max_hm_walk[key] = 0
        queue.append((start_x, start_y, sf, 0, 0, 1.0))
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
            min_mp_d = min_mp_run if is_run else min_mp_walk
            max_hm_d = max_hm_run if is_run else max_hm_walk

            # Update output dict (best path for this walk/run category)
            existing = best.get(key)
            if existing is None or _is_better_output(hm, psr, existing[1], existing[2]):
                best[key] = (new_mp, hm, psr)

            # Re-explore if cheaper MP or more hexes (multi-objective gate)
            prev_min = min_mp_d.get(key)
            prev_max = max_hm_d.get(key)
            should_explore = False
            if prev_min is None or new_mp < prev_min:
                min_mp_d[key] = new_mp
                should_explore = True
            if prev_max is None or hm > prev_max:
                max_hm_d[key] = hm
                should_explore = True
            if should_explore:
                queue.append((cx, cy, new_facing, new_mp, hm, psr))

        # --- FORWARD transition (move in facing direction) ---
        nx, ny = neighbor(cx, cy, cf)
        if not (0 <= nx < WIDTH and 0 <= ny < HEIGHT):
            continue

        # Can't enter enemy-occupied hex (MoveStep.java:3396-3401)
        if blocked_hex is not None and (nx, ny) == blocked_hex:
            continue

        cost = _hex_mp_cost(board, cx, cy, nx, ny)
        if cost < 0:
            continue  # Impassable

        new_mp = mp + cost
        if new_mp > run_mp:
            continue

        new_psr = psr
        elev_diff = abs(board.elevation(nx, ny) - board.elevation(cx, cy))
        if elev_diff >= 2:
            new_psr *= _PSR_PROBS.get(piloting, 0.0)

        new_hm = hm + 1
        key = (nx, ny, cf)  # After FORWARD, facing stays the same
        is_run = new_mp > walk_mp
        best = best_run if is_run else best_walk
        min_mp_d = min_mp_run if is_run else min_mp_walk
        max_hm_d = max_hm_run if is_run else max_hm_walk

        # Update output dict (best path for this walk/run category)
        existing = best.get(key)
        if existing is None or _is_better_output(new_hm, new_psr, existing[1], existing[2]):
            best[key] = (new_mp, new_hm, new_psr)

        # Re-explore if cheaper MP or more hexes (multi-objective gate)
        prev_min = min_mp_d.get(key)
        prev_max = max_hm_d.get(key)
        should_explore = False
        if prev_min is None or new_mp < prev_min:
            min_mp_d[key] = new_mp
            should_explore = True
        if prev_max is None or new_hm > prev_max:
            max_hm_d[key] = new_hm
            should_explore = True
        if should_explore:
            queue.append((nx, ny, cf, new_mp, new_hm, new_psr))

    return best_walk, best_run


# ---------------------------------------------------------------------------
# Deque-based relaxation pathfinder (Java LongestPathFinder port)
# ---------------------------------------------------------------------------

def _candidate_is_worse(deque: list[tuple[int, int, float]],
                        cand_mp: int, cand_hm: int) -> bool:
    """Check if candidate should be rejected from a node's Pareto deque.

    Faithful translation of Java's ``candidateIsWorseThanTopOfStack``.
    The deque is sorted by mp ascending; each entry has strictly more hexes
    than all prior entries.  We compare against the *last* entry (highest mp
    and highest hexes).

    The FIFO counter in the heap tuple ensures that equal ``(mpUsed, -hexesMoved)``
    entries are popped in insertion order, approximating Java's PriorityQueue
    binary-heap behavior (approximately FIFO for comparator-equal entries).

    Simplified vs Java: no prone/hull-down comparison (handled externally),
    no waypoint logic, no forward/backward preservation (separate passes).
    """
    if not deque:
        return False

    top_mp, top_hm, _top_psr = deque[-1]

    # PQ ordering guarantees top_mp <= cand_mp; if violated, reject
    if top_mp > cand_mp:
        return True

    # Top already reached more hexes at same-or-lower MP
    if top_hm > cand_hm:
        return True

    # Same hexes — reject (no forward/backward distinction within a pass)
    if top_hm == cand_hm:
        return True

    # top_hm < cand_hm: candidate went farther.
    # Accept only if candidate used strictly MORE mp (extends Pareto frontier).
    # If same mp, this shouldn't happen (PQ processes equal-mp in hex order)
    # but reject to be safe.
    return top_mp == cand_mp


def _facing_bfs_deque(board: Board, start_x: int, start_y: int,
                      start_facing: int,
                      walk_mp: int, run_mp: int, piloting: int,
                      free_turns: bool = False,
                      backward: bool = False,
                      start_facings: list[int] | None = None,
                      blocked_hex: tuple[int, int] | None = None,
                      has_just_stood: bool = False,
                      ) -> tuple[dict, dict]:
    """Deque-relaxation pathfinder matching Java's LongestPathFinder.

    Returns ``(best_walk, best_run)`` dicts mapping
    ``(x, y, facing) -> (mp_used, hexes_moved, psr_prob)``.

    Uses **lazy relaxation** matching Java's ``AbstractPathFinder.run()``:
    candidates are pushed to the priority queue unconditionally when their
    parent is accepted, and only relaxed (checked against the node's Pareto
    deque) when popped.  This prevents over-exploration from eagerly-accepted
    intermediate paths whose deque state may change before they're processed.

    When *backward* is True, movement steps go to the hex behind the unit
    (``neighbor(x, y, (facing+3)%6)``) while facing is unchanged, matching
    Java's ``MoveStepType.BACKWARDS`` pass.  Backward movement prohibits
    running (``BackwardStep.setRunProhibited(true)`` in Java), so the
    effective MP limit is *walk_mp* — only walk-speed paths are produced.

    *start_facings*: seed the pathfinder from multiple facings at the start
    hex (all at mp=0).  Used for prone paths where GET_UP allows free
    initial turns: seeding all 6 facings in a single run avoids the
    over-generation from merging 6 independent runs.
    """
    turn_cost = 0 if free_turns else 1

    # Backward movement prohibits running in Java (BackwardStep.java)
    eff_run_mp = walk_mp if backward else run_mp

    # Output dicts (best path per walk/run, using isBetterPath criterion)
    best_walk: dict[tuple[int, int, int], tuple[int, int, float]] = {}
    best_run: dict[tuple[int, int, int], tuple[int, int, float]] = {}

    # Per-node Pareto deques: (x,y,facing) -> list[(mp, hm, psr)]
    # Sorted by mp ascending; each entry has strictly more hm than prior.
    node_deques: dict[tuple[int, int, int], list[tuple[int, int, float]]] = {}

    # Seed from one or more start facings.
    # Matching Java: starting edges go into the priority queue only;
    # deques start empty and are populated via lazy relaxation at pop time.
    facings = start_facings if start_facings is not None else [start_facing]
    initial_step = _STEP_NONE
    if has_just_stood:
        # Match Java: start from a single facing (like MovePath with GET_UP).
        # The has_just_stood state is tracked via _STEP_JS_TURN_* types so
        # the free-turn chain preserves turn constraints exactly.
        facings = [start_facing]
        initial_step = _STEP_FORWARD  # GET_UP behaves like a forward step
        # for turn constraint purposes: hasJustStood allows both turns

    # FIFO counter for tie-breaking: when (mp, -hm) are equal, earlier-inserted
    # entries pop first, matching Java's PriorityQueue approximately-FIFO
    # behavior for comparator-equal entries.
    seq = 0
    heap: list = []
    for sf in facings:
        heap.append((0, 0, seq, start_x, start_y, sf, 1.0, initial_step, 0))
        seq += 1
    heapq.heapify(heap)

    while heap:
        mp, neg_hm, _seq, cx, cy, cf, psr, last_step, consec = heapq.heappop(heap)
        hm = -neg_hm
        key = (cx, cy, cf)

        # --- Lazy relaxation: relax the popped candidate against its node ---
        # Matches Java's AbstractPathFinder.run() which calls
        # edgeRelaxer.doRelax(cost, e, comparator) at pop time.
        dq = node_deques.get(key)
        if dq is None:
            node_deques[key] = [(mp, hm, psr)]
        elif _candidate_is_worse(dq, mp, hm):
            continue  # Rejected — skip neighbor generation
        else:
            dq.append((mp, hm, psr))

        # Update output dicts
        is_run = mp > walk_mp
        best = best_run if is_run else best_walk
        existing = best.get(key)
        if existing is None or _is_better_output(hm, psr,
                                                 existing[1], existing[2]):
            best[key] = (mp, hm, psr)

        # --- Generate neighbor candidates and push unconditionally ---
        # Matches Java: accepted path generates neighbors, all pushed to queue.

        # Determine if this step is in the "just stood" phase.
        # hasJustStood is true for GET_UP and propagates through turns,
        # cleared by the first FORWARD step.
        is_js = last_step in (_STEP_JS_TURN_LEFT, _STEP_JS_TURN_RIGHT)
        # The initial seed (FORWARD after has_just_stood) also has hasJustStood
        is_js = is_js or (has_just_stood and last_step == _STEP_FORWARD
                          and hm == 0 and cx == start_x and cy == start_y)

        # Map JS turn types to their base direction for turn constraints
        if last_step == _STEP_JS_TURN_LEFT:
            eff_last = _STEP_TURN_LEFT
        elif last_step == _STEP_JS_TURN_RIGHT:
            eff_last = _STEP_TURN_RIGHT
        else:
            eff_last = last_step

        # TURN_RIGHT (facing + 1)
        if eff_last != _STEP_TURN_LEFT and not (
            eff_last == _STEP_TURN_RIGHT and consec >= _MAX_CONSEC_TURNS
        ):
            # In "just stood" phase, turns are free (0 MP)
            actual_turn_cost = 0 if is_js else turn_cost
            new_mp_t = mp + actual_turn_cost
            if new_mp_t <= eff_run_mp:
                new_consec = (consec + 1) if eff_last == _STEP_TURN_RIGHT else 1
                step_type = _STEP_JS_TURN_RIGHT if is_js else _STEP_TURN_RIGHT
                seq += 1
                heapq.heappush(heap, (
                    new_mp_t, -hm, seq, cx, cy, (cf + 1) % 6, psr,
                    step_type, new_consec,
                ))

        # TURN_LEFT (facing - 1 = facing + 5)
        if eff_last != _STEP_TURN_RIGHT and not (
            eff_last == _STEP_TURN_LEFT and consec >= _MAX_CONSEC_TURNS
        ):
            actual_turn_cost = 0 if is_js else turn_cost
            new_mp_t = mp + actual_turn_cost
            if new_mp_t <= eff_run_mp:
                new_consec = (consec + 1) if eff_last == _STEP_TURN_LEFT else 1
                step_type = _STEP_JS_TURN_LEFT if is_js else _STEP_TURN_LEFT
                seq += 1
                heapq.heappush(heap, (
                    new_mp_t, -hm, seq, cx, cy, (cf + 5) % 6, psr,
                    step_type, new_consec,
                ))

        # FORWARD / BACKWARD movement (clears hasJustStood)
        if backward:
            nx, ny = neighbor(cx, cy, (cf + 3) % 6)
        else:
            nx, ny = neighbor(cx, cy, cf)

        if 0 <= nx < WIDTH and 0 <= ny < HEIGHT:
            if blocked_hex is not None and (nx, ny) == blocked_hex:
                pass  # skip — can't enter enemy-occupied hex (MoveStep:3396)
            elif backward and board.elevation(nx, ny) != board.elevation(cx, cy):
                pass  # skip — movement impossible
            else:
                cost = _hex_mp_cost(board, cx, cy, nx, ny)
                if cost >= 0:
                    new_mp_f = mp + cost
                    if new_mp_f <= eff_run_mp:
                        new_psr = psr
                        elev_diff = abs(board.elevation(nx, ny) - board.elevation(cx, cy))
                        if elev_diff >= 2:
                            new_psr *= _PSR_PROBS.get(piloting, 0.0)
                        seq += 1
                        heapq.heappush(heap, (
                            new_mp_f, -(hm + 1), seq, nx, ny, cf, new_psr,
                            _STEP_FORWARD, 0,
                        ))

    return best_walk, best_run


def _emit_walk_run(moves: list[dict], key: tuple[int, int, int],
                   walk_data: tuple | None, run_data: tuple | None) -> None:
    """Append walk and/or run move dicts, applying Java's run-only-if-more-hexes filter."""
    x, y, f = key
    if walk_data and run_data and run_data[1] <= walk_data[1]:
        run_data = None
    if walk_data:
        moves.append({"dest_x": x, "dest_y": y, "facing": f, "mp_used": walk_data[0],
                       "hexes_moved": walk_data[1], "success_probability": walk_data[2]})
    if run_data:
        moves.append({"dest_x": x, "dest_y": y, "facing": f, "mp_used": run_data[0],
                       "hexes_moved": run_data[1], "success_probability": run_data[2]})


def _enumerate_standing_moves(unit: Unit, board: Board,
                              algorithm: str = "bfs",
                              enemy_xy: tuple[int, int] | None = None,
                              ) -> list[dict]:
    """Enumerate moves for a standing mech using facing-aware BFS."""
    walk_mp = unit.walk_mp
    run_mp = unit.run_mp

    if walk_mp <= 0:
        return _stand_still_moves(unit)

    if algorithm == "deque":
        fwd_w, fwd_r = _facing_bfs_deque(
            board, unit.x, unit.y, unit.facing,
            walk_mp, run_mp, unit.template.piloting,
            backward=False, blocked_hex=enemy_xy,
        )
        bwd_w, bwd_r = _facing_bfs_deque(
            board, unit.x, unit.y, unit.facing,
            walk_mp, run_mp, unit.template.piloting,
            backward=True, blocked_hex=enemy_xy,
        )
        return _build_moves_deque(unit, fwd_w, fwd_r, bwd_w, bwd_r)

    best_walk, best_run = _facing_bfs(
        board, unit.x, unit.y, unit.facing,
        walk_mp, run_mp, unit.template.piloting,
        blocked_hex=enemy_xy,
    )

    # Build move dicts
    moves: list[dict] = []

    # Standing still at current position (original facing only, matching Java's
    # empty MovePath which preserves entity facing)
    moves.append({"dest_x": unit.x, "dest_y": unit.y, "facing": unit.facing, "mp_used": 0,
                   "hexes_moved": 0, "success_probability": 1.0})

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
                moves.append({"dest_x": x, "dest_y": y, "facing": f, "mp_used": walk_data[0],
                               "hexes_moved": walk_data[1], "success_probability": walk_data[2]})
            # Run data at start hex unlikely but handle it
            run_data = best_run.get(key)
            if run_data and run_data[0] > 0:
                walk_hm = walk_data[1] if walk_data else -1
                if run_data[1] > walk_hm:
                    moves.append({"dest_x": x, "dest_y": y, "facing": f, "mp_used": run_data[0],
                                   "hexes_moved": run_data[1], "success_probability": run_data[2]})
            continue

        walk_data = best_walk.get(key)
        run_data = best_run.get(key)

        # Skip run if it doesn't provide more hexes than walk
        if walk_data and run_data:
            if run_data[1] <= walk_data[1]:
                run_data = None

        if walk_data:
            moves.append({"dest_x": x, "dest_y": y, "facing": f, "mp_used": walk_data[0],
                           "hexes_moved": walk_data[1], "success_probability": walk_data[2]})
        if run_data:
            moves.append({"dest_x": x, "dest_y": y, "facing": f, "mp_used": run_data[0],
                           "hexes_moved": run_data[1], "success_probability": run_data[2]})

    return moves


def _build_moves_deque(unit: Unit,
                       fwd_w: dict, fwd_r: dict,
                       bwd_w: dict, bwd_r: dict) -> list[dict]:
    """Build move list from deque forward+backward results.

    Matches Java's output structure: the empty path (standing still) is
    added once, then ``addWalkAndRunPaths`` is called separately for
    forward and backward results, each applying its own
    run-only-if-more-hexes dedup.  This means a destination can have up
    to 4 entries: fwd_walk, fwd_run, bwd_walk, bwd_run.
    """
    moves: list[dict] = []
    sx, sy, sf = unit.x, unit.y, unit.facing

    # Java: paths.add(new MovePath(game, entity)) — empty path, original facing
    moves.append({"dest_x": sx, "dest_y": sy, "facing": sf, "mp_used": 0,
                   "hexes_moved": 0, "success_probability": 1.0})

    # Collect all keys across both passes
    all_keys = set(fwd_w) | set(fwd_r) | set(bwd_w) | set(bwd_r)

    for key in sorted(all_keys):
        x, y, f = key

        # Java's addWalkAndRunPaths for forward pass
        fw = fwd_w.get(key)
        fr = fwd_r.get(key)

        # Java's addWalkAndRunPaths for backward pass
        bw = bwd_w.get(key)
        br = bwd_r.get(key)

        # Skip start-hex entries at mp=0 (already added as empty path above)
        if fw and fw[0] == 0 and x == sx and y == sy and f == sf:
            fw = None
        if bw and bw[0] == 0 and x == sx and y == sy and f == sf:
            bw = None

        # Per-pass run dedup: run only if more hexes than walk in SAME pass
        _emit_walk_run(moves, key, fw, fr)
        _emit_walk_run(moves, key, bw, br)

    return moves


def _enumerate_prone_moves(unit: Unit, board: Board,
                           algorithm: str = "bfs",
                           enemy_xy: tuple[int, int] | None = None,
                           ) -> list[dict]:
    """Enumerate moves for a prone mech."""
    moves: list[dict] = []

    # Stay prone (stand still) — original facing only, matching Java's empty MovePath
    moves.append({"dest_x": unit.x, "dest_y": unit.y, "facing": unit.facing, "mp_used": 0,
                   "hexes_moved": 0, "success_probability": 1.0})

    # GET_UP costs 2 MP (or 1 if run_mp == 1), matching Java's GetUpStep.java
    stand_cost = 1 if unit.run_mp == 1 else 2
    remaining_walk = unit.walk_mp - stand_cost

    if remaining_walk < 0:
        return moves

    remaining_run = unit.run_mp - stand_cost
    eff_run_mp = max(remaining_walk, remaining_run)

    if eff_run_mp <= 0:
        # No movement possible after standing — just facing changes
        for f in range(6):
            moves.append({
                "dest_x": unit.x, "dest_y": unit.y, "facing": f,
                "mp_used": stand_cost,
                "hexes_moved": 0, "success_probability": 1.0,
            })
        return moves

    # Prone pathfinding matching Java's hasJustStood behavior:
    # After GET_UP, turns are free until the first FORWARD step.
    # Java starts from ONE facing (the entity's current facing) and lets the
    # free-turn chain explore other facings organically, with proper turn
    # constraints (no opposite turn, max 3 consecutive).  The "deque"
    # algorithm uses has_just_stood=True to model this; the "bfs" algorithm
    # seeds all 6 facings (faster approximation, ~97% match).
    all_facings = list(range(6))

    if algorithm == "deque":
        fwd_w, fwd_r = _facing_bfs_deque(
            board, unit.x, unit.y, unit.facing,
            remaining_walk, eff_run_mp, unit.template.piloting,
            free_turns=False, backward=False,
            blocked_hex=enemy_xy, has_just_stood=True,
        )
        bwd_w, bwd_r = _facing_bfs_deque(
            board, unit.x, unit.y, unit.facing,
            remaining_walk, eff_run_mp, unit.template.piloting,
            free_turns=False, backward=True,
            blocked_hex=enemy_xy, has_just_stood=True,
        )
        return _build_prone_moves_deque(
            unit, stand_cost, fwd_w, fwd_r, bwd_w, bwd_r, moves)

    best_walk, best_run = _facing_bfs(
        board, unit.x, unit.y, unit.facing,
        remaining_walk, eff_run_mp,
        unit.template.piloting,
        free_turns=False, start_facings=all_facings,
        blocked_hex=enemy_xy,
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
                    "hexes_moved": walk_data[1], "success_probability": walk_data[2],
                })
        if run_data:
            total_mp = run_data[0] + stand_cost
            moves.append({
                "dest_x": x, "dest_y": y, "facing": f,
                "mp_used": total_mp,
                "hexes_moved": run_data[1], "success_probability": run_data[2],
            })

    return moves


def _build_prone_moves_deque(unit: Unit, stand_cost: int,
                             fwd_w: dict, fwd_r: dict,
                             bwd_w: dict, bwd_r: dict,
                             moves: list[dict]) -> list[dict]:
    """Build prone move list from deque results, keeping fwd/bwd separate."""
    sx, sy = unit.x, unit.y

    all_keys = set(fwd_w) | set(fwd_r) | set(bwd_w) | set(bwd_r)

    for key in sorted(all_keys):
        x, y, f = key

        for w, r in [(fwd_w.get(key), fwd_r.get(key)),
                      (bwd_w.get(key), bwd_r.get(key))]:
            # Per-pass run dedup
            if w and r and r[1] <= w[1]:
                r = None
            if w:
                total_mp = w[0] + stand_cost
                if not (x == sx and y == sy and total_mp == 0):
                    moves.append({
                        "dest_x": x, "dest_y": y, "facing": f,
                        "mp_used": total_mp,
                        "hexes_moved": w[1], "success_probability": w[2],
                    })
            if r:
                total_mp = r[0] + stand_cost
                moves.append({
                    "dest_x": x, "dest_y": y, "facing": f,
                    "mp_used": total_mp,
                    "hexes_moved": r[1], "success_probability": r[2],
                })

    return moves


def _stand_still_moves(unit: Unit) -> list[dict]:
    return [
        {"dest_x": unit.x, "dest_y": unit.y, "facing": unit.facing, "mp_used": 0,
         "hexes_moved": 0, "success_probability": 1.0}
    ]
