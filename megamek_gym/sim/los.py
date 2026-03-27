"""Line-of-sight computation for hex grids.

Precomputes a LOS table for all hex pairs on the board.
Matches Java MegaMek's LosEffects algorithm: traces a hex line,
accumulates woods points along intervening hexes (with elevation gating),
and blocks LOS when total woods points >= 3.
"""

from __future__ import annotations

import math

from megamek_gym.sim.board import Board, Terrain


# -- Hex line tracing (cube coordinate interpolation) -------------------------

def _to_cube(x: int, y: int) -> tuple[int, int, int]:
    """Convert odd-column offset to cube coordinates."""
    q = x
    r = y - (x - (x & 1)) // 2
    s = -q - r
    return q, r, s


def _from_cube(q: int, r: int) -> tuple[int, int]:
    """Convert cube coordinates back to odd-column offset."""
    x = q
    y = r + (q - (q & 1)) // 2
    return x, y


def _cube_round(fq: float, fr: float, fs: float) -> tuple[int, int, int]:
    """Round fractional cube coordinates to nearest hex."""
    rq = round(fq)
    rr = round(fr)
    rs = round(fs)

    dq = abs(rq - fq)
    dr = abs(rr - fr)
    ds = abs(rs - fs)

    if dq > dr and dq > ds:
        rq = -rr - rs
    elif dr > ds:
        rr = -rq - rs
    else:
        rs = -rq - rr

    return rq, rr, rs


def _hex_linedraw(x1: int, y1: int, x2: int, y2: int) -> list[tuple[int, int]]:
    """Draw a line between two hexes using cube coordinate interpolation.

    Returns list of hex coordinates (inclusive of endpoints).
    Uses a small epsilon nudge to avoid landing exactly on hex edges,
    which helps produce a consistent single line (non-divided case).
    """
    q1, r1, s1 = _to_cube(x1, y1)
    q2, r2, s2 = _to_cube(x2, y2)

    n = max(abs(q2 - q1), abs(r2 - r1), abs(s2 - s1))
    if n == 0:
        return [(x1, y1)]

    # Nudge to avoid exact hex-edge ambiguity
    eps = 1e-6
    fq1 = q1 + eps
    fr1 = r1 + eps
    fs1 = s1 - 2 * eps
    fq2 = q2 + eps
    fr2 = r2 + eps
    fs2 = s2 - 2 * eps

    results = []
    seen = set()
    for i in range(n + 1):
        t = i / n
        fq = fq1 + (fq2 - fq1) * t
        fr = fr1 + (fr2 - fr1) * t
        fs = fs1 + (fs2 - fs1) * t
        rq, rr, rs = _cube_round(fq, fr, fs)
        ox, oy = _from_cube(rq, rr)
        key = (ox, oy)
        if key not in seen:
            seen.add(key)
            results.append(key)

    return results


def _hex_linedraw_divided(x1: int, y1: int, x2: int, y2: int) -> list[tuple[int, int]]:
    """Draw the alternate line (opposite nudge direction) for divided-line LOS.

    For lines that pass along hex edges, Java checks both sides and takes
    the worse (more blocking) result.
    """
    q1, r1, s1 = _to_cube(x1, y1)
    q2, r2, s2 = _to_cube(x2, y2)

    n = max(abs(q2 - q1), abs(r2 - r1), abs(s2 - s1))
    if n == 0:
        return [(x1, y1)]

    # Opposite nudge direction
    eps = 1e-6
    fq1 = q1 - eps
    fr1 = r1 - eps
    fs1 = s1 + 2 * eps
    fq2 = q2 - eps
    fr2 = r2 - eps
    fs2 = s2 + 2 * eps

    results = []
    seen = set()
    for i in range(n + 1):
        t = i / n
        fq = fq1 + (fq2 - fq1) * t
        fr = fr1 + (fr2 - fr1) * t
        fs = fs1 + (fs2 - fs1) * t
        rq, rr, rs = _cube_round(fq, fr, fs)
        ox, oy = _from_cube(rq, rr)
        key = (ox, oy)
        if key not in seen:
            seen.add(key)
            results.append(key)

    return results


def _is_divided_line(x1: int, y1: int, x2: int, y2: int) -> bool:
    """Check if the line between two hexes passes along hex edges.

    In hex grids, lines at angles that are multiples of 60 degrees
    (offset by 30 from vertex directions) run along shared edges.
    This happens when the cube coordinate deltas have a zero component.
    """
    q1, r1, s1 = _to_cube(x1, y1)
    q2, r2, s2 = _to_cube(x2, y2)
    dq = q2 - q1
    dr = r2 - r1
    ds = s2 - s1
    return dq == 0 or dr == 0 or ds == 0


# -- LOS computation matching Java's LosEffects ------------------------------

def _los_woods_score(board: Board, line: list[tuple[int, int]],
                     src_abs_height: float, tgt_abs_height: float,
                     x1: int, y1: int, x2: int, y2: int) -> int | None:
    """Compute accumulated woods score along a hex line.

    Returns None if LOS is hard-blocked by elevation (terrain floor exceeds
    both unit heights). Otherwise returns the total woods score.

    Matches Java's LosEffects.losForCoords() in non-diagram mode:
    - Bare elevation blocks if hex_elev > max(src, tgt) absolute height
    - Woods on the Woodland board have foliageElev=2, so both light and
      heavy woods are checked at terrainEl = hexEl + 2
    - The "affects LOS" gate uses strict > (not >=)
    - Light woods: +1 point (lightWoods). Heavy woods: heavyWoods (×2 in formula)
    - Final: lightWoods + heavyWoods*2 >= 3 → blocked
    """
    max_unit_height = max(src_abs_height, tgt_abs_height)
    total_distance = len(line) - 1
    if total_distance <= 0:
        return 0

    light_woods = 0
    heavy_woods = 0

    for step_idx, (hx, hy) in enumerate(line):
        # Skip endpoints (Java: coords.equals(attackPos/targetPos) → return)
        if step_idx == 0 or step_idx == total_distance:
            continue
        if not board.in_bounds(hx, hy):
            continue

        hex_elev = board.elevation(hx, hy)
        terrain = board.terrain(hx, hy)

        is_adjacent_to_src = (step_idx == 1)
        is_adjacent_to_tgt = (step_idx == total_distance - 1)

        # Bare elevation blocking: hex floor exceeds both units' height
        # Java: totalEl > maxUnitHeight (with adjacent variants)
        blocked = (hex_elev > max_unit_height or
                   (is_adjacent_to_src and hex_elev > src_abs_height) or
                   (is_adjacent_to_tgt and hex_elev > tgt_abs_height))
        if blocked:
            return None  # Hard blocked by hill

        # Woods: foliageElev=2 on Woodland board, so check at hexEl + 2
        # Java: terrainEl = hexEl + 2, affectsLos = terrainEl > maxUnitHeight || adjacency
        if terrain in (Terrain.LIGHT_WOODS, Terrain.HEAVY_WOODS):
            terrain_el = hex_elev + 2
            affects_los = (terrain_el > max_unit_height or
                          (is_adjacent_to_src and terrain_el > src_abs_height) or
                          (is_adjacent_to_tgt and terrain_el > tgt_abs_height))
            if affects_los:
                if terrain == Terrain.LIGHT_WOODS:
                    light_woods += 1
                else:
                    heavy_woods += 1

    # Java: (lightWoods + lightSmoke) + (heavyWoods + heavySmoke) * 2 + ultraWoods * 3 >= 3
    return light_woods + heavy_woods * 2


def compute_los(board: Board, x1: int, y1: int, x2: int, y2: int) -> bool:
    """Check if there is line of sight between two hexes.

    Matches Java MegaMek's LosEffects algorithm:
    1. Trace hex line(s) between source and target
    2. For divided lines (along hex edges), check both sides
    3. Accumulate woods points with elevation gating
    4. Block if woods score >= 3 or elevation hard-blocks

    Assumes standing mechs (height = elevation + 1).
    """
    if x1 == x2 and y1 == y2:
        return True

    src_abs_height = board.elevation(x1, y1) + 1.0
    tgt_abs_height = board.elevation(x2, y2) + 1.0

    line1 = _hex_linedraw(x1, y1, x2, y2)
    score1 = _los_woods_score(board, line1, src_abs_height, tgt_abs_height,
                               x1, y1, x2, y2)

    if _is_divided_line(x1, y1, x2, y2):
        # Check alternate line and take the worse (higher score = more blocking)
        line2 = _hex_linedraw_divided(x1, y1, x2, y2)
        score2 = _los_woods_score(board, line2, src_abs_height, tgt_abs_height,
                                   x1, y1, x2, y2)
        # None means hard-blocked by elevation
        if score1 is None and score2 is None:
            return False
        elif score1 is None:
            return score2 < 3
        elif score2 is None:
            return score1 < 3
        else:
            # Take the worse (higher) score — favor defender
            return max(score1, score2) < 3
    else:
        if score1 is None:
            return False
        return score1 < 3


def compute_terrain_modifier(board: Board, x1: int, y1: int,
                              x2: int, y2: int) -> int:
    """Compute terrain to-hit modifier for firing from (x1,y1) to (x2,y2).

    Woods at the target hex provide cover.
    """
    target_terrain = board.terrain(x2, y2)
    if target_terrain == Terrain.HEAVY_WOODS:
        return 2
    elif target_terrain == Terrain.LIGHT_WOODS:
        return 1

    # Also check for intervening woods (simplified: just target hex)
    return 0


class LosTable:
    """Precomputed LOS lookup table for all hex pairs on a board."""

    def __init__(self, board: Board) -> None:
        self.board = board
        w, h = board.width, board.height
        # Store as flat boolean array for speed
        self._table = [False] * (w * h * w * h)
        self._w = w
        self._h = h
        self._compute()

    def _compute(self) -> None:
        w, h = self._w, self._h
        for y1 in range(h):
            for x1 in range(w):
                idx1 = y1 * w + x1
                for y2 in range(h):
                    for x2 in range(w):
                        idx2 = y2 * w + x2
                        self._table[idx1 * w * h + idx2] = compute_los(
                            self.board, x1, y1, x2, y2
                        )

    def has_los(self, x1: int, y1: int, x2: int, y2: int) -> bool:
        idx1 = y1 * self._w + x1
        idx2 = y2 * self._w + x2
        return self._table[idx1 * self._w * self._h + idx2]
