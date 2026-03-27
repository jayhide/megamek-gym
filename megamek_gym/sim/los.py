"""Line-of-sight computation for hex grids.

Precomputes a LOS table for all hex pairs on the board.
Simplified algorithm: trace a hex line between source and target,
check for blocking terrain along the way.
"""

from __future__ import annotations

import math

from megamek_gym.sim.board import Board, Terrain


def _hex_linedraw(x1: int, y1: int, x2: int, y2: int) -> list[tuple[int, int]]:
    """Draw a line between two hexes using cube coordinate interpolation.

    Returns list of hex coordinates (inclusive of endpoints).
    """
    # Convert to cube coordinates
    q1, r1, s1 = _to_cube(x1, y1)
    q2, r2, s2 = _to_cube(x2, y2)

    n = max(abs(q2 - q1), abs(r2 - r1), abs(s2 - s1))
    if n == 0:
        return [(x1, y1)]

    results = []
    for i in range(n + 1):
        t = i / n
        # Interpolate in cube coordinates
        fq = q1 + (q2 - q1) * t
        fr = r1 + (r2 - r1) * t
        fs = s1 + (s2 - s1) * t
        # Round to nearest hex
        rq, rr, rs = _cube_round(fq, fr, fs)
        # Convert back to offset
        ox, oy = _from_cube(rq, rr)
        results.append((ox, oy))

    return results


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


def compute_los(board: Board, x1: int, y1: int, x2: int, y2: int) -> bool:
    """Check if there is line of sight between two hexes.

    Assumes standing mechs (height = elevation + 1 for mech eye level).
    LOS is blocked if intervening terrain/elevation blocks the sight line.
    """
    if x1 == x2 and y1 == y2:
        return True

    line = _hex_linedraw(x1, y1, x2, y2)
    if len(line) <= 2:
        return True  # Adjacent hexes always have LOS

    src_height = board.elevation(x1, y1) + 1.0  # Mech eye level
    tgt_height = board.elevation(x2, y2) + 1.0

    # Check each intermediate hex (not endpoints)
    for hx, hy in line[1:-1]:
        if not board.in_bounds(hx, hy):
            continue

        hex_elev = board.elevation(hx, hy)
        terrain = board.terrain(hx, hy)

        # Terrain adds effective height
        if terrain == Terrain.HEAVY_WOODS:
            blocking_height = hex_elev + 2.0
        elif terrain == Terrain.LIGHT_WOODS:
            blocking_height = hex_elev + 1.0
        else:
            blocking_height = hex_elev + 0.0

        # Check if this hex blocks the sight line
        # Interpolate the sight line height at this hex's position
        dist_total = len(line) - 1
        dist_to_hex = line.index((hx, hy))
        t = dist_to_hex / dist_total

        sight_height = src_height + (tgt_height - src_height) * t

        if blocking_height > sight_height:
            return False

    return True


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
