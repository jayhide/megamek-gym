"""Line-of-sight computation for hex grids.

Precomputes a LOS table for all hex pairs on the board.
Matches Java MegaMek's LosEffects algorithm: uses geometric hex-line
tracing (ported from IdealHex/Coords.intervening), accumulates woods
points along intervening hexes (with elevation gating), and blocks LOS
when total woods points >= 3.
"""

from __future__ import annotations

import math

from megamek_gym.sim.board import Board, Terrain, neighbor
from megamek_gym.reward import hex_distance


# -- Geometric hex primitives (ported from IdealHex.java / Coords.java) ------

_TAN30 = math.tan(math.pi / 6.0)  # ~0.577350269
_HEX_SIDE = math.pi / 3.0  # 60 degrees in radians

# Vertex offsets relative to hex origin (ox, oy).
# From IdealHex constructor: x[0..5], y[0..5]
_VERTEX_DX = (
    _TAN30,        # v0
    _TAN30 * 3,    # v1
    _TAN30 * 4,    # v2
    _TAN30 * 3,    # v3
    _TAN30,        # v4
    0.0,           # v5
)
_VERTEX_DY = (
    0.0,   # v0
    0.0,   # v1
    1.0,   # v2
    2.0,   # v3
    2.0,   # v4
    1.0,   # v5
)

# Cross-product epsilon matching Java's IdealHex.turns()
_EPS = 0.000001


def _ideal_hex_center(x: int, y: int) -> tuple[float, float]:
    """Continuous-space center of hex (x, y). Matches IdealHex constructor."""
    ox = x * _TAN30 * 3
    oy = y * 2 + (1 if x & 1 else 0)
    return ox + _TAN30 * 2, oy + 1.0


def _is_intersected_by(hx: int, hy: int,
                        x0: float, y0: float,
                        x1: float, y1: float) -> bool:
    """Check if the line from (x0,y0) to (x1,y1) intersects hex (hx,hy).

    Port of IdealHex.isIntersectedBy(). Tests all 6 vertices against the
    line using cross-product sign. If any vertex is STRAIGHT (on the line)
    or has a different sign than the first, the hex is intersected.
    """
    ox = hx * _TAN30 * 3
    oy = hy * 2 + (1 if hx & 1 else 0)

    # Inline cross-product for first vertex
    vx = ox + _VERTEX_DX[0]
    vy = oy + _VERTEX_DY[0]
    cross = (x1 - x0) * (vy - y0) - (vx - x0) * (y1 - y0)
    if cross > _EPS:
        side1 = 1
    elif cross < -_EPS:
        side1 = -1
    else:
        return True  # STRAIGHT

    for i in range(1, 6):
        vx = ox + _VERTEX_DX[i]
        vy = oy + _VERTEX_DY[i]
        cross = (x1 - x0) * (vy - y0) - (vx - x0) * (y1 - y0)
        if cross > _EPS:
            side = 1
        elif cross < -_EPS:
            side = -1
        else:
            return True  # STRAIGHT
        if side != side1:
            return True

    return False


def _radian(x1: int, y1: int, x2: int, y2: int) -> float:
    """Angle in radians from hex (x1,y1) to hex (x2,y2).

    Port of Coords.radian(). Uses IdealHex centers.
    """
    src_cx, src_cy = _ideal_hex_center(x1, y1)
    dst_cx, dst_cy = _ideal_hex_center(x2, y2)

    if src_cy == dst_cy:
        return math.pi / 2.0 if src_cx < dst_cx else math.pi * 1.5

    r = math.atan((dst_cx - src_cx) / (src_cy - dst_cy))
    if src_cy < dst_cy:
        r = (r + math.pi) % (math.pi * 2)
    if r < 0:
        r += math.pi * 2
    return r


def _degree(x1: int, y1: int, x2: int, y2: int) -> int:
    """Integer degree angle from hex (x1,y1) to hex (x2,y2).

    Port of Coords.degree().
    """
    return int(round((180.0 / math.pi) * _radian(x1, y1, x2, y2)))


def _direction(x1: int, y1: int, x2: int, y2: int) -> int:
    """Primary direction (0-5) from hex (x1,y1) to hex (x2,y2).

    Port of Coords.direction().
    """
    return int(round(_radian(x1, y1, x2, y2) / _HEX_SIDE)) % 6


# -- Geometric hex-line tracing (port of Coords.intervening/nextHex) ---------

def _intervening(x1: int, y1: int, x2: int, y2: int) -> list[tuple[int, int]]:
    """Trace the hex line from (x1,y1) to (x2,y2) using geometric intersection.

    Port of Coords.intervening(src, dest, false). Returns ordered list of
    hex coordinates including endpoints. When the line clips a hex boundary,
    the side neighbors are checked before center (matching Java's priority).
    """
    if x1 == x2 and y1 == y2:
        return [(x1, y1)]

    src_cx, src_cy = _ideal_hex_center(x1, y1)
    dst_cx, dst_cy = _ideal_hex_center(x2, y2)

    center_dir = _direction(x1, y1, x2, y2)
    # Java checks sides first: (center+1)%6 and (center+5)%6, then center
    dirs = [
        (center_dir + 1) % 6,  # right side
        (center_dir + 5) % 6,  # left side
        center_dir,             # center last
    ]

    result = [(x1, y1)]
    cx, cy = x1, y1
    max_iter = abs(x2 - x1) + abs(y2 - y1) + max(abs(x2 - x1), abs(y2 - y1)) + 2
    for _ in range(max_iter):
        if cx == x2 and cy == y2:
            break
        found = False
        for d in dirs:
            nx, ny = neighbor(cx, cy, d)
            if _is_intersected_by(nx, ny, src_cx, src_cy, dst_cx, dst_cy):
                result.append((nx, ny))
                cx, cy = nx, ny
                found = True
                break
        if not found:
            # Fallback: move in center direction
            nx, ny = neighbor(cx, cy, center_dir)
            result.append((nx, ny))
            cx, cy = nx, ny

    return result


def _intervening_split(x1: int, y1: int, x2: int, y2: int) -> list[tuple[int, int]]:
    """Trace hex line in split mode for divided lines.

    Port of Coords.intervening(src, dest, true). Identical walk to
    _intervening() but with a tiny radian nudge on centerDirection for
    reliable left-before-right ordering. The triplet pattern [side1, side2,
    center, side1, side2, center, ...] emerges naturally from the geometry:
    when the line passes between two hexes, both are intersected and get
    visited consecutively before the algorithm advances to the center hex.
    """
    if x1 == x2 and y1 == y2:
        return [(x1, y1)]

    src_cx, src_cy = _ideal_hex_center(x1, y1)
    dst_cx, dst_cy = _ideal_hex_center(x2, y2)

    # Java's split-mode hack: slight radian nudge for left-before-right ordering
    rad = _radian(x1, y1, x2, y2)
    center_dir = int(round(rad + 0.0001 / _HEX_SIDE)) % 6

    dirs = [
        (center_dir + 1) % 6,  # side 1
        (center_dir + 5) % 6,  # side 2
        center_dir,             # center last
    ]

    result = [(x1, y1)]
    cx, cy = x1, y1
    max_iter = abs(x2 - x1) + abs(y2 - y1) + max(abs(x2 - x1), abs(y2 - y1)) + 5
    for _ in range(max_iter):
        if cx == x2 and cy == y2:
            break
        found = False
        for d in dirs:
            nx, ny = neighbor(cx, cy, d)
            if _is_intersected_by(nx, ny, src_cx, src_cy, dst_cx, dst_cy):
                result.append((nx, ny))
                cx, cy = nx, ny
                found = True
                break
        if not found:
            nx, ny = neighbor(cx, cy, center_dir)
            result.append((nx, ny))
            cx, cy = nx, ny

    return result


# -- LOS computation matching Java's LosEffects ------------------------------

def _los_hex_score(board: Board, hx: int, hy: int,
                   src_abs_height: float, tgt_abs_height: float,
                   max_unit_height: float,
                   x1: int, y1: int, x2: int, y2: int) -> tuple[bool, int, int]:
    """Evaluate a single intervening hex for LOS effects.

    Uses actual hex distance for adjacency (matching Java's
    LosEffects.losForCoords which checks attackPos.distance(coords) == 1).
    Returns (hard_blocked, light_woods_count, heavy_woods_count).
    """
    if not board.in_bounds(hx, hy):
        return False, 0, 0

    hex_elev = board.elevation(hx, hy)
    attacker_adjacent = hex_distance(x1, y1, hx, hy) == 1
    target_adjacent = hex_distance(x2, y2, hx, hy) == 1

    # Bare elevation blocking
    if (hex_elev > max_unit_height or
            (attacker_adjacent and hex_elev > src_abs_height) or
            (target_adjacent and hex_elev > tgt_abs_height)):
        return True, 0, 0

    terrain = board.terrain(hx, hy)
    light = 0
    heavy = 0
    if terrain in (Terrain.LIGHT_WOODS, Terrain.HEAVY_WOODS):
        terrain_el = hex_elev + 2
        affects = (terrain_el > max_unit_height or
                   (attacker_adjacent and terrain_el > src_abs_height) or
                   (target_adjacent and terrain_el > tgt_abs_height))
        if affects:
            if terrain == Terrain.LIGHT_WOODS:
                light = 1
            else:
                heavy = 1

    return False, light, heavy


def _los_woods_score(board: Board, line: list[tuple[int, int]],
                     src_abs_height: float, tgt_abs_height: float,
                     x1: int, y1: int, x2: int, y2: int) -> int | None:
    """Compute accumulated woods score along a straight hex line.

    Returns None if hard-blocked by elevation. Otherwise returns the total
    woods score (lightWoods + heavyWoods * 2).
    """
    max_unit_height = max(src_abs_height, tgt_abs_height)
    total_distance = len(line) - 1
    if total_distance <= 0:
        return 0

    light_woods = 0
    heavy_woods = 0

    for step_idx, (hx, hy) in enumerate(line):
        # Skip endpoints
        if step_idx == 0 or step_idx == total_distance:
            continue

        blocked, lw, hw = _los_hex_score(
            board, hx, hy, src_abs_height, tgt_abs_height,
            max_unit_height, x1, y1, x2, y2)
        if blocked:
            return None
        light_woods += lw
        heavy_woods += hw

    return light_woods + heavy_woods * 2


def _los_divided(board: Board, triplets: list[tuple[int, int]],
                 src_abs_height: float, tgt_abs_height: float,
                 x1: int, y1: int, x2: int, y2: int) -> bool:
    """Evaluate divided-line LOS using Java's triplet structure.

    Port of LosEffects.losDivided(). The triplet list from
    _intervening_split has structure:
    - Index 0: source
    - Then groups of 3: [left_split, right_split, non_split]

    For non-split hexes, both paths accumulate equally.
    For split pairs, left path sees left hex, right path sees right hex.
    Final: take worse (higher) accumulated score. If both paths hard-blocked,
    no LOS. If one path hard-blocked, use the other.
    """
    max_unit_height = max(src_abs_height, tgt_abs_height)
    n = len(triplets)

    # Count "real" hexes to determine adjacency
    # The total path length (for adjacency checks) is based on non-duplicate
    # hex count along either path. We approximate using the triplet count:
    # Each group of 3 after src represents one step. Total steps = (n-1)//3
    total_steps = (n - 1) // 3 if n > 1 else 0
    if total_steps <= 0:
        return True

    left_light = 0
    left_heavy = 0
    left_blocked = False
    right_light = 0
    right_heavy = 0
    right_blocked = False

    # Process non-split hexes first (indices 3, 6, 9, ...)
    # Java: for (int i = 3; i < in.size() - 2; i += 3)
    # These affect both paths equally
    for group_idx in range(total_steps):
        non_split_idx = 3 + group_idx * 3
        if non_split_idx >= n - 2:  # Java: i < in.size() - 2
            break
        hx, hy = triplets[non_split_idx]
        if (hx, hy) == (x1, y1) or (hx, hy) == (x2, y2):
            continue

        blocked, lw, hw = _los_hex_score(
            board, hx, hy, src_abs_height, tgt_abs_height,
            max_unit_height, x1, y1, x2, y2)
        if blocked:
            left_blocked = True
            right_blocked = True
        else:
            left_light += lw
            left_heavy += hw
            right_light += lw
            right_heavy += hw

    # If already fully blocked by non-split hexes, no LOS
    if left_blocked and right_blocked:
        return False

    # Process split pairs (indices 1-2, 4-5, 7-8, ...)
    # Java: for (int i = 1; i < in.size() - 2; i += 3)
    for group_idx in range(total_steps):
        left_idx = 1 + group_idx * 3
        right_idx = 2 + group_idx * 3
        if left_idx >= n - 2:  # Java: i < in.size() - 2
            break

        lhx, lhy = triplets[left_idx]
        rhx, rhy = triplets[right_idx]

        # Left path
        if not left_blocked:
            if (lhx, lhy) != (x1, y1) and (lhx, lhy) != (x2, y2):
                bl, lw, hw = _los_hex_score(
                    board, lhx, lhy, src_abs_height, tgt_abs_height,
                    max_unit_height, x1, y1, x2, y2)
                if bl:
                    left_blocked = True
                else:
                    left_light += lw
                    left_heavy += hw

        # Right path
        if not right_blocked:
            if (rhx, rhy) != (x1, y1) and (rhx, rhy) != (x2, y2):
                bl, lw, hw = _los_hex_score(
                    board, rhx, rhy, src_abs_height, tgt_abs_height,
                    max_unit_height, x1, y1, x2, y2)
                if bl:
                    right_blocked = True
                else:
                    right_light += lw
                    right_heavy += hw

    # Java combines: los (non-split) + worse-of(totalLeft, totalRight)
    # The worse side is determined by losModifiers().getValue():
    # - IMPOSSIBLE if blocked or woods score > 2
    # - Otherwise the sum of modifiers (lightWoods + heavyWoods * 2)
    # Then the combined score is checked against the threshold.

    # Compute split-only scores (before combining with non-split)
    left_split_score = left_light + left_heavy * 2 if not left_blocked else None
    right_split_score = right_light + right_heavy * 2 if not right_blocked else None

    # Determine which side is worse (Java: lVal > rVal picks left)
    # None (blocked) is always worse than any numeric score
    if left_split_score is None and right_split_score is None:
        # Both split paths blocked → use either (both blocked)
        worse_light, worse_heavy, worse_blocked = left_light, left_heavy, True
    elif left_split_score is None:
        worse_light, worse_heavy, worse_blocked = left_light, left_heavy, True
    elif right_split_score is None:
        worse_light, worse_heavy, worse_blocked = right_light, right_heavy, True
    elif left_split_score > right_split_score:
        worse_light, worse_heavy, worse_blocked = left_light, left_heavy, False
    else:
        worse_light, worse_heavy, worse_blocked = right_light, right_heavy, False

    # Combine non-split + worse split
    if left_blocked and right_blocked and worse_blocked:
        return False

    # If the non-split portion blocked both paths, no LOS
    total_blocked = (left_blocked and right_blocked)
    if total_blocked:
        return False

    # Combined score = non-split woods + worse-side split woods
    # (non-split already accumulated into both left/right above)
    # Actually, we need the non-split contribution separately.
    # Let's extract: left = non_split + left_split, right = non_split + right_split
    # But we already accumulated non-split into both left and right.
    # So left_light/left_heavy already include non-split.
    # The "worse side" was chosen based on split-only scores though.
    # We need to add the worse split to the non-split base.

    # Recompute: non-split is shared, we need it separately
    # Actually the current code accumulates non-split into BOTH left/right.
    # So the total for each path = non_split + split.
    # Java: los (non-split) + worse(totalLeft, totalRight)
    # = non_split + worse(left_split, right_split)
    # = worse(non_split + left_split, non_split + right_split)
    # Which is the same as max(left_total, right_total).

    left_total = (left_light + left_heavy * 2) if not left_blocked else None
    right_total = (right_light + right_heavy * 2) if not right_blocked else None

    if left_total is None and right_total is None:
        return False
    elif left_total is None:
        return right_total < 3
    elif right_total is None:
        return left_total < 3
    else:
        return max(left_total, right_total) < 3


def compute_los(board: Board, x1: int, y1: int, x2: int, y2: int) -> bool:
    """Check if there is line of sight between two hexes.

    Matches Java MegaMek's LosEffects algorithm using geometric hex-line
    tracing (ported from IdealHex/Coords.intervening):
    1. Compute degree angle between hex centers
    2. If degree % 60 == 30: divided line — trace with split mode, evaluate
       per-segment taking worse of left/right path
    3. Otherwise: straight line — trace and accumulate woods score
    4. Block if woods score >= 3 or elevation hard-blocks

    Assumes standing mechs (height = elevation + 1).
    """
    if x1 == x2 and y1 == y2:
        return True

    src_abs_height = board.elevation(x1, y1) + 1.0
    tgt_abs_height = board.elevation(x2, y2) + 1.0

    deg = _degree(x1, y1, x2, y2)
    if deg % 60 == 30:
        triplets = _intervening_split(x1, y1, x2, y2)
        return _los_divided(board, triplets, src_abs_height, tgt_abs_height,
                            x1, y1, x2, y2)
    else:
        line = _intervening(x1, y1, x2, y2)
        score = _los_woods_score(board, line, src_abs_height, tgt_abs_height,
                                  x1, y1, x2, y2)
        if score is None:
            return False
        return score < 3


def _terrain_modifier_for_line(board: Board, line: list[tuple[int, int]],
                                src_abs_height: float, tgt_abs_height: float,
                                x1: int, y1: int, x2: int, y2: int) -> int:
    """Compute terrain to-hit modifier along a straight hex line."""
    max_unit_height = max(src_abs_height, tgt_abs_height)
    total_distance = len(line) - 1
    modifier = 0

    for step_idx, (hx, hy) in enumerate(line):
        if step_idx == 0:
            continue
        if not board.in_bounds(hx, hy):
            continue

        hex_elev = board.elevation(hx, hy)
        terrain = board.terrain(hx, hy)

        if terrain in (Terrain.LIGHT_WOODS, Terrain.HEAVY_WOODS):
            terrain_el = hex_elev + 2
            is_adjacent_to_src = (step_idx == 1)
            is_adjacent_to_tgt = (step_idx == total_distance - 1)
            is_target = (step_idx == total_distance)
            affects = (terrain_el > max_unit_height or
                       is_target or
                       (is_adjacent_to_src and terrain_el > src_abs_height) or
                       (is_adjacent_to_tgt and terrain_el > tgt_abs_height))
            if affects:
                modifier += 1 if terrain == Terrain.LIGHT_WOODS else 2

    return modifier


def _terrain_modifier_divided(board: Board, triplets: list[tuple[int, int]],
                               src_abs_height: float, tgt_abs_height: float,
                               x1: int, y1: int, x2: int, y2: int) -> int:
    """Compute terrain modifier for divided lines — take worse of two paths."""
    max_unit_height = max(src_abs_height, tgt_abs_height)
    n = len(triplets)
    total_steps = (n - 1) // 3 if n > 1 else 0
    if total_steps <= 0:
        return 0

    left_mod = 0
    right_mod = 0

    # Non-split hexes (shared by both paths)
    for group_idx in range(total_steps):
        non_split_idx = 3 + group_idx * 3
        if non_split_idx >= n:
            break
        hx, hy = triplets[non_split_idx]
        if (hx, hy) == (x1, y1):
            continue
        if not board.in_bounds(hx, hy):
            continue
        hex_elev = board.elevation(hx, hy)
        terrain = board.terrain(hx, hy)
        if terrain in (Terrain.LIGHT_WOODS, Terrain.HEAVY_WOODS):
            terrain_el = hex_elev + 2
            is_target = ((hx, hy) == (x2, y2))
            step_in_path = group_idx + 1
            is_adj_src = (step_in_path == 1)
            is_adj_tgt = (step_in_path == total_steps)
            affects = (terrain_el > max_unit_height or
                       is_target or
                       (is_adj_src and terrain_el > src_abs_height) or
                       (is_adj_tgt and terrain_el > tgt_abs_height))
            if affects:
                val = 1 if terrain == Terrain.LIGHT_WOODS else 2
                left_mod += val
                right_mod += val

    # Split pairs
    for group_idx in range(total_steps):
        left_idx = 1 + group_idx * 3
        right_idx = 2 + group_idx * 3
        if right_idx >= n:
            break

        step_in_path = group_idx + 1
        is_adj_src = (step_in_path == 1)
        is_adj_tgt = (step_in_path == total_steps)

        for side_idx, side_mod_ref in [(left_idx, 'left'), (right_idx, 'right')]:
            hx, hy = triplets[side_idx]
            if (hx, hy) == (x1, y1):
                continue
            if not board.in_bounds(hx, hy):
                continue
            hex_elev = board.elevation(hx, hy)
            terrain = board.terrain(hx, hy)
            if terrain in (Terrain.LIGHT_WOODS, Terrain.HEAVY_WOODS):
                terrain_el = hex_elev + 2
                is_target = ((hx, hy) == (x2, y2))
                affects = (terrain_el > max_unit_height or
                           is_target or
                           (is_adj_src and terrain_el > src_abs_height) or
                           (is_adj_tgt and terrain_el > tgt_abs_height))
                if affects:
                    val = 1 if terrain == Terrain.LIGHT_WOODS else 2
                    if side_mod_ref == 'left':
                        left_mod += val
                    else:
                        right_mod += val

    # Defender's choice: take worse (higher) modifier
    return max(left_mod, right_mod)


def compute_terrain_modifier(board: Board, x1: int, y1: int,
                              x2: int, y2: int) -> int:
    """Compute terrain to-hit modifier for firing from (x1,y1) to (x2,y2).

    Counts intervening and target hex woods along the LOS path.
    Each light woods = +1, heavy woods = +2.
    Skips the attacker hex. Includes the target hex.
    For divided lines, takes the worse (higher) modifier of both paths.
    """
    if x1 == x2 and y1 == y2:
        return 0

    src_abs_height = board.elevation(x1, y1) + 1.0
    tgt_abs_height = board.elevation(x2, y2) + 1.0

    deg = _degree(x1, y1, x2, y2)
    if deg % 60 == 30:
        triplets = _intervening_split(x1, y1, x2, y2)
        return _terrain_modifier_divided(board, triplets, src_abs_height,
                                          tgt_abs_height, x1, y1, x2, y2)
    else:
        line = _intervening(x1, y1, x2, y2)
        return _terrain_modifier_for_line(board, line, src_abs_height,
                                           tgt_abs_height, x1, y1, x2, y2)


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
