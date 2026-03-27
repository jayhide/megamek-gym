"""Simplified Princess AI opponent for movement and firing decisions."""

from __future__ import annotations

from megamek_gym.reward import cover_value, hex_distance, range_quality
from megamek_gym.sim.board import Board
from megamek_gym.sim.los import LosTable


def score_move(move: dict, unit_obs: dict, enemy_obs: dict,
               board: Board, board_hexes: list[dict],
               los_table: LosTable) -> float:
    """Score a move for the Princess AI.

    Higher score = better move. Considers:
    - Range quality (weapon effectiveness at destination)
    - Cover (woods at destination)
    - LOS to enemy
    - TMM (hexes moved ~ mp_used as proxy)
    """
    dest_x = move["dest_x"]
    dest_y = move["dest_y"]
    facing = move["facing"]
    mp_used = move["mp_used"]

    ex = enemy_obs.get("x", -1)
    ey = enemy_obs.get("y", -1)

    if ex < 0 or ey < 0:
        return 0.0

    score = 0.0

    # Range quality: how well our weapons perform from this position
    dist = hex_distance(dest_x, dest_y, ex, ey)
    rq = range_quality(
        unit_obs, dist,
        target_x=ex, target_y=ey,
        unit_x=dest_x, unit_y=dest_y,
        unit_facing=facing,
    )
    score += rq * 3.0  # Weight range quality heavily

    # Enemy range quality (lower is better for us)
    enemy_facing = enemy_obs.get("facing", 0)
    erq = range_quality(
        enemy_obs, dist,
        target_x=dest_x, target_y=dest_y,
        unit_x=ex, unit_y=ey,
        unit_facing=enemy_facing,
    )
    score -= erq * 1.0

    # Cover bonus
    cv = cover_value(board_hexes, dest_x, dest_y)
    score += cv * 0.5

    # LOS: prefer positions with LOS to enemy
    if los_table.has_los(dest_x, dest_y, ex, ey):
        score += 1.0

    # TMM bonus (more movement = harder to hit)
    # Rough proxy: mp_used correlates with hexes moved
    tmm_proxy = min(mp_used, 9) / 9.0
    score += tmm_proxy * 0.5

    # Elevation advantage
    dest_elev = board.elevation(dest_x, dest_y)
    enemy_elev = board.elevation(ex, ey)
    score += (dest_elev - enemy_elev) * 0.3

    # Small preference for not running (no +2 to-hit penalty on own weapons)
    walk_mp = unit_obs.get("mp_walk", 5)
    if mp_used > walk_mp:
        score -= 0.3

    return score


def select_move(moves: list[dict], unit_obs: dict, enemy_obs: dict,
                board: Board, board_hexes: list[dict],
                los_table: LosTable) -> int:
    """Select the best move index for Princess AI.

    Returns the index of the selected move.
    """
    if not moves:
        return 0

    best_idx = 0
    best_score = float("-inf")

    for i, move in enumerate(moves):
        s = score_move(move, unit_obs, enemy_obs, board, board_hexes, los_table)
        if s > best_score:
            best_score = s
            best_idx = i

    return best_idx
