"""Tier 2: Princess AI move selection comparison.

Compares the Python sim's simplified Princess heuristic against Java's
actual Princess AI by inferring opponent moves from consecutive observations
and comparing to what the Python heuristic would choose.

This is a measurement test (always passes) — it establishes a baseline
for how often the two AIs agree, not an exact-match requirement.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from megamek_gym.reward import hex_distance
from megamek_gym.sim.board import BOARD
from megamek_gym.sim.los import LosTable
from megamek_gym.sim.movement import enumerate_moves
from megamek_gym.sim.princess import score_move, select_move
from tests.sim_validation.collector import GameTrace
from tests.sim_validation.reconstruct import extract_unit_state, reconstruct_unit


@dataclass
class OpponentMoveObservation:
    """Extracted opponent move from consecutive Java observations."""
    round_num: int
    step_idx: int
    pre_move_x: int
    pre_move_y: int
    pre_move_facing: int
    post_move_x: int
    post_move_y: int
    post_move_facing: int
    rl_x: int  # RL position that Princess sees when deciding
    rl_y: int
    rl_facing: int
    rl_moves_first: bool
    opp_unit_dict: dict  # opponent unit dict at pre-move state
    enemy_unit_dict: dict  # RL unit dict (the "enemy" from opponent's POV)
    friends_coords: tuple[int, int] | None = None  # None in round 1 (Java behavior)


@dataclass
class PrincessComparison:
    """Result of comparing Python vs Java Princess for one round."""
    round_num: int
    java_dest: tuple[int, int, int]  # (x, y, facing)
    python_dest: tuple[int, int, int]  # (x, y, facing)
    python_score_of_java: float | None  # None if java dest not reachable
    python_score_of_python: float
    hex_match: bool
    full_match: bool  # hex + facing
    java_dest_reachable: bool
    n_python_moves: int
    disagree_distance: float  # hex distance between java and python dest
    java_dist_to_enemy: float
    python_dist_to_enemy: float
    skipped: bool = False
    skip_reason: str = ""


@dataclass
class PrincessBehaviorSummary:
    """Aggregate metrics across a full game."""
    total_rounds: int = 0
    compared_rounds: int = 0
    skipped_rounds: int = 0
    skip_reasons: dict[str, int] = field(default_factory=dict)
    hex_match_count: int = 0
    full_match_count: int = 0
    java_dest_reachable_count: int = 0
    java_closer_count: int = 0
    python_closer_count: int = 0
    same_distance_count: int = 0
    total_disagree_distance: float = 0.0
    disagree_count: int = 0
    python_scores_of_java: list[float] = field(default_factory=list)
    python_scores_of_python: list[float] = field(default_factory=list)
    comparisons: list[PrincessComparison] = field(default_factory=list)

    @property
    def hex_match_rate(self) -> float:
        return self.hex_match_count / self.compared_rounds if self.compared_rounds else 0.0

    @property
    def full_match_rate(self) -> float:
        return self.full_match_count / self.compared_rounds if self.compared_rounds else 0.0

    @property
    def mean_disagree_distance(self) -> float:
        return self.total_disagree_distance / self.disagree_count if self.disagree_count else 0.0


def extract_opponent_moves(
    trace: GameTrace,
) -> list[OpponentMoveObservation]:
    """Extract opponent moves from consecutive observations in a Java trace.

    Uses prev_round_enemy_* fields to determine pre/post movement positions.
    For round 1, falls back to units[opp] from the initial observation.
    """
    opp_owner_id = 1 - trace.rl_owner_id
    observations = []
    # Track friends_coords: None in round 1, then opponent's pre-move position
    prev_opp_post_move: tuple[int, int] | None = None

    for i in range(len(trace.steps) - 1):
        curr_step = trace.steps[i]
        next_step = trace.steps[i + 1]
        curr_obs = curr_step.raw_obs
        next_obs = next_step.raw_obs

        # Skip terminal observations
        if curr_obs.get("terminated") or curr_obs.get("truncated"):
            break

        # Get round number from next observation (it reflects state after this round)
        round_num = curr_obs.get("round", 0)

        # Opponent pre-move position: from prev_round_enemy_* in current obs
        # (= post-movement from previous round = pre-movement for this round)
        pre_x = curr_obs.get("prev_round_enemy_x", -1)
        pre_y = curr_obs.get("prev_round_enemy_y", -1)
        pre_facing = curr_obs.get("prev_round_enemy_facing", -1)

        if pre_x < 0 or pre_y < 0:
            # Round 1: use opponent's initial position from this observation
            opp_unit = extract_unit_state(curr_obs, opp_owner_id)
            if opp_unit is None:
                continue
            pre_x = opp_unit["x"]
            pre_y = opp_unit["y"]
            pre_facing = opp_unit["facing"]

        # Opponent post-move position: from prev_round_enemy_* in next obs
        post_x = next_obs.get("prev_round_enemy_x", -1)
        post_y = next_obs.get("prev_round_enemy_y", -1)
        post_facing = next_obs.get("prev_round_enemy_facing", -1)

        if post_x < 0 or post_y < 0:
            # Can't determine where opponent moved — skip
            continue

        # Determine RL position that Princess sees when deciding
        rl_moves_first = curr_obs.get("rl_moves_first", False)
        rl_unit = extract_unit_state(curr_obs, trace.rl_owner_id)
        if rl_unit is None:
            continue

        if rl_moves_first:
            # RL moved before Princess — Princess sees RL's post-move position
            # Use next observation's RL unit position (ground truth from Java)
            next_rl_unit = extract_unit_state(next_obs, trace.rl_owner_id)
            if next_rl_unit:
                rl_x = next_rl_unit["x"]
                rl_y = next_rl_unit["y"]
                rl_facing = next_rl_unit["facing"]
            else:
                rl_x = rl_unit["x"]
                rl_y = rl_unit["y"]
                rl_facing = rl_unit["facing"]
        else:
            # Princess moved before RL — sees RL's pre-move position
            rl_x = rl_unit["x"]
            rl_y = rl_unit["y"]
            rl_facing = rl_unit["facing"]

        # Build opponent unit dict at pre-move state
        # Start from the current observation's opponent data, override position
        opp_unit = extract_unit_state(curr_obs, opp_owner_id)
        if opp_unit is None:
            continue

        # Create a copy with pre-move position
        opp_unit_premove = dict(opp_unit)
        opp_unit_premove["x"] = pre_x
        opp_unit_premove["y"] = pre_y
        opp_unit_premove["facing"] = pre_facing

        # Build RL unit dict at the position Princess sees
        enemy_dict = dict(rl_unit)
        enemy_dict["x"] = rl_x
        enemy_dict["y"] = rl_y
        enemy_dict["facing"] = rl_facing

        observations.append(OpponentMoveObservation(
            round_num=round_num,
            step_idx=i,
            pre_move_x=pre_x,
            pre_move_y=pre_y,
            pre_move_facing=pre_facing,
            post_move_x=post_x,
            post_move_y=post_y,
            post_move_facing=post_facing,
            rl_x=rl_x,
            rl_y=rl_y,
            rl_facing=rl_facing,
            rl_moves_first=rl_moves_first,
            opp_unit_dict=opp_unit_premove,
            enemy_unit_dict=enemy_dict,
            friends_coords=prev_opp_post_move,
        ))
        # After this round, friends_coords = opponent's post-move position
        # (Java's friendsCoords = unit's position at start of NEXT round's evaluation
        #  = where it ended up after moving THIS round)
        prev_opp_post_move = (post_x, post_y)

    return observations


def validate_princess_single_round(
    obs: OpponentMoveObservation,
    board: object,
    board_hexes: list[dict],
    los_table: LosTable,
    template_name: str = "Trebuchet TBT-5S",
) -> PrincessComparison:
    """Compare Python Princess vs Java Princess for one round."""

    # Check if opponent is destroyed or prone (uninformative)
    if obs.opp_unit_dict.get("destroyed", False):
        return PrincessComparison(
            round_num=obs.round_num, java_dest=(0, 0, 0), python_dest=(0, 0, 0),
            python_score_of_java=None, python_score_of_python=0.0,
            hex_match=False, full_match=False, java_dest_reachable=False,
            n_python_moves=0, disagree_distance=0.0,
            java_dist_to_enemy=0.0, python_dist_to_enemy=0.0,
            skipped=True, skip_reason="destroyed",
        )

    if obs.opp_unit_dict.get("prone", False):
        return PrincessComparison(
            round_num=obs.round_num, java_dest=(0, 0, 0), python_dest=(0, 0, 0),
            python_score_of_java=None, python_score_of_python=0.0,
            hex_match=False, full_match=False, java_dest_reachable=False,
            n_python_moves=0, disagree_distance=0.0,
            java_dist_to_enemy=0.0, python_dist_to_enemy=0.0,
            skipped=True, skip_reason="prone",
        )

    # Reconstruct opponent unit at pre-move state
    sim_opp = reconstruct_unit(obs.opp_unit_dict, template_name)

    # Reconstruct RL unit (the "enemy" from opponent's perspective)
    sim_rl = reconstruct_unit(obs.enemy_unit_dict, template_name)

    # Enumerate opponent's legal moves (deque matches Java's LongestPathFinder)
    opp_moves = enumerate_moves(sim_opp, board, sim_rl, algorithm="deque")

    if not opp_moves:
        return PrincessComparison(
            round_num=obs.round_num, java_dest=(0, 0, 0), python_dest=(0, 0, 0),
            python_score_of_java=None, python_score_of_python=0.0,
            hex_match=False, full_match=False, java_dest_reachable=False,
            n_python_moves=0, disagree_distance=0.0,
            java_dist_to_enemy=0.0, python_dist_to_enemy=0.0,
            skipped=True, skip_reason="no_moves",
        )

    # Build obs dicts for princess scoring
    unit_obs = sim_opp.to_obs_dict()
    enemy_obs = sim_rl.to_obs_dict()

    # Run Python Princess
    # If RL moved first, RL is the "enemy" from Princess's POV and has already moved
    enemy_has_moved = obs.rl_moves_first

    # Pre-compute enemy reachable hexes for unmoved enemy evaluation
    enemy_reachable = None
    if not enemy_has_moved:
        rl_moves = enumerate_moves(sim_rl, board, sim_opp, algorithm="deque")
        enemy_reachable = {(m["dest_x"], m["dest_y"]) for m in rl_moves}

    python_idx = select_move(opp_moves, unit_obs, enemy_obs, board, board_hexes, los_table,
                             enemy_has_moved=enemy_has_moved,
                             friends_coords=obs.friends_coords,
                             enemy_reachable_hexes=enemy_reachable)
    python_move = opp_moves[python_idx]
    python_dest = (python_move["dest_x"], python_move["dest_y"], python_move["facing"])
    python_score = score_move(python_move, unit_obs, enemy_obs, board, board_hexes, los_table,
                              enemy_has_moved=enemy_has_moved,
                              friends_coords=obs.friends_coords,
                              enemy_reachable_hexes=enemy_reachable)

    # Java's actual destination
    java_dest = (obs.post_move_x, obs.post_move_y, obs.post_move_facing)

    # Check if Java's destination is in Python's legal move set
    java_dest_reachable = False
    java_score = None
    for m in opp_moves:
        if (m["dest_x"], m["dest_y"], m["facing"]) == java_dest:
            java_dest_reachable = True
            java_score = score_move(m, unit_obs, enemy_obs, board, board_hexes, los_table,
                                    enemy_has_moved=enemy_has_moved,
                                    friends_coords=obs.friends_coords,
                                    enemy_reachable_hexes=enemy_reachable)
            break

    # If exact (x,y,facing) not found, check just hex match
    if not java_dest_reachable:
        for m in opp_moves:
            if m["dest_x"] == java_dest[0] and m["dest_y"] == java_dest[1]:
                java_dest_reachable = True
                java_score = score_move(m, unit_obs, enemy_obs, board, board_hexes, los_table,
                                        enemy_has_moved=enemy_has_moved,
                                        friends_coords=obs.friends_coords,
                                        enemy_reachable_hexes=enemy_reachable)
                break

    hex_match = (python_dest[0], python_dest[1]) == (java_dest[0], java_dest[1])
    full_match = python_dest == java_dest

    disagree_distance = 0.0 if hex_match else hex_distance(
        python_dest[0], python_dest[1], java_dest[0], java_dest[1],
    )

    java_dist = hex_distance(java_dest[0], java_dest[1], obs.rl_x, obs.rl_y)
    python_dist = hex_distance(python_dest[0], python_dest[1], obs.rl_x, obs.rl_y)

    return PrincessComparison(
        round_num=obs.round_num,
        java_dest=java_dest,
        python_dest=python_dest,
        python_score_of_java=java_score,
        python_score_of_python=python_score,
        hex_match=hex_match,
        full_match=full_match,
        java_dest_reachable=java_dest_reachable,
        n_python_moves=len(opp_moves),
        disagree_distance=disagree_distance,
        java_dist_to_enemy=java_dist,
        python_dist_to_enemy=python_dist,
    )


def validate_princess_behavior(
    trace: GameTrace,
    template_name: str = "Trebuchet TBT-5S",
) -> PrincessBehaviorSummary:
    """Compare Python Princess vs Java Princess across a full game trace."""
    board = BOARD
    board_hexes = board.to_obs_hexes()
    los_table = LosTable(board)

    move_obs = extract_opponent_moves(trace)
    summary = PrincessBehaviorSummary(total_rounds=len(move_obs))

    for obs in move_obs:
        comp = validate_princess_single_round(
            obs, board, board_hexes, los_table, template_name,
        )

        if comp.skipped:
            summary.skipped_rounds += 1
            reason = comp.skip_reason
            summary.skip_reasons[reason] = summary.skip_reasons.get(reason, 0) + 1
            summary.comparisons.append(comp)
            continue

        summary.compared_rounds += 1
        summary.comparisons.append(comp)

        if comp.hex_match:
            summary.hex_match_count += 1
        if comp.full_match:
            summary.full_match_count += 1
        if comp.java_dest_reachable:
            summary.java_dest_reachable_count += 1

        # Directional bias
        if comp.java_dist_to_enemy < comp.python_dist_to_enemy:
            summary.java_closer_count += 1
        elif comp.python_dist_to_enemy < comp.java_dist_to_enemy:
            summary.python_closer_count += 1
        else:
            summary.same_distance_count += 1

        # Disagree distance
        if not comp.hex_match:
            summary.disagree_count += 1
            summary.total_disagree_distance += comp.disagree_distance

        # Score tracking
        if comp.python_score_of_java is not None:
            summary.python_scores_of_java.append(comp.python_score_of_java)
        summary.python_scores_of_python.append(comp.python_score_of_python)

    return summary
