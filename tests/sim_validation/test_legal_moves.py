"""Tier 1: Legal move set comparison — highest priority validator.

Compares the set of legal moves enumerated by the Python sim's BFS
against the Java MegaMek LongestPathFinder output.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from megamek_gym.sim.board import BOARD
from megamek_gym.sim.movement import enumerate_moves
from tests.sim_validation.reconstruct import reconstruct_unit, extract_unit_state


@dataclass
class LegalMoveResult:
    """Results of comparing legal moves for one observation."""
    step_idx: int = 0
    unit_pos: tuple[int, int, int] = (0, 0, 0)  # x, y, facing
    unit_prone: bool = False
    java_move_count: int = 0
    sim_move_count: int = 0

    # Level 1: reachable hex set {(x, y)}
    java_only_hexes: set[tuple[int, int]] = field(default_factory=set)
    sim_only_hexes: set[tuple[int, int]] = field(default_factory=set)
    shared_hexes: int = 0

    # Level 2: (x, y, facing) tuples
    java_only_moves: set[tuple[int, int, int]] = field(default_factory=set)
    sim_only_moves: set[tuple[int, int, int]] = field(default_factory=set)
    shared_moves: int = 0

    # Level 3: walk/run classification mismatches at shared (x, y, facing)
    walk_run_mismatches: list[str] = field(default_factory=list)

    # Level 4: mp_used disagreements at shared (x, y, facing)
    mp_mismatches: list[str] = field(default_factory=list)

    @property
    def hex_match_rate(self) -> float:
        total = self.shared_hexes + len(self.java_only_hexes) + len(self.sim_only_hexes)
        return self.shared_hexes / total if total > 0 else 1.0

    @property
    def move_match_rate(self) -> float:
        total = self.shared_moves + len(self.java_only_moves) + len(self.sim_only_moves)
        return self.shared_moves / total if total > 0 else 1.0

    @property
    def passed(self) -> bool:
        return (not self.java_only_hexes and not self.sim_only_hexes
                and not self.java_only_moves and not self.sim_only_moves)


@dataclass
class LegalMoveSummary:
    """Aggregate results across all steps."""
    per_step: list[LegalMoveResult] = field(default_factory=list)
    total_java_moves: int = 0
    total_sim_moves: int = 0
    total_shared_hexes: int = 0
    total_java_only_hexes: int = 0
    total_sim_only_hexes: int = 0
    total_shared_moves: int = 0
    total_java_only_moves: int = 0
    total_sim_only_moves: int = 0

    @property
    def hex_match_rate(self) -> float:
        total = self.total_shared_hexes + self.total_java_only_hexes + self.total_sim_only_hexes
        return self.total_shared_hexes / total if total > 0 else 1.0

    @property
    def move_match_rate(self) -> float:
        total = self.total_shared_moves + self.total_java_only_moves + self.total_sim_only_moves
        return self.total_shared_moves / total if total > 0 else 1.0

    @property
    def passed(self) -> bool:
        return self.total_java_only_hexes == 0 and self.total_sim_only_hexes == 0


def validate_legal_moves(java_obs: dict, rl_owner_id: int,
                         step_idx: int = 0) -> LegalMoveResult:
    """Compare Python-enumerated moves against Java legal_moves for one step."""
    result = LegalMoveResult(step_idx=step_idx)

    # Extract Java moves
    java_moves = java_obs.get("legal_moves", [])
    result.java_move_count = len(java_moves)

    if not java_moves:
        return result

    # Get RL unit state from Java obs
    java_unit = extract_unit_state(java_obs, rl_owner_id)
    if java_unit is None:
        return result

    result.unit_pos = (java_unit["x"], java_unit["y"], java_unit["facing"])
    result.unit_prone = java_unit.get("prone", False)

    # Reconstruct sim unit
    sim_unit = reconstruct_unit(java_unit, "Trebuchet TBT-5S")

    # Get enemy for LOS (not needed for move enumeration, but passed through)
    enemy_unit = None
    for u in java_obs.get("units", []):
        if u.get("owner") != rl_owner_id:
            enemy_unit = u
            break

    # Build a minimal enemy Unit for the sim (only position matters for
    # move enumeration, but we pass None since enumerate_moves doesn't use
    # the enemy for movement calculation)
    sim_moves = enumerate_moves(sim_unit, BOARD, enemy=None)
    result.sim_move_count = len(sim_moves)

    # --- Level 1: reachable hex set ---
    java_hexes = {(m["dest_x"], m["dest_y"]) for m in java_moves}
    sim_hexes = {(m["dest_x"], m["dest_y"]) for m in sim_moves}

    result.java_only_hexes = java_hexes - sim_hexes
    result.sim_only_hexes = sim_hexes - java_hexes
    result.shared_hexes = len(java_hexes & sim_hexes)

    # --- Level 2: (x, y, facing) tuples ---
    java_xyf = {(m["dest_x"], m["dest_y"], m["facing"]) for m in java_moves}
    sim_xyf = {(m["dest_x"], m["dest_y"], m["facing"]) for m in sim_moves}

    result.java_only_moves = java_xyf - sim_xyf
    result.sim_only_moves = sim_xyf - java_xyf
    result.shared_moves = len(java_xyf & sim_xyf)

    # --- Level 3: walk/run classification ---
    walk_mp = sim_unit.walk_mp

    # Build lookup: (x, y, facing) -> mp_used for Java
    java_mp = {}
    for m in java_moves:
        key = (m["dest_x"], m["dest_y"], m["facing"])
        mp = m["mp_used"]
        java_mp.setdefault(key, []).append(mp)

    # Build lookup for sim
    sim_mp = {}
    for m in sim_moves:
        key = (m["dest_x"], m["dest_y"], m["facing"])
        mp = m["mp_used"]
        sim_mp.setdefault(key, []).append(mp)

    # Compare at shared moves
    for key in java_xyf & sim_xyf:
        j_mps = sorted(java_mp.get(key, []))
        s_mps = sorted(sim_mp.get(key, []))

        # Walk/run classification
        j_has_walk = any(mp <= walk_mp for mp in j_mps)
        j_has_run = any(mp > walk_mp for mp in j_mps)
        s_has_walk = any(mp <= walk_mp for mp in s_mps)
        s_has_run = any(mp > walk_mp for mp in s_mps)

        if j_has_walk != s_has_walk or j_has_run != s_has_run:
            x, y, f = key
            result.walk_run_mismatches.append(
                f"({x},{y},f={f}): java walk={j_has_walk}/run={j_has_run} "
                f"sim walk={s_has_walk}/run={s_has_run}"
            )

        # --- Level 4: mp_used comparison ---
        if j_mps != s_mps:
            x, y, f = key
            result.mp_mismatches.append(
                f"({x},{y},f={f}): java_mp={j_mps} sim_mp={s_mps}"
            )

    return result


def validate_legal_moves_trace(trace_steps: list, rl_owner_id: int) -> LegalMoveSummary:
    """Validate legal moves across all non-terminal steps in a game trace."""
    summary = LegalMoveSummary()

    for step in trace_steps:
        raw_obs = step.raw_obs
        if raw_obs.get("terminated") or raw_obs.get("truncated"):
            continue
        if not raw_obs.get("legal_moves"):
            continue

        result = validate_legal_moves(raw_obs, rl_owner_id, step.step_idx)
        summary.per_step.append(result)

        summary.total_java_moves += result.java_move_count
        summary.total_sim_moves += result.sim_move_count
        summary.total_shared_hexes += result.shared_hexes
        summary.total_java_only_hexes += len(result.java_only_hexes)
        summary.total_sim_only_hexes += len(result.sim_only_hexes)
        summary.total_shared_moves += result.shared_moves
        summary.total_java_only_moves += len(result.java_only_moves)
        summary.total_sim_only_moves += len(result.sim_only_moves)

    return summary
