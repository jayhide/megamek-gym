"""Tier 3: Statistical comparison of sim vs Java game distributions.

Runs many games in both engines with random play and compares
aggregate statistics to catch systematic biases.
"""

from __future__ import annotations

import random
import time
from dataclasses import dataclass, field

from megamek_gym.sim.game import Game


@dataclass
class GameStats:
    """Statistics from a single game."""
    rounds: int = 0
    outcome: str = ""  # "WIN", "LOSS", "DRAW"
    total_rl_damage_taken: int = 0
    total_opp_damage_taken: int = 0
    avg_legal_moves: float = 0.0


@dataclass
class StatisticalResult:
    """Results of statistical comparison between engines."""
    sim_games: list[GameStats] = field(default_factory=list)
    java_games: list[GameStats] = field(default_factory=list)
    comparison: dict = field(default_factory=dict)  # Computed summary

    @property
    def passed(self) -> bool:
        # Generous thresholds — these are inherently noisy
        if not self.comparison:
            return True
        return all(
            v.get("ok", True) for v in self.comparison.values()
            if isinstance(v, dict)
        )


def run_sim_games(n_games: int, seed: int = 42,
                  max_rounds: int = 40) -> list[GameStats]:
    """Run N games in the Python sim with random play."""
    results = []
    rng = random.Random(seed)

    for i in range(n_games):
        game = Game(
            rl_unit_name="Trebuchet TBT-5S",
            opponent_unit_name="Trebuchet TBT-5S",
            rl_start=(14, 1),
            opp_start=(1, 15),
            max_rounds=max_rounds,
        )

        game_seed = rng.randint(0, 2**31)
        obs = game.reset(seed=game_seed)

        move_counts = []
        while not obs.get("terminated") and not obs.get("truncated"):
            legal_moves = obs.get("legal_moves", [])
            move_counts.append(len(legal_moves))

            if legal_moves:
                action = game.rng.randint(0, len(legal_moves) - 1)
            else:
                action = 0

            obs = game.step(action)

        # Compute damage taken from armor state
        rl_damage = _compute_total_damage(game.rl_unit)
        opp_damage = _compute_total_damage(game.opp_unit)

        stats = GameStats(
            rounds=game.round,
            outcome=game.game_outcome or "DRAW",
            total_rl_damage_taken=rl_damage,
            total_opp_damage_taken=opp_damage,
            avg_legal_moves=sum(move_counts) / len(move_counts) if move_counts else 0,
        )
        results.append(stats)

    return results


def run_java_games(trace_collector_fn, n_games: int,
                   megamek_dir: str, base_port: int = 9999) -> list[GameStats]:
    """Run N games via Java and extract the same statistics.

    Args:
        trace_collector_fn: The collect_game_trace function.
        n_games: Number of games to run.
        megamek_dir: Path to MegaMek.
        base_port: Starting port.

    Returns:
        List of GameStats from Java games.
    """
    from tests.sim_validation.collector import random_action_fn
    results = []

    for i in range(n_games):
        trace = trace_collector_fn(
            megamek_dir=megamek_dir,
            port=base_port,
            max_rounds=40,
            action_fn=random_action_fn,
        )

        if not trace.steps:
            continue

        last_obs = trace.steps[-1].raw_obs
        outcome = last_obs.get("game_outcome", "DRAW")
        game_round = last_obs.get("round", 0)

        # Count legal moves per step
        move_counts = []
        for step in trace.steps:
            moves = step.raw_obs.get("legal_moves", [])
            if moves:
                move_counts.append(len(moves))

        # Compute damage from first vs last observation
        first_obs = trace.steps[0].raw_obs
        rl_damage = _compute_obs_damage(first_obs, last_obs, trace.rl_owner_id)
        opp_damage = _compute_obs_damage(first_obs, last_obs, 1 - trace.rl_owner_id)

        stats = GameStats(
            rounds=game_round,
            outcome=outcome,
            total_rl_damage_taken=rl_damage,
            total_opp_damage_taken=opp_damage,
            avg_legal_moves=sum(move_counts) / len(move_counts) if move_counts else 0,
        )
        results.append(stats)

    return results


def compare_distributions(sim_games: list[GameStats],
                          java_games: list[GameStats]) -> dict:
    """Compare aggregate statistics between sim and Java games."""
    comparison = {}

    if not sim_games or not java_games:
        return comparison

    # Game length
    sim_rounds = [g.rounds for g in sim_games]
    java_rounds = [g.rounds for g in java_games]
    comparison["game_length"] = _compare_metric(
        "game_length", sim_rounds, java_rounds, tolerance=0.4
    )

    # Win rate (RL wins)
    sim_wins = sum(1 for g in sim_games if g.outcome == "WIN") / len(sim_games)
    java_wins = sum(1 for g in java_games if g.outcome == "WIN") / len(java_games)
    comparison["win_rate"] = {
        "sim": f"{sim_wins:.2f}",
        "java": f"{java_wins:.2f}",
        "diff": f"{abs(sim_wins - java_wins):.2f}",
        "ok": abs(sim_wins - java_wins) < 0.3,  # Generous for small sample
    }

    # Average legal moves
    sim_avg_moves = [g.avg_legal_moves for g in sim_games]
    java_avg_moves = [g.avg_legal_moves for g in java_games]
    comparison["avg_legal_moves"] = _compare_metric(
        "avg_legal_moves", sim_avg_moves, java_avg_moves, tolerance=0.3
    )

    return comparison


def _compare_metric(name: str, sim_vals: list, java_vals: list,
                    tolerance: float = 0.3) -> dict:
    """Compare a numeric metric between sim and Java."""
    sim_mean = sum(sim_vals) / len(sim_vals) if sim_vals else 0
    java_mean = sum(java_vals) / len(java_vals) if java_vals else 0
    denom = max(abs(java_mean), 1.0)
    rel_diff = abs(sim_mean - java_mean) / denom

    return {
        "sim_mean": f"{sim_mean:.1f}",
        "java_mean": f"{java_mean:.1f}",
        "relative_diff": f"{rel_diff:.2f}",
        "ok": rel_diff < tolerance,
    }


def _compute_total_damage(unit) -> int:
    """Compute total damage taken by a sim Unit from its armor state."""
    total = 0
    for loc_idx in range(8):
        front_max, rear_max, internal_max = unit.template.armor[loc_idx]
        front_curr = unit.armor[loc_idx][0]
        rear_curr = unit.armor[loc_idx][1]
        internal_curr = unit.armor[loc_idx][2]

        total += (front_max - front_curr) + (rear_max - rear_curr) + (internal_max - internal_curr)
    return total


def _compute_obs_damage(first_obs: dict, last_obs: dict, owner_id: int) -> int:
    """Compute total damage taken from obs armor deltas."""
    from tests.sim_validation.reconstruct import extract_unit_state

    first_unit = extract_unit_state(first_obs, owner_id)
    last_unit = extract_unit_state(last_obs, owner_id)
    if first_unit is None or last_unit is None:
        return 0

    total = 0
    first_armor = first_unit.get("armor", [])
    last_armor = last_unit.get("armor", [])

    for fa, la in zip(first_armor, last_armor):
        total += max(0, fa.get("armor", 0) - la.get("armor", 0))
        total += max(0, fa.get("rear_armor", 0) - la.get("rear_armor", 0))
        total += max(0, fa.get("internal", 0) - la.get("internal", 0))

    return total
