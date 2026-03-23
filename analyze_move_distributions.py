#!/usr/bin/env python3
"""Analyze legal move set distributions across multiple games.

Plays N games with random actions, collecting raw legal move data at every
movement step. Computes comprehensive statistics and generates matplotlib
visualizations.

Usage:
    poetry run python analyze_move_distributions.py --megamek-dir ../megamek --games 5
    poetry run python analyze_move_distributions.py --megamek-dir ../megamek --games 10 --output moves.png
"""

import argparse
import time
from collections import defaultdict
from dataclasses import dataclass, field

import gymnasium
import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

import megamek_gym  # noqa: F401, E402
from megamek_gym.config import MegaMekConfig  # noqa: E402

# Steps with n_legal <= this are considered "immobile" (prone, shutdown, etc.)
MOBILE_THRESHOLD = 10


@dataclass
class StepRecord:
    game_idx: int = 0
    step: int = 0
    game_round: int = 0
    phase: str = ""
    n_legal: int = 0
    unique_dest_hexes: int = 0
    board_total: int = 272
    reachable_pct: float = 0.0
    # Ground movement categories (disjoint partition of ground hexes)
    walk_only_hexes: int = 0
    run_only_hexes: int = 0
    walk_and_run_hexes: int = 0
    # Jump categories
    jump_only_hexes: int = 0
    jump_total_hexes: int = 0
    # Facing coverage (walk moves only)
    all_facings_walk: int = 0
    partial_facings_walk: int = 0
    # All-mode facing coverage
    all_facings_any: int = 0
    partial_facings_any: int = 0
    # Counts by move type
    n_walk: int = 0
    n_run: int = 0
    n_jump: int = 0
    n_prone: int = 0
    n_stand_still: int = 0
    # MP distribution
    mp_values: list[int] = field(default_factory=list)

    @property
    def is_mobile(self) -> bool:
        return self.n_legal > MOBILE_THRESHOLD

    @property
    def walk_total_hexes(self) -> int:
        return self.walk_only_hexes + self.walk_and_run_hexes


def classify_step(moves: list[dict], walk_mp: int, board_total: int) -> StepRecord:
    """Classify a set of legal moves into a StepRecord."""
    rec = StepRecord(board_total=board_total)
    rec.n_legal = len(moves)

    walk_hexes: set[tuple[int, int]] = set()
    run_hexes: set[tuple[int, int]] = set()
    jump_hexes: set[tuple[int, int]] = set()
    walk_hex_facings: dict[tuple[int, int], set[int]] = defaultdict(set)
    any_hex_facings: dict[tuple[int, int], set[int]] = defaultdict(set)

    for m in moves:
        dx, dy = m.get("dest_x", -1), m.get("dest_y", -1)
        facing = m.get("facing", 0)
        mp = m.get("mp_used", 0)
        jumping = m.get("jumping", False)
        prone = m.get("prone", False)

        rec.mp_values.append(mp)
        any_hex_facings[(dx, dy)].add(facing)

        if prone:
            rec.n_prone += 1
        if mp == 0 and not jumping:
            rec.n_stand_still += 1

        if jumping:
            jump_hexes.add((dx, dy))
            rec.n_jump += 1
        elif mp <= walk_mp:
            walk_hexes.add((dx, dy))
            walk_hex_facings[(dx, dy)].add(facing)
            rec.n_walk += 1
        else:
            run_hexes.add((dx, dy))
            rec.n_run += 1

    all_dest = walk_hexes | run_hexes | jump_hexes
    rec.unique_dest_hexes = len(all_dest)
    rec.reachable_pct = 100.0 * len(all_dest) / board_total if board_total > 0 else 0.0

    # Disjoint ground hex categories
    rec.walk_only_hexes = len(walk_hexes - run_hexes)
    rec.run_only_hexes = len(run_hexes - walk_hexes)
    rec.walk_and_run_hexes = len(walk_hexes & run_hexes)

    # Jump categories
    rec.jump_total_hexes = len(jump_hexes)
    ground_hexes = walk_hexes | run_hexes
    rec.jump_only_hexes = len(jump_hexes - ground_hexes)

    # Facing coverage (walk moves only)
    rec.all_facings_walk = sum(1 for fs in walk_hex_facings.values() if len(fs) == 6)
    rec.partial_facings_walk = sum(1 for fs in walk_hex_facings.values() if len(fs) < 6)

    # Facing coverage (any move type)
    rec.all_facings_any = sum(1 for fs in any_hex_facings.values() if len(fs) == 6)
    rec.partial_facings_any = sum(1 for fs in any_hex_facings.values() if len(fs) < 6)

    return rec


def fmt_stats(vals: list[float], fmt: str = ".0f") -> str:
    """Format min/max/mean/median stats."""
    if not vals:
        return "n/a"
    return (f"min={min(vals):{fmt}}, max={max(vals):{fmt}}, "
            f"mean={np.mean(vals):{fmt}}, median={np.median(vals):{fmt}}")


def print_text_summary(records: list[StepRecord], games_played: int) -> str:
    """Print and return aggregate text summary."""
    if not records:
        msg = "No movement observations collected!"
        print(msg)
        return msg

    lines = []

    def p(s=""):
        lines.append(s)
        print(s)

    mobile = [r for r in records if r.is_mobile]
    immobile = [r for r in records if not r.is_mobile]
    n = len(records)

    p(f"{'=' * 70}")
    p(f"LEGAL MOVE DISTRIBUTION ANALYSIS")
    p(f"  {n} movement steps across {games_played} games")
    p(f"  Board: {records[0].board_total} hexes")
    p(f"  Mobile steps (>{MOBILE_THRESHOLD} moves): {len(mobile)} "
      f"({100*len(mobile)/n:.0f}%)")
    p(f"  Immobile steps (<={MOBILE_THRESHOLD} moves): {len(immobile)} "
      f"({100*len(immobile)/n:.0f}%)")
    p(f"{'=' * 70}")

    # --- All steps: move count distribution ---
    counts = [r.n_legal for r in records]
    p(f"\nALL STEPS — Legal moves per step:")
    p(f"  {fmt_stats(counts)}")

    # --- Mobile steps only: detailed analysis ---
    if mobile:
        p(f"\n{'─' * 70}")
        p(f"MOBILE STEPS ONLY ({len(mobile)} steps)")
        p(f"{'─' * 70}")

        mc = [r.n_legal for r in mobile]
        p(f"\nLegal moves per step:")
        p(f"  {fmt_stats(mc)}")

        # Reachable hex stats
        pcts = [r.reachable_pct for r in mobile]
        p(f"\nReachable hexes (% of board):")
        p(f"  {fmt_stats(pcts, '.1f')}")
        unique_hexes = [r.unique_dest_hexes for r in mobile]
        p(f"  Hex count: {fmt_stats([float(x) for x in unique_hexes])}")

        # Walk+run overlap
        wr = [r.walk_and_run_hexes for r in mobile]
        p(f"\nHexes with BOTH walk and run paths:")
        p(f"  {fmt_stats([float(x) for x in wr])}")
        pct_wr = [100.0 * r.walk_and_run_hexes / r.unique_dest_hexes
                  for r in mobile if r.unique_dest_hexes > 0]
        if pct_wr:
            p(f"  As % of reachable: mean={np.mean(pct_wr):.1f}%, "
              f"median={np.median(pct_wr):.1f}%")

        # Facing coverage
        af = [r.all_facings_walk for r in mobile]
        pf = [r.partial_facings_walk for r in mobile]
        p(f"\nWalk hexes with all 6 facings:")
        p(f"  {fmt_stats([float(x) for x in af])}")
        pct_af = [100.0 * r.all_facings_walk / r.walk_total_hexes
                  for r in mobile if r.walk_total_hexes > 0]
        if pct_af:
            p(f"  As % of walk hexes: mean={np.mean(pct_af):.1f}%, "
              f"median={np.median(pct_af):.1f}%")
        p(f"Walk hexes with <6 facings:")
        p(f"  {fmt_stats([float(x) for x in pf])}")

        # Any-mode facing coverage
        af_any = [r.all_facings_any for r in mobile]
        p(f"\nAll hexes with all 6 facings (any move type):")
        p(f"  {fmt_stats([float(x) for x in af_any])}")
        pct_af_any = [100.0 * r.all_facings_any / r.unique_dest_hexes
                      for r in mobile if r.unique_dest_hexes > 0]
        if pct_af_any:
            p(f"  As % of reachable: mean={np.mean(pct_af_any):.1f}%, "
              f"median={np.median(pct_af_any):.1f}%")

        # Hex category breakdown
        wo = [r.walk_only_hexes for r in mobile]
        ro = [r.run_only_hexes for r in mobile]
        jo = [r.jump_only_hexes for r in mobile]
        p(f"\nHex category means (mobile steps):")
        p(f"  Walk-only:    {np.mean(wo):.1f} hexes")
        p(f"  Run-only:     {np.mean(ro):.1f} hexes")
        p(f"  Walk+Run:     {np.mean(wr):.1f} hexes")
        p(f"  Jump-only:    {np.mean(jo):.1f} hexes")
        jt = [r.jump_total_hexes for r in mobile]
        if any(j > 0 for j in jt):
            p(f"  Jump total:   {np.mean(jt):.1f} hexes")

        # Move type breakdown
        nw = [r.n_walk for r in mobile]
        nr = [r.n_run for r in mobile]
        nj = [r.n_jump for r in mobile]
        p(f"\nMove type counts (mean per mobile step):")
        p(f"  Walk moves:   {np.mean(nw):.0f}  ({100*np.sum(nw)/np.sum(mc):.0f}%)")
        p(f"  Run moves:    {np.mean(nr):.0f}  ({100*np.sum(nr)/np.sum(mc):.0f}%)")
        if any(j > 0 for j in nj):
            p(f"  Jump moves:   {np.mean(nj):.0f}  ({100*np.sum(nj)/np.sum(mc):.0f}%)")
        p(f"  Stand-still:  {np.mean([r.n_stand_still for r in mobile]):.1f}")

    # Per-round breakdown (all steps)
    round_groups: dict[int, list[StepRecord]] = defaultdict(list)
    for r in records:
        round_groups[r.game_round].append(r)

    p(f"\n{'─' * 70}")
    p(f"PER-ROUND BREAKDOWN (all steps)")
    p(f"  {'Rnd':>3}  {'N':>3}  {'Mob':>3}  {'Moves':>12}  {'Reach%':>10}  "
      f"{'W+R hex':>8}  {'6-face':>7}")
    for rnd in sorted(round_groups.keys()):
        recs = round_groups[rnd]
        mob = [r for r in recs if r.is_mobile]
        mc = [r.n_legal for r in recs]
        rp = [r.reachable_pct for r in recs]
        wrh = [r.walk_and_run_hexes for r in recs]
        afh = [r.all_facings_walk for r in recs]
        p(f"  {rnd:>3}  {len(recs):>3}  {len(mob):>3}  "
          f"{np.mean(mc):>5.0f}+/-{np.std(mc):>4.0f}  "
          f"{np.mean(rp):>5.1f}+/-{np.std(rp):>3.1f}  "
          f"{np.mean(wrh):>5.1f}+/-{np.std(wrh):>2.0f}  "
          f"{np.mean(afh):>4.1f}+/-{np.std(afh):>2.0f}")

    # Per-round detail for mobile steps only
    if mobile:
        p(f"\n{'─' * 70}")
        p(f"SAMPLE ROUNDS — detailed mobile step breakdown")
        # Show up to 5 individual mobile steps from different rounds
        shown = 0
        seen_rounds = set()
        for r in mobile:
            if r.game_round in seen_rounds:
                continue
            seen_rounds.add(r.game_round)
            p(f"\n  Game {r.game_idx+1}, Round {r.game_round}, Step {r.step}:")
            p(f"    Legal moves: {r.n_legal}")
            p(f"    Reachable hexes: {r.unique_dest_hexes}/{r.board_total} "
              f"({r.reachable_pct:.1f}%)")
            p(f"    Walk hexes: {r.walk_total_hexes} "
              f"(walk-only={r.walk_only_hexes}, walk+run={r.walk_and_run_hexes})")
            p(f"    Run-only hexes: {r.run_only_hexes}")
            if r.jump_total_hexes > 0:
                p(f"    Jump hexes: {r.jump_total_hexes} "
                  f"(jump-only={r.jump_only_hexes})")
            p(f"    All-6-facings (walk): {r.all_facings_walk} / "
              f"{r.walk_total_hexes} walk hexes "
              f"({100*r.all_facings_walk/r.walk_total_hexes:.0f}%)"
              if r.walk_total_hexes > 0 else
              f"    All-6-facings (walk): 0")
            p(f"    All-6-facings (any):  {r.all_facings_any} / "
              f"{r.unique_dest_hexes} hexes "
              f"({100*r.all_facings_any/r.unique_dest_hexes:.0f}%)"
              if r.unique_dest_hexes > 0 else
              f"    All-6-facings (any):  0")
            p(f"    Move types: walk={r.n_walk}, run={r.n_run}, "
              f"jump={r.n_jump}, stand_still={r.n_stand_still}")
            mp = r.mp_values
            if mp:
                p(f"    MP used: range [{min(mp)}, {max(mp)}], "
                  f"median={sorted(mp)[len(mp)//2]}")
            shown += 1
            if shown >= 5:
                break

    return "\n".join(lines)


def create_visualization(records: list[StepRecord], games_played: int,
                         config: MegaMekConfig, output_path: str) -> None:
    """Create 8-panel matplotlib visualization."""
    mobile = [r for r in records if r.is_mobile]
    counts = [r.n_legal for r in records]

    fig = plt.figure(figsize=(16, 18))
    gs = fig.add_gridspec(4, 2, hspace=0.35, wspace=0.3)

    # --- [0,0] Histogram of legal move counts (all steps) ---
    ax = fig.add_subplot(gs[0, 0])
    ax.hist(counts, bins=30, edgecolor="black", alpha=0.7, color="#4C72B0")
    mean_c = np.mean(counts)
    med_c = np.median(counts)
    ax.axvline(mean_c, color="red", linestyle="--", linewidth=1.5,
               label=f"Mean={mean_c:.0f}")
    ax.axvline(med_c, color="blue", linestyle="--", linewidth=1.5,
               label=f"Median={med_c:.0f}")
    ax.set_xlabel("Legal Moves per Step")
    ax.set_ylabel("Frequency")
    ax.set_title("Distribution of Legal Move Set Size (All Steps)")
    ax.legend()

    # --- [0,1] Histogram of mobile steps only ---
    ax = fig.add_subplot(gs[0, 1])
    if mobile:
        mc = [r.n_legal for r in mobile]
        ax.hist(mc, bins=25, edgecolor="black", alpha=0.7, color="#55A868")
        mean_m = np.mean(mc)
        med_m = np.median(mc)
        ax.axvline(mean_m, color="red", linestyle="--", linewidth=1.5,
                   label=f"Mean={mean_m:.0f}")
        ax.axvline(med_m, color="blue", linestyle="--", linewidth=1.5,
                   label=f"Median={med_m:.0f}")
        ax.legend()
    ax.set_xlabel("Legal Moves per Step")
    ax.set_ylabel("Frequency")
    ax.set_title(f"Mobile Steps Only (n>{MOBILE_THRESHOLD}, N={len(mobile)})")

    # --- [1,0] Box plot by game round ---
    ax = fig.add_subplot(gs[1, 0])
    round_groups: dict[int, list[int]] = defaultdict(list)
    for r in records:
        round_groups[r.game_round].append(r.n_legal)
    sorted_rounds = sorted(round_groups.keys())
    if sorted_rounds:
        data_by_round = [round_groups[rnd] for rnd in sorted_rounds]
        bp = ax.boxplot(data_by_round, positions=list(range(len(sorted_rounds))),
                        widths=0.6, patch_artist=True)
        for patch in bp["boxes"]:
            patch.set_facecolor("#4C72B0")
            patch.set_alpha(0.7)
        # Add sample counts above each box
        for i, rnd in enumerate(sorted_rounds):
            n_samples = len(round_groups[rnd])
            ax.text(i, max(round_groups[rnd]) + 5, f"n={n_samples}",
                    ha="center", va="bottom", fontsize=7, color="gray")
        ax.set_xticks(list(range(len(sorted_rounds))))
        ax.set_xticklabels([str(r) for r in sorted_rounds], fontsize=8)
    ax.set_xlabel("Game Round")
    ax.set_ylabel("Legal Moves")
    ax.set_title("Legal Moves by Game Round")

    # --- [1,1] Reachable hex % over time (line per game) ---
    ax = fig.add_subplot(gs[1, 1])
    cmap = matplotlib.colormaps["tab10"]
    game_indices = sorted(set(r.game_idx for r in records))
    for gi in game_indices:
        game_recs = [r for r in records if r.game_idx == gi]
        # Only plot games that have at least one mobile step (interesting games)
        has_mobile = any(r.is_mobile for r in game_recs)
        steps = list(range(len(game_recs)))
        reach_pcts = [r.reachable_pct for r in game_recs]
        color = cmap(gi % 10)
        alpha = 0.8 if has_mobile else 0.15
        lw = 1.5 if has_mobile else 0.5
        label = f"G{gi+1}" if has_mobile else None
        ax.plot(steps, reach_pcts, marker="." if has_mobile else None,
                markersize=3, alpha=alpha, linewidth=lw,
                color=color, label=label)
    ax.set_xlabel("Step within Game")
    ax.set_ylabel("Reachable Hexes (%)")
    ax.set_title("Reachable Hex % Over Time")
    handles, labels = ax.get_legend_handles_labels()
    if labels:
        ax.legend(fontsize=7, ncol=2)
    ax.set_ylim(bottom=0)

    # --- [2,0] Hex categories stacked bar (mobile steps only) ---
    ax = fig.add_subplot(gs[2, 0])
    if mobile:
        # One bar per mobile step, sorted by game/step order
        n_bars = min(len(mobile), 50)
        if len(mobile) > n_bars:
            bin_size = len(mobile) // n_bars
            binned = []
            for i in range(n_bars):
                chunk = mobile[i * bin_size:(i + 1) * bin_size]
                binned.append({
                    "walk_only": np.mean([r.walk_only_hexes for r in chunk]),
                    "walk_and_run": np.mean([r.walk_and_run_hexes for r in chunk]),
                    "run_only": np.mean([r.run_only_hexes for r in chunk]),
                    "jump_only": np.mean([r.jump_only_hexes for r in chunk]),
                })
        else:
            binned = [
                {"walk_only": r.walk_only_hexes, "walk_and_run": r.walk_and_run_hexes,
                 "run_only": r.run_only_hexes, "jump_only": r.jump_only_hexes}
                for r in mobile
            ]

        x = np.arange(len(binned))
        wo = [b["walk_only"] for b in binned]
        wr = [b["walk_and_run"] for b in binned]
        ro = [b["run_only"] for b in binned]
        jo = [b["jump_only"] for b in binned]

        ax.bar(x, wo, label="Walk only", color="#4C72B0", alpha=0.8)
        ax.bar(x, wr, bottom=wo, label="Walk+Run", color="#55A868", alpha=0.8)
        bottom2 = [a + b for a, b in zip(wo, wr)]
        ax.bar(x, ro, bottom=bottom2, label="Run only", color="#C44E52", alpha=0.8)
        bottom3 = [a + b for a, b in zip(bottom2, ro)]
        ax.bar(x, jo, bottom=bottom3, label="Jump only", color="#8172B2", alpha=0.8)
        ax.legend(fontsize=8)
    ax.set_xlabel("Mobile Step Index" + (" (binned)" if mobile and len(mobile) > 50 else ""))
    ax.set_ylabel("Hexes")
    ax.set_title("Reachable Hex Categories (Mobile Steps)")

    # --- [2,1] Facing coverage: scatter of all-6 vs total walk hexes ---
    ax = fig.add_subplot(gs[2, 1])
    if mobile:
        walk_totals = [r.walk_total_hexes for r in mobile]
        af_walk = [r.all_facings_walk for r in mobile]
        af_any = [r.all_facings_any for r in mobile]
        unique_hexes = [r.unique_dest_hexes for r in mobile]

        ax.scatter(walk_totals, af_walk, alpha=0.5, s=20, color="#55A868",
                   label="Walk: 6-facing", zorder=3)
        ax.scatter(unique_hexes, af_any, alpha=0.5, s=20, color="#4C72B0",
                   marker="^", label="Any: 6-facing", zorder=3)
        # Reference line: y = x (all hexes have all facings)
        max_val = max(max(unique_hexes), max(walk_totals)) if unique_hexes else 10
        ax.plot([0, max_val], [0, max_val], "k--", alpha=0.3, label="y=x (100%)")
        ax.legend(fontsize=8)
    ax.set_xlabel("Total Hexes (walk / any)")
    ax.set_ylabel("Hexes with All 6 Facings")
    ax.set_title("Facing Coverage vs Reachable Hexes")

    # --- [3,0] Walk/Run move count scatter ---
    ax = fig.add_subplot(gs[3, 0])
    if mobile:
        n_walk = [r.n_walk for r in mobile]
        n_run = [r.n_run for r in mobile]
        ax.scatter(n_walk, n_run, alpha=0.5, s=20, color="#4C72B0", zorder=3)
        ax.set_xlabel("Walk Moves")
        ax.set_ylabel("Run Moves")
        ax.set_title("Walk vs Run Move Counts (Mobile Steps)")
        # Add diagonal reference
        max_val = max(max(n_walk), max(n_run))
        ax.plot([0, max_val], [0, max_val], "k--", alpha=0.3, label="1:1")
        ax.legend(fontsize=8)

    # --- [3,1] Summary stats text ---
    ax = fig.add_subplot(gs[3, 1])
    ax.axis("off")
    board_total = records[0].board_total

    n_immobile = len([r for r in records if not r.is_mobile])
    summary = (
        f"Summary ({len(records)} steps, {games_played} games)\n"
        f"{'─' * 44}\n"
        f"Board size:            {board_total} hexes\n"
        f"Mobile / Immobile:     {len(mobile)} / {n_immobile}\n"
    )

    if mobile:
        mc = [r.n_legal for r in mobile]
        pcts = [r.reachable_pct for r in mobile]
        wr_vals = [r.walk_and_run_hexes for r in mobile]
        af_vals = [r.all_facings_walk for r in mobile]
        walk_totals = [r.walk_total_hexes for r in mobile]
        pct_wr = [100.0 * r.walk_and_run_hexes / r.unique_dest_hexes
                  for r in mobile if r.unique_dest_hexes > 0]
        pct_af = [100.0 * r.all_facings_walk / wt
                  for r, wt in zip(mobile, walk_totals) if wt > 0]

        summary += (
            f"\nMOBILE STEPS ONLY:\n"
            f"Legal moves/step:      {np.mean(mc):.0f} mean, "
            f"{np.median(mc):.0f} med\n"
            f"                       [{min(mc)}, {max(mc)}] range\n"
            f"\n"
            f"Reachable hexes:       {np.mean(pcts):.1f}% mean\n"
            f"                       [{min(pcts):.1f}%, {max(pcts):.1f}%]\n"
            f"\n"
            f"Walk+Run hexes:        {np.mean(wr_vals):.1f} mean"
        )
        if pct_wr:
            summary += f" ({np.mean(pct_wr):.0f}% of reachable)\n"
        else:
            summary += "\n"
        summary += f"All-6-facings (walk):  {np.mean(af_vals):.1f} mean"
        if pct_af:
            summary += f" ({np.mean(pct_af):.0f}% of walk)\n"
        else:
            summary += "\n"
        summary += (
            f"\nMove type means:\n"
            f"  Walk: {np.mean([r.n_walk for r in mobile]):.0f}  "
            f"Run: {np.mean([r.n_run for r in mobile]):.0f}  "
            f"Jump: {np.mean([r.n_jump for r in mobile]):.0f}"
        )

    ax.text(0.05, 0.95, summary, transform=ax.transAxes, fontsize=10,
            verticalalignment="top", fontfamily="monospace",
            bbox=dict(boxstyle="round,pad=0.5", facecolor="lightyellow", alpha=0.8))

    fig.suptitle(
        f"Legal Move Distributions — {config.rl_unit} vs {config.opponent_unit}",
        fontsize=14, fontweight="bold"
    )
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"\nSaved visualization to {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Analyze legal move distributions across multiple games"
    )
    parser.add_argument("--megamek-dir", type=str, default="../megamek")
    parser.add_argument("--config", type=str, default=None,
                        help="YAML config file")
    parser.add_argument("--port", type=int, default=None)
    parser.add_argument("--games", type=int, default=10,
                        help="Number of games to play")
    parser.add_argument("--max-steps", type=int, default=80,
                        help="Max steps per game")
    parser.add_argument("--walk-mp", type=int, default=4,
                        help="Walk MP threshold (default 4 for Commando COM-2D)")
    parser.add_argument("--output", type=str, default="move_distributions.png",
                        help="Output PNG path")
    parser.add_argument("--no-show", action="store_true",
                        help="Don't call plt.show()")
    args = parser.parse_args()

    if args.config:
        config = MegaMekConfig.load(args.config)
    else:
        config = MegaMekConfig()
        config.rl_unit = "Commando COM-2D"
        config.opponent_unit = "Commando COM-2D"
        config.max_game_rounds = 40
        config.firing_strategy = "naive"
        config.max_rotating_round_saves = 0

    config.megamek_dir = args.megamek_dir
    if args.port is not None:
        config.rl_port = args.port

    board_total = config.resolved_board_width * config.resolved_board_height

    print(f"Unit: {config.rl_unit} vs {config.opponent_unit}")
    print(f"Board: {config.board} ({config.resolved_board_width}x"
          f"{config.resolved_board_height} = {board_total} hexes)")
    print(f"Games: {args.games}, max steps/game: {args.max_steps}")
    print(f"Walk MP threshold: {args.walk_mp}")

    env = gymnasium.make("MegaMekGym/MegaMek-v0", config=config)

    records: list[StepRecord] = []
    games_played = 0
    t0 = time.monotonic()

    for game_idx in range(args.games):
        print(f"\n--- Game {game_idx + 1}/{args.games} ---")

        obs, info = env.reset()
        n_legal = info.get("n_legal_moves", 0)

        raw_obs = env.unwrapped._last_raw_obs
        legal_moves = raw_obs.get("legal_moves", [])
        if legal_moves:
            rec = classify_step(legal_moves, args.walk_mp, board_total)
            rec.game_idx = game_idx
            rec.step = 0
            rec.game_round = raw_obs.get("round", 0)
            rec.phase = raw_obs.get("phase", "")
            records.append(rec)

        step = 0
        while step < args.max_steps:
            if n_legal > 0:
                action = np.random.randint(0, n_legal)
            else:
                action = 0

            obs, reward, terminated, truncated, info = env.step(action)
            step += 1
            n_legal = info.get("n_legal_moves", 0)

            if terminated or truncated:
                outcome = "terminated" if terminated else "truncated"
                print(f"  Game ended ({outcome}) at step {step}")
                break

            raw_obs = env.unwrapped._last_raw_obs
            legal_moves = raw_obs.get("legal_moves", [])
            if legal_moves:
                rec = classify_step(legal_moves, args.walk_mp, board_total)
                rec.game_idx = game_idx
                rec.step = step
                rec.game_round = raw_obs.get("round", 0)
                rec.phase = raw_obs.get("phase", "")
                records.append(rec)

            if (step <= 2 or step % 10 == 0) and legal_moves:
                print(f"  step {step}: n_legal={n_legal}, "
                      f"reach={records[-1].reachable_pct:.1f}%")

        games_played += 1

    elapsed = time.monotonic() - t0
    print(f"\nPlayed {games_played} games, collected {len(records)} movement "
          f"observations in {elapsed:.1f}s")

    env.close()

    # Text summary
    print()
    print_text_summary(records, games_played)

    # Visualization
    if records:
        create_visualization(records, games_played, config, args.output)
        if not args.no_show:
            plt.show()


if __name__ == "__main__":
    main()
