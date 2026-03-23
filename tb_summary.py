#!/usr/bin/env python3
"""Read TensorBoard event files and produce compact training summaries.

Usage:
    poetry run python tb_summary.py runs/sanity-check__1__*
    poetry run python tb_summary.py runs/run_a runs/run_b          # compare
    poetry run python tb_summary.py runs/latest --tail 20          # last 20%
    poetry run python tb_summary.py runs/latest --diagnostics-only
    poetry run python tb_summary.py runs/latest --raw charts/win_rate
"""

import argparse
import glob
import os
import sys

import numpy as np
import yaml
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

# ---------------------------------------------------------------------------
# Metric groups
# ---------------------------------------------------------------------------
PERFORMANCE_METRICS = [
    "charts/win_rate", "charts/episodic_return", "charts/game_rounds",
    "charts/episodic_length", "charts/reward_mean",
]
LOSS_METRICS = [
    "losses/policy_loss", "losses/value_loss", "losses/entropy",
    "losses/approx_kl", "losses/clipfrac", "losses/explained_variance",
]
TIMING_METRICS = [
    "timing/rollout_seconds", "timing/train_seconds",
    "timing/episodes_per_rollout", "charts/SPS",
]
STABILITY_METRICS = [
    "charts/java_crashes", "charts/early_terminations",
]
ALL_GROUPS = {
    "performance": PERFORMANCE_METRICS,
    "losses": LOSS_METRICS,
    "timing": TIMING_METRICS,
    "stability": STABILITY_METRICS,
}

# ---------------------------------------------------------------------------
# Diagnostic thresholds
# ---------------------------------------------------------------------------
ENTROPY_COLLAPSE_RATIO = 0.2
KL_SPIKE_MULTIPLIER = 2.0
CLIPFRAC_HIGH = 0.3
CLIPFRAC_ZERO_THRESHOLD = 0.001
EXPLAINED_VAR_LOW = 0.1
SPS_DEGRADATION_RATIO = 0.7
WIN_RATE_PLATEAU_THRESHOLD = 0.02
CRASH_RATE_WARN = 0.05
KL_ZERO_THRESHOLD = 0.0005


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------
def load_run(run_dir):
    """Load TensorBoard scalars and optional config from a run directory."""
    events_files = sorted(glob.glob(os.path.join(run_dir, "events.out.tfevents.*")))
    if not events_files:
        print(f"ERROR: no event files in {run_dir}", file=sys.stderr)
        sys.exit(1)

    ea = EventAccumulator(run_dir, size_guidance={"scalars": 0})
    ea.Reload()

    tags = {}
    for tag in ea.Tags().get("scalars", []):
        events = ea.Scalars(tag)
        tags[tag] = [(e.step, e.wall_time, e.value) for e in events]

    config = None
    config_path = os.path.join(run_dir, "config.yaml")
    if os.path.exists(config_path):
        with open(config_path) as f:
            config = yaml.safe_load(f)

    return {
        "tags": tags,
        "config": config,
        "run_name": os.path.basename(run_dir),
        "run_dir": run_dir,
    }


def filter_tail(tags, tail_pct):
    """Keep only the last tail_pct% of data by step."""
    if tail_pct >= 100:
        return tags
    max_step = max(
        (ev[0] for evs in tags.values() for ev in evs), default=0
    )
    cutoff = max_step * (1 - tail_pct / 100.0)
    return {
        tag: [(s, w, v) for s, w, v in evs if s >= cutoff]
        for tag, evs in tags.items()
    }


# ---------------------------------------------------------------------------
# Windowed analysis
# ---------------------------------------------------------------------------
def make_windows(events, n_windows):
    """Split events into n_windows by step range, return per-window stats."""
    if not events:
        return []
    steps = np.array([e[0] for e in events])
    vals = np.array([e[2] for e in events])
    lo, hi = steps.min(), steps.max()
    if lo == hi:
        return [{"step_lo": lo, "step_hi": hi, "mean": vals.mean(),
                 "std": vals.std(), "min": vals.min(), "max": vals.max(),
                 "count": len(vals)}]
    edges = np.linspace(lo, hi, n_windows + 1)
    windows = []
    for i in range(n_windows):
        if i < n_windows - 1:
            mask = (steps >= edges[i]) & (steps < edges[i + 1])
        else:
            mask = steps >= edges[i]
        wv = vals[mask]
        if len(wv) == 0:
            continue
        windows.append({
            "step_lo": int(edges[i]),
            "step_hi": int(edges[i + 1]),
            "mean": float(wv.mean()),
            "std": float(wv.std()),
            "min": float(wv.min()),
            "max": float(wv.max()),
            "count": int(len(wv)),
        })
    return windows


def trend(windows):
    """Return (delta, direction_str) comparing first to last window."""
    if len(windows) < 2:
        return 0.0, "insufficient data"
    first, last = windows[0]["mean"], windows[-1]["mean"]
    delta = last - first
    if abs(delta) < 1e-6:
        return delta, "flat"
    return delta, "improving" if delta > 0 else "declining"


# ---------------------------------------------------------------------------
# Formatting helpers
# ---------------------------------------------------------------------------
def fmt_step(s):
    if s >= 1_000_000:
        return f"{s / 1e6:.1f}M"
    if s >= 1_000:
        return f"{s / 1e3:.0f}k"
    return str(s)


def fmt_val(v, tag=""):
    if "win_rate" in tag:
        return f"{v * 100:.1f}%"
    if "SPS" in tag or "episodes_per" in tag:
        return f"{v:.0f}"
    if "seconds" in tag:
        return f"{v:.1f}s"
    if abs(v) >= 100:
        return f"{v:.1f}"
    if abs(v) >= 1:
        return f"{v:.2f}"
    return f"{v:.4f}"


def short_tag(tag):
    return tag.split("/")[-1]


def hms(seconds):
    h, rem = divmod(int(seconds), 3600)
    m, s = divmod(rem, 60)
    return f"{h:02d}:{m:02d}:{s:02d}"


# ---------------------------------------------------------------------------
# Printing sections
# ---------------------------------------------------------------------------
def print_config(config):
    if not config:
        print("  (no config.yaml found)")
        return
    important = [
        "rl_unit", "opponent_unit", "board",
        "num_envs", "num_steps", "hidden_size",
        "learning_rate", "ent_coef", "gamma", "gae_lambda",
        "total_timesteps", "max_game_rounds",
    ]
    for key in important:
        if key in config:
            print(f"  {key}: {config[key]}")


def print_progress(tags):
    # Steps
    max_step = max((ev[0] for evs in tags.values() for ev in evs), default=0)
    print(f"  Steps: {max_step:,}")

    # Episodes (count of episodic_return events)
    er = tags.get("charts/episodic_return", [])
    print(f"  Episodes: {len(er):,}")

    # Wall time
    all_times = [ev[1] for evs in tags.values() for ev in evs]
    if all_times:
        elapsed = max(all_times) - min(all_times)
        print(f"  Wall time: {hms(elapsed)}")

    # SPS
    sps = tags.get("charts/SPS", [])
    if sps:
        avg_sps = np.mean([v for _, _, v in sps])
        print(f"  Avg SPS: {avg_sps:.0f}")


def print_game_summary(tags):
    """Print W/L/D/C/E totals from per-episode metrics."""
    outcomes = tags.get("charts/game_outcome", [])
    if not outcomes:
        return
    vals = [v for _, _, v in outcomes]
    wins = sum(1 for v in vals if v > 0.5)
    losses = sum(1 for v in vals if v < -0.5)
    draws = sum(1 for v in vals if -0.5 <= v <= 0.5)
    total = len(vals)
    parts = [f"W:{wins}", f"L:{losses}", f"D:{draws}"]

    crashes = tags.get("charts/java_crashes", [])
    if crashes:
        parts.append(f"C:{int(crashes[-1][2])}")
    early = tags.get("charts/early_terminations", [])
    if early:
        parts.append(f"E:{int(early[-1][2])}")

    wr = wins / total * 100 if total else 0
    print(f"  Games: {total:,} ({' '.join(parts)} — {wr:.1f}% win rate)")


def print_windowed_section(tags, metric_list, n_windows, section_name):
    available = [m for m in metric_list if m in tags and len(tags[m]) > 0]
    if not available:
        return

    print(f"\n--- {section_name} ({n_windows} windows) ---")
    # Header
    col_w = 12
    header = f"  {'Window':<12}"
    for m in available:
        header += f"{short_tag(m):>{col_w}}"
    print(header)

    # Build windows for each metric
    all_wins = {m: make_windows(tags[m], n_windows) for m in available}
    max_rows = max(len(w) for w in all_wins.values())

    for i in range(max_rows):
        step_label = ""
        row = ""
        for m in available:
            wins = all_wins[m]
            if i < len(wins):
                w = wins[i]
                if not step_label:
                    step_label = f"{fmt_step(w['step_lo'])}-{fmt_step(w['step_hi'])}"
                row += f"{fmt_val(w['mean'], m):>{col_w}}"
            else:
                row += f"{'—':>{col_w}}"
        print(f"  {step_label:<12}{row}")

    # Trends
    trends = []
    for m in available:
        wins = all_wins[m]
        delta, direction = trend(wins)
        if direction not in ("flat", "insufficient data"):
            trends.append(f"{short_tag(m)}: {delta:+.4f} ({direction})")
    if trends:
        print(f"  Trends: {'; '.join(trends)}")


def print_timing(tags):
    print("\n--- Timing ---")
    for m in TIMING_METRICS:
        if m in tags and tags[m]:
            vals = [v for _, _, v in tags[m]]
            print(f"  {short_tag(m)}: avg={fmt_val(np.mean(vals), m)}, "
                  f"last={fmt_val(vals[-1], m)}")


# ---------------------------------------------------------------------------
# Diagnostics
# ---------------------------------------------------------------------------
def run_diagnostics(tags, config):
    """Return list of (level, message) diagnostic tuples."""
    diags = []
    n_w = 5  # use 5 windows for diagnostics

    # --- Win rate ---
    if "charts/win_rate" in tags and len(tags["charts/win_rate"]) > 10:
        wins = make_windows(tags["charts/win_rate"], n_w)
        if len(wins) >= 2:
            delta, direction = trend(wins)
            if direction == "improving" and abs(delta) > WIN_RATE_PLATEAU_THRESHOLD:
                diags.append(("OK", f"Win rate improving ({wins[0]['mean']*100:.1f}% → {wins[-1]['mean']*100:.1f}%)"))
            elif abs(delta) <= WIN_RATE_PLATEAU_THRESHOLD:
                diags.append(("WARN", f"Win rate plateaued ({wins[0]['mean']*100:.1f}% → {wins[-1]['mean']*100:.1f}%)"))
            else:
                diags.append(("WARN", f"Win rate declining ({wins[0]['mean']*100:.1f}% → {wins[-1]['mean']*100:.1f}%)"))

    # --- Entropy ---
    if "losses/entropy" in tags and len(tags["losses/entropy"]) > 2:
        ent_vals = [v for _, _, v in tags["losses/entropy"]]
        first_ent = np.mean(ent_vals[:max(1, len(ent_vals) // 10)])
        last_ent = np.mean(ent_vals[-max(1, len(ent_vals) // 10):])
        if first_ent > 0:
            ratio = last_ent / first_ent
            if ratio < ENTROPY_COLLAPSE_RATIO:
                diags.append(("WARN", f"Entropy collapsed ({first_ent:.3f} → {last_ent:.3f}, ratio {ratio:.2f})"))
            elif 0.95 < ratio < 1.05:
                diags.append(("INFO", f"Entropy unchanged ({first_ent:.3f} → {last_ent:.3f}) — policy may not be learning"))
            else:
                diags.append(("OK", f"Entropy declining gradually ({first_ent:.3f} → {last_ent:.3f}, ratio {ratio:.2f})"))

    # --- KL ---
    if "losses/approx_kl" in tags and len(tags["losses/approx_kl"]) > 2:
        kl_vals = [v for _, _, v in tags["losses/approx_kl"]]
        target_kl = (config or {}).get("target_kl", 0.03)
        last_kl = np.mean(kl_vals[-max(1, len(kl_vals) // 10):])
        max_kl = max(kl_vals)
        if last_kl < KL_ZERO_THRESHOLD:
            diags.append(("WARN", f"KL near zero ({last_kl:.6f}) — policy stopped updating"))
        elif max_kl > target_kl * KL_SPIKE_MULTIPLIER:
            diags.append(("WARN", f"KL spike detected (max={max_kl:.4f}, target={target_kl})"))
        else:
            diags.append(("OK", f"KL stable (last={last_kl:.4f}, max={max_kl:.4f})"))

    # --- Clipfrac ---
    if "losses/clipfrac" in tags and len(tags["losses/clipfrac"]) > 2:
        cf_vals = [v for _, _, v in tags["losses/clipfrac"]]
        last_cf = np.mean(cf_vals[-max(1, len(cf_vals) // 10):])
        if last_cf < CLIPFRAC_ZERO_THRESHOLD:
            diags.append(("WARN", f"Clipfrac near zero ({last_cf:.5f}) — updates too small to clip"))
        elif last_cf > CLIPFRAC_HIGH:
            diags.append(("WARN", f"Clipfrac high ({last_cf:.3f}) — LR may be too high"))
        else:
            diags.append(("OK", f"Clipfrac healthy ({last_cf:.3f})"))

    # --- Explained variance ---
    if "losses/explained_variance" in tags and len(tags["losses/explained_variance"]) > 2:
        ev_vals = [v for _, _, v in tags["losses/explained_variance"]]
        first_ev = np.mean(ev_vals[:max(1, len(ev_vals) // 10)])
        last_ev = np.mean(ev_vals[-max(1, len(ev_vals) // 10):])
        if last_ev < 0:
            diags.append(("WARN", f"Explained variance negative ({last_ev:.3f}) — critic worse than mean prediction"))
        elif last_ev < EXPLAINED_VAR_LOW:
            diags.append(("WARN", f"Explained variance low ({last_ev:.3f}) — critic underfitting"))
        elif last_ev > first_ev + 0.05:
            diags.append(("OK", f"Explained variance improving ({first_ev:.3f} → {last_ev:.3f})"))
        else:
            diags.append(("INFO", f"Explained variance flat ({first_ev:.3f} → {last_ev:.3f})"))

    # --- SPS ---
    if "charts/SPS" in tags and len(tags["charts/SPS"]) > 2:
        sps_vals = [v for _, _, v in tags["charts/SPS"]]
        first_sps = np.mean(sps_vals[:max(1, len(sps_vals) // 10)])
        last_sps = np.mean(sps_vals[-max(1, len(sps_vals) // 10):])
        if first_sps > 0 and last_sps / first_sps < SPS_DEGRADATION_RATIO:
            diags.append(("WARN", f"SPS degraded ({first_sps:.0f} → {last_sps:.0f})"))

    # --- Crashes ---
    crashes = tags.get("charts/java_crashes", [])
    total_crashes = int(crashes[-1][2]) if crashes else 0
    total_episodes = len(tags.get("charts/episodic_return", []))
    if total_crashes == 0:
        diags.append(("OK", "No Java crashes"))
    elif total_episodes > 0 and total_crashes / total_episodes > CRASH_RATE_WARN:
        diags.append(("WARN", f"High crash rate: {total_crashes} crashes in {total_episodes} episodes ({total_crashes/total_episodes*100:.1f}%)"))
    else:
        diags.append(("INFO", f"{total_crashes} Java crash(es)"))

    # --- Early terminations ---
    early = tags.get("charts/early_terminations", [])
    total_early = int(early[-1][2]) if early else 0
    if total_early > 0 and total_episodes > 0:
        pct = total_early / total_episodes * 100
        level = "WARN" if pct > 20 else "INFO"
        diags.append((level, f"{total_early} early terminations ({pct:.1f}% of episodes)"))

    return diags


def print_diagnostics(diags):
    print("\n--- Diagnostics ---")
    for level, msg in diags:
        print(f"  [{level:>4}] {msg}")


# ---------------------------------------------------------------------------
# Comparison
# ---------------------------------------------------------------------------
def print_comparison(runs):
    print("=== RUN COMPARISON ===\n")
    names = [r["run_name"][:40] for r in runs]
    col_w = max(len(n) for n in names) + 2

    # Final metrics (last 20%)
    print("--- Final Metrics (last 20%) ---")
    header = f"  {'Metric':<28}" + "".join(f"{n:>{col_w}}" for n in names)
    print(header)

    compare_metrics = [
        ("charts/win_rate", "win_rate"),
        ("charts/episodic_return", "episodic_return"),
        ("losses/explained_variance", "explained_var"),
        ("losses/entropy", "entropy"),
        ("losses/policy_loss", "policy_loss"),
        ("losses/value_loss", "value_loss"),
        ("losses/approx_kl", "approx_kl"),
        ("losses/clipfrac", "clipfrac"),
        ("charts/SPS", "SPS"),
    ]

    for tag, label in compare_metrics:
        row = f"  {label:<28}"
        for r in runs:
            evs = r["tags"].get(tag, [])
            if evs:
                tail = evs[max(0, int(len(evs) * 0.8)):]
                avg = np.mean([v for _, _, v in tail])
                row += f"{fmt_val(avg, tag):>{col_w}}"
            else:
                row += f"{'—':>{col_w}}"
        print(row)

    # Total episodes / games
    row_eps = f"  {'episodes':<28}"
    for r in runs:
        n = len(r["tags"].get("charts/episodic_return", []))
        row_eps += f"{n:>{col_w},}"
    print(row_eps)

    # Config diff
    configs = [r["config"] for r in runs if r["config"]]
    if len(configs) >= 2:
        all_keys = set()
        for c in configs:
            all_keys.update(c.keys())
        diffs = []
        for k in sorted(all_keys):
            vals = [str(c.get(k, "—")) for c in configs]
            if len(set(vals)) > 1:
                diffs.append((k, vals))
        if diffs:
            print(f"\n--- Config Differences ---")
            header = f"  {'Key':<28}" + "".join(f"{n:>{col_w}}" for n in names)
            print(header)
            for k, vals in diffs:
                row = f"  {k:<28}" + "".join(f"{v:>{col_w}}" for v in vals)
                print(row)


# ---------------------------------------------------------------------------
# Raw dump
# ---------------------------------------------------------------------------
def print_raw(tags, metric):
    if metric not in tags:
        print(f"ERROR: metric '{metric}' not found", file=sys.stderr)
        print(f"Available: {', '.join(sorted(tags.keys()))}", file=sys.stderr)
        sys.exit(1)
    for step, _, val in tags[metric]:
        print(f"{step}\t{val}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="TensorBoard training run summarizer")
    parser.add_argument("run_dirs", nargs="+", help="Run directories (or glob patterns)")
    parser.add_argument("--windows", "-w", type=int, default=10, help="Number of summary windows (default 10)")
    parser.add_argument("--tail", "-t", type=float, default=100, help="Only analyze last N%% of training (default 100)")
    parser.add_argument("--diagnostics-only", "-d", action="store_true", help="Only show diagnostics")
    parser.add_argument("--groups", "-g", help="Comma-separated groups: performance,losses,timing,stability")
    parser.add_argument("--raw", "-r", help="Dump raw values for a specific metric")
    parser.add_argument("--list-metrics", "-l", action="store_true", help="List available metrics and exit")
    args = parser.parse_args()

    # Expand glob patterns in run_dirs
    expanded = []
    for pattern in args.run_dirs:
        matches = sorted(glob.glob(pattern))
        if matches:
            expanded.extend(m for m in matches if os.path.isdir(m))
        elif os.path.isdir(pattern):
            expanded.append(pattern)
        else:
            print(f"WARNING: no match for '{pattern}'", file=sys.stderr)
    if not expanded:
        print("ERROR: no valid run directories found", file=sys.stderr)
        sys.exit(1)

    runs = [load_run(d) for d in expanded]

    # --list-metrics
    if args.list_metrics:
        for r in runs:
            print(f"{r['run_name']}:")
            for tag in sorted(r["tags"].keys()):
                print(f"  {tag} ({len(r['tags'][tag])} events)")
        return

    # --raw
    if args.raw:
        for r in runs:
            if len(runs) > 1:
                print(f"# {r['run_name']}")
            print_raw(r["tags"], args.raw)
        return

    # Multi-run comparison
    if len(runs) > 1:
        for r in runs:
            r["tags"] = filter_tail(r["tags"], args.tail)
        print_comparison(runs)
        # Diagnostics for each
        for r in runs:
            print(f"\n--- Diagnostics: {r['run_name'][:50]} ---")
            diags = run_diagnostics(r["tags"], r["config"])
            for level, msg in diags:
                print(f"  [{level:>4}] {msg}")
        return

    # Single run
    run = runs[0]
    tags = filter_tail(run["tags"], args.tail)

    print(f"=== {run['run_name']} ===\n")

    if not args.diagnostics_only:
        print("--- Config ---")
        print_config(run["config"])

        print("\n--- Progress ---")
        print_progress(tags)
        print_game_summary(tags)

        groups = args.groups.split(",") if args.groups else ["performance", "losses", "timing"]
        if "performance" in groups:
            print_windowed_section(tags, PERFORMANCE_METRICS, args.windows, "Performance")
        if "losses" in groups:
            print_windowed_section(tags, LOSS_METRICS, args.windows, "Losses")
        if "timing" in groups:
            print_timing(tags)
        if "stability" in groups:
            print_windowed_section(tags, STABILITY_METRICS, args.windows, "Stability")

    diags = run_diagnostics(tags, run["config"])
    print_diagnostics(diags)


if __name__ == "__main__":
    main()
