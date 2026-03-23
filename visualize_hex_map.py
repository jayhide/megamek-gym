#!/usr/bin/env python3
"""Interactive hex map visualizer for MegaMek RL environment.

Plays a game with random actions, collects board/unit/move snapshots at every
movement step, then generates a self-contained HTML file with SVG frames.
Open in browser and use arrow keys to step through turns.

Controls (in browser):
    Right / N   - next step
    Left  / P   - previous step
    Home        - first step
    End         - last step

Usage:
    poetry run python visualize_hex_map.py --megamek-dir ../megamek
    poetry run python visualize_hex_map.py --megamek-dir ../megamek --config configs/default.yaml
    poetry run python visualize_hex_map.py --megamek-dir ../megamek --show-coords --show-elevation
"""

import argparse
import copy
import math
import time
from dataclasses import dataclass, field

import gymnasium
import matplotlib
matplotlib.use("Agg")
import matplotlib.patches as mpatches  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.patches import FancyArrow, Polygon  # noqa: E402

import megamek_gym  # noqa: F401
from megamek_gym.config import MegaMekConfig


# --- Terrain colors ---
TERRAIN_COLORS = {
    "Heavy Woods": "#1a5c1a",
    "Light Woods": "#4da64d",
    "Water": "#4488cc",
    "Rough": "#c4a55a",
    "Pavement": "#888888",
}
DEFAULT_TERRAIN_COLOR = "#d4d4a0"

# --- Reachable hex colors ---
WALK_COLOR = "#3366cc"
RUN_COLOR = "#dd8800"
WALK_RUN_COLOR = "#339933"
REACHABLE_ALPHA = 0.4

# --- Unit colors ---
RL_COLOR = "#2255cc"
ENEMY_COLOR = "#cc2222"


# ---------------------------------------------------------------------------
# Hex geometry
# ---------------------------------------------------------------------------

def hex_to_pixel(x: int, y: int, size: float = 1.0) -> tuple[float, float]:
    """Convert odd-q offset coords to pixel center (flat-top hex).

    MegaMek: odd x columns shift down by half a hex. Y is negated so North
    (facing 0) points upward in matplotlib.
    """
    px = x * size * 1.5
    py = y * size * math.sqrt(3)
    if x % 2 == 1:
        py += size * math.sqrt(3) / 2
    return px, -py


def hex_vertices(cx: float, cy: float, size: float = 1.0) -> list[tuple[float, float]]:
    """Return 6 vertices of a flat-top hexagon centered at (cx, cy)."""
    return [
        (cx + size * math.cos(math.radians(60 * i)),
         cy + size * math.sin(math.radians(60 * i)))
        for i in range(6)
    ]


def facing_angle(facing: int) -> float:
    """Convert MegaMek facing (0=N, 1=NE, ..., 5=NW) to degrees from +x axis."""
    return 90 - 60 * facing


# ---------------------------------------------------------------------------
# Terrain helpers
# ---------------------------------------------------------------------------

def terrain_color(terrain_str: str, elevation: int = 0) -> str:
    """Pick a base color for a hex based on its terrain string and elevation."""
    color = DEFAULT_TERRAIN_COLOR
    for key, val in TERRAIN_COLORS.items():
        if key in terrain_str:
            color = val
            break
    # Darken by elevation
    if elevation != 0:
        r, g, b = int(color[1:3], 16), int(color[3:5], 16), int(color[5:7], 16)
        factor = max(0.4, 1.0 - abs(elevation) * 0.08)
        r, g, b = int(r * factor), int(g * factor), int(b * factor)
        color = f"#{r:02x}{g:02x}{b:02x}"
    return color


# ---------------------------------------------------------------------------
# Data collection
# ---------------------------------------------------------------------------

@dataclass
class StepSnapshot:
    step_idx: int = 0
    game_round: int = 0
    phase: str = ""
    board: dict = field(default_factory=dict)
    units: list[dict] = field(default_factory=list)
    legal_moves: list[dict] = field(default_factory=list)
    n_legal: int = 0
    action_taken: int = -1
    rl_owner_id: int = -1


def detect_walk_mp(units: list[dict], rl_owner_id: int) -> int:
    """Auto-detect walk MP from the RL unit's stats."""
    for u in units:
        if u.get("owner") == rl_owner_id:
            return u.get("mp_walk", 4)
    return 4


def collect_game(env, max_steps: int = 80) -> list[StepSnapshot]:
    """Play one game with random actions, collecting snapshots at each step."""
    snapshots: list[StepSnapshot] = []

    obs, info = env.reset()
    n_legal = info.get("n_legal_moves", 0)
    raw_obs = env.unwrapped._last_raw_obs

    # Identify RL owner from the active entity
    rl_owner_id = -1
    active_id = raw_obs.get("active_entity_id", -1)
    for u in raw_obs.get("units", []):
        if u.get("id") == active_id:
            rl_owner_id = u.get("owner", -1)
            break

    if raw_obs.get("legal_moves"):
        snap = StepSnapshot(
            step_idx=0,
            game_round=raw_obs.get("round", 0),
            phase=raw_obs.get("phase", ""),
            board=copy.deepcopy(raw_obs.get("board", {})),
            units=copy.deepcopy(raw_obs.get("units", [])),
            legal_moves=copy.deepcopy(raw_obs.get("legal_moves", [])),
            n_legal=n_legal,
            rl_owner_id=rl_owner_id,
        )
        snapshots.append(snap)

    for step in range(1, max_steps + 1):
        action = np.random.randint(0, max(n_legal, 1))

        obs, reward, terminated, truncated, info = env.step(action)
        n_legal = info.get("n_legal_moves", 0)

        if terminated or truncated:
            status = "terminated" if terminated else "truncated"
            print(f"  Game ended ({status}) at step {step}")
            break

        raw_obs = env.unwrapped._last_raw_obs
        if raw_obs.get("legal_moves"):
            # Re-detect RL owner if needed
            if rl_owner_id == -1:
                active_id = raw_obs.get("active_entity_id", -1)
                for u in raw_obs.get("units", []):
                    if u.get("id") == active_id:
                        rl_owner_id = u.get("owner", -1)
                        break

            snap = StepSnapshot(
                step_idx=step,
                game_round=raw_obs.get("round", 0),
                phase=raw_obs.get("phase", ""),
                board=copy.deepcopy(raw_obs.get("board", {})),
                units=copy.deepcopy(raw_obs.get("units", [])),
                legal_moves=copy.deepcopy(raw_obs.get("legal_moves", [])),
                n_legal=n_legal,
                action_taken=action,
                rl_owner_id=rl_owner_id,
            )
            snapshots.append(snap)

        if step <= 2 or step % 10 == 0:
            print(f"  step {step}: n_legal={n_legal}")

    return snapshots


# ---------------------------------------------------------------------------
# Drawing functions
# ---------------------------------------------------------------------------

def draw_board(ax, board_data: dict, hex_size: float,
               show_coords: bool = False, show_elevation: bool = False):
    """Draw the base hex grid with terrain coloring."""
    hexes = board_data.get("hexes", [])
    for h in hexes:
        x, y = h["x"], h["y"]
        elev = h.get("elevation", 0)
        terrain = h.get("terrain", "")
        cx, cy = hex_to_pixel(x, y, hex_size)
        verts = hex_vertices(cx, cy, hex_size * 0.97)  # slight gap between hexes
        color = terrain_color(terrain, elev)
        poly = Polygon(verts, closed=True, facecolor=color,
                       edgecolor="#888888", linewidth=0.5)
        ax.add_patch(poly)

        if show_coords:
            ax.text(cx, cy + hex_size * 0.25, f"{x},{y}",
                    ha="center", va="center", fontsize=5, color="#555555")
        if show_elevation and elev != 0:
            ax.text(cx, cy - hex_size * 0.25, f"e{elev}",
                    ha="center", va="center", fontsize=5, color="#333333",
                    fontweight="bold")


def draw_reachable(ax, legal_moves: list[dict], walk_mp: int, hex_size: float):
    """Overlay reachable hexes color-coded by movement type."""
    walk_hexes: set[tuple[int, int]] = set()
    run_hexes: set[tuple[int, int]] = set()

    for m in legal_moves:
        dx, dy = m.get("dest_x", -1), m.get("dest_y", -1)
        mp = m.get("mp_used", 0)
        if mp <= walk_mp:
            walk_hexes.add((dx, dy))
        else:
            run_hexes.add((dx, dy))

    walk_only = walk_hexes - run_hexes
    run_only = run_hexes - walk_hexes
    walk_and_run = walk_hexes & run_hexes

    for hx, hy in walk_only:
        cx, cy = hex_to_pixel(hx, hy, hex_size)
        verts = hex_vertices(cx, cy, hex_size * 0.97)
        poly = Polygon(verts, closed=True, facecolor=WALK_COLOR,
                       alpha=REACHABLE_ALPHA, edgecolor=WALK_COLOR,
                       linewidth=1.0)
        ax.add_patch(poly)

    for hx, hy in run_only:
        cx, cy = hex_to_pixel(hx, hy, hex_size)
        verts = hex_vertices(cx, cy, hex_size * 0.97)
        poly = Polygon(verts, closed=True, facecolor=RUN_COLOR,
                       alpha=REACHABLE_ALPHA, edgecolor=RUN_COLOR,
                       linewidth=1.0)
        ax.add_patch(poly)

    for hx, hy in walk_and_run:
        cx, cy = hex_to_pixel(hx, hy, hex_size)
        verts = hex_vertices(cx, cy, hex_size * 0.97)
        poly = Polygon(verts, closed=True, facecolor=WALK_RUN_COLOR,
                       alpha=REACHABLE_ALPHA, edgecolor=WALK_RUN_COLOR,
                       linewidth=1.0)
        ax.add_patch(poly)

    return len(walk_only), len(run_only), len(walk_and_run)


def draw_units(ax, units: list[dict], rl_owner_id: int, hex_size: float):
    """Draw unit markers with facing arrows."""
    for u in units:
        x, y = u.get("x", -1), u.get("y", -1)
        if x < 0 or y < 0:
            continue
        if u.get("destroyed", False):
            continue

        is_rl = u.get("owner") == rl_owner_id
        color = RL_COLOR if is_rl else ENEMY_COLOR
        label = u.get("chassis", "?")

        cx, cy = hex_to_pixel(x, y, hex_size)

        # Unit circle
        circle = plt.Circle((cx, cy), hex_size * 0.3, facecolor=color,
                             edgecolor="white", linewidth=1.5, zorder=10)
        ax.add_patch(circle)

        # Facing arrow
        facing = u.get("facing", 0)
        angle_deg = facing_angle(facing)
        angle_rad = math.radians(angle_deg)
        arrow_len = hex_size * 0.45
        dx = arrow_len * math.cos(angle_rad)
        dy = arrow_len * math.sin(angle_rad)
        arrow = FancyArrow(cx, cy, dx, dy, width=hex_size * 0.12,
                           head_width=hex_size * 0.25, head_length=hex_size * 0.12,
                           facecolor="white", edgecolor=color, linewidth=0.8,
                           zorder=11)
        ax.add_patch(arrow)

        # Label
        label_side = "RL" if is_rl else "Opp"
        ax.text(cx, cy - hex_size * 0.55, f"{label_side}: {label}",
                ha="center", va="top", fontsize=6, color=color,
                fontweight="bold", zorder=12,
                bbox=dict(boxstyle="round,pad=0.15", facecolor="white",
                          alpha=0.8, edgecolor="none"))


def render_step(ax, snapshot: StepSnapshot, walk_mp: int, hex_size: float,
                show_coords: bool = False, show_elevation: bool = False):
    """Render a complete frame for one step."""
    ax.clear()
    ax.set_aspect("equal")
    ax.axis("off")

    draw_board(ax, snapshot.board, hex_size, show_coords, show_elevation)
    n_walk, n_run, n_both = draw_reachable(ax, snapshot.legal_moves, walk_mp, hex_size)
    draw_units(ax, snapshot.units, snapshot.rl_owner_id, hex_size)

    # Compute axis limits from board dimensions
    bw = snapshot.board.get("width", 16)
    bh = snapshot.board.get("height", 17)
    x_min, _ = hex_to_pixel(0, 0, hex_size)
    x_max, _ = hex_to_pixel(bw - 1, 0, hex_size)
    _, y_max = hex_to_pixel(0, 0, hex_size)
    _, y_min = hex_to_pixel(0, bh - 1, hex_size)
    # Also check odd column shift
    _, y_min2 = hex_to_pixel(1, bh - 1, hex_size)
    y_min = min(y_min, y_min2)

    margin = hex_size * 1.5
    ax.set_xlim(x_min - margin, x_max + margin)
    ax.set_ylim(y_min - margin, y_max + margin)

    # Title with step info
    total_reachable = n_walk + n_run + n_both
    ax.set_title(
        f"Round {snapshot.game_round}  |  Step {snapshot.step_idx}  |  "
        f"Legal moves: {snapshot.n_legal}  |  "
        f"Reachable hexes: {total_reachable} "
        f"(walk={n_walk + n_both}, run={n_run + n_both})",
        fontsize=10, pad=10,
    )

    # Legend
    legend_patches = [
        # Terrain
        mpatches.Patch(facecolor=DEFAULT_TERRAIN_COLOR, edgecolor="#888888",
                       linewidth=0.5, label="Clear"),
        mpatches.Patch(facecolor=TERRAIN_COLORS["Light Woods"], edgecolor="#888888",
                       linewidth=0.5, label="Light Woods"),
        mpatches.Patch(facecolor=TERRAIN_COLORS["Heavy Woods"], edgecolor="#888888",
                       linewidth=0.5, label="Heavy Woods"),
        mpatches.Patch(facecolor=TERRAIN_COLORS["Rough"], edgecolor="#888888",
                       linewidth=0.5, label="Rough"),
        mpatches.Patch(facecolor=TERRAIN_COLORS["Water"], edgecolor="#888888",
                       linewidth=0.5, label="Water"),
        mpatches.Patch(facecolor=TERRAIN_COLORS["Pavement"], edgecolor="#888888",
                       linewidth=0.5, label="Pavement"),
        # Reachable hexes
        mpatches.Patch(facecolor=WALK_COLOR, alpha=REACHABLE_ALPHA,
                       edgecolor=WALK_COLOR, label=f"Walk only ({n_walk})"),
        mpatches.Patch(facecolor=RUN_COLOR, alpha=REACHABLE_ALPHA,
                       edgecolor=RUN_COLOR, label=f"Run only ({n_run})"),
        mpatches.Patch(facecolor=WALK_RUN_COLOR, alpha=REACHABLE_ALPHA,
                       edgecolor=WALK_RUN_COLOR, label=f"Walk+Run ({n_both})"),
        # Units
        mpatches.Patch(facecolor=RL_COLOR, label="RL unit"),
        mpatches.Patch(facecolor=ENEMY_COLOR, label="Enemy"),
    ]
    ax.legend(handles=legend_patches, loc="upper left", bbox_to_anchor=(1.01, 1.0),
              fontsize=7, framealpha=0.9, borderaxespad=0)


# ---------------------------------------------------------------------------
# HTML viewer
# ---------------------------------------------------------------------------

def render_all_svgs(snapshots: list[StepSnapshot], walk_mp: int,
                    hex_size: float, show_coords: bool,
                    show_elevation: bool) -> list[str]:
    """Render each snapshot to an SVG string."""
    import io
    svgs = []
    for i, snap in enumerate(snapshots):
        fig, ax = plt.subplots(1, 1, figsize=(12, 10))
        render_step(ax, snap, walk_mp, hex_size, show_coords, show_elevation)
        fig.suptitle(
            f"Step {i + 1}/{len(snapshots)}",
            fontsize=9, color="#666666",
        )
        buf = io.StringIO()
        fig.savefig(buf, format="svg", bbox_inches="tight")
        plt.close(fig)
        svgs.append(buf.getvalue())
    return svgs


def build_html(svgs: list[str], title: str) -> str:
    """Build a self-contained HTML file with arrow-key navigation."""
    # Escape SVGs for embedding (they're already valid XML)
    frames_js = []
    for i, svg in enumerate(svgs):
        # Extract just the <svg ...>...</svg> content
        start = svg.index("<svg")
        escaped = svg[start:].replace("\\", "\\\\").replace("`", "\\`")
        frames_js.append(escaped)

    frames_array = ",\n".join(f"`{f}`" for f in frames_js)

    return f"""<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<title>{title}</title>
<style>
  body {{
    margin: 0; padding: 20px; background: #1a1a1a; color: #ccc;
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
    display: flex; flex-direction: column; align-items: center;
    min-height: 100vh;
  }}
  #controls {{
    display: flex; align-items: center; gap: 16px;
    margin-bottom: 12px; font-size: 14px;
  }}
  #controls button {{
    background: #333; color: #ccc; border: 1px solid #555;
    border-radius: 4px; padding: 6px 14px; cursor: pointer;
    font-size: 14px;
  }}
  #controls button:hover {{ background: #444; }}
  #controls button:disabled {{ opacity: 0.3; cursor: default; }}
  #step-label {{ font-size: 16px; font-weight: bold; min-width: 120px; text-align: center; }}
  #frame-container {{
    background: white; border-radius: 8px; padding: 10px;
    box-shadow: 0 2px 12px rgba(0,0,0,0.5);
  }}
  #frame-container svg {{ display: block; max-width: 90vw; height: auto; }}
  #help {{ margin-top: 12px; font-size: 12px; color: #888; }}
</style>
</head>
<body>
<div id="controls">
  <button onclick="goFirst()" title="Home">&laquo; First</button>
  <button onclick="goPrev()" id="btn-prev" title="Left arrow">&larr; Prev</button>
  <span id="step-label"></span>
  <button onclick="goNext()" id="btn-next" title="Right arrow">Next &rarr;</button>
  <button onclick="goLast()" title="End">Last &raquo;</button>
</div>
<div id="frame-container"></div>
<div id="help">Arrow keys: navigate &nbsp;|&nbsp; Home/End: jump to first/last</div>
<script>
const frames = [{frames_array}];
let idx = 0;

function show(i) {{
  idx = Math.max(0, Math.min(frames.length - 1, i));
  document.getElementById("frame-container").innerHTML = frames[idx];
  document.getElementById("step-label").textContent = "Step " + (idx+1) + " / " + frames.length;
  document.getElementById("btn-prev").disabled = (idx === 0);
  document.getElementById("btn-next").disabled = (idx === frames.length - 1);
}}

function goNext() {{ show(idx + 1); }}
function goPrev() {{ show(idx - 1); }}
function goFirst() {{ show(0); }}
function goLast() {{ show(frames.length - 1); }}

document.addEventListener("keydown", function(e) {{
  if (e.key === "ArrowRight" || e.key === "n") goNext();
  else if (e.key === "ArrowLeft" || e.key === "p") goPrev();
  else if (e.key === "Home") goFirst();
  else if (e.key === "End") goLast();
}});

show(0);
</script>
</body>
</html>"""


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Interactive hex map visualizer for MegaMek RL environment"
    )
    parser.add_argument("--megamek-dir", type=str, default="../megamek")
    parser.add_argument("--config", type=str, default=None,
                        help="YAML config file")
    parser.add_argument("--port", type=int, default=None)
    parser.add_argument("--walk-mp", type=int, default=None,
                        help="Walk MP threshold (auto-detected from unit if not set)")
    parser.add_argument("--max-steps", type=int, default=80,
                        help="Max steps per game")
    parser.add_argument("--hex-size", type=float, default=1.0,
                        help="Hex rendering size")
    parser.add_argument("--show-coords", action="store_true",
                        help="Label hexes with (x,y) coordinates")
    parser.add_argument("--show-elevation", action="store_true",
                        help="Label hexes with elevation values")
    parser.add_argument("--output", type=str, default="hex_map.html",
                        help="Output HTML file path (default: hex_map.html)")
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

    print(f"Unit: {config.rl_unit} vs {config.opponent_unit}")
    print(f"Board: {config.board}")
    print(f"Playing game with random actions (max {args.max_steps} steps)...")

    env = gymnasium.make("MegaMekGym/MegaMek-v0", config=config)

    t0 = time.monotonic()
    snapshots = collect_game(env, args.max_steps)
    elapsed = time.monotonic() - t0

    env.close()

    print(f"\nCollected {len(snapshots)} movement snapshots in {elapsed:.1f}s")

    if not snapshots:
        print("No movement snapshots collected!")
        return

    # Auto-detect walk MP from the RL unit if not specified
    walk_mp = args.walk_mp
    if walk_mp is None:
        walk_mp = detect_walk_mp(snapshots[0].units, snapshots[0].rl_owner_id)
        print(f"Auto-detected walk MP: {walk_mp}")

    print(f"Rendering {len(snapshots)} frames to SVG...")
    svgs = render_all_svgs(snapshots, walk_mp, args.hex_size,
                           args.show_coords, args.show_elevation)

    title = f"Hex Map - {config.rl_unit} vs {config.opponent_unit}"
    html = build_html(svgs, title)

    with open(args.output, "w") as f:
        f.write(html)
    print(f"Saved to {args.output}")

    # Open in browser (WSL2-aware)
    import os
    import shutil
    import subprocess
    abs_path = os.path.abspath(args.output)

    opened = False
    # WSL2: use wslview or explorer.exe to open in Windows browser
    if shutil.which("wslview"):
        subprocess.Popen(["wslview", abs_path],
                         stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        opened = True
    elif shutil.which("explorer.exe"):
        # Convert WSL path to Windows path
        try:
            win_path = subprocess.check_output(
                ["wslpath", "-w", abs_path], text=True
            ).strip()
            subprocess.Popen(["explorer.exe", win_path],
                             stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            opened = True
        except (subprocess.CalledProcessError, FileNotFoundError):
            pass

    if not opened:
        import webbrowser
        webbrowser.open("file://" + abs_path)

    print("Opened in browser. Arrow keys to navigate, Home/End to jump.")


if __name__ == "__main__":
    main()
