"""Combined hex map + transcript viewer for MegaMek RL games.

Plays a game, collects step-level board snapshots and round-level transcript
data (from save files), then generates a self-contained HTML file with the
hex map on the left and annotated transcript on the right.

Usage:
    # Random actions
    poetry run python game_viewer.py --megamek-dir ../megamek --config configs/default.yaml

    # Trained policy
    poetry run python game_viewer.py --megamek-dir ../megamek --config configs/default.yaml \
        --checkpoint runs/megamek-ppo__1__*/checkpoints/latest.pt --deterministic

    # With hex labels
    poetry run python game_viewer.py --megamek-dir ../megamek --show-coords --show-elevation
"""

import argparse
import copy
import json
import re
import random
import time
from distutils.util import strtobool
from pathlib import Path

import gymnasium
import numpy as np

import megamek_gym  # noqa: F401 — registers the env
from megamek_gym.agent import load_agent, select_action, OUTCOME_MAP
from megamek_gym.config import MegaMekConfig
from megamek_gym.reward import CompositeReward

from inspect_games import collect_saves, clear_saves
from transcript import (
    parse_save, get_round_reports, build_round_transcript, determine_end_condition,
)
from visualize_hex_map import (
    StepSnapshot, detect_walk_mp, render_all_svgs,
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Combined hex map + transcript viewer for MegaMek RL games"
    )
    # Game config
    parser.add_argument("--megamek-dir", type=str, default="../megamek")
    parser.add_argument("--config", type=str, default=None, help="YAML config file")
    parser.add_argument("--port", type=int, default=None)
    parser.add_argument("--seed", type=int, default=1)

    # Policy
    parser.add_argument("--checkpoint", type=str, default=None,
                        help="Path to .pt checkpoint for trained policy")
    parser.add_argument("--deterministic", type=lambda x: bool(strtobool(x)),
                        default=False, nargs="?", const=True,
                        help="Use greedy action selection (only with --checkpoint)")

    # Hex rendering
    parser.add_argument("--hex-size", type=float, default=1.0)
    parser.add_argument("--show-coords", action="store_true")
    parser.add_argument("--show-elevation", action="store_true")
    parser.add_argument("--max-steps", type=int, default=80)

    # Output
    parser.add_argument("--output", type=str, default="game_viewer.html")
    parser.add_argument("--verbose", "-v", action="store_true",
                        help="Show damage details in transcript panel")

    return parser.parse_args()


def play_game(env, agent, device, deterministic, max_steps):
    """Play one game, collecting step snapshots and step log simultaneously.

    Returns (snapshots, step_log, outcome, game_rounds).
    """
    snapshots = []
    step_log = []

    obs, info = env.reset()
    n_legal = info.get("n_legal_moves", 0)
    raw_obs = env.unwrapped._last_raw_obs
    episode_return = 0.0

    # Identify RL owner
    rl_owner_id = -1
    active_id = raw_obs.get("active_entity_id", -1)
    for u in raw_obs.get("units", []):
        if u.get("id") == active_id:
            rl_owner_id = u.get("owner", -1)
            break

    # Initial snapshot
    if raw_obs.get("legal_moves"):
        snapshots.append(StepSnapshot(
            step_idx=0,
            game_round=raw_obs.get("round", 0),
            phase=raw_obs.get("phase", ""),
            board=copy.deepcopy(raw_obs.get("board", {})),
            units=copy.deepcopy(raw_obs.get("units", [])),
            legal_moves=copy.deepcopy(raw_obs.get("legal_moves", [])),
            n_legal=n_legal,
            rl_owner_id=rl_owner_id,
        ))

    for step in range(1, max_steps + 1):
        if agent is not None:
            action = select_action(agent, obs, info["action_mask"], device, deterministic)
        else:
            action = np.random.randint(0, max(n_legal, 1))

        obs, reward, terminated, truncated, info = env.step(action)
        n_legal = info.get("n_legal_moves", 0)
        episode_return += reward

        # Capture reward breakdown
        reward_fn = env.unwrapped.reward_fn
        reward_details = []
        if isinstance(reward_fn, CompositeReward):
            reward_details = list(reward_fn.last_details)

        step_log.append({
            "round": info.get("round", 0),
            "phase": info.get("phase", ""),
            "action": int(action),
            "n_legal_moves": info.get("n_legal_moves", 0),
            "reward": reward,
            "reward_details": reward_details,
            "cumulative_return": episode_return,
            "early_termination": info.get("early_termination", 0),
            "java_crash": info.get("java_crash", 0),
        })

        if terminated or truncated:
            status = "terminated" if terminated else "truncated"
            print(f"  Game ended ({status}) at step {step}")
            break

        raw_obs = env.unwrapped._last_raw_obs
        if raw_obs.get("legal_moves"):
            if rl_owner_id == -1:
                active_id = raw_obs.get("active_entity_id", -1)
                for u in raw_obs.get("units", []):
                    if u.get("id") == active_id:
                        rl_owner_id = u.get("owner", -1)
                        break

            snapshots.append(StepSnapshot(
                step_idx=step,
                game_round=raw_obs.get("round", 0),
                phase=raw_obs.get("phase", ""),
                board=copy.deepcopy(raw_obs.get("board", {})),
                units=copy.deepcopy(raw_obs.get("units", [])),
                legal_moves=copy.deepcopy(raw_obs.get("legal_moves", [])),
                n_legal=n_legal,
                action_taken=action,
                rl_owner_id=rl_owner_id,
            ))

        if step <= 2 or step % 10 == 0:
            print(f"  step {step}: n_legal={n_legal}")

    outcome = OUTCOME_MAP.get(info.get("game_outcome", 0), "UNKNOWN")
    game_rounds = info.get("game_rounds", "?")
    return snapshots, step_log, outcome, game_rounds


def build_combined_html(svgs, step_meta, round_data, outcome, title, verbose):
    """Build self-contained HTML with hex map + transcript side by side."""
    # Embed SVGs as JS array
    frames_js = []
    for svg in svgs:
        start = svg.index("<svg")
        escaped = svg[start:].replace("\\", "\\\\").replace("`", "\\`")
        frames_js.append(escaped)
    frames_array = ",\n".join(f"`{f}`" for f in frames_js)

    step_meta_json = json.dumps(step_meta)
    round_data_json = json.dumps(round_data)

    verbose_js = "true" if verbose else "false"

    return f"""<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<title>{title}</title>
<style>
  * {{ box-sizing: border-box; margin: 0; padding: 0; }}
  body {{
    background: #1a1a1a; color: #ccc;
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
    display: flex; flex-direction: column; height: 100vh; overflow: hidden;
  }}
  #controls {{
    display: flex; align-items: center; justify-content: center; gap: 16px;
    padding: 12px; background: #222; border-bottom: 1px solid #333;
    font-size: 14px; flex-shrink: 0;
  }}
  #controls button {{
    background: #333; color: #ccc; border: 1px solid #555;
    border-radius: 4px; padding: 6px 14px; cursor: pointer; font-size: 14px;
  }}
  #controls button:hover {{ background: #444; }}
  #controls button:disabled {{ opacity: 0.3; cursor: default; }}
  #step-label {{ font-size: 15px; font-weight: bold; min-width: 280px; text-align: center; }}
  #main {{
    display: flex; flex: 1; overflow: hidden;
  }}
  #map-panel {{
    flex: 3; display: flex; align-items: center; justify-content: center;
    background: white; overflow: auto; padding: 8px;
  }}
  #map-panel svg {{ max-width: 100%; height: auto; }}
  #transcript-panel {{
    flex: 2; overflow-y: auto; padding: 16px; background: #1e1e1e;
    border-left: 2px solid #333; font-size: 13px; line-height: 1.5;
    min-width: 320px;
  }}
  .round-header {{
    font-size: 16px; font-weight: bold; color: #e0e0e0;
    margin: 12px 0 8px 0; padding-bottom: 4px;
    border-bottom: 1px solid #444;
  }}
  .round-header:first-child {{ margin-top: 0; }}
  .section-label {{ color: #6ab0de; font-weight: bold; margin-top: 8px; }}
  .initiative {{ color: #888; font-style: italic; }}
  .combat-hit {{ color: #5cb85c; }}
  .combat-miss {{ color: #888; }}
  .damage-line {{ color: #d9534f; }}
  .unit-status {{ margin: 2px 0; }}
  .unit-name {{ font-weight: bold; }}
  .armor-good {{ color: #5cb85c; }}
  .armor-mid {{ color: #f0ad4e; }}
  .armor-low {{ color: #d9534f; }}
  .rl-section {{ background: #252535; border-radius: 4px; padding: 6px 8px; margin: 6px 0; }}
  .rl-step {{ margin: 2px 0; }}
  .rl-step.current {{ background: #333355; border-radius: 3px; padding: 2px 4px; }}
  .reward-pos {{ color: #5cb85c; }}
  .reward-neg {{ color: #d9534f; }}
  .reward-zero {{ color: #888; }}
  .components {{ color: #999; font-size: 12px; margin-left: 12px; }}
  .loc-destroyed {{ color: #d9534f; font-weight: bold; }}
  .loc-internal {{ color: #f0ad4e; }}
  .prone-tag {{ color: #f0ad4e; font-weight: bold; }}
  .destroyed-tag {{ color: #d9534f; font-weight: bold; }}
  .outcome-bar {{
    padding: 8px; text-align: center; font-weight: bold; font-size: 15px;
    border-top: 1px solid #333;
  }}
  .outcome-WIN {{ background: #1a3a1a; color: #5cb85c; }}
  .outcome-LOSS {{ background: #3a1a1a; color: #d9534f; }}
  .outcome-DRAW {{ background: #3a3a1a; color: #f0ad4e; }}
  .outcome-UNKNOWN {{ background: #2a2a2a; color: #888; }}
  #help {{ padding: 6px; text-align: center; font-size: 11px; color: #666; }}
  @media (max-width: 900px) {{
    #main {{ flex-direction: column; }}
    #transcript-panel {{ border-left: none; border-top: 2px solid #333; max-height: 40vh; }}
  }}
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
<div id="main">
  <div id="map-panel"></div>
  <div id="transcript-panel"></div>
</div>
<div id="outcome-bar" class="outcome-bar"></div>
<div id="help">Arrow keys: navigate &nbsp;|&nbsp; Home/End: jump &nbsp;|&nbsp; Hex map + transcript synced by round</div>
<script>
const frames = [{frames_array}];
const stepMeta = {step_meta_json};
const roundData = {round_data_json};
const showVerbose = {verbose_js};
const gameOutcome = {json.dumps(outcome)};

// Build round lookup: round_num -> roundData entries
const roundLookup = {{}};
for (const rd of roundData) {{
  const key = rd.round;
  if (!roundLookup[key]) roundLookup[key] = [];
  roundLookup[key].push(rd);
}}

let idx = 0;

function escHtml(s) {{
  return String(s).replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;');
}}

function rewardClass(v) {{
  return v > 0.0001 ? 'reward-pos' : (v < -0.0001 ? 'reward-neg' : 'reward-zero');
}}

function armorClass(pct) {{
  return pct <= 25 ? 'armor-low' : (pct <= 50 ? 'armor-mid' : 'armor-good');
}}

function renderRound(rd, currentStepIdx) {{
  let h = '';
  const label = rd.is_final ? `Final attacks (Round ${{rd.round}})` : `Round ${{rd.round}}`;
  h += `<div class="round-header">${{label}}</div>`;

  // Initiative
  if (rd.initiative) {{
    const rolls = Object.entries(rd.initiative.rolls).map(([n,r]) => `${{n}}=${{r}}`).join(', ');
    const first = rd.initiative.first_mover ? `${{rd.initiative.first_mover}} moves first` : '';
    h += `<div class="initiative">${{first}} (rolled ${{rolls}})</div>`;
  }}

  // Movement
  if (rd.movement && rd.movement.length > 0) {{
    h += `<div class="section-label">Movement</div>`;
    for (const m of rd.movement) {{
      let line = escHtml(m.name) + ': ';
      if (m.prone_change === 'fell') line += '<span class="prone-tag">FELL PRONE</span> ';
      else if (m.prone_change === 'stood') line += '<span class="armor-good">stood up</span> ';
      if (m.from_pos && m.to_pos) {{
        line += `(${{m.from_pos[0]}},${{m.from_pos[1]}}) &rarr; (${{m.to_pos[0]}},${{m.to_pos[1]}}), `;
      }}
      line += `facing ${{m.facing}}, ${{m.move_type}}`;
      h += `<div>${{line}}</div>`;
    }}
  }}

  // Combat
  if (rd.combat && rd.combat.length > 0) {{
    h += `<div class="section-label">Weapons fire</div>`;
    for (const c of rd.combat) {{
      const tohit = c.tohit ? `needs ${{c.tohit}}` : '?';
      const roll = c.roll ? `rolls ${{c.roll}}` : '?';
      const resultCls = c.result === 'HIT' ? 'combat-hit' : 'combat-miss';
      let extra = '';
      if (c.result === 'HIT' && c.location) extra = ` (${{escHtml(c.location)}})`;
      if (c.result === 'HIT' && c.missiles) extra = ` (${{c.missiles}} missile(s))`;
      h += `<div class="${{resultCls}}">${{escHtml(c.attacker)}} fires ${{escHtml(c.weapon)}} at ${{escHtml(c.target)}}: ${{tohit}}, ${{roll}} &rarr; ${{c.result}}${{extra}}</div>`;
    }}
  }}

  // Damage (verbose)
  if (showVerbose && rd.damage && rd.damage.length > 0) {{
    h += `<div class="section-label">Damage</div>`;
    for (const d of rd.damage) {{
      h += `<div class="damage-line">${{escHtml(d.entity)}} takes ${{d.amount}} to ${{escHtml(d.location)}}</div>`;
    }}
  }}

  // Unit status
  if (rd.unit_status && rd.unit_status.length > 0) {{
    h += `<div class="section-label">Unit status</div>`;
    for (const u of rd.unit_status) {{
      let line = `<span class="unit-name">${{escHtml(u.name)}}</span>: `;
      line += `armor ${{u.armor_current}}/${{u.armor_max}} (<span class="${{armorClass(u.pct)}}">${{u.pct}}%</span>), `;
      line += `IS ${{u.internal_current}}/${{u.internal_max}}, heat ${{u.heat}}`;
      if (u.destroyed) line += ' <span class="destroyed-tag">DESTROYED</span>';
      if (u.prone) line += ' <span class="prone-tag">PRONE</span>';
      if (u.damaged_locs && u.damaged_locs.length > 0) {{
        const parts = u.damaged_locs.map(d => {{
          if (d.type === 'internal' && d.destroyed) return `<span class="loc-destroyed">${{d.loc}}:DESTROYED</span>`;
          if (d.type === 'internal') return `<span class="loc-internal">${{d.loc}}(IS):${{d.current}}/${{d.max}}</span>`;
          return `${{d.loc}}:${{d.current}}/${{d.max}}`;
        }});
        line += ` [${{parts.join(', ')}}]`;
      }}
      h += `<div class="unit-status">${{line}}</div>`;
    }}
  }}

  // RL steps
  if (rd.rl_steps && rd.rl_steps.length > 0) {{
    h += `<div class="section-label">RL Agent</div><div class="rl-section">`;
    for (let si = 0; si < rd.rl_steps.length; si++) {{
      const s = rd.rl_steps[si];
      const cls = 'rl-step';
      const rCls = rewardClass(s.reward);
      let line = `${{s.phase}}: action ${{s.action}}/${{s.n_legal_moves}} legal`;
      line += ` &rarr; reward <span class="${{rCls}}">${{s.reward >= 0 ? '+' : ''}}${{s.reward.toFixed(3)}}</span>`;
      h += `<div class="${{cls}}">${{line}}</div>`;
      if (s.components && s.components.length > 0) {{
        const parts = s.components.map(c => `${{c.name}}=${{c.value >= 0 ? '+' : ''}}${{c.value.toFixed(3)}}`);
        h += `<div class="components">(${{parts.join(', ')}})</div>`;
      }}
    }}
    // Cumulative from last step
    const last = rd.rl_steps[rd.rl_steps.length - 1];
    const cumCls = rewardClass(last.cumulative);
    h += `<div>Cumulative: <span class="${{cumCls}}">${{last.cumulative >= 0 ? '+' : ''}}${{last.cumulative.toFixed(3)}}</span></div>`;
    h += `</div>`;
  }}

  return h;
}}

function show(i) {{
  idx = Math.max(0, Math.min(frames.length - 1, i));
  document.getElementById("map-panel").innerHTML = frames[idx];
  document.getElementById("btn-prev").disabled = (idx === 0);
  document.getElementById("btn-next").disabled = (idx === frames.length - 1);

  // Determine current round from stepMeta
  const meta = stepMeta[idx] || {{}};
  const roundNum = meta.round || 0;
  const phase = meta.phase || '';
  document.getElementById("step-label").textContent =
    `Step ${{idx+1}} / ${{frames.length}}  (Round ${{roundNum}}${{phase ? ', ' + phase : ''}})`;

  // Render transcript for this round (and previous rounds for context)
  let html = '';
  const allRounds = [...new Set(roundData.map(r => r.round))].sort((a,b) => a - b);
  for (const rn of allRounds) {{
    const entries = roundLookup[rn] || [];
    for (const rd of entries) {{
      html += renderRound(rd, idx);
    }}
    if (rn >= roundNum && !entries.some(e => e.is_final)) break;
  }}

  const tp = document.getElementById("transcript-panel");
  tp.innerHTML = html;

  // Auto-scroll to current round header
  const headers = tp.querySelectorAll('.round-header');
  for (const h of headers) {{
    if (h.textContent.includes('Round ' + roundNum)) {{
      h.scrollIntoView({{ behavior: 'smooth', block: 'start' }});
      break;
    }}
  }}

  // Outcome bar
  const bar = document.getElementById("outcome-bar");
  bar.className = 'outcome-bar outcome-' + gameOutcome;
  bar.textContent = gameOutcome;
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


def main():
    args = parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)

    # Load config and enable saves
    if args.config:
        cfg = MegaMekConfig.load(args.config)
    else:
        cfg = MegaMekConfig()
        cfg.rl_unit = "Commando COM-2D"
        cfg.opponent_unit = "Commando COM-2D"
        cfg.max_game_rounds = 40
        cfg.firing_strategy = "naive"

    cfg.megamek_dir = args.megamek_dir
    if args.port is not None:
        cfg.rl_port = args.port
    cfg.env_index = 0
    cfg.max_rotating_round_saves = 100
    cfg.save_budget_mb = 10000
    cfg.enable_game_reports = True

    env = gymnasium.make("MegaMekGym/MegaMek-v0", config=cfg)

    # Load trained policy if provided
    agent = None
    device = None
    if args.checkpoint:
        obs_size = env.observation_space.shape[0]
        action_size = env.action_space.n
        agent, checkpoint, device = load_agent(args.checkpoint, obs_size, action_size)
        print(f"Loaded checkpoint: {args.checkpoint}")
        print(f"  global_step={checkpoint.get('global_step', '?')}, "
              f"deterministic={args.deterministic}")

    print(f"Playing game: {cfg.rl_unit} vs {cfg.opponent_unit}")
    print(f"  config: {args.config or 'defaults'}")
    print(f"  policy: {'checkpoint' if agent else 'random'}")

    # Clear stale saves
    port = args.port or cfg.rl_port
    clear_saves(args.megamek_dir, port)

    t0 = time.monotonic()
    snapshots, step_log, outcome, game_rounds = play_game(
        env, agent, device, args.deterministic, args.max_steps
    )
    elapsed = time.monotonic() - t0

    print(f"\nGame: {outcome} in {game_rounds} rounds, {len(snapshots)} snapshots, "
          f"{len(step_log)} steps ({elapsed:.1f}s)")

    # Collect saves into a descriptive directory
    output_stem = Path(args.output).stem
    save_dir = Path(f"{output_stem}_saves_{outcome}")
    n_saves = collect_saves(args.megamek_dir, port, save_dir)
    print(f"Collected {n_saves} save files → {save_dir}/")

    env.close()

    if not snapshots:
        print("No movement snapshots collected!")
        return

    # Parse saves for transcript data
    save_files = sorted(
        save_dir.glob("Round-*.sav.gz"),
        key=lambda p: int(re.match(r"Round-(\d+)-", p.name).group(1)),
    )
    saves = [parse_save(sf) for sf in save_files]

    autosave_files = sorted(save_dir.glob("autosave_*.sav.gz"))
    autosave_path = autosave_files[-1] if autosave_files else None

    round_data = build_round_transcript(saves, step_log, autosave_path)

    # Build step metadata for JS
    step_meta = []
    for snap in snapshots:
        step_meta.append({
            "round": snap.game_round,
            "phase": snap.phase,
        })

    # Auto-detect walk MP
    walk_mp = detect_walk_mp(snapshots[0].units, snapshots[0].rl_owner_id)
    print(f"Walk MP: {walk_mp}")

    # Render hex map SVGs
    print(f"Rendering {len(snapshots)} hex map frames...")
    svgs = render_all_svgs(snapshots, walk_mp, args.hex_size,
                           args.show_coords, args.show_elevation)

    # Generate HTML
    title = f"Game Viewer - {cfg.rl_unit} vs {cfg.opponent_unit}"
    html = build_combined_html(svgs, step_meta, round_data, outcome, title, args.verbose)

    with open(args.output, "w") as f:
        f.write(html)
    print(f"Saved to {args.output}")
    print(f"Save files: {save_dir}/")

    # Open in browser (WSL2-aware)
    import os
    import shutil
    import subprocess
    abs_path = os.path.abspath(args.output)

    opened = False
    if shutil.which("wslview"):
        subprocess.Popen(["wslview", abs_path],
                         stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        opened = True
    elif shutil.which("explorer.exe"):
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

    print("Opened in browser. Arrow keys to navigate.")


if __name__ == "__main__":
    main()
