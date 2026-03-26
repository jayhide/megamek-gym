#!/usr/bin/env python3
"""Dump the exact observation vector the model receives, with labeled indices.

Usage:
    poetry run python test_hierarchical_obs.py --megamek-dir ../megamek --steps 3
    poetry run python test_hierarchical_obs.py --megamek-dir ../megamek --steps 3 --flat
"""

import argparse
import numpy as np
import gymnasium

import megamek_gym  # noqa: F401
from megamek_gym.config import MegaMekConfig
from megamek_gym.observation import (
    _board_block_size, _group_moves_by_destination, format_observation,
    UNIT_FEATURES, GLOBAL_FEATURES, DEST_FEATURES, FACING_FEATURES,
    MOVE_FEATURES, MOVE_FEATURE_NAMES, MAX_ARMOR_LOCATIONS, MAX_WEAPONS,
)

_FACING_NAMES = ["N", "NE", "SE", "S", "SW", "NW"]

# Armor location names in MegaMek order (matches Java serialization)
_LOC_NAMES = ["HD", "CT", "RT", "LT", "RA", "LA", "RL", "LL"]


def _print_board(obs, bw, bh):
    """Print board elevation grid with indices."""
    board_block = _board_block_size(bw, bh)
    print(f"\n{'=' * 70}")
    if board_block == 0:
        print(f"BOARD ELEVATIONS  (disabled — INCLUDE_BOARD_ELEVATION=False)")
        print(f"{'=' * 70}")
        return
    print(f"BOARD ELEVATIONS  obs[0:{board_block}]  ({bw}x{bh} = {board_block} floats, normalized /10)")
    print(f"{'=' * 70}")
    board = obs[0:board_block].reshape(bh, bw)
    has_nonzero = False
    for y in range(bh):
        row = board[y]
        if np.any(row != 0.0):
            vals = " ".join(f"{v:+.1f}" if v != 0.0 else "  . " for v in row)
            base_idx = y * bw
            print(f"  y={y:2d} obs[{base_idx:3d}:{base_idx + bw:3d}]  {vals}")
            has_nonzero = True
    if not has_nonzero:
        print("  (all zero — flat board)")


def _print_unit(obs, offset, label, bw, bh, raw_unit=None):
    """Print all 55 unit features with indices, labels, and denormalized values."""
    end = offset + UNIT_FEATURES
    print(f"\n{'=' * 70}")
    name = ""
    if raw_unit:
        chassis = raw_unit.get("chassis", "?")
        model = raw_unit.get("model", "")
        name = f" — {chassis} {model}".strip()
    print(f"{label} FEATURES  obs[{offset}:{end}]  ({UNIT_FEATURES} floats){name}")
    print(f"{'=' * 70}")

    i = offset

    # Position (2)
    x_norm, y_norm = obs[i], obs[i + 1]
    print(f"  obs[{i:4d}] x/W              = {x_norm:.4f}  (x={x_norm * bw:.0f})")
    print(f"  obs[{i + 1:4d}] y/H              = {y_norm:.4f}  (y={y_norm * bh:.0f})")
    i += 2

    # Facing one-hot (6)
    active = [f for f in range(6) if obs[i + f] == 1.0]
    facing_str = _FACING_NAMES[active[0]] if active else "none"
    for f in range(6):
        marker = " <--" if obs[i + f] == 1.0 else ""
        print(f"  obs[{i + f:4d}] facing_{_FACING_NAMES[f]:<3s}        = {obs[i + f]:.0f}{marker}")
    i += 6

    # Movement points (3)
    print(f"  obs[{i:4d}] mp_walk/20       = {obs[i]:.4f}  (walk={obs[i] * 20:.0f})")
    print(f"  obs[{i + 1:4d}] mp_run/20        = {obs[i + 1]:.4f}  (run={obs[i + 1] * 20:.0f})")
    print(f"  obs[{i + 2:4d}] mp_jump/20       = {obs[i + 2]:.4f}  (jump={obs[i + 2] * 20:.0f})")
    i += 3

    # Heat (1)
    print(f"  obs[{i:4d}] heat/30          = {obs[i]:.4f}  (heat={obs[i] * 30:.0f})")
    i += 1

    # Status flags (4)
    flag_names = ["prone", "destroyed", "deployed", "retreated"]
    for fi, fname in enumerate(flag_names):
        val = obs[i + fi]
        marker = " <-- TRUE" if val == 1.0 else ""
        print(f"  obs[{i + fi:4d}] {fname:<16s} = {val:.0f}{marker}")
    i += 4

    # Armor locations (8 × 4 = 32)
    loc_names = _LOC_NAMES
    raw_armor = raw_unit.get("armor", []) if raw_unit else []
    for loc_idx in range(MAX_ARMOR_LOCATIONS):
        loc_name = loc_names[loc_idx] if loc_idx < len(loc_names) else f"L{loc_idx}"
        a_frac = obs[i]
        is_frac = obs[i + 1]
        ra_frac = obs[i + 2]
        destroyed = obs[i + 3]

        # Denormalize using raw data if available
        if loc_idx < len(raw_armor):
            rl = raw_armor[loc_idx]
            a_max = rl.get("armor_max", 1)
            i_max = rl.get("internal_max", 1)
            ra_max = rl.get("rear_armor_max", 0)
            a_str = f"  ({a_frac * max(a_max, 1):.0f}/{a_max})"
            is_str = f"  ({is_frac * max(i_max, 1):.0f}/{i_max})"
            ra_str = f"  ({ra_frac * max(ra_max, 1):.0f}/{ra_max})" if ra_max > 0 else ""
        else:
            a_str = is_str = ra_str = ""

        dest_marker = " [DESTROYED]" if destroyed == 1.0 else ""
        print(f"  obs[{i:4d}] {loc_name:<3s} armor        = {a_frac:.4f}{a_str}")
        print(f"  obs[{i + 1:4d}] {loc_name:<3s} internal     = {is_frac:.4f}{is_str}")
        if ra_frac != 0.0 or (loc_idx < len(raw_armor) and raw_armor[loc_idx].get("rear_armor_max", 0) > 0):
            print(f"  obs[{i + 2:4d}] {loc_name:<3s} rear_armor   = {ra_frac:.4f}{ra_str}")
        else:
            print(f"  obs[{i + 2:4d}] {loc_name:<3s} rear_armor   = {ra_frac:.4f}")
        print(f"  obs[{i + 3:4d}] {loc_name:<3s} loc_destroyed = {destroyed:.0f}{dest_marker}")
        i += 4

    # Weapon destroyed flags (7)
    raw_weapons = raw_unit.get("weapons", []) if raw_unit else []
    for w_idx in range(MAX_WEAPONS):
        val = obs[i]
        if w_idx < len(raw_weapons):
            wname = raw_weapons[w_idx].get("name", "?")
            marker = " [DESTROYED]" if val == 1.0 else ""
            print(f"  obs[{i:4d}] wpn{w_idx} destroyed   = {val:.0f}  ({wname}){marker}")
        else:
            print(f"  obs[{i:4d}] wpn{w_idx} destroyed   = {val:.0f}  (no weapon)")
        i += 1

    assert i == end, f"Unit feature count mismatch: expected {end}, got {i}"


def _print_global(obs, offset):
    """Print global features."""
    print(f"\n{'=' * 70}")
    print(f"GLOBAL FEATURES  obs[{offset}:{offset + GLOBAL_FEATURES}]  ({GLOBAL_FEATURES} float)")
    print(f"{'=' * 70}")
    val = obs[offset]
    label = "RL moves FIRST" if val == 1.0 else "RL moves SECOND"
    print(f"  obs[{offset:4d}] rl_moves_first   = {val:.0f}  ({label})")


def _print_dest_features(obs, offset, max_dest, bw, bh, mask):
    """Print per-destination features with indices."""
    end = offset + max_dest * DEST_FEATURES
    n_valid = int(mask["dest_mask"].sum()) if isinstance(mask, dict) else int(mask.sum())
    print(f"\n{'=' * 70}")
    print(f"PER-DESTINATION FEATURES  obs[{offset}:{end}]  "
          f"({max_dest} slots x {DEST_FEATURES} features = {max_dest * DEST_FEATURES} floats, "
          f"{n_valid} valid)")
    print(f"{'=' * 70}")
    print(f"  {'':4s} {'idx range':>13s}  {'x/W':>6s} {'y/H':>6s} {'mp/20':>6s} "
          f"{'dist':>6s} {'cover':>6s} {'elev':>6s} {'los':>5s} "
          f"{'en_rq':>6s} {'best':>6s}  denorm")

    max_dim = max(bw, bh)
    for i in range(max_dest):
        d_base = offset + i * DEST_FEATURES
        vals = obs[d_base:d_base + DEST_FEATURES]
        if np.all(vals == 0.0):
            remaining = max_dest - i
            if remaining > 0:
                print(f"  ... remaining {remaining} slots are zero-padded "
                      f"obs[{d_base}:{end}]")
            break

        dest_mask = mask["dest_mask"] if isinstance(mask, dict) else mask
        masked = "MASKED" if not dest_mask[i] else ""

        # Denormalize
        dx = vals[0] * bw
        dy = vals[1] * bh
        mp = vals[2] * 20
        dist = vals[3] * max_dim
        cover_raw = vals[4] * 2
        elev_raw = vals[5] * 10
        los = vals[6]
        en_rq = vals[7]
        best_rq = vals[8]

        denorm = (f"({dx:.0f},{dy:.0f}) mp={mp:.0f} dist={dist:.0f} "
                  f"cover={cover_raw:.1f} elev_diff={elev_raw:+.0f} "
                  f"los={'Y' if los else 'N'} en_rq={en_rq:+.3f} best_rq={best_rq:+.3f}")

        print(f"  D{i:<3d} obs[{d_base:4d}:{d_base + DEST_FEATURES:4d}]  "
              f"{vals[0]:6.3f} {vals[1]:6.3f} {vals[2]:6.3f} "
              f"{vals[3]:6.3f} {vals[4]:6.3f} {vals[5]:6.3f} {vals[6]:5.1f} "
              f"{vals[7]:6.3f} {vals[8]:6.3f}  "
              f"{denorm}  {masked}")


def _print_facing_features(obs, facing_offset, max_dest, mask, selected_dest=None):
    """Print per-facing features with indices.

    If selected_dest is given, only show facings for that destination.
    Otherwise show all destinations (used for the initial observation).
    """
    end = facing_offset + max_dest * 6 * FACING_FEATURES
    facing_mask = mask["facing_mask"] if isinstance(mask, dict) else None

    if selected_dest is not None:
        # Show only the selected destination's facings
        print(f"\n{'=' * 70}")
        print(f"FACING FEATURES FOR SELECTED DEST {selected_dest}  "
              f"(facing head input: shared_features[512] + dest_features[{DEST_FEATURES}])")
        print(f"{'=' * 70}")
        print(f"  {'':4s} {'face':>4s} {'idx range':>13s}  {'rl_rq':>7s}  mask")

        for f in range(6):
            f_base = facing_offset + selected_dest * 6 * FACING_FEATURES + f * FACING_FEATURES
            vals = obs[f_base:f_base + FACING_FEATURES]
            is_masked_in = facing_mask[selected_dest, f] if facing_mask is not None else True
            fn = _FACING_NAMES[f]
            mask_str = "valid" if is_masked_in else "MASKED"
            anomaly = ""
            if not is_masked_in and np.any(vals != 0.0):
                anomaly = " !! MASKED BUT NONZERO"
            print(f"  D{selected_dest:<3d} {fn:>4s} obs[{f_base:4d}:{f_base + FACING_FEATURES:4d}]  "
                  f"{vals[0]:+7.3f}  {mask_str}{anomaly}")
    else:
        # Show all destinations (initial obs, no action yet)
        print(f"\n{'=' * 70}")
        print(f"PER-FACING FEATURES  obs[{facing_offset}:{end}]  "
              f"({max_dest} dests x 6 facings x {FACING_FEATURES} features "
              f"= {max_dest * 6 * FACING_FEATURES} floats)")
        print(f"{'=' * 70}")
        print(f"  {'':4s} {'face':>4s} {'idx range':>13s}  {'rl_rq':>7s}  mask")

        any_printed = False
        for i in range(max_dest):
            dest_has_data = False
            for f in range(6):
                f_base = facing_offset + i * 6 * FACING_FEATURES + f * FACING_FEATURES
                vals = obs[f_base:f_base + FACING_FEATURES]
                is_masked_in = facing_mask[i, f] if facing_mask is not None else True
                has_data = np.any(vals != 0.0)

                if has_data or is_masked_in:
                    fn = _FACING_NAMES[f]
                    mask_str = "valid" if is_masked_in else "MASKED"
                    anomaly = ""
                    if not is_masked_in and has_data:
                        anomaly = " !! MASKED BUT NONZERO"
                    print(f"  D{i:<3d} {fn:>4s} obs[{f_base:4d}:{f_base + FACING_FEATURES:4d}]  "
                          f"{vals[0]:+7.3f}  {mask_str}{anomaly}")
                    any_printed = True
                    dest_has_data = True

            # Stop after seeing an all-zero destination
            if not dest_has_data:
                remaining = max_dest - i
                if remaining > 0:
                    print(f"  ... remaining {remaining} destinations have no facing data")
                break

        if not any_printed:
            print("  (all zero)")


def _print_flat_move_features(obs, offset, max_moves, bw, bh, mask):
    """Print per-move features for flat action space."""
    end = offset + max_moves * MOVE_FEATURES
    n_valid = int(mask.sum())
    print(f"\n{'=' * 70}")
    print(f"PER-MOVE FEATURES  obs[{offset}:{end}]  "
          f"({max_moves} slots x {MOVE_FEATURES} features = {max_moves * MOVE_FEATURES} floats, "
          f"{n_valid} valid)")
    print(f"{'=' * 70}")
    short_names = ["dst_x", "dst_y", "facing", "mp", "dist", "rl_rq", "en_rq", "cover", "elev", "los"]
    header = " ".join(f"{n:>7s}" for n in short_names)
    print(f"  {'':4s} {'idx range':>15s}  {header}")
    print(f"  {'':4s} {'':>15s}  {'x/W':>7s} {'y/H':>7s} {'/5':>7s} {'/20':>7s} "
          f"{'/maxD':>7s} {'[-½,1]':>7s} {'[-½,1]':>7s} {'/2':>7s} {'/10':>7s} {'0|1':>7s}")

    max_dim = max(bw, bh)
    for i in range(max_moves):
        m_base = offset + i * MOVE_FEATURES
        vals = obs[m_base:m_base + MOVE_FEATURES]
        if np.all(vals == 0.0):
            remaining = max_moves - i
            if remaining > 0:
                print(f"  ... remaining {remaining} slots are zero-padded "
                      f"obs[{m_base}:{end}]")
            break

        is_valid = mask[i] if i < len(mask) else False
        masked_str = "" if is_valid else "MASKED"

        # Denormalize
        dx = vals[0] * bw
        dy = vals[1] * bh
        facing = vals[2] * 5
        mp = vals[3] * 20

        vals_str = " ".join(f"{v:+7.3f}" for v in vals)
        denorm = (f"({dx:.0f},{dy:.0f}) f={_FACING_NAMES[int(round(facing))]} "
                  f"mp={mp:.0f}")

        print(f"  M{i:<3d} obs[{m_base:4d}:{m_base + MOVE_FEATURES:4d}]  "
              f"{vals_str}  {denorm}  {masked_str}")


def _print_sanity_checks(obs, obs_size, mask, cfg, hierarchical):
    """Print sanity checks to flag anomalies."""
    print(f"\n{'=' * 70}")
    print("SANITY CHECKS")
    print(f"{'=' * 70}")

    issues = []

    # 1. Obs size
    if len(obs) != obs_size:
        issues.append(f"  FAIL: obs length {len(obs)} != expected {obs_size}")
    else:
        print(f"  OK: obs length = {obs_size}")

    bw = cfg.resolved_board_width
    bh = cfg.resolved_board_height
    base = _board_block_size(bw, bh) + 2 * UNIT_FEATURES + GLOBAL_FEATURES

    if hierarchical:
        max_dest = cfg.max_destinations
        dest_mask = mask["dest_mask"]
        facing_mask = mask["facing_mask"]
        n_valid_dests = int(dest_mask.sum())

        # Count non-zero destinations in obs
        n_nonzero = 0
        for i in range(max_dest):
            d_base = base + i * DEST_FEATURES
            if np.any(obs[d_base:d_base + DEST_FEATURES] != 0.0):
                n_nonzero += 1
            else:
                break

        if n_nonzero == n_valid_dests:
            print(f"  OK: {n_valid_dests} valid dests match non-zero feature count")
        else:
            issues.append(f"  FAIL: dest_mask has {n_valid_dests} valid but "
                          f"{n_nonzero} non-zero feature blocks")

        # Check facing consistency: masked-out facings should not have data
        facing_offset = base + max_dest * DEST_FEATURES
        masked_nonzero = 0
        for i in range(n_valid_dests):
            for f in range(6):
                f_base = facing_offset + i * 6 * FACING_FEATURES + f * FACING_FEATURES
                vals = obs[f_base:f_base + FACING_FEATURES]
                has_data = np.any(vals != 0.0)
                is_valid = facing_mask[i, f]
                # Valid facings with zero features are normal (long range = 0.0)
                if not is_valid and has_data:
                    masked_nonzero += 1
                    issues.append(f"  FAIL: D{i} {_FACING_NAMES[f]} masked out "
                                  f"but has nonzero features")
        if masked_nonzero == 0:
            print(f"  OK: facing features consistent with mask")

        # Check mask beyond valid range is all False
        if np.any(dest_mask[n_valid_dests:]):
            issues.append(f"  FAIL: dest_mask has True values beyond valid range")
        else:
            print(f"  OK: dest_mask clean beyond valid range")

    else:
        n_valid = int(mask.sum())
        n_nonzero = 0
        for i in range(cfg.max_legal_moves):
            m_base = base + i * MOVE_FEATURES
            if np.any(obs[m_base:m_base + MOVE_FEATURES] != 0.0):
                n_nonzero += 1
            else:
                break

        if n_nonzero == n_valid:
            print(f"  OK: {n_valid} valid moves match non-zero feature count")
        else:
            issues.append(f"  FAIL: mask has {n_valid} valid but "
                          f"{n_nonzero} non-zero feature blocks")

    if issues:
        print()
        for issue in issues:
            print(issue)
    else:
        print("  ALL CHECKS PASSED")


def _print_raw_summary(raw_obs, rl_owner_id):
    """Print a condensed raw JSON summary for cross-reference."""
    print(f"\n{'=' * 70}")
    print("RAW JSON SUMMARY (for cross-reference with obs vector)")
    print(f"{'=' * 70}")

    rnd = raw_obs.get("round", "?")
    phase = raw_obs.get("phase", "?")
    print(f"  Round {rnd}, Phase {phase}")

    units = raw_obs.get("units", [])
    for u in units:
        is_rl = u.get("owner") == rl_owner_id
        label = "RL" if is_rl else "EN"
        chassis = u.get("chassis", "?")
        model = u.get("model", "")
        x, y = u.get("x", -1), u.get("y", -1)
        facing = u.get("facing", 0)
        fn = _FACING_NAMES[facing] if 0 <= facing < 6 else str(facing)
        mp_w = u.get("mp_walk", 0)
        mp_r = u.get("mp_run", 0)
        heat = u.get("heat", 0)
        prone = u.get("prone", False)

        armor_locs = u.get("armor", [])
        armor_parts = []
        for loc in armor_locs:
            abbr = loc.get("location", "?")
            a = loc.get("armor", 0)
            a_max = loc.get("armor_max", 0)
            i_val = loc.get("internal", 0)
            i_max = loc.get("internal_max", 0)
            destroyed = a <= 0 and i_val <= 0
            part = f"{abbr}={a}/{a_max}"
            if destroyed:
                part += "[X]"
            armor_parts.append(part)

        print(f"  [{label}] {chassis} {model}: ({x},{y}) f={fn} "
              f"mp={mp_w}/{mp_r} heat={heat}"
              f"{' PRONE' if prone else ''}")
        print(f"       Armor: {' '.join(armor_parts)}")

    moves = raw_obs.get("legal_moves", [])
    print(f"  Legal moves: {len(moves)}")
    print(f"  rl_moves_first: {raw_obs.get('rl_moves_first', '?')}")

    prev_ex = raw_obs.get("prev_round_enemy_x", -1)
    prev_ey = raw_obs.get("prev_round_enemy_y", -1)
    prev_ef = raw_obs.get("prev_round_enemy_facing", -1)
    if prev_ex >= 0:
        fn = _FACING_NAMES[prev_ef] if 0 <= prev_ef < 6 else str(prev_ef)
        print(f"  prev_round_enemy: ({prev_ex},{prev_ey}) f={fn}")


def print_step(obs, mask, cfg, raw_obs, rl_owner_id, hierarchical, selected_dest=None):
    """Print full annotated observation for one step."""
    bw = cfg.resolved_board_width
    bh = cfg.resolved_board_height
    board_block = _board_block_size(bw, bh)
    base = board_block + 2 * UNIT_FEATURES + GLOBAL_FEATURES

    # Raw JSON summary first
    _print_raw_summary(raw_obs, rl_owner_id)

    # Board
    _print_board(obs, bw, bh)

    # Units from raw obs for denormalization context
    units = raw_obs.get("units", [])
    rl_unit = enemy_unit = None
    for u in units:
        if u.get("owner") == rl_owner_id:
            rl_unit = u
        else:
            enemy_unit = u

    # RL unit features
    rl_offset = board_block
    _print_unit(obs, rl_offset, "RL UNIT", bw, bh, rl_unit)

    # Enemy unit features
    en_offset = rl_offset + UNIT_FEATURES
    _print_unit(obs, en_offset, "ENEMY UNIT", bw, bh, enemy_unit)

    # Global
    global_offset = en_offset + UNIT_FEATURES
    _print_global(obs, global_offset)

    # Move/dest features
    if hierarchical:
        _print_dest_features(obs, base, cfg.max_destinations, bw, bh, mask)
        facing_offset = base + cfg.max_destinations * DEST_FEATURES
        _print_facing_features(obs, facing_offset, cfg.max_destinations, mask, selected_dest)

        if hierarchical:
            from megamek_gym.observation import compute_obs_size_hierarchical
            expected = compute_obs_size_hierarchical(bw, bh, cfg.max_destinations)
    else:
        _print_flat_move_features(obs, base, cfg.max_legal_moves, bw, bh, mask)
        from megamek_gym.observation import compute_obs_size
        expected = compute_obs_size(bw, bh, cfg.max_legal_moves)

    # Sanity checks
    _print_sanity_checks(obs, expected, mask, cfg, hierarchical)


def main():
    parser = argparse.ArgumentParser(
        description="Dump the exact observation vector the model receives, with labeled indices."
    )
    parser.add_argument("--megamek-dir", type=str, default=None,
                        help="Path to megamek repo (overrides config)")
    parser.add_argument("--port", type=int, default=None,
                        help="Port for Java bridge (overrides config)")
    parser.add_argument("--config", type=str, default=None,
                        help="YAML config file (e.g. configs/default.yaml)")
    parser.add_argument("--steps", type=int, default=5, help="Number of steps to show")
    parser.add_argument("--flat", action="store_true",
                        help="Use flat action space instead of hierarchical")
    args = parser.parse_args()

    # Load config: from YAML if provided, otherwise defaults
    if args.config:
        cfg = MegaMekConfig.load(args.config)
    else:
        cfg = MegaMekConfig(
            max_game_rounds=10,
            rl_fixed_coords=(8, 2),
            opponent_fixed_coords=(8, 14),
        )

    # CLI overrides
    if args.megamek_dir is not None:
        cfg.megamek_dir = args.megamek_dir
    if args.port is not None:
        cfg.rl_port = args.port
    if args.flat:
        cfg.action_space_type = "flat"

    hierarchical = cfg.action_space_type == "hierarchical"

    env = gymnasium.make("MegaMekGym/MegaMek-v0", config=cfg)
    try:
        obs, info = env.reset()
        inner = env.unwrapped
        raw_obs = inner._last_raw_obs
        rl_owner_id = inner._rl_owner_id

        if hierarchical:
            mask = info["action_mask"]  # {"dest_mask": ..., "facing_mask": ...}
        else:
            mask = info["action_mask"]  # flat bool array

        print("=" * 70)
        print(f"OBSERVATION INSPECTOR — {'HIERARCHICAL' if hierarchical else 'FLAT'} MODE")
        print(f"  config: {args.config or '(defaults)'}")
        print(f"  units: {cfg.rl_unit} vs {cfg.opponent_unit}")
        print(f"  obs vector length: {len(obs)}")
        print(f"  board: {cfg.resolved_board_width}x{cfg.resolved_board_height}")
        if hierarchical:
            print(f"  max_destinations: {cfg.max_destinations}")
        else:
            print(f"  max_legal_moves: {cfg.max_legal_moves}")
        print("=" * 70)

        print_step(obs, mask, cfg, raw_obs, rl_owner_id, hierarchical)

        # Take some steps
        for step in range(1, args.steps + 1):
            if hierarchical:
                dest_mask = mask["dest_mask"]
                facing_mask = mask["facing_mask"]
                valid_dests = np.where(dest_mask)[0]
                dest = np.random.choice(valid_dests) if len(valid_dests) > 0 else 0
                valid_facings = np.where(facing_mask[dest])[0]
                facing = np.random.choice(valid_facings) if len(valid_facings) > 0 else 0
                action = np.array([dest, facing])
                action_str = f"dest={dest} ({_FACING_NAMES[facing]})"
            else:
                valid_moves = np.where(mask)[0]
                action = np.random.choice(valid_moves) if len(valid_moves) > 0 else 0
                action_str = f"move={action}"

            obs, reward, terminated, truncated, info = env.step(action)
            mask = info["action_mask"]

            print(f"\n\n{'#' * 70}")
            print(f"STEP {step}: action={action_str}, "
                  f"reward={reward:+.4f}, terminated={terminated}, truncated={truncated}")
            print(f"{'#' * 70}")

            if terminated or truncated:
                raw_obs = inner._last_raw_obs
                if raw_obs:
                    outcome = raw_obs.get("game_outcome", "?")
                    print(f"\n  Game ended: {outcome}")
                break

            raw_obs = inner._last_raw_obs
            chosen_dest = dest if hierarchical else None
            print_step(obs, mask, cfg, raw_obs, rl_owner_id, hierarchical, chosen_dest)

    finally:
        env.close()


if __name__ == "__main__":
    main()
