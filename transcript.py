#!/usr/bin/env python3
"""Generate human-readable transcripts from MegaMek save files.

Usage:
    poetry run python transcript.py inspected_games/game_001_DRAW/
    poetry run python transcript.py inspected_games/game_001_LOSS/ --verbose
"""

import argparse
import gzip
import html
import os
import re
import sys
from pathlib import Path


# ANSI color codes (disabled when output is not a terminal)
_USE_COLOR = sys.stdout.isatty() and os.environ.get("NO_COLOR") is None

def _ansi(code: str, text: str) -> str:
    return f"\033[{code}m{text}\033[0m" if _USE_COLOR else text

def red(text: str) -> str:
    return _ansi("1;31", text)

def green(text: str) -> str:
    return _ansi("32", text)

def yellow(text: str) -> str:
    return _ansi("33", text)

def cyan(text: str) -> str:
    return _ansi("36", text)

def bold(text: str) -> str:
    return _ansi("1", text)

def dim(text: str) -> str:
    return _ansi("2", text)


# BattleTech mech locations (indices 0-7 matching armor/internal arrays)
LOCATIONS = ["Head", "CT", "RT", "LT", "RA", "LA", "RL", "LL"]

# Movement type display names
MOVE_NAMES = {
    "MOVE_NONE": "stood still",
    "MOVE_WALK": "walked",
    "MOVE_RUN": "ran",
    "MOVE_JUMP": "jumped",
    "MOVE_SPRINT": "sprinted",
}

# Facing direction names (0-5)
FACING_NAMES = ["N", "NE", "SE", "S", "SW", "NW"]


def strip_html(s: str) -> str:
    """Strip HTML tags from a string, returning plain text."""
    # First unescape HTML entities
    s = html.unescape(s)
    # Remove all HTML tags
    return re.sub(r"<[^>]+>", "", s)


def extract_tooltip_number(s: str) -> str:
    """Extract the display number from a tooltip link like <a href='#tooltip:...'>11</a>."""
    s = html.unescape(s)
    m = re.search(r"<a[^>]*>(\d+)</a>", s)
    return m.group(1) if m else strip_html(s)


def extract_entity_name(s: str) -> str:
    """Extract entity name from span like <span class='entity-name'...><a...>Name</a></span>."""
    return strip_html(s)


def parse_int_array(text: str, tag: str) -> list[int]:
    """Parse an array of <int> values from an XML block like <armor id="..."><int>9</int>...</armor>."""
    # Find the tag block (non-greedy to get first match)
    pattern = rf"<{tag}\s[^>]*>(.*?)</{tag}>"
    m = re.search(pattern, text, re.DOTALL)
    if not m:
        return []
    return [int(x) for x in re.findall(r"<int>(-?\d+)</int>", m.group(1))]


def extract_units(text: str) -> list[dict]:
    """Extract unit state for all non-observer entities from save file XML."""
    units = []

    # Find all entity blocks (BipedMek, QuadMek, etc.)
    # They're inside <inGameObjects> with tags like <megamek.common.units.BipedMek id="...">
    entity_pattern = r"<megamek\.common\.units\.\w+Mek\s+id=\"\d+\">(.*?)</megamek\.common\.units\.\w+Mek>"
    for m in re.finditer(entity_pattern, text, re.DOTALL):
        block = m.group(1)

        # Display name
        name_m = re.search(r"<displayName>(.*?)</displayName>", block)
        if not name_m:
            continue
        name = name_m.group(1)

        # Owner ID
        owner_m = re.search(r"<ownerId>(\d+)</ownerId>", block)

        # Position
        pos_m = re.search(r"<position[^>]*>\s*<x>(\d+)</x>\s*<y>(\d+)</y>", block, re.DOTALL)
        pos = (int(pos_m.group(1)), int(pos_m.group(2))) if pos_m else None

        # Facing - the first <facing> after </position> that isn't -1
        # Find the position block end, then the next facing
        pos_end = block.find("</position>")
        if pos_end >= 0:
            facing_m = re.search(r"<facing>(\d+)</facing>", block[pos_end:])
            facing = int(facing_m.group(1)) if facing_m else 0
        else:
            facing = 0

        # Heat
        heat_m = re.search(r"<heat>(\d+)</heat>", block)
        heat = int(heat_m.group(1)) if heat_m else 0

        # Movement type (moved is reset at start of round; movedLastRound has prior)
        moved_m = re.search(r"<moved>(MOVE_\w+)</moved>", block)
        moved = moved_m.group(1) if moved_m else "MOVE_NONE"
        moved_last_m = re.search(r"<movedLastRound>(MOVE_\w+)</movedLastRound>", block)
        moved_last = moved_last_m.group(1) if moved_last_m else "MOVE_NONE"

        # Prone status
        prone_m = re.search(r"<prone>(true|false)</prone>", block)
        prone = prone_m is not None and prone_m.group(1) == "true"

        # Destroyed status
        destroyed_m = re.search(r"<destroyed>(true|false)</destroyed>", block)
        destroyed = destroyed_m and destroyed_m.group(1) == "true"

        # Armor array - need the top-level one, not nested equipment ones
        # The pattern is <armor id="NNN"><int>... right after <exposure>...</exposure>
        armor = parse_int_array(block, "armor")
        internal = parse_int_array(block, "internal")
        orig_armor = parse_int_array(block, "orig__armor")
        orig_internal = parse_int_array(block, "orig__internal")

        # Removal condition flags (bitfield)
        rc_m = re.search(r"<removalCondition>(\d+)</removalCondition>", block)
        removal_condition = int(rc_m.group(1)) if rc_m else 0

        # Retreat direction
        rd_m = re.search(r"<retreatedDirection>(\w+)</retreatedDirection>", block)
        retreated_direction = rd_m.group(1) if rd_m else "NONE"

        units.append({
            "name": name,
            "owner_id": int(owner_m.group(1)) if owner_m else -1,
            "pos": pos,
            "facing": facing,
            "heat": heat,
            "moved": moved,
            "moved_last": moved_last,
            "prone": prone,
            "destroyed": destroyed,
            "removal_condition": removal_condition,
            "retreated_direction": retreated_direction,
            "armor": armor,
            "internal": internal,
            "orig_armor": orig_armor,
            "orig_internal": orig_internal,
        })

    return units


def _unit_condition(unit: dict) -> str | None:
    """Determine what happened to a unit from its state, or None if alive/unknown."""
    # Direct evidence: destroyed flag
    if unit.get("destroyed"):
        return "destroyed"

    rc = unit.get("removal_condition", 0)
    # Removal condition bitfield values from MegaMek IEntityRemovalConditions:
    # 0x0100 (256) = retreat, 0x0110 (272) = pushed off map
    # 0x0200 (512) = salvageable (destroyed but salvageable)
    # 0x0210 (528) = ejection
    # 0x0400 (1024) = devastated (destroyed, not salvageable)
    if rc & 0x0400:
        return "devastated"
    if rc & 0x0200:
        if rc & 0x0010:
            return "pilot ejected"
        return "destroyed (salvageable)"
    if rc & 0x0100:
        if rc & 0x0010:
            return "pushed off map"
        return "fled the battlefield"

    rd = unit.get("retreated_direction", "NONE")
    if rd != "NONE":
        return "fled the battlefield"

    return None


def determine_end_condition(units: list[dict], outcome: str) -> str:
    """Determine why the game ended based on unit states and outcome.

    Args:
        units: Unit dicts from the final save file.
        outcome: "WIN", "LOSS", "DRAW", or "Unknown" (from directory name).

    Returns:
        Human-readable string describing the end condition.
    """
    if len(units) < 2:
        return ""

    # Identify RL bot vs opponent by name convention
    rl_unit = next((u for u in units if "(RLBot)" in u["name"]), units[0])
    opp_unit = next((u for u in units if "(Princess)" in u["name"]), units[1])

    rl_cond = _unit_condition(rl_unit)
    opp_cond = _unit_condition(opp_unit)

    # Direct evidence available
    if outcome == "WIN" and opp_cond:
        return f"opponent {opp_cond}"
    if outcome == "LOSS" and rl_cond:
        return f"mech {rl_cond}"
    if outcome == "DRAW":
        if rl_cond and opp_cond:
            return "mutual destruction"
        if not rl_cond and not opp_cond:
            return "timeout (both alive)"
        return rl_cond or opp_cond or ""

    # No direct evidence — infer from damage (saves are at round start,
    # so the final round's destruction usually isn't captured)
    if outcome in ("WIN", "LOSS"):
        # Compare CT internal structure ratios as a heuristic
        loser = opp_unit if outcome == "WIN" else rl_unit
        winner = rl_unit if outcome == "WIN" else opp_unit
        loser_label = "opponent" if outcome == "WIN" else "mech"

        loser_ct_ratio = _ct_ratio(loser)
        winner_ct_ratio = _ct_ratio(winner)
        total_dmg_loser = _total_damage(loser)
        total_dmg_winner = _total_damage(winner)

        # If loser's CT is more damaged, mention it
        if loser_ct_ratio is not None and loser_ct_ratio < 1.0:
            return f"{loser_label} destroyed (inferred — CT at {loser_ct_ratio:.0%})"
        if total_dmg_loser > total_dmg_winner:
            return f"{loser_label} destroyed (inferred)"
        return f"{loser_label} destroyed (inferred from final round)"

    return ""


def _ct_ratio(unit: dict) -> float | None:
    """Return CT internal structure as fraction of original, or None."""
    internal = unit.get("internal", [])
    orig = unit.get("orig_internal", [])
    if len(internal) > 1 and len(orig) > 1 and orig[1] > 0:
        return max(0, internal[1]) / orig[1]
    return None


def _total_damage(unit: dict) -> int:
    """Return total armor + internal damage taken."""
    armor = unit.get("armor", [])
    orig_armor = unit.get("orig_armor", [])
    internal = unit.get("internal", [])
    orig_internal = unit.get("orig_internal", [])
    dmg = 0
    if armor and orig_armor:
        dmg += sum(orig_armor) - sum(max(0, a) for a in armor)
    if internal and orig_internal:
        dmg += sum(orig_internal) - sum(max(0, s) for s in internal)
    return dmg


def extract_reports(text: str) -> list[dict]:
    """Extract all Report entries from gameReports section."""
    reports = []

    # Find the gameReports section
    gr_m = re.search(r"<gameReports[^>]*>(.*?)</gameReports>", text, re.DOTALL)
    if not gr_m:
        return reports

    gr_text = gr_m.group(1)

    # Parse each Report entry
    for rm in re.finditer(
        r"<megamek\.common\.Report[^>]*>(.*?)</megamek\.common\.Report>",
        gr_text,
        re.DOTALL,
    ):
        rblock = rm.group(1)

        msg_m = re.search(r"<messageId>(\d+)</messageId>", rblock)
        if not msg_m:
            continue

        msg_id = int(msg_m.group(1))

        # Extract tagData strings
        tag_data = []
        td_m = re.search(r"<tagData[^>]*>(.*?)</tagData>", rblock, re.DOTALL)
        if td_m:
            tag_data = re.findall(r"<string>(.*?)</string>", td_m.group(1), re.DOTALL)

        reports.append({"messageId": msg_id, "tagData": tag_data})

    return reports


def _tag_name(entity_name: str, player_name: str | None) -> str:
    """Append a short player tag to an entity name for disambiguation.

    Strips HTML from both, then appends '(RLBot)' or '(Princess)' etc.
    """
    if not player_name:
        return entity_name
    player = strip_html(player_name).strip()
    if not player:
        return entity_name
    return f"{entity_name} ({player})"


def decode_combat_reports(reports: list[dict]) -> list[str]:
    """Decode weapon attacks from report entries into human-readable lines.

    Processes reports sequentially:
    - 3100: firing header (attacker name)
    - 3115: weapon declaration (weapon name, target name)
    - 3150: to-hit number
    - 3155: dice roll
    - 3220: miss
    - 3325: missile hit (count)
    - 3345: hit end marker
    - 3405: direct hit (location)
    - 6065: damage to location
    """
    lines = []
    current_attacker = None
    current_weapon = None
    current_target = None
    current_tohit = None
    current_roll = None

    for r in reports:
        mid = r["messageId"]
        td = r["tagData"]

        if mid == 3100:
            # Weapons fire header: [entity_name_html, player_name_html]
            if td:
                player = td[1] if len(td) >= 2 else None
                current_attacker = _tag_name(extract_entity_name(td[0]), player)
            current_weapon = None
            current_target = None
            current_tohit = None
            current_roll = None

        elif mid == 3115:
            # Weapon attack: [weapon_name, target_html, target_player_html]
            if len(td) >= 2:
                current_weapon = td[0]
                target_player = td[2] if len(td) >= 3 else None
                current_target = _tag_name(extract_entity_name(td[1]), target_player)
            current_tohit = None
            current_roll = None

        elif mid == 3150:
            # To-hit number
            if td:
                current_tohit = extract_tooltip_number(td[0])

        elif mid == 3155:
            # Dice roll
            if td:
                current_roll = extract_tooltip_number(td[0])

        elif mid == 3220:
            # Miss
            if current_attacker and current_weapon and current_target:
                tohit_str = f"needs {current_tohit}" if current_tohit else "?"
                roll_str = f"rolls {current_roll}" if current_roll else "?"
                lines.append(
                    dim(f"    {current_attacker} fires {current_weapon} at "
                        f"{current_target}: {tohit_str}, {roll_str} -> MISS")
                )
            current_weapon = None

        elif mid == 3325:
            # Missile hit: [count, "missile(s)", side_table_info]
            count = td[0] if td else "?"
            side = ""
            if len(td) >= 3 and td[2].strip():
                side = f" {strip_html(td[2]).strip()}"
            if current_attacker and current_weapon and current_target:
                tohit_str = f"needs {current_tohit}" if current_tohit else "?"
                roll_str = f"rolls {current_roll}" if current_roll else "?"
                lines.append(
                    f"    {current_attacker} fires {cyan(current_weapon)} at "
                    f"{current_target}: {tohit_str}, {roll_str} -> {green('HIT')} "
                    f"({count} missile(s)){side}"
                )
            current_weapon = None

        elif mid == 3405:
            # Direct hit: [extra, location]
            loc = td[1] if len(td) >= 2 else ""
            if current_attacker and current_weapon and current_target:
                tohit_str = f"needs {current_tohit}" if current_tohit else "?"
                roll_str = f"rolls {current_roll}" if current_roll else "?"
                loc_str = f" ({loc})" if loc else ""
                lines.append(
                    f"    {current_attacker} fires {cyan(current_weapon)} at "
                    f"{current_target}: {tohit_str}, {roll_str} -> {green('HIT')}{loc_str}"
                )
            current_weapon = None

    return lines


def decode_damage_reports(reports: list[dict]) -> list[str]:
    """Extract damage reports (6065) into human-readable lines.

    Uses surrounding combat reports (3100/3115) to determine which player
    owns the damaged entity, since 6065 reports don't include player info.
    """
    lines = []
    # Track entity->player mapping from combat reports for disambiguation
    entity_player_map: dict[str, str] = {}
    for r in reports:
        mid = r["messageId"]
        td = r["tagData"]
        if mid == 3100 and len(td) >= 2:
            # Attacker: td[0]=entity, td[1]=player
            name = extract_entity_name(td[0])
            entity_player_map[name] = td[1]
        elif mid == 3115 and len(td) >= 3:
            # Target: td[1]=entity, td[2]=player
            name = extract_entity_name(td[1])
            entity_player_map[name] = td[2]
        elif mid == 6065 and len(td) >= 4:
            entity = extract_entity_name(td[0])
            player = entity_player_map.get(entity)
            tagged = _tag_name(entity, player)
            dmg = td[2]
            loc = td[3]
            lines.append(f"    {tagged} takes {dmg} damage to {loc}")
    return lines


def diff_units(prev_units: list[dict], curr_units: list[dict]) -> list[str]:
    """Compare unit state between rounds, produce movement lines."""
    lines = []

    prev_by_name = {u["name"]: u for u in prev_units}

    for curr in curr_units:
        name = curr["name"]
        prev = prev_by_name.get(name)
        if not prev:
            continue

        # Prone transitions
        was_prone = prev.get("prone", False)
        is_prone = curr.get("prone", False)
        if is_prone and not was_prone:
            lines.append(f"  {name}: {red('FELL PRONE')}")
        elif not is_prone and was_prone:
            lines.append(f"  {name}: {green('stood up')}")

        # Movement - use movedLastRound since saves are taken at start of round
        effective_moved = curr.get("moved_last", curr["moved"])
        if effective_moved == "MOVE_NONE":
            effective_moved = curr["moved"]
        if curr["pos"] != prev["pos"] or effective_moved != "MOVE_NONE":
            move_type = MOVE_NAMES.get(effective_moved, effective_moved)
            facing_str = FACING_NAMES[curr["facing"]] if 0 <= curr["facing"] < 6 else "?"
            prone_tag = f" {yellow('[PRONE]')}" if is_prone else ""
            if prev["pos"] and curr["pos"]:
                lines.append(
                    f"  {name}: ({prev['pos'][0]},{prev['pos'][1]}) -> "
                    f"({curr['pos'][0]},{curr['pos'][1]}), "
                    f"facing {facing_str}, {move_type}{prone_tag}"
                )
            elif curr["pos"]:
                lines.append(
                    f"  {name}: -> ({curr['pos'][0]},{curr['pos'][1]}), "
                    f"facing {facing_str}, {move_type}{prone_tag}"
                )

    return lines


def armor_summary(unit: dict) -> str:
    """Generate armor/internal summary string for a unit."""
    if not unit["armor"] or not unit["orig_armor"]:
        return f"{unit['name']}: no armor data"

    total_armor = sum(max(0, v) for v in unit["armor"])
    total_orig = sum(unit["orig_armor"])
    total_internal = sum(max(0, v) for v in unit["internal"]) if unit["internal"] else 0
    total_orig_internal = sum(unit["orig_internal"]) if unit["orig_internal"] else 0

    pct = (total_armor / total_orig * 100) if total_orig > 0 else 0

    # Color-code armor percentage
    pct_str = f"{pct:.0f}%"
    if pct <= 25:
        pct_str = red(pct_str)
    elif pct <= 50:
        pct_str = yellow(pct_str)
    elif pct < 100:
        pct_str = green(pct_str)

    # Color-code heat
    heat_str = str(unit["heat"])
    if unit["heat"] >= 14:
        heat_str = red(heat_str)
    elif unit["heat"] >= 5:
        heat_str = yellow(heat_str)

    # Show damaged locations
    damaged = []
    for i, loc in enumerate(LOCATIONS):
        if i < len(unit["armor"]) and i < len(unit["orig_armor"]):
            a = max(0, unit["armor"][i])
            orig_a = unit["orig_armor"][i]
            if a < orig_a:
                damaged.append(f"{loc}:{a}/{orig_a}")
        if i < len(unit["internal"]) and i < len(unit["orig_internal"]):
            s = max(0, unit["internal"][i])
            orig_s = unit["orig_internal"][i]
            if s < orig_s:
                if s == 0:
                    damaged.append(red(f"{loc}:DESTROYED"))
                else:
                    damaged.append(yellow(f"{loc}(IS):{s}/{orig_s}"))

    dmg_str = f" [{', '.join(damaged)}]" if damaged else ""
    status_tags = []
    if unit.get("destroyed"):
        status_tags.append(red("DESTROYED"))
    if unit.get("prone"):
        status_tags.append(yellow("PRONE"))
    status_str = f" ** {', '.join(status_tags)} **" if status_tags else ""

    return (
        f"  {bold(unit['name'])}: armor {total_armor}/{total_orig} ({pct_str}), "
        f"internals {total_internal}/{total_orig_internal}, "
        f"heat {heat_str}{dmg_str}{status_str}"
    )


def extract_initiative(text: str) -> dict | None:
    """Extract initiative rolls and first mover from save file XML.

    Returns dict with:
        rolls: {player_name: roll_value}
        first_mover_name: name of the player who moves first
        first_mover_id: player ID of the first mover
    Or None if initiative data is not found.
    """
    # Build player ID -> name mapping (skip watcher/observer)
    players = {}
    for m in re.finditer(
        r"<megamek\.common\.Player\s+id=\"\d+\">(.*?)</megamek\.common\.Player>",
        text, re.DOTALL,
    ):
        block = m.group(1)
        name_m = re.search(r"<name>(.*?)</name>", block)
        id_m = re.search(r"<id>(\d+)</id>", block)
        if not name_m or not id_m:
            continue
        pid = int(id_m.group(1))
        name = name_m.group(1)
        if name == "watcher" or pid == 0:
            continue

        # Get the last roll value (current round)
        rolls_m = re.search(r"<rolls[^>]*>(.*?)</rolls>", block, re.DOTALL)
        roll = None
        if rolls_m:
            ints = re.findall(r"<int>(-?\d+)</int>", rolls_m.group(1))
            if ints:
                roll = int(ints[-1])

        players[pid] = {"name": name, "roll": roll}

    if not players:
        return None

    # Determine first mover from turnVector
    tv_m = re.search(r"<turnVector[^>]*>(.*?)</turnVector>", text, re.DOTALL)
    first_mover_id = None
    if tv_m:
        pid_matches = re.findall(r"<playerId>(\d+)</playerId>", tv_m.group(1))
        if pid_matches:
            first_mover_id = int(pid_matches[0])

    rolls = {p["name"]: p["roll"] for p in players.values() if p["roll"] is not None}
    first_mover_name = players[first_mover_id]["name"] if first_mover_id in players else None

    return {
        "rolls": rolls,
        "first_mover_name": first_mover_name,
        "first_mover_id": first_mover_id,
    }


def parse_save(path: Path) -> dict:
    """Parse a .sav.gz file, return round info with units and reports."""
    # Extract round number from filename
    fname = path.name
    round_m = re.match(r"Round-(\d+)-", fname)
    round_num = int(round_m.group(1)) if round_m else -1

    with gzip.open(path, "rt", encoding="utf-8") as f:
        text = f.read()

    units = extract_units(text)
    reports = extract_reports(text)
    initiative = extract_initiative(text)

    return {
        "round": round_num,
        "units": units,
        "reports": reports,
        "n_reports": len(reports),
        "initiative": initiative,
    }


def get_round_reports(prev_save: dict, curr_save: dict) -> list[dict]:
    """Get only the new reports added in this round (reports are cumulative)."""
    prev_count = prev_save["n_reports"] if prev_save else 0
    return curr_save["reports"][prev_count:]


# ---------------------------------------------------------------------------
# Structured (JSON-serializable) variants for HTML viewers
# ---------------------------------------------------------------------------

def decode_combat_reports_structured(reports: list[dict]) -> list[dict]:
    """Decode weapon attacks into structured dicts (no ANSI formatting).

    Returns list of dicts with keys: attacker, weapon, target, tohit, roll,
    result ("HIT"/"MISS"), location, missiles.
    """
    results = []
    current_attacker = None
    current_weapon = None
    current_target = None
    current_tohit = None
    current_roll = None

    for r in reports:
        mid = r["messageId"]
        td = r["tagData"]

        if mid == 3100:
            if td:
                player = td[1] if len(td) >= 2 else None
                current_attacker = _tag_name(extract_entity_name(td[0]), player)
            current_weapon = None
            current_target = None
            current_tohit = None
            current_roll = None

        elif mid == 3115:
            if len(td) >= 2:
                current_weapon = td[0]
                target_player = td[2] if len(td) >= 3 else None
                current_target = _tag_name(extract_entity_name(td[1]), target_player)
            current_tohit = None
            current_roll = None

        elif mid == 3150:
            if td:
                current_tohit = extract_tooltip_number(td[0])

        elif mid == 3155:
            if td:
                current_roll = extract_tooltip_number(td[0])

        elif mid == 3220:
            if current_attacker and current_weapon and current_target:
                results.append({
                    "attacker": current_attacker, "weapon": current_weapon,
                    "target": current_target, "tohit": current_tohit,
                    "roll": current_roll, "result": "MISS",
                    "location": None, "missiles": None,
                })
            current_weapon = None

        elif mid == 3325:
            count = td[0] if td else None
            if current_attacker and current_weapon and current_target:
                results.append({
                    "attacker": current_attacker, "weapon": current_weapon,
                    "target": current_target, "tohit": current_tohit,
                    "roll": current_roll, "result": "HIT",
                    "location": None, "missiles": count,
                })
            current_weapon = None

        elif mid == 3405:
            loc = td[1] if len(td) >= 2 else None
            if current_attacker and current_weapon and current_target:
                results.append({
                    "attacker": current_attacker, "weapon": current_weapon,
                    "target": current_target, "tohit": current_tohit,
                    "roll": current_roll, "result": "HIT",
                    "location": loc, "missiles": None,
                })
            current_weapon = None

    return results


def decode_damage_reports_structured(reports: list[dict]) -> list[dict]:
    """Extract damage reports into structured dicts.

    Returns list of dicts with keys: entity, amount, location.
    """
    results = []
    entity_player_map: dict[str, str] = {}
    for r in reports:
        mid = r["messageId"]
        td = r["tagData"]
        if mid == 3100 and len(td) >= 2:
            entity_player_map[extract_entity_name(td[0])] = td[1]
        elif mid == 3115 and len(td) >= 3:
            entity_player_map[extract_entity_name(td[1])] = td[2]
        elif mid == 6065 and len(td) >= 4:
            entity = extract_entity_name(td[0])
            player = entity_player_map.get(entity)
            results.append({
                "entity": _tag_name(entity, player),
                "amount": td[2],
                "location": td[3],
            })
    return results


def diff_units_structured(prev_units: list[dict], curr_units: list[dict]) -> list[dict]:
    """Compare unit state between rounds, return structured movement dicts.

    Returns list of dicts with keys: name, from_pos, to_pos, facing, move_type,
    prone_change ("fell"/"stood"/None).
    """
    results = []
    prev_by_name = {u["name"]: u for u in prev_units}

    for curr in curr_units:
        name = curr["name"]
        prev = prev_by_name.get(name)
        if not prev:
            continue

        was_prone = prev.get("prone", False)
        is_prone = curr.get("prone", False)
        prone_change = None
        if is_prone and not was_prone:
            prone_change = "fell"
        elif not is_prone and was_prone:
            prone_change = "stood"

        effective_moved = curr.get("moved_last", curr["moved"])
        if effective_moved == "MOVE_NONE":
            effective_moved = curr["moved"]

        if curr["pos"] != prev["pos"] or effective_moved != "MOVE_NONE" or prone_change:
            move_type = MOVE_NAMES.get(effective_moved, effective_moved)
            facing_str = FACING_NAMES[curr["facing"]] if 0 <= curr["facing"] < 6 else "?"
            results.append({
                "name": name,
                "from_pos": list(prev["pos"]) if prev["pos"] else None,
                "to_pos": list(curr["pos"]) if curr["pos"] else None,
                "facing": facing_str,
                "move_type": move_type,
                "prone_change": prone_change,
            })

    return results


def armor_summary_structured(unit: dict) -> dict:
    """Return structured armor/internal summary for a unit.

    Returns dict with keys: name, armor_current, armor_max, pct,
    internal_current, internal_max, heat, prone, destroyed, damaged_locs.
    """
    armor = unit.get("armor", [])
    orig_armor = unit.get("orig_armor", [])
    internal = unit.get("internal", [])
    orig_internal = unit.get("orig_internal", [])

    total_armor = sum(max(0, v) for v in armor) if armor else 0
    total_orig = sum(orig_armor) if orig_armor else 0
    total_internal = sum(max(0, v) for v in internal) if internal else 0
    total_orig_internal = sum(orig_internal) if orig_internal else 0
    pct = (total_armor / total_orig * 100) if total_orig > 0 else 0

    damaged_locs = []
    for i, loc in enumerate(LOCATIONS):
        if i < len(armor) and i < len(orig_armor):
            a = max(0, armor[i])
            if a < orig_armor[i]:
                damaged_locs.append({"loc": loc, "type": "armor",
                                     "current": a, "max": orig_armor[i]})
        if i < len(internal) and i < len(orig_internal):
            s = max(0, internal[i])
            if s < orig_internal[i]:
                damaged_locs.append({"loc": loc, "type": "internal",
                                     "current": s, "max": orig_internal[i],
                                     "destroyed": s == 0})

    return {
        "name": unit["name"],
        "armor_current": total_armor,
        "armor_max": total_orig,
        "pct": round(pct, 1),
        "internal_current": total_internal,
        "internal_max": total_orig_internal,
        "heat": unit.get("heat", 0),
        "prone": unit.get("prone", False),
        "destroyed": unit.get("destroyed", False),
        "damaged_locs": damaged_locs,
    }


def build_round_transcript(saves: list[dict], step_log: list[dict],
                           autosave_path=None) -> list[dict]:
    """Build JSON-serializable transcript data from parsed saves and step_log.

    Args:
        saves: List of parsed save dicts (from parse_save()).
        step_log: List of per-step dicts with round, phase, action, reward, etc.
        autosave_path: Optional path to autosave file for final round combat data.

    Returns list of round dicts, each with keys: round, initiative, movement,
    combat, damage, unit_status, rl_steps.
    """
    # Group step_log by round
    steps_by_round: dict[int, list[dict]] = {}
    for s in step_log:
        steps_by_round.setdefault(s["round"], []).append(s)

    rounds = []
    for i, save in enumerate(saves):
        round_num = save["round"]
        prev_save = saves[i - 1] if i > 0 else None

        rd: dict = {"round": round_num}

        # Initiative
        init = save.get("initiative")
        if init and init.get("rolls"):
            rd["initiative"] = {
                "rolls": init["rolls"],
                "first_mover": init.get("first_mover_name"),
            }
        else:
            rd["initiative"] = None

        # Movement
        if prev_save:
            rd["movement"] = diff_units_structured(prev_save["units"], save["units"])
        else:
            rd["movement"] = []

        # Combat
        round_reports = get_round_reports(prev_save, save)
        rd["combat"] = decode_combat_reports_structured(round_reports)
        rd["damage"] = decode_damage_reports_structured(round_reports)

        # Unit status
        rd["unit_status"] = [armor_summary_structured(u) for u in save["units"]]

        # RL steps for this round
        round_steps = steps_by_round.get(round_num, [])
        rd["rl_steps"] = []
        for s in round_steps:
            step_data = {
                "phase": "GAME_END" if s.get("phase") == "VICTORY" else s.get("phase", ""),
                "action": s.get("action"),
                "n_legal_moves": s.get("n_legal_moves", 0),
                "reward": round(s.get("reward", 0), 4),
                "cumulative": round(s.get("cumulative_return", 0), 4),
            }
            details = s.get("reward_details", [])
            nonzero = [(name, round(weighted, 4))
                       for name, raw, weighted in details if abs(weighted) > 1e-6]
            if nonzero:
                step_data["components"] = [
                    {"name": name.replace("Reward", "").replace("DamageDelta", "Damage")
                     .replace("LocationDestruction", "LocDestroy")
                     .replace("RangeAdvantage", "Range"),
                     "value": val}
                    for name, val in nonzero
                ]
            rd["rl_steps"].append(step_data)

        rounds.append(rd)

    # Handle autosave (final round combat data)
    if autosave_path and saves:
        final_save = parse_save(autosave_path)
        last_round_save = saves[-1]
        final_reports = get_round_reports(last_round_save, final_save)

        if final_reports:
            final_rd = {
                "round": last_round_save["round"],
                "is_final": True,
                "initiative": None,
                "movement": [],
                "combat": decode_combat_reports_structured(final_reports),
                "damage": decode_damage_reports_structured(final_reports),
                "unit_status": [armor_summary_structured(u) for u in final_save["units"]],
                "rl_steps": [],
            }
            rounds.append(final_rd)

    return rounds


def main():
    parser = argparse.ArgumentParser(description="Generate transcript from MegaMek save files")
    parser.add_argument("save_dir", type=Path, help="Directory containing .sav.gz files")
    parser.add_argument("--verbose", "-v", action="store_true", help="Show damage details")
    parser.add_argument("--no-color", action="store_true", help="Disable colored output")
    args = parser.parse_args()

    global _USE_COLOR
    if args.no_color:
        _USE_COLOR = False

    if not args.save_dir.is_dir():
        print(f"Error: {args.save_dir} is not a directory", file=sys.stderr)
        sys.exit(1)

    # Find and sort save files by round number
    save_files = sorted(
        args.save_dir.glob("Round-*.sav.gz"),
        key=lambda p: int(re.match(r"Round-(\d+)-", p.name).group(1)),
    )

    if not save_files:
        print(f"No save files found in {args.save_dir}", file=sys.stderr)
        sys.exit(1)

    print(f"Loading {len(save_files)} save files...", file=sys.stderr)

    # Parse all saves
    saves = []
    for sf in save_files:
        saves.append(parse_save(sf))

    # Print header using first save's unit names
    if saves and saves[0]["units"]:
        unit_names = [u["name"] for u in saves[0]["units"]]
        print(bold(f"\n=== Game: {' vs '.join(unit_names)} ==="))
    else:
        print(bold("\n=== Game Transcript ==="))

    # Determine game outcome from directory name
    dirname = args.save_dir.name
    if "WIN" in dirname:
        outcome_str = "WIN"
        outcome_display = green("WIN (RLBot)")
    elif "LOSS" in dirname:
        outcome_str = "LOSS"
        outcome_display = red("LOSS (RLBot)")
    elif "DRAW" in dirname:
        outcome_str = "DRAW"
        outcome_display = yellow("DRAW")
    else:
        outcome_str = "Unknown"
        outcome_display = "Unknown"

    # Determine end condition from final save's unit states
    final_units = saves[-1]["units"] if saves else []
    end_condition = determine_end_condition(final_units, outcome_str)
    condition_suffix = f" — {end_condition}" if end_condition else ""
    print(f"    Outcome: {outcome_display}{condition_suffix}")
    print(f"    Rounds: {saves[-1]['round']}")

    # Process each round
    for i, save in enumerate(saves):
        round_num = save["round"]
        prev_save = saves[i - 1] if i > 0 else None

        print(bold(f"\n--- Round {round_num} ---"))

        # Initiative
        init = save.get("initiative")
        if init and init["rolls"]:
            rolls = init["rolls"]
            first = init.get("first_mover_name")
            roll_parts = [f"{name}={roll}" for name, roll in rolls.items()]
            first_str = f"{first} moves first" if first else ""
            roll_str = ", ".join(roll_parts)
            print(f"  {dim(f'Initiative: {first_str} (rolled {roll_str})')}")

        # Movement diff
        if prev_save:
            move_lines = diff_units(prev_save["units"], save["units"])
            if move_lines:
                print(f"  {cyan('Movement:')}")
                for line in move_lines:
                    print(line)

        # Combat reports (only new ones from this round)
        round_reports = get_round_reports(prev_save, save)
        combat_lines = decode_combat_reports(round_reports)
        if combat_lines:
            print(f"\n  {cyan('Weapons fire:')}")
            for line in combat_lines:
                print(line)

        # Damage details (verbose mode)
        if args.verbose:
            damage_lines = decode_damage_reports(round_reports)
            if damage_lines:
                print(f"\n  {cyan('Damage:')}")
                for line in damage_lines:
                    print(line)

        # Unit status summary
        if save["units"]:
            print()
            for unit in save["units"]:
                print(armor_summary(unit))

    # Check for autosave (end-of-game state with final round's combat reports)
    autosave_files = sorted(args.save_dir.glob("autosave_*.sav.gz"))
    if autosave_files and saves:
        final_save = parse_save(autosave_files[-1])
        last_round_save = saves[-1]

        # Extract combat reports from the final round (diff against last Round save)
        final_reports = get_round_reports(last_round_save, final_save)

        if final_reports:
            print(bold(f"\n--- Final attacks (Round {last_round_save['round']}) ---"))

            combat_lines = decode_combat_reports(final_reports)
            if combat_lines:
                print(f"\n  {cyan('Weapons fire:')}")
                for line in combat_lines:
                    print(line)

            if args.verbose:
                damage_lines = decode_damage_reports(final_reports)
                if damage_lines:
                    print(f"\n  {cyan('Damage:')}")
                    for line in damage_lines:
                        print(line)

            # Show final unit states from autosave
            if final_save["units"]:
                print()
                for unit in final_save["units"]:
                    print(armor_summary(unit))

            # Re-determine end condition with autosave's more complete unit states
            end_condition = determine_end_condition(final_save["units"], outcome_str)
            if end_condition:
                print(f"\n    End condition: {end_condition}")

    print()


if __name__ == "__main__":
    main()
