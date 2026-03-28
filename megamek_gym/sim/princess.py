"""Princess AI opponent — ports Java's BasicPathRanker scoring for 1v1.

Implements Java's BasicPathRanker.rankPath() for 1v1:
    utility = -fallMod + braveryMod - aggressionMod - herdingMod - facingMod

In 1v1, getFriendEntities() returns the unit itself (it's not an enemy
of its own player), so herding = distance from dest to own current
position × herdMentality (default 1.0). This acts as a "don't stray
too far" penalty that tempers aggression.

Components that are always 0 in 1v1 with default settings:
crowding (disabled), self-preservation (no forced withdrawal),
off-board (ground units), movement mod (enemies visible + favorHigherTMM=0).
"""

from __future__ import annotations

from megamek_gym.reward import hex_bearing, hex_distance, in_firing_arc_with_twist
from megamek_gym.sim.board import Board
from megamek_gym.sim.los import LosTable, compute_terrain_modifier

# Java default behavior settings (all index 5)
_FALL_SHAME = 500.0
_UNIT_DESTRUCTION_FACTOR = 1000.0
_BRAVERY = 1.5
_HYPER_AGGRESSION = 2.5
_FACING_MOD_MULTIPLIER = 50.0
_QUICK_DAMAGE_DISCOUNT = 0.42  # Java's approximate average hit probability
_UNMOVED_DAMAGE_DISCOUNT = 0.25  # Discount for unmoved enemy uncertainty
_FACING_TOLERANCE = 1  # Java's DEFAULT_ALLOW_FACING_TOLERANCE
_HERD_MENTALITY = 1.0  # Default index 5
_KICK_DAMAGE = 10.0  # ceil(50 tons / 5) for TBT-5S
_KICK_HIT_PROB = 30 / 36  # P(2d6 >= piloting_skill=5) = 0.833


def _hex_direction(x1: int, y1: int, x2: int, y2: int) -> int:
    """Hex direction (0-5) from (x1,y1) to (x2,y2), matching Java Coords.direction()."""
    bearing = hex_bearing(x1, y1, x2, y2)
    return round(bearing / 60) % 6


def _armor_bias(unit_obs: dict) -> int:
    """Armor-aware facing bias: +1 to face stronger left side, -1 for right, 0 if equal.

    Matches Java's FacingDiffCalculator.getBiasTowardsFacing().
    """
    left = 0
    right = 0
    for loc in unit_obs.get("armor", []):
        name = loc.get("location", "")
        armor = loc.get("armor", 0) + loc.get("rear_armor", 0)
        if name in ("LA", "LL", "LT"):
            left += armor
        elif name in ("RA", "RL", "RT"):
            right += armor
    if left > right:
        return 1  # Bias counterclockwise (face strong left toward enemy)
    elif right > left:
        return -1  # Bias clockwise (face strong right toward enemy)
    return 0


def _facing_diff(dest_x: int, dest_y: int, dest_facing: int,
                 target_x: int, target_y: int,
                 unit_obs: dict | None = None) -> int:
    """Facing difference after tolerance, matching Java's FacingDiffCalculator.

    Raw diff (0-3) has tolerance subtracted and is clamped to >= 0.
    Default tolerance=1 means being off by 1 hexside is free.
    Desired facing includes armor-distribution bias (±1 hexside).
    """
    if dest_x == target_x and dest_y == target_y:
        return 0
    desired = _hex_direction(dest_x, dest_y, target_x, target_y)
    if unit_obs is not None:
        desired = (desired + _armor_bias(unit_obs)) % 6
    diff = abs(dest_facing - desired)
    if diff > 3:
        diff = 6 - diff
    return max(0, diff - _FACING_TOLERANCE)


def _kick_damage() -> float:
    """Kick damage for TBT-5S: ceil(50/5) = 10.

    Java's calculateKickDamagePotential returns expectedDamageOnHit * probToHit.
    For a healthy pilot (piloting 5), kick TN=5, prob=0.72, so 10*0.72=7.2.
    But from the debug logs, Java uses 10.0 for physical damage, suggesting
    the full damage is used. We match the observed Java behavior.
    """
    return _KICK_DAMAGE


def _sum_weapon_damage(unit_obs: dict, dist: int) -> float:
    """Sum weapon damage for all weapons in range. No LOS check.

    Matches Java's FireControl.getMaxDamageAtRange() which is purely
    range-based (no firing arc, no LOS). Cluster weapons use rackSize
    (missile count), not full effective damage. TACOPS_RANGE (extreme
    range) is disabled by default and not set by RLGameRunner, so only
    long_range is checked.
    """
    if dist <= 0:
        return 0.0

    total = 0.0

    for w in unit_obs.get("weapons", []):
        if w.get("destroyed", False):
            continue
        damage = w.get("damage", 0)
        if damage <= 0:
            continue

        # Range check only (Java's getMaxDamageAtRange ignores firing arc)
        long_range = w.get("long_range", 0)
        if dist > long_range:
            continue

        # Cluster weapons: Java uses rackSize, not full effective damage
        # Our obs dict "damage" is rackSize * perMissileDmg (e.g. SRM-2 = 4)
        # Java's weaponDamage for clusters = rackSize (e.g. SRM-2 = 2)
        name = w.get("name", "")
        if "SRM" in name or "LRM" in name:
            # Extract rack size from name (SRM 2 → 2, LRM 15 → 15)
            parts = name.split()
            try:
                rack_size = int(parts[-1])
                total += rack_size
            except (ValueError, IndexError):
                total += damage
        else:
            total += damage

    return total


def _max_damage_at_range(unit_obs: dict, from_x: int, from_y: int,
                         from_facing: int, target_x: int, target_y: int,
                         board: Board, los_table: LosTable) -> float:
    """Max damage with terrain LOS check. Used by evaluateMovedEnemy path.

    Checks LOS first (returns 0 if blocked), then delegates to
    _sum_weapon_damage for the range-based damage calculation.
    """
    if not los_table.has_los(from_x, from_y, target_x, target_y):
        return 0.0

    dist = hex_distance(from_x, from_y, target_x, target_y)
    return _sum_weapon_damage(unit_obs, dist)


def _is_in_forward_arc(dest_x: int, dest_y: int, facing: int,
                       target_x: int, target_y: int) -> bool:
    """Check if target is in forward 240-degree cone (not in rear arc).

    Approximates Java's isInMyLoS for evaluateUnmovedEnemy. Java checks
    whether the enemy's movable area intersects a 240-degree forward cone
    bounded by HexLines from the hex behind the unit. We approximate by
    checking the enemy's current position against the cone.

    For mechs that canChangeSecondaryFacing() (standard mechs), the cone
    extends +-120 degrees from facing (i.e., excludes only the rear 120-degree arc).
    """
    if dest_x == target_x and dest_y == target_y:
        return True
    bearing = hex_bearing(dest_x, dest_y, target_x, target_y)
    facing_angle = facing * 60
    relative = (round(bearing) - facing_angle) % 360
    # Rear arc is the 120-degree zone directly behind: (120, 240) exclusive.
    # Forward cone = everything NOT in rear arc.
    return relative <= 120 or relative >= 240


def _approx_closest_range(dest_x: int, dest_y: int,
                          enemy_x: int, enemy_y: int,
                          enemy_run_mp: int) -> int:
    """Approximate range to the closest hex an unmoved enemy can reach.

    Fallback when enemy reachable hexes aren't available. Uses linear
    approximation: max(1, dist - run_mp). Prefer _closest_reachable_range
    with actual reachable hexes when available.
    """
    current_dist = hex_distance(dest_x, dest_y, enemy_x, enemy_y)
    return max(1, current_dist - enemy_run_mp)


def _closest_reachable_range(dest_x: int, dest_y: int,
                             reachable_hexes: set[tuple[int, int]]) -> int:
    """Find actual range to closest hex in the enemy's reachable set.

    Matches Java's ConvexBoardArea.getClosestCoordsTo() but uses exact
    reachable hexes from move enumeration instead of a convex hull.
    Returns 0 if dest is in the reachable set (enemy can reach this hex).
    """
    min_dist = 999
    for rx, ry in reachable_hexes:
        d = hex_distance(dest_x, dest_y, rx, ry)
        if d == 0:
            return 0
        if d < min_dist:
            min_dist = d
    return min_dist


def score_move(move: dict, unit_obs: dict, enemy_obs: dict,
               board: Board, board_hexes: list[dict],
               los_table: LosTable,
               enemy_has_moved: bool = True,
               friends_coords: tuple[int, int] | None = ...,
               enemy_reachable_hexes: set[tuple[int, int]] | None = None) -> float:
    """Score a move using Java Princess's BasicPathRanker formula (1v1 simplified).

    utility = -fallMod + braveryMod - aggressionMod - herdingMod - facingMod

    Java uses two code paths depending on whether the enemy has moved:
    - evaluateMovedEnemy (enemy_has_moved=True): Uses actual positions, terrain
      LOS check, discount=0.42. Includes own physical (kick) damage if adjacent.
      Includes enemy kick (kickDmg * hitProb) if adjacent.
    - evaluateUnmovedEnemy (enemy_has_moved=False): Uses closest-reachable-hex
      range, forward arc check for own damage (no terrain LOS), NO LOS check
      for enemy damage, discount=0.25. No own physical damage.

    friends_coords: Position for herding calculation. Defaults to unit's own
        position (unit_obs x/y). Pass None to skip herding (Java round 1 behavior
        where friendsCoords is uninitialized).
    enemy_reachable_hexes: Pre-computed set of (x, y) hexes the enemy can reach.
        Used for closest-range calculation in unmoved path. Falls back to linear
        approximation (max(1, dist - run_mp)) if not provided.
    """
    dest_x = move["dest_x"]
    dest_y = move["dest_y"]
    facing = move["facing"]
    mp_used = move["mp_used"]
    success_prob = move.get("success_probability", 1.0)

    ex = enemy_obs.get("x", -1)
    ey = enemy_obs.get("y", -1)

    if ex < 0 or ey < 0:
        return 0.0

    enemy_facing = enemy_obs.get("facing", 0)

    # --- fallMod ---
    if success_prob <= 0.0:
        fall_mod = _UNIT_DESTRUCTION_FACTOR
    else:
        fall_mod = (1.0 - success_prob) * _FALL_SHAME

    # --- braveryMod = successProb * maxDamageDone * bravery - expectedDamageTaken ---
    dist_to_enemy = hex_distance(dest_x, dest_y, ex, ey)

    if enemy_has_moved:
        # === evaluateMovedEnemy path ===
        # Uses actual positions, terrain LOS, discount=0.42
        discount = _QUICK_DAMAGE_DISCOUNT

        my_firing_damage = _max_damage_at_range(
            unit_obs, dest_x, dest_y, facing, ex, ey, board, los_table,
        ) * discount

        my_physical_damage = 0.0
        if dist_to_enemy <= 1:
            my_physical_damage = _kick_damage()

        max_damage_done = my_firing_damage + my_physical_damage

        enemy_firing_damage = _max_damage_at_range(
            enemy_obs, ex, ey, enemy_facing, dest_x, dest_y, board, los_table,
        )
        expected_damage_taken = enemy_firing_damage * discount

        # Enemy kick: Java adds calculateKickDamagePotential when adjacent
        # = kickDmg * probToHit. For piloting 5: P(2d6 >= 5) = 30/36 = 0.833.
        if dist_to_enemy <= 1:
            expected_damage_taken += _kick_damage() * _KICK_HIT_PROB

    else:
        # === evaluateUnmovedEnemy path ===
        # Uses closest-reachable-hex range, forward arc check (not LOS),
        # no LOS check for enemy damage, discount=0.25.
        # Java's canFlankAndKick (for enemy kick) checks specific
        # CoordFacingCombos in the enemy's movable area, which we can't
        # replicate without full path enumeration. Omitted to avoid
        # false positives.
        discount = _UNMOVED_DAMAGE_DISCOUNT

        if enemy_reachable_hexes is not None:
            approx_range = _closest_reachable_range(dest_x, dest_y, enemy_reachable_hexes)
        else:
            enemy_run_mp = enemy_obs.get("mp_run", 0)
            approx_range = _approx_closest_range(dest_x, dest_y, ex, ey, enemy_run_mp)

        # Own damage: forward arc check replaces terrain LOS.
        # Java's isInMyLoS requires the ENTIRE enemy movable area to be within
        # the forward 240° cone. If any reachable hex is in the rear arc,
        # isInMyLoS returns false and own damage = 0.
        my_firing_damage = 0.0
        if enemy_reachable_hexes is not None:
            in_arc = all(
                _is_in_forward_arc(dest_x, dest_y, facing, rx, ry)
                for rx, ry in enemy_reachable_hexes
            )
        else:
            in_arc = _is_in_forward_arc(dest_x, dest_y, facing, ex, ey)
        if in_arc:
            my_firing_damage = _sum_weapon_damage(
                unit_obs, approx_range,
            ) * discount

        # No own physical damage in evaluateUnmovedEnemy
        max_damage_done = my_firing_damage

        # Enemy damage: unconditional (no LOS check), uses closest-reachable range
        expected_damage_taken = _sum_weapon_damage(enemy_obs, approx_range) * discount

    bravery_mod = success_prob * max_damage_done * _BRAVERY - expected_damage_taken

    # --- aggressionMod = distToEnemy * hyperAggression ---
    # Java's distanceToClosestEnemy returns raw hex distance (walkMP modifier
    # is only used for multi-enemy "which is closest" comparison, not the value).
    if dist_to_enemy == 0:
        dist_to_enemy = 2  # Java: minimum 2 for non-infantry at same hex
    aggression_mod = dist_to_enemy * _HYPER_AGGRESSION

    # --- herdingMod = distToFriends * herdMentality ---
    # In 1v1, friendsCoords = unit's own current position (self-herding).
    # Java's friendsCoords is null in round 1 → herdingMod=0.
    if friends_coords is ...:
        # Default: use unit's own position (Java round 2+ behavior)
        fx = unit_obs.get("x", -1)
        fy = unit_obs.get("y", -1)
    elif friends_coords is None:
        fx, fy = -1, -1
    else:
        fx, fy = friends_coords

    if fx >= 0 and fy >= 0:
        herding_mod = hex_distance(dest_x, dest_y, fx, fy) * _HERD_MENTALITY
    else:
        herding_mod = 0.0

    # --- facingMod = 50 * facingDiff ---
    facing_mod = _FACING_MOD_MULTIPLIER * _facing_diff(dest_x, dest_y, facing, ex, ey, unit_obs)

    return -fall_mod + bravery_mod - aggression_mod - herding_mod - facing_mod


def select_move(moves: list[dict], unit_obs: dict, enemy_obs: dict,
                board: Board, board_hexes: list[dict],
                los_table: LosTable,
                enemy_has_moved: bool = True,
                friends_coords: tuple[int, int] | None = ...,
                enemy_reachable_hexes: set[tuple[int, int]] | None = None) -> int:
    """Select the best move index for Princess AI.

    enemy_reachable_hexes: Pre-computed set of (x, y) hexes the enemy can
        reach this turn. Used for accurate closest-range calculation when
        the enemy hasn't moved. Compute from enumerate_moves() on the enemy
        unit. Falls back to linear approximation if not provided.
    """
    if not moves:
        return 0

    best_idx = 0
    best_score = float("-inf")

    for i, move in enumerate(moves):
        s = score_move(move, unit_obs, enemy_obs, board, board_hexes, los_table,
                       enemy_has_moved, friends_coords, enemy_reachable_hexes)
        if s > best_score:
            best_score = s
            best_idx = i

    return best_idx
