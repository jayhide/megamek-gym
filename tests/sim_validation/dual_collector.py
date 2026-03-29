"""Dual-bot game trace collector for scripted movement validation.

Runs a Java game with both units controlled via the RL bridge (opponentType=rl),
following scripted waypoint paths. Bypasses MegaMekEnv since it assumes single-bot
control.
"""

from __future__ import annotations

import copy
import json
import logging
import socket
import time
from dataclasses import dataclass, field

from megamek_gym.config import MegaMekConfig
from megamek_gym.java_process import JavaProcess
from megamek_gym.reward import hex_distance

logger = logging.getLogger(__name__)


@dataclass
class DualStepTrace:
    """One step of a dual-bot game."""
    step_idx: int
    raw_obs: dict
    action_taken: int
    active_entity_id: int


@dataclass
class DualGameTrace:
    """Complete trace of a dual-bot game."""
    steps: list[DualStepTrace] = field(default_factory=list)
    entity_owners: dict[int, int] = field(default_factory=dict)  # entity_id -> owner
    config: MegaMekConfig | None = None

    @property
    def rl_owner_id(self) -> int:
        """Identify RL player's owner from starting position."""
        if not self.steps or not self.config or not self.config.rl_fixed_coords:
            return -1
        rl_x, rl_y = self.config.rl_fixed_coords
        first_obs = self.steps[0].raw_obs
        for unit in first_obs.get("units", []):
            if unit["x"] == rl_x and unit["y"] == rl_y:
                return unit["owner"]
        return min(self.entity_owners.values()) if self.entity_owners else -1


class WaypointController:
    """Selects moves that steer a unit toward a sequence of waypoints."""

    def __init__(self, waypoints: list[tuple[int, int]], prefer_run: bool = False):
        self.waypoints = waypoints
        self.target_idx = 0
        self.prefer_run = prefer_run

    def select_move(self, legal_moves: list[dict], walk_mp: int) -> int:
        """Pick the legal move index closest to the current waypoint target.

        If prefer_run is True, prefer run-speed moves (mp_used > walk_mp).
        Advances to the next waypoint when within 1 hex.
        """
        if not legal_moves:
            return 0

        target_x, target_y = self.waypoints[self.target_idx]

        best_idx = 0
        best_dist = float("inf")
        best_is_preferred = False

        for i, move in enumerate(legal_moves):
            dist = hex_distance(move["dest_x"], move["dest_y"], target_x, target_y)
            is_run = move["mp_used"] > walk_mp
            is_preferred = (is_run == self.prefer_run) if self.prefer_run else True

            # Prefer moves of the right speed class, then closest to target
            if is_preferred and not best_is_preferred:
                best_idx = i
                best_dist = dist
                best_is_preferred = True
            elif is_preferred == best_is_preferred and dist < best_dist:
                best_idx = i
                best_dist = dist
                best_is_preferred = is_preferred

        # Advance waypoint if close enough
        dest = legal_moves[best_idx]
        dist_after = hex_distance(dest["dest_x"], dest["dest_y"], target_x, target_y)
        if dist_after <= 1:
            self.target_idx = (self.target_idx + 1) % len(self.waypoints)

        return best_idx


# Waypoint paths on the 16x17 Woodland board (diagonal traversal through
# varied terrain and elevation). Spaced ~5 hexes apart so each leg takes
# 1-2 turns at walk speed.
WALK_WAYPOINTS_A = [(14, 1), (10, 5), (6, 9), (2, 13), (6, 9), (10, 5)]
WALK_WAYPOINTS_B = [(1, 15), (5, 11), (9, 7), (13, 3), (9, 7), (5, 11)]

# Same paths work for running — with run_mp=8 units just cover more ground per turn
RUN_WAYPOINTS_A = WALK_WAYPOINTS_A
RUN_WAYPOINTS_B = WALK_WAYPOINTS_B

# Firing validation waypoints: converge to positions with light woods between
# them at (6,8), ensuring partial cover triggers. Units orbit at distance ~3
# across the woods line, with excursions to different ranges for diversity.
# All waypoint hexes are clear terrain at elevation 0.
FIRING_WAYPOINTS_A = [(10, 5), (8, 8), (10, 8), (8, 8), (10, 10), (8, 8)]
FIRING_WAYPOINTS_B = [(5, 11), (5, 8), (3, 8), (5, 8), (3, 10), (5, 8)]


def _find_walk_mp(obs: dict, entity_id: int) -> int:
    """Extract walk_mp for the given entity from the observation."""
    for unit in obs.get("units", []):
        if unit.get("id") == entity_id:
            return unit.get("walk_mp", 5)
    return 5  # fallback


def _find_entity_owner(obs: dict, entity_id: int) -> int:
    """Find the owner of a given entity."""
    for unit in obs.get("units", []):
        if unit.get("id") == entity_id:
            return unit.get("owner", -1)
    return -1


def _resolve_entity_ids(obs: dict, state_spec: dict) -> dict:
    """Build inject_state message with entity IDs resolved from observation.

    The state_spec uses "owner" to identify entities. This function maps
    owner IDs to entity IDs from the observation's units list.

    Args:
        obs: The first observation from Java (contains units with id/owner).
        state_spec: State injection spec with "entities" list, each having
            "owner" and optional state fields.

    Returns:
        inject_state message dict with entity IDs filled in.
    """
    # Build owner -> entity_id mapping
    owner_to_id = {}
    for unit in obs.get("units", []):
        owner_to_id[unit["owner"]] = unit["id"]

    entities = []
    for spec in state_spec.get("entities", []):
        owner = spec.get("owner")
        if owner is None:
            continue
        entity_id = owner_to_id.get(owner)
        if entity_id is None:
            logger.warning("inject_state: no entity for owner %d", owner)
            continue
        # Copy spec and replace owner with resolved id
        entry = {k: v for k, v in spec.items() if k != "owner"}
        entry["id"] = entity_id
        entities.append(entry)

    return {"type": "inject_state", "entities": entities}


def _send_inject_state(
    sock: socket.socket,
    reader,
    obs: dict,
    state_spec: dict,
) -> dict:
    """Send inject_state message and read the updated observation.

    Args:
        sock: Connected socket to Java bridge.
        reader: Line-buffered reader on the socket.
        obs: Current observation (used to resolve entity IDs).
        state_spec: State injection spec (owner-based).

    Returns:
        Updated observation dict reflecting the injected state.
    """
    msg = _resolve_entity_ids(obs, state_spec)
    logger.info("Sending inject_state: %d entities", len(msg.get("entities", [])))
    payload = json.dumps(msg) + "\n"
    sock.sendall(payload.encode("utf-8"))

    # Read the updated observation (Java re-enumerates and re-sends)
    line = reader.readline()
    if not line:
        logger.warning("No response after inject_state")
        return obs
    updated_obs = json.loads(line)
    logger.info("Received updated observation after inject_state")
    return updated_obs


def collect_dual_game_trace(
    megamek_dir: str,
    port: int = 9999,
    max_rounds: int = 15,
    movement_mode: str = "walk",
    rl_unit: str = "Trebuchet TBT-5S",
    opponent_unit: str = "Trebuchet TBT-5S",
    board: str = "Map Set 6/16x17 Woodland",
    rl_fixed_coords: tuple[int, int] = (14, 1),
    opponent_fixed_coords: tuple[int, int] = (1, 15),
    initial_state: dict | None = None,
    waypoints_a: list[tuple[int, int]] | None = None,
    waypoints_b: list[tuple[int, int]] | None = None,
    firing_strategy: str = "none",
) -> DualGameTrace:
    """Play a dual-bot Java game with scripted waypoint movement.

    Both units are RL-controlled (opponentType=rl). Returns traces for
    all movement steps.

    Args:
        megamek_dir: Path to MegaMek checkout.
        port: TCP port for the Java bridge.
        max_rounds: Max game rounds before truncation.
        movement_mode: "walk" or "run" — controls waypoint controller preference.
        rl_unit: RL player's mech.
        opponent_unit: Opponent mech.
        board: Board name.
        rl_fixed_coords: Fixed starting hex for RL unit.
        opponent_fixed_coords: Fixed starting hex for opponent.
        initial_state: Optional state injection spec. Dict with "entities" list,
            each containing "owner" (int) and optional "heat", "prone", "armor",
            "internal" fields. Entity IDs are resolved from the first observation.
            Example: {"entities": [{"owner": 0, "prone": True, "heat": 10}]}
        waypoints_a: Custom waypoints for first player (default: diagonal patrol).
        waypoints_b: Custom waypoints for second player (default: diagonal patrol).
        firing_strategy: Firing strategy for both bots ("none", "naive", "princess").
    """
    prefer_run = movement_mode == "run"

    config = MegaMekConfig(
        megamek_dir=megamek_dir,
        rl_port=port,
        rl_unit=rl_unit,
        opponent_unit=opponent_unit,
        board=board,
        max_game_rounds=max_rounds,
        rl_fixed_coords=rl_fixed_coords,
        opponent_fixed_coords=opponent_fixed_coords,
        firing_strategy=firing_strategy,
        opponent_type="rl",
        max_rotating_round_saves=0,
        auto_wake_pilot=True,
    )

    # Start Java process
    java = JavaProcess(
        megamek_dir=config.megamek_dir,
        rl_unit=config.rl_unit,
        opponent_unit=config.opponent_unit,
        board=config.board,
        port=config.rl_port,
        timeout_minutes=config.java_timeout_minutes,
        max_rotating_round_saves=config.max_rotating_round_saves,
        firing_strategy=config.firing_strategy,
        max_game_rounds=config.max_game_rounds,
        rl_fixed_coords=config.rl_fixed_coords,
        opponent_fixed_coords=config.opponent_fixed_coords,
        opponent_type=config.opponent_type,
        auto_wake_pilot=config.auto_wake_pilot,
    )
    java.start()

    trace = DualGameTrace(config=config)

    # Connect to the bridge socket
    sock = None
    try:
        # Wait for Java to start listening
        for attempt in range(60):
            try:
                sock = socket.create_connection(("localhost", port), timeout=5)
                break
            except (ConnectionRefusedError, OSError):
                time.sleep(1)
        else:
            raise ConnectionError(f"Could not connect to Java bridge on port {port}")

        sock.settimeout(60)
        reader = sock.makefile("r", encoding="utf-8")

        # Waypoint controllers — keyed by owner ID once we learn them
        controllers: dict[int, WaypointController] = {}
        # We'll assign controllers by order of first appearance:
        # first entity seen gets waypoints A, second gets waypoints B
        wp_a = waypoints_a if waypoints_a is not None else (
            RUN_WAYPOINTS_A if prefer_run else WALK_WAYPOINTS_A)
        wp_b = waypoints_b if waypoints_b is not None else (
            RUN_WAYPOINTS_B if prefer_run else WALK_WAYPOINTS_B)
        owners_seen: list[int] = []

        state_injected = False
        step_idx = 0

        while True:
            line = reader.readline()
            if not line:
                logger.warning("Java bridge closed connection")
                break

            obs = json.loads(line)

            # Terminal observation
            if obs.get("terminated") or obs.get("truncated"):
                trace.steps.append(DualStepTrace(
                    step_idx=step_idx,
                    raw_obs=copy.deepcopy(obs),
                    action_taken=-1,
                    active_entity_id=obs.get("active_entity_id", -1),
                ))
                break

            # Inject initial state on first non-terminal observation
            if initial_state and not state_injected:
                obs = _send_inject_state(
                    sock, reader, obs, initial_state,
                )
                state_injected = True

            active_id = obs.get("active_entity_id", -1)
            owner = _find_entity_owner(obs, active_id)
            legal_moves = obs.get("legal_moves", [])

            # Record entity owners
            if active_id >= 0 and active_id not in trace.entity_owners:
                trace.entity_owners[active_id] = owner

            # Assign controller on first appearance
            if owner >= 0 and owner not in owners_seen:
                owners_seen.append(owner)
                if len(owners_seen) == 1:
                    controllers[owner] = WaypointController(wp_a, prefer_run)
                else:
                    controllers[owner] = WaypointController(wp_b, prefer_run)

            # Select action
            if legal_moves and owner in controllers:
                walk_mp = _find_walk_mp(obs, active_id)
                action = controllers[owner].select_move(legal_moves, walk_mp)
            else:
                action = 0

            # Record trace
            trace.steps.append(DualStepTrace(
                step_idx=step_idx,
                raw_obs=copy.deepcopy(obs),
                action_taken=action,
                active_entity_id=active_id,
            ))

            # Send action
            action_msg = json.dumps({"type": "action", "move_index": action}) + "\n"
            sock.sendall(action_msg.encode("utf-8"))

            step_idx += 1

    finally:
        if sock is not None:
            try:
                sock.close()
            except OSError:
                pass
        java.stop()

    logger.info(
        "Dual-bot trace: %d steps, %d entities, mode=%s",
        len(trace.steps), len(trace.entity_owners), movement_mode,
    )
    return trace
