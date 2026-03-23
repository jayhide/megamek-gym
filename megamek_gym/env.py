"""MegaMek Gymnasium environment."""

from __future__ import annotations

import json
import logging
import os
import signal
import socket
import time
from pathlib import Path

logger = logging.getLogger(__name__)

import gymnasium
import numpy as np
from gymnasium import spaces

from megamek_gym.config import MegaMekConfig
from megamek_gym.java_process import JavaProcess
from megamek_gym.observation import (
    compute_obs_size,
    flatten_observation,
    identify_rl_owner,
)
from megamek_gym.reward import CompositeReward, RewardFunction


class MegaMekEnv(gymnasium.Env):
    metadata = {"render_modes": []}

    def __init__(
        self,
        config: MegaMekConfig | None = None,
        reward_fn: RewardFunction | None = None,
        **kwargs,
    ):
        if config is not None and kwargs:
            raise ValueError(
                "Pass either config or keyword arguments, not both"
            )

        if config is None:
            config = MegaMekConfig(**kwargs)

        # Pop gymnasium's render_mode before it reaches super().__init__
        super().__init__()

        self.config = config
        self.reward_fn = reward_fn or CompositeReward()

        obs_size = compute_obs_size(
            config.resolved_board_width, config.resolved_board_height,
            config.max_legal_moves,
        )
        self.observation_space = spaces.Box(
            low=-1.0, high=1.0, shape=(obs_size,), dtype=np.float32
        )
        self.action_space = spaces.Discrete(config.max_legal_moves)

        self._java: JavaProcess | None = None
        self._sock: socket.socket | None = None
        self._reader = None
        self._rl_owner_id: int | None = None
        self._last_raw_obs: dict | None = None
        self._last_flat_obs: np.ndarray | None = None
        self._legal_moves: list = []
        self._reset_timing: dict | None = None
        self._java_crashed: bool = False
        self._reset_count: int = 0

    @property
    def _port(self) -> int:
        return self.config.rl_port + self.config.env_index

    def _cleanup_saves(self):
        """Delete autosave_* files (not needed for replay, which uses Round-* files).
        Also enforce save_budget_mb by deleting oldest Round-* files if over budget.
        Checks both the shared savegames dir and per-JVM run_{port}/savegames dirs."""
        base = Path(self.config.megamek_dir) / "megamek"
        # Check both the legacy shared dir and the per-JVM directory
        dirs_to_check = [
            base / "savegames",
            base / f"run_{self._port}" / "savegames",
        ]

        all_round_files = []
        for savegames_dir in dirs_to_check:
            if not savegames_dir.is_dir():
                continue

            # Remove autosave_* files (written at VICTORY, not used by replay UI)
            for f in savegames_dir.glob("autosave_*.sav.gz"):
                try:
                    f.unlink()
                except OSError:
                    pass  # Another env may have already deleted it

            all_round_files.extend(savegames_dir.glob("Round-*.sav.gz"))

        # Enforce save budget on Round-* files across all dirs
        budget_bytes = self.config.save_budget_mb * 1024 * 1024
        round_files = sorted(all_round_files, key=lambda f: f.stat().st_mtime)
        total = sum(f.stat().st_size for f in round_files)
        while total > budget_bytes and round_files:
            oldest = round_files.pop(0)
            try:
                total -= oldest.stat().st_size
                oldest.unlink()
            except OSError:
                pass

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self._reset_count += 1
        if self._reset_count % 10 == 1:  # First reset + every 10th
            self._cleanup_saves()

        # Try persistent reset if we have a live connection
        if (self._sock is not None and self._java is not None
                and self._java.is_alive() and not self._java_crashed):
            try:
                return self._reset_persistent()
            except Exception as e:
                logger.warning(
                    "[port:%d] Persistent reset failed (%s), falling back to cold restart",
                    self._port, e,
                )
                self._cleanup()

        return self._reset_cold()

    def _reset_persistent(self):
        """Reset by sending a reset message over the existing socket.
        Java tears down the current game and starts a new one without restarting the JVM."""
        t0 = time.monotonic()
        logger.info("[port:%d] Persistent reset (reusing JVM)", self._port)

        # Send reset message to Java
        reset_msg = json.dumps({"type": "reset"}) + "\n"
        self._sock.sendall(reset_msg.encode("utf-8"))

        # Use longer timeout for reset (Java needs to set up new game)
        self._sock.settimeout(360)

        # Read first observation from new game
        raw_obs = self._read_obs()
        logger.debug(
            "[port:%d] First obs after reset: keys=%s, phase=%s, terminated=%s",
            self._port, list(raw_obs.keys()), raw_obs.get("phase"), raw_obs.get("terminated"),
        )

        t_done = time.monotonic()
        self._reset_timing = {
            "java_start_s": 0.0,
            "connect_s": 0.0,
            "first_obs_s": t_done - t0,
            "total_reset_s": t_done - t0,
        }
        logger.info(
            "[port:%d] Persistent reset complete (%.1fs)",
            self._port, t_done - t0,
        )

        # Switch to step timeout
        self._sock.settimeout(self.config.step_timeout_seconds)
        self._java_crashed = False

        return self._process_first_obs(raw_obs)

    def _reset_cold(self):
        """Full cold restart: kill JVM, start new one, connect, read first obs."""
        logger.info("Cold reset on port %d (env_index=%d)", self._port, self.config.env_index)
        self._cleanup()

        cfg = self.config
        t0 = time.monotonic()

        # Start Java
        self._java = JavaProcess(
            megamek_dir=cfg.megamek_dir,
            rl_unit=cfg.rl_unit,
            opponent_unit=cfg.opponent_unit,
            board=cfg.board,
            port=self._port,
            timeout_minutes=cfg.java_timeout_minutes,
            max_rotating_round_saves=cfg.max_rotating_round_saves,
            paranoid_autosave=cfg.paranoid_autosave,
            rl_starting_pos=cfg.rl_starting_pos,
            opponent_starting_pos=cfg.opponent_starting_pos,
            rl_deployment=cfg.rl_deployment,
            firing_strategy=cfg.firing_strategy,
            max_game_rounds=cfg.max_game_rounds,
            perf_log=cfg.perf_log,
            opponent_type=cfg.opponent_type,
            force_gc=cfg.force_gc,
            mem_log=cfg.mem_log,
            auto_wake_pilot=cfg.auto_wake_pilot,
            force_unconscious_on_turn=cfg.force_unconscious_on_turn,
            rl_fixed_coords=cfg.rl_fixed_coords,
            opponent_fixed_coords=cfg.opponent_fixed_coords,
            enable_game_reports=cfg.enable_game_reports,
        )
        self._java.start()

        t_java_started = time.monotonic()
        logger.debug("JVM started on port %d (%.1fs)", self._port, t_java_started - t0)

        # Connect with retry
        self._sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        connected = False
        timeout_s = cfg.connection_retries * cfg.connection_retry_delay
        for _ in range(cfg.connection_retries):
            try:
                self._sock.connect(("localhost", self._port))
                connected = True
                break
            except ConnectionRefusedError:
                if not self._java.is_alive():
                    raise RuntimeError(
                        f"Java process died before accepting connection.\n"
                        f"Java log tail (rl_java_{self._port}.log):\n"
                        f"  {self._java_log_tail()}"
                    )
                time.sleep(cfg.connection_retry_delay)
        if not connected:
            self._cleanup()
            raise RuntimeError(
                f"Could not connect to Java on port {self._port} after {timeout_s:.0f}s.\n"
                f"Java log tail (rl_java_{self._port}.log):\n"
                f"  {self._java_log_tail()}"
            )

        t_connected = time.monotonic()
        logger.debug("Connected to JVM on port %d (%.1fs)", self._port, t_connected - t_java_started)

        # Save partial timing so it's available even if _read_obs() fails
        self._reset_timing = {
            "java_start_s": t_java_started - t0,
            "connect_s": t_connected - t_java_started,
            "first_obs_s": None,
            "total_reset_s": None,
        }

        self._sock.settimeout(360)  # 6 min — longer than Java's 5-min socket timeout
        self._reader = self._sock.makefile("r", encoding="utf-8")

        # Read first observation
        raw_obs = self._read_obs()

        t_first_obs = time.monotonic()
        self._reset_timing["first_obs_s"] = t_first_obs - t_connected
        self._reset_timing["total_reset_s"] = t_first_obs - t0
        logger.info(
            "First obs received on port %d (%.1fs, total reset: %.1fs)",
            self._port, t_first_obs - t_connected, t_first_obs - t0,
        )
        # Switch to shorter step timeout now that connection is established
        self._sock.settimeout(self.config.step_timeout_seconds)
        self._java_crashed = False

        return self._process_first_obs(raw_obs)

    def _process_first_obs(self, raw_obs):
        """Common logic for processing the first observation after any reset."""
        cfg = self.config

        if raw_obs.get("terminated", False):
            raise ConnectionError(
                f"First obs after reset is terminal (phase={raw_obs.get('phase')}), "
                f"game likely crashed on startup"
            )

        active_id = raw_obs.get("active_entity_id")
        if active_id is None:
            raise KeyError(
                f"Missing 'active_entity_id' in observation "
                f"(phase={raw_obs.get('phase')}, terminated={raw_obs.get('terminated')}, "
                f"keys={list(raw_obs.keys())})"
            )

        self._rl_owner_id = identify_rl_owner(
            raw_obs, active_id
        )

        self.reward_fn.reset()
        if hasattr(self.reward_fn, "set_rl_owner"):
            self.reward_fn.set_rl_owner(self._rl_owner_id)

        self._last_raw_obs = raw_obs
        self._legal_moves = raw_obs.get("legal_moves", [])
        flat = flatten_observation(
            raw_obs, self._rl_owner_id,
            cfg.resolved_board_width, cfg.resolved_board_height,
            legal_moves=self._legal_moves,
            max_legal_moves=cfg.max_legal_moves,
        )
        self._last_flat_obs = flat

        info = self._build_info(raw_obs)
        return flat, info

    def step(self, action):
        # If Java already crashed, keep returning terminal until reset() is called
        if self._java_crashed:
            return self._handle_crash("Java already crashed, awaiting reset")

        action_msg = json.dumps({"type": "action", "move_index": int(action)}) + "\n"
        try:
            self._sock.sendall(action_msg.encode("utf-8"))
        except (BrokenPipeError, ConnectionError, OSError) as e:
            return self._handle_crash(f"Failed to send action: {e}")

        try:
            raw_obs = self._read_obs()
        except ConnectionError as e:
            return self._handle_crash(f"Failed to read observation: {e}")

        terminated = raw_obs.get("terminated", False)
        truncated = raw_obs.get("truncated", False)

        # Early termination: prone with destroyed leg(s) = unrecoverable
        if not terminated and not truncated and self._check_early_termination(raw_obs):
            logger.info(
                "[port:%d] Early termination: RL unit is prone with destroyed leg(s)",
                self._port,
            )
            # Send forfeit to Java so it can end the game cleanly
            # and we can use persistent reset instead of cold restart
            try:
                forfeit_msg = json.dumps({"type": "forfeit"}) + "\n"
                self._sock.sendall(forfeit_msg.encode("utf-8"))
                terminal_obs = self._read_obs()  # read Java's terminal response
                raw_obs = terminal_obs
            except (BrokenPipeError, ConnectionError, OSError, Exception) as e:
                logger.warning(
                    "[port:%d] Forfeit exchange failed (%s), falling back to cold restart",
                    self._port, e,
                )
                self._java_crashed = True  # fallback to cold restart
            raw_obs["game_outcome"] = "LOSS"
            raw_obs["terminated"] = True
            terminated = True
            early_term = True
        else:
            early_term = False

        prev_raw = self._last_raw_obs
        reward = self.reward_fn.compute(prev_raw, raw_obs, terminated)

        self._last_raw_obs = raw_obs
        self._legal_moves = raw_obs.get("legal_moves", [])

        if terminated or truncated:
            # Terminal obs may have empty data — reuse last valid flat obs
            flat = self._last_flat_obs
        else:
            flat = flatten_observation(
                raw_obs, self._rl_owner_id,
                self.config.resolved_board_width,
                self.config.resolved_board_height,
                legal_moves=self._legal_moves,
                max_legal_moves=self.config.max_legal_moves,
            )
            self._last_flat_obs = flat

        info = self._build_info(raw_obs)
        if early_term:
            info["early_termination"] = 1
        return flat, reward, terminated, truncated, info

    def _handle_crash(self, reason: str):
        """Return a graceful terminal step when Java crashes mid-game."""
        logger.error(
            "[port:%d] Java crash detected: %s\nJava log tail:\n  %s",
            self._port, reason, self._java_log_tail(20),
        )
        self._java_crashed = True
        self._legal_moves = []

        crash_obs = {
            "type": "observation",
            "phase": "CRASH",
            "round": self._last_raw_obs.get("round", 0) if self._last_raw_obs else 0,
            "board": {},
            "units": [],
            "legal_moves": [],
            "terminated": True,
            "truncated": False,
        }

        info = self._build_info(crash_obs)
        info["java_crash"] = 1
        return self._last_flat_obs, 0.0, True, False, info

    def _check_early_termination(self, raw_obs: dict) -> bool:
        """Return True if RL unit is prone with at least one destroyed leg.

        A mech that is prone with a destroyed leg cannot stand, making it
        effectively immobilized.  Ending the episode early avoids wasting
        training time on hopeless states.
        """
        units = raw_obs.get("units", [])
        for unit in units:
            if unit.get("owner") != self._rl_owner_id:
                continue
            if not unit.get("prone", False):
                return False
            for loc in unit.get("armor", []):
                if loc.get("location") in ("LL", "RL") and loc.get("internal", 1) <= 0:
                    return True
            return False
        return False

    def _build_info(self, raw_obs: dict) -> dict:
        # Gymnasium's AsyncVectorEnv._add_info cannot merge nested dicts/lists
        # across envs during auto-reset. Only include scalars and numpy arrays
        # at the top level. Complex objects are JSON-serialized as strings.
        terminated = raw_obs.get("terminated", False)
        game_round = raw_obs.get("round", 0)

        # Determine game outcome: 1 = win, -1 = loss, 0 = draw/ongoing
        game_outcome = 0
        if terminated:
            outcome_str = raw_obs.get("game_outcome")
            if outcome_str == "WIN":
                game_outcome = 1
            elif outcome_str == "LOSS":
                game_outcome = -1

        n_legal = len(self._legal_moves)
        moves_truncated = max(0, n_legal - self.config.max_legal_moves)

        info = {
            "action_mask": self.action_masks(),
            "round": game_round,
            "phase": raw_obs.get("phase", ""),
            "n_legal_moves": n_legal,
            "moves_truncated": moves_truncated,
            "game_outcome": game_outcome,
            "game_rounds": game_round,
            "java_crash": 0,
            "early_termination": 0,
            "auto_wake_count": raw_obs.get("auto_wake_count", 0),
        }
        return info

    def action_masks(self) -> np.ndarray:
        mask = np.zeros(self.config.max_legal_moves, dtype=bool)
        n = min(len(self._legal_moves), self.config.max_legal_moves)
        if n > 0:
            mask[:n] = True
        return mask

    def close(self):
        self._cleanup()

    def __del__(self):
        self._cleanup()

    def _cleanup(self):
        port = self._port if hasattr(self, 'config') and self.config else '?'
        logger.debug("[cleanup:%s] START", port)

        if self._reader is not None:
            try:
                self._reader.close()
            except Exception:
                pass
            self._reader = None
            logger.debug("[cleanup:%s] reader closed", port)

        if self._sock is not None:
            try:
                self._sock.shutdown(socket.SHUT_RDWR)
            except Exception:
                pass
            logger.debug("[cleanup:%s] sock.shutdown done", port)
            try:
                self._sock.close()
            except Exception:
                pass
            self._sock = None
            logger.debug("[cleanup:%s] sock.close done", port)

        if self._java is not None:
            logger.debug("[cleanup:%s] java.stop() START", port)
            self._java.stop()
            logger.debug("[cleanup:%s] java.stop() DONE", port)
            self._java = None

        logger.debug("[cleanup:%s] COMPLETE", port)

    def _check_port_available(self, port: int) -> None:
        """Fail fast if port is already in use."""
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
                s.bind(("localhost", port))
        except OSError as e:
            raise RuntimeError(
                f"Port {port} is already in use (cannot start RL bridge). "
                f"Kill the process using the port or choose a different port."
            ) from e

    def _java_log_tail(self, n: int = 10) -> str:
        """Return the last n lines of the Java log for this port, if available."""
        log_path = Path(self.config.megamek_dir) / f"rl_java_{self._port}.log"
        try:
            lines = log_path.read_text().splitlines()
            tail = lines[-n:] if len(lines) > n else lines
            return "\n  ".join(tail)
        except OSError:
            return f"(log not found: {log_path})"

    def _request_thread_dump(self):
        """Send SIGQUIT to the JVM to trigger a thread dump (written to stderr log)."""
        if self._java and self._java._process and self._java._process.poll() is None:
            try:
                os.kill(self._java._process.pid, signal.SIGQUIT)
                logger.info("[port:%d] SIGQUIT sent for thread dump", self._port)
                # Give JVM time to write the dump
                time.sleep(1)
            except (ProcessLookupError, OSError) as e:
                logger.warning("[port:%d] Failed to send SIGQUIT: %s", self._port, e)

    def _read_obs(self) -> dict:
        t0 = time.monotonic()
        try:
            line = self._reader.readline()
        except socket.timeout:
            alive = self._java.is_alive() if self._java else False
            status = "running" if alive else "dead"
            if alive:
                self._request_thread_dump()
            raise ConnectionError(
                f"Timed out waiting for Java observation (Java process is {status}).\n"
                f"Check rl_java_{self._port}.log for thread dump.\n"
                f"Java log tail (rl_java_{self._port}.log):\n"
                f"  {self._java_log_tail(20)}"
            )
        elapsed = time.monotonic() - t0
        if not line:
            alive = self._java.is_alive() if self._java else False
            status = "running" if alive else "dead"
            raise ConnectionError(
                f"Java process closed the connection after {elapsed:.3f}s (process is {status}).\n"
                f"Java log tail (rl_java_{self._port}.log):\n"
                f"  {self._java_log_tail()}"
            )
        obs = json.loads(line)
        terminated = obs.get("terminated", False)
        logger.debug(
            "[port:%d] _read_obs: %d chars in %.3fs (terminated=%s, round=%s)",
            self._port, len(line), elapsed, terminated, obs.get("round"),
        )
        if terminated:
            logger.info(
                "[port:%d] Terminal observation received (%.3fs)", self._port, elapsed,
            )
        return obs
