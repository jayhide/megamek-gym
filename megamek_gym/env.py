"""MegaMek Gymnasium environment."""

from __future__ import annotations

import json
import socket
import time

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
            config.resolved_board_width, config.resolved_board_height
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

    @property
    def _port(self) -> int:
        return self.config.rl_port + self.config.env_index

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)

        self._cleanup()

        cfg = self.config

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
        )
        self._java.start()

        # Connect with retry
        self._sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        connected = False
        for _ in range(60):
            try:
                self._sock.connect(("localhost", self._port))
                connected = True
                break
            except ConnectionRefusedError:
                if not self._java.is_alive():
                    raise RuntimeError("Java process died before accepting connection")
                time.sleep(1.0)
        if not connected:
            self._cleanup()
            raise RuntimeError(
                f"Could not connect to Java on port {self._port} after 60s"
            )

        self._reader = self._sock.makefile("r", encoding="utf-8")

        # Read first observation
        raw_obs = self._read_obs()
        self._rl_owner_id = identify_rl_owner(
            raw_obs, raw_obs["active_entity_id"]
        )

        self.reward_fn.reset()
        if hasattr(self.reward_fn, "set_rl_owner"):
            self.reward_fn.set_rl_owner(self._rl_owner_id)

        self._last_raw_obs = raw_obs
        self._legal_moves = raw_obs.get("legal_moves", [])
        flat = flatten_observation(
            raw_obs, self._rl_owner_id,
            cfg.resolved_board_width, cfg.resolved_board_height,
        )
        self._last_flat_obs = flat

        info = self._build_info(raw_obs)
        return flat, info

    def step(self, action):
        action_msg = json.dumps({"type": "action", "move_index": int(action)}) + "\n"
        self._sock.sendall(action_msg.encode("utf-8"))

        raw_obs = self._read_obs()
        terminated = raw_obs.get("terminated", False)
        truncated = raw_obs.get("truncated", False)

        prev_raw = self._last_raw_obs
        reward = self.reward_fn.compute(prev_raw, raw_obs, terminated)

        if terminated or truncated:
            # Terminal obs may have empty data — reuse last valid flat obs
            flat = self._last_flat_obs
        else:
            flat = flatten_observation(
                raw_obs, self._rl_owner_id,
                self.config.resolved_board_width,
                self.config.resolved_board_height,
            )
            self._last_flat_obs = flat

        self._last_raw_obs = raw_obs
        self._legal_moves = raw_obs.get("legal_moves", [])

        info = self._build_info(raw_obs)
        return flat, reward, terminated, truncated, info

    def _build_info(self, raw_obs: dict) -> dict:
        units = raw_obs.get("units", [])
        rl_unit = None
        enemy_unit = None
        for u in units:
            if u.get("owner") == self._rl_owner_id:
                rl_unit = u
            else:
                enemy_unit = u
        return {
            "legal_moves": self._legal_moves,
            "action_mask": self.action_masks(),
            "round": raw_obs.get("round", 0),
            "phase": raw_obs.get("phase", ""),
            "rl_unit": rl_unit,
            "enemy_unit": enemy_unit,
        }

    def action_masks(self) -> np.ndarray:
        mask = np.zeros(self.config.max_legal_moves, dtype=bool)
        n = len(self._legal_moves)
        if n > 0:
            mask[:n] = True
        return mask

    def close(self):
        self._cleanup()

    def _cleanup(self):
        if self._reader is not None:
            try:
                self._reader.close()
            except Exception:
                pass
            self._reader = None
        if self._sock is not None:
            try:
                self._sock.close()
            except Exception:
                pass
            self._sock = None
        if self._java is not None:
            self._java.stop()
            self._java = None

    def _read_obs(self) -> dict:
        line = self._reader.readline()
        if not line:
            raise ConnectionError("Java process closed the connection")
        return json.loads(line)
