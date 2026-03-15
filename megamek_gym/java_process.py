"""Manage the MegaMek Java subprocess."""

from __future__ import annotations

import subprocess
from pathlib import Path


class JavaProcess:
    """Wraps a MegaMek RLGameRunner subprocess."""

    def __init__(
        self,
        megamek_dir: str,
        rl_unit: str,
        opponent_unit: str,
        board: str,
        port: int,
        timeout_minutes: int,
        max_rotating_round_saves: int = 100,
        paranoid_autosave: bool = False,
        rl_starting_pos: int = 2,
        opponent_starting_pos: int = 6,
        rl_deployment: bool = False,
    ):
        self.megamek_dir = Path(megamek_dir).resolve()
        self.rl_unit = rl_unit
        self.opponent_unit = opponent_unit
        self.board = board
        self.port = port
        self.timeout_minutes = timeout_minutes
        self.max_rotating_round_saves = max_rotating_round_saves
        self.paranoid_autosave = paranoid_autosave
        self.rl_starting_pos = rl_starting_pos
        self.opponent_starting_pos = opponent_starting_pos
        self.rl_deployment = rl_deployment
        self._process: subprocess.Popen | None = None

    def start(self) -> None:
        rl_args = "|".join([
            self.rl_unit,
            self.opponent_unit,
            self.board,
            str(self.port),
            str(self.timeout_minutes),
            str(self.max_rotating_round_saves),
            str(self.paranoid_autosave).lower(),
            str(self.rl_starting_pos),
            str(self.opponent_starting_pos),
            str(self.rl_deployment).lower(),
        ])
        cmd = [
            "./gradlew",
            ":megamek:runRLGameRunner",
            f"-PrlArgs={rl_args}",
        ]
        self._process = subprocess.Popen(
            cmd,
            cwd=self.megamek_dir,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )

    def stop(self) -> None:
        if self._process is None:
            return
        self._process.terminate()
        try:
            self._process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            self._process.kill()
            self._process.wait()
        self._process = None

    def is_alive(self) -> bool:
        return self._process is not None and self._process.poll() is None
