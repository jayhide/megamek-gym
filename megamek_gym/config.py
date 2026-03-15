"""MegaMek environment configuration."""

from __future__ import annotations

import dataclasses
import re
from pathlib import Path

import yaml


def parse_board_dimensions(board_name: str) -> tuple[int, int] | None:
    """Extract WxH from board name like 'Map Set 6/16x17 BattleForce 2'."""
    match = re.search(r"(\d+)x(\d+)", board_name)
    if match:
        return int(match.group(1)), int(match.group(2))
    return None


@dataclasses.dataclass
class MegaMekConfig:
    """Configuration for a MegaMek RL environment instance."""

    megamek_dir: str = "../megamek"
    rl_unit: str = "Firestarter FS9-H"
    opponent_unit: str = "Commando COM-2D"
    board: str = "Map Set 6/16x17 BattleForce 2"
    board_width: int | None = None
    board_height: int | None = None
    rl_port: int = 9999
    env_index: int = 0
    java_timeout_minutes: int = 10
    max_legal_moves: int = 1000
    max_rotating_round_saves: int = 100
    paranoid_autosave: bool = False
    rl_starting_pos: int = 2
    opponent_starting_pos: int = 6
    rl_deployment: bool = False
    firing_strategy: str = "princess"

    def __post_init__(self):
        # Validate that board dimensions can be resolved
        _ = self.resolved_board_width
        _ = self.resolved_board_height

    @property
    def resolved_board_width(self) -> int:
        if self.board_width is not None:
            return self.board_width
        dims = parse_board_dimensions(self.board)
        if dims is None:
            raise ValueError(
                f"Cannot derive board width from '{self.board}'. "
                "Set board_width explicitly."
            )
        return dims[0]

    @property
    def resolved_board_height(self) -> int:
        if self.board_height is not None:
            return self.board_height
        dims = parse_board_dimensions(self.board)
        if dims is None:
            raise ValueError(
                f"Cannot derive board height from '{self.board}'. "
                "Set board_height explicitly."
            )
        return dims[1]

    def save(self, path: str | Path) -> None:
        """Save configuration to a YAML file."""
        d = dataclasses.asdict(self)
        # Omit None board dimensions (they auto-derive)
        d = {k: v for k, v in d.items() if v is not None}
        Path(path).write_text(yaml.safe_dump(d, sort_keys=False))

    @classmethod
    def load(cls, path: str | Path) -> MegaMekConfig:
        """Load configuration from a YAML file."""
        data = yaml.safe_load(Path(path).read_text())
        return cls(**data)
