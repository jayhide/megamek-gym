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
    rl_unit: str = "Commando COM-2D"
    opponent_unit: str = "Commando COM-2D"
    board: str = "Map Set 6/16x17 BattleForce 2"
    board_width: int | None = None
    board_height: int | None = None
    rl_port: int = 9999
    env_index: int = 0
    java_timeout_minutes: int = 10
    connection_retries: int = 60
    connection_retry_delay: float = 1.0
    max_legal_moves: int = 400
    max_rotating_round_saves: int = 0
    paranoid_autosave: bool = False
    save_budget_mb: int = 1000
    rl_starting_pos: int = 2
    opponent_starting_pos: int = 6
    rl_deployment: bool = False
    firing_strategy: str = "princess"
    step_timeout_seconds: int = 30
    max_game_rounds: int = 50
    perf_log: bool = False
    opponent_type: str = "princess"
    force_gc: bool = False
    mem_log: int = 0
    auto_wake_pilot: bool = True
    force_unconscious_on_turn: int = 0
    rl_fixed_coords: tuple[int, int] | None = None
    opponent_fixed_coords: tuple[int, int] | None = None
    enable_game_reports: bool = False
    validate_caches: bool = False
    # Training hyperparameters (used by train_ppo.py)
    exp_name: str = "megamek-ppo"
    seed: int = 1
    cuda: bool = True
    num_envs: int = 8
    stagger_delay: float = 3.0
    total_timesteps: int = 500_000
    num_steps: int = 256
    num_minibatches: int = 4
    update_epochs: int = 4
    learning_rate: float = 3e-4
    anneal_lr: bool = True
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_coef: float = 0.2
    clip_vloss: bool = False
    ent_coef: float = 0.05
    vf_coef: float = 1.0
    max_grad_norm: float = 0.5
    target_kl: float = 0.03
    hidden_size: int = 512
    save_interval: int = 50

    def __post_init__(self):
        # YAML deserializes [x, y] as list; convert to tuple
        if isinstance(self.rl_fixed_coords, list):
            self.rl_fixed_coords = tuple(self.rl_fixed_coords)
        if isinstance(self.opponent_fixed_coords, list):
            self.opponent_fixed_coords = tuple(self.opponent_fixed_coords)
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
