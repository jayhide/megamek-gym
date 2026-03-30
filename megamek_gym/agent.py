import numpy as np
import torch
import torch.nn as nn
from torch.distributions import Categorical

from megamek_gym.observation import (
    BOARD_CHANNELS, BOARD_HEIGHT, BOARD_WIDTH, DEST_FEATURES, FACING_FEATURES, OBS_SIZE,
)


OUTCOME_MAP = {1: "WIN", -1: "LOSS", 0: "DRAW"}


class Agent(nn.Module):
    def __init__(self, obs_size, action_size, hidden_size=512, critic_obs_size=None):
        super().__init__()
        self.critic_obs_size = critic_obs_size
        critic_input = critic_obs_size if critic_obs_size is not None else obs_size
        self.critic = nn.Sequential(
            nn.Linear(critic_input, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 1),
        )
        self.actor = nn.Sequential(
            nn.Linear(obs_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, action_size),
        )

    def get_value(self, obs):
        if self.critic_obs_size is not None:
            obs = obs[:, :self.critic_obs_size]
        return self.critic(obs)

    def get_action_and_value(self, obs, action_mask, action=None):
        logits = self.actor(obs)
        invalid_mask = ~action_mask
        logits = logits.masked_fill(invalid_mask, -1e8)

        probs = Categorical(logits=logits)
        if action is None:
            action = probs.sample()
        return action, probs.log_prob(action), probs.entropy(), self.get_value(obs)


class HierarchicalAgent(nn.Module):
    """Autoregressive agent: pick destination, then pick facing conditioned on destination."""

    def __init__(self, obs_size, max_destinations, num_facings=6, hidden_size=512, critic_obs_size=None):
        super().__init__()
        self.max_destinations = max_destinations
        self.num_facings = num_facings
        self.critic_obs_size = critic_obs_size
        self.dest_block_offset = OBS_SIZE  # where dest features start in flat obs
        self.facing_block_offset = OBS_SIZE + max_destinations * DEST_FEATURES

        # Shared feature extractor
        self.feature_net = nn.Sequential(
            nn.Linear(obs_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
        )

        # Destination selection head
        self.dest_head = nn.Linear(hidden_size, max_destinations)

        # Facing selection head (conditioned on dest features + per-facing features)
        facing_input_size = hidden_size + DEST_FEATURES + num_facings * FACING_FEATURES
        self.facing_head = nn.Sequential(
            nn.Linear(facing_input_size, 64),
            nn.ReLU(),
            nn.Linear(64, num_facings),
        )

        # Value head (state-only input — no action-space features)
        critic_input = critic_obs_size if critic_obs_size is not None else obs_size
        self.critic = nn.Sequential(
            nn.Linear(critic_input, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 1),
        )

    def get_value(self, obs):
        if self.critic_obs_size is not None:
            obs = obs[:, :self.critic_obs_size]
        return self.critic(obs)

    def get_action_and_value(self, obs, dest_mask, facing_mask, action=None):
        """Autoregressive action selection: destination then facing.

        Args:
            obs: (batch, obs_size) float tensor
            dest_mask: (batch, max_destinations) bool tensor
            facing_mask: (batch, max_destinations, 6) bool tensor
            action: optional (batch, 2) long tensor [dest_idx, facing]

        Returns:
            action: (batch, 2) long tensor
            log_prob: (batch,) joint log probability
            entropy: (batch,) joint entropy
            value: (batch, 1) state value
        """
        features = self.feature_net(obs)

        # Stage 1: destination selection
        dest_logits = self.dest_head(features)
        dest_logits = dest_logits.masked_fill(~dest_mask, -1e8)
        dest_dist = Categorical(logits=dest_logits)

        if action is None:
            dest_action = dest_dist.sample()
        else:
            dest_action = action[:, 0]

        # Stage 2: facing selection (conditioned on dest features + per-facing features)
        off = self.dest_block_offset
        dest_start = off + dest_action * DEST_FEATURES  # (batch,)
        idx = dest_start.unsqueeze(1) + torch.arange(DEST_FEATURES, device=obs.device)
        dest_feats = obs.gather(1, idx)  # (batch, DEST_FEATURES)

        # Gather per-facing features for the selected destination
        f_off = self.facing_block_offset
        n_facing_feats = self.num_facings * FACING_FEATURES
        facing_start = f_off + dest_action * n_facing_feats  # (batch,)
        f_idx = facing_start.unsqueeze(1) + torch.arange(n_facing_feats, device=obs.device)
        facing_feats = obs.gather(1, f_idx)  # (batch, num_facings * FACING_FEATURES)

        facing_input = torch.cat([features, dest_feats, facing_feats], dim=-1)
        facing_logits = self.facing_head(facing_input)

        # Gather the facing mask for each sample's chosen destination
        batch_idx = torch.arange(dest_action.shape[0], device=dest_action.device)
        per_sample_facing_mask = facing_mask[batch_idx, dest_action]  # (batch, 6)
        facing_logits = facing_logits.masked_fill(~per_sample_facing_mask, -1e8)
        facing_dist = Categorical(logits=facing_logits)

        if action is None:
            facing_action = facing_dist.sample()
        else:
            facing_action = action[:, 1]

        # Joint log prob and entropy (autoregressive factorization)
        log_prob = dest_dist.log_prob(dest_action) + facing_dist.log_prob(facing_action)
        entropy = dest_dist.entropy() + facing_dist.entropy()

        combined_action = torch.stack([dest_action, facing_action], dim=-1)
        value = self.get_value(obs)

        return combined_action, log_prob, entropy, value


class SpatialEncoder(nn.Module):
    """CNN that produces full-resolution per-hex feature maps from board channels."""

    def __init__(self, in_channels: int = BOARD_CHANNELS, hidden_channels: int = 32):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, hidden_channels, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(hidden_channels, hidden_channels, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(hidden_channels, hidden_channels, 3, padding=1),
            nn.ReLU(),
        )

    def forward(self, board: torch.Tensor) -> torch.Tensor:
        """Process board tensor through conv layers.

        Args:
            board: (batch, in_channels, H, W)
        Returns:
            (batch, hidden_channels, H, W) — full-resolution feature maps
        """
        return self.conv(board)


class SpatialHierarchicalAgent(nn.Module):
    """Spatial CNN agent: pick destination hex from the board grid, then pick facing.

    Destination selection uses a 1×1 conv over per-hex CNN feature maps.
    Facing selection uses features at the chosen hex + base scalar features.
    Critic uses global-avg-pooled CNN features + base features.
    """

    def __init__(self, board_h: int = BOARD_HEIGHT, board_w: int = BOARD_WIDTH,
                 cnn_channels: int = BOARD_CHANNELS, hidden_channels: int = 32,
                 hidden_size: int = 512):
        super().__init__()
        self.board_h = board_h
        self.board_w = board_w
        self.n_hexes = board_h * board_w
        self.cnn_channels = cnn_channels
        self.hidden_channels = hidden_channels
        self.board_channel_size = cnn_channels * board_h * board_w

        # Spatial encoder (preserves H×W resolution)
        self.spatial_encoder = SpatialEncoder(cnn_channels, hidden_channels)

        # Destination head: per-hex logits via 1×1 conv
        self.dest_head = nn.Conv2d(hidden_channels, 1, 1)

        # Facing head: hex features + base scalar features → 6 facing logits
        facing_input = hidden_channels + OBS_SIZE
        self.facing_head = nn.Sequential(
            nn.Linear(facing_input, 64),
            nn.ReLU(),
            nn.Linear(64, 6),
        )

        # Critic: pooled CNN features + base features → value
        critic_input = hidden_channels + OBS_SIZE
        self.critic = nn.Sequential(
            nn.Linear(critic_input, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 1),
        )
        self.pool = nn.AdaptiveAvgPool2d((1, 1))

    def _split_obs(self, obs: torch.Tensor):
        """Split flat observation into base scalar features and board tensor."""
        base = obs[:, :OBS_SIZE]
        board_flat = obs[:, OBS_SIZE:]
        board = board_flat.view(-1, self.cnn_channels, self.board_h, self.board_w)
        return base, board

    def get_value(self, obs: torch.Tensor) -> torch.Tensor:
        base, board = self._split_obs(obs)
        hex_features = self.spatial_encoder(board)
        pooled = self.pool(hex_features).squeeze(-1).squeeze(-1)
        critic_input = torch.cat([pooled, base], dim=-1)
        return self.critic(critic_input)

    def get_action_and_value(self, obs, dest_mask, facing_mask, action=None):
        """Spatial action selection: pick hex from grid, then pick facing.

        Args:
            obs: (batch, obs_size) float tensor
            dest_mask: (batch, H*W) bool tensor — which hexes are reachable
            facing_mask: (batch, H*W, 6) bool tensor — which facings valid per hex
            action: optional (batch, 2) long tensor [hex_idx, facing]

        Returns:
            action: (batch, 2) long tensor
            log_prob: (batch,) joint log probability
            entropy: (batch,) joint entropy
            value: (batch, 1) state value
        """
        base, board = self._split_obs(obs)
        hex_features = self.spatial_encoder(board)  # (batch, C, H, W)

        # Stage 1: destination (hex) selection
        dest_logits = self.dest_head(hex_features)  # (batch, 1, H, W)
        dest_logits = dest_logits.view(-1, self.n_hexes)  # (batch, H*W)
        dest_logits = dest_logits.masked_fill(~dest_mask, -1e8)
        dest_dist = Categorical(logits=dest_logits)

        if action is None:
            dest_action = dest_dist.sample()  # (batch,) flat hex index
        else:
            dest_action = action[:, 0]

        # Stage 2: facing selection (conditioned on chosen hex features)
        # Gather the feature vector at the selected hex
        hex_y = dest_action // self.board_w
        hex_x = dest_action % self.board_w
        # hex_features is (batch, C, H, W) — gather at (hex_y, hex_x) for each batch
        batch_size = obs.shape[0]
        batch_idx = torch.arange(batch_size, device=obs.device)
        selected_hex_feats = hex_features[batch_idx, :, hex_y, hex_x]  # (batch, C)

        facing_input = torch.cat([selected_hex_feats, base], dim=-1)
        facing_logits = self.facing_head(facing_input)

        # Gather facing mask for chosen hex
        per_sample_facing_mask = facing_mask[batch_idx, dest_action]  # (batch, 6)
        facing_logits = facing_logits.masked_fill(~per_sample_facing_mask, -1e8)
        facing_dist = Categorical(logits=facing_logits)

        if action is None:
            facing_action = facing_dist.sample()
        else:
            facing_action = action[:, 1]

        # Joint log prob and entropy
        log_prob = dest_dist.log_prob(dest_action) + facing_dist.log_prob(facing_action)
        entropy = dest_dist.entropy() + facing_dist.entropy()

        combined_action = torch.stack([dest_action, facing_action], dim=-1)

        # Critic
        pooled = self.pool(hex_features).squeeze(-1).squeeze(-1)
        critic_input = torch.cat([pooled, base], dim=-1)
        value = self.critic(critic_input)

        return combined_action, log_prob, entropy, value


def load_config_from_checkpoint(checkpoint_path, device=None):
    """Load the MegaMekConfig dict stored in a checkpoint."""
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    return checkpoint.get("config", {})


def load_agent(checkpoint_path, obs_size, action_size, device=None):
    """Load a trained Agent from a checkpoint file.

    Returns (agent, checkpoint_dict, device).
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    cfg = checkpoint.get("config", checkpoint.get("args", {}))
    hidden_size = cfg.get("hidden_size", 512)
    # Old checkpoints lack critic_obs_size — fall back to full obs (no slicing)
    critic_obs_size = cfg.get("critic_obs_size", None)
    agent = Agent(obs_size, action_size, hidden_size=hidden_size, critic_obs_size=critic_obs_size).to(device)
    agent.load_state_dict(checkpoint["model"])
    agent.eval()
    return agent, checkpoint, device


def load_hierarchical_agent(checkpoint_path, obs_size, max_destinations, device=None):
    """Load a trained HierarchicalAgent from a checkpoint file.

    Returns (agent, checkpoint_dict, device).
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    cfg = checkpoint.get("config", checkpoint.get("args", {}))
    hidden_size = cfg.get("hidden_size", 512)
    # Old checkpoints lack critic_obs_size — fall back to full obs (no slicing)
    critic_obs_size = cfg.get("critic_obs_size", None)
    agent = HierarchicalAgent(obs_size, max_destinations, hidden_size=hidden_size, critic_obs_size=critic_obs_size).to(device)
    agent.load_state_dict(checkpoint["model"])
    agent.eval()
    return agent, checkpoint, device


def load_spatial_agent(checkpoint_path, board_h=BOARD_HEIGHT, board_w=BOARD_WIDTH, device=None):
    """Load a trained SpatialHierarchicalAgent from a checkpoint file.

    Returns (agent, checkpoint_dict, device).
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    cfg = checkpoint.get("config", checkpoint.get("args", {}))
    hidden_size = cfg.get("hidden_size", 512)
    hidden_channels = cfg.get("hidden_channels", 32)
    agent = SpatialHierarchicalAgent(
        board_h=board_h, board_w=board_w,
        hidden_channels=hidden_channels, hidden_size=hidden_size,
    ).to(device)
    agent.load_state_dict(checkpoint["model"])
    agent.eval()
    return agent, checkpoint, device


def select_action(agent, obs, action_mask, device, deterministic=False):
    """Select an action using the agent for a single observation.

    Returns an int action index.
    """
    obs_t = torch.tensor(obs, dtype=torch.float32).unsqueeze(0).to(device)
    mask_t = torch.tensor(action_mask, dtype=torch.bool).unsqueeze(0).to(device)

    with torch.no_grad():
        if deterministic:
            logits = agent.actor(obs_t)
            logits = logits.masked_fill(~mask_t, -1e8)
            return logits.argmax(dim=1).item()
        else:
            action, _, _, _ = agent.get_action_and_value(obs_t, mask_t)
            return action.item()
