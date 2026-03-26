import numpy as np
import torch
import torch.nn as nn
from torch.distributions import Categorical

from megamek_gym.observation import DEST_FEATURES, FACING_FEATURES, OBS_SIZE


OUTCOME_MAP = {1: "WIN", -1: "LOSS", 0: "DRAW"}


class Agent(nn.Module):
    def __init__(self, obs_size, action_size, hidden_size=512):
        super().__init__()
        self.critic = nn.Sequential(
            nn.Linear(obs_size, hidden_size),
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

    def __init__(self, obs_size, max_destinations, num_facings=6, hidden_size=512):
        super().__init__()
        self.max_destinations = max_destinations
        self.num_facings = num_facings
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

        # Value head (independent)
        self.critic = nn.Sequential(
            nn.Linear(obs_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 1),
        )

    def get_value(self, obs):
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
        value = self.critic(obs)

        return combined_action, log_prob, entropy, value


def load_agent(checkpoint_path, obs_size, action_size, device=None):
    """Load a trained Agent from a checkpoint file.

    Returns (agent, checkpoint_dict, device).
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    hidden_size = checkpoint.get("config", checkpoint.get("args", {})).get("hidden_size", 512)
    agent = Agent(obs_size, action_size, hidden_size=hidden_size).to(device)
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
    hidden_size = checkpoint.get("config", checkpoint.get("args", {})).get("hidden_size", 512)
    agent = HierarchicalAgent(obs_size, max_destinations, hidden_size=hidden_size).to(device)
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
