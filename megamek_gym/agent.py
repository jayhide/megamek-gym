import numpy as np
import torch
import torch.nn as nn
from torch.distributions import Categorical


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


def load_agent(checkpoint_path, obs_size, action_size, device=None):
    """Load a trained Agent from a checkpoint file.

    Returns (agent, checkpoint_dict, device).
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    hidden_size = checkpoint.get("args", {}).get("hidden_size", 512)
    agent = Agent(obs_size, action_size, hidden_size=hidden_size).to(device)
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
