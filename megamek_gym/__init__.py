from gymnasium.envs.registration import register

from megamek_gym.config import MegaMekConfig  # noqa: F401


def register_envs():
    pass


register(
    id="MegaMekGym/MegaMek-v0",
    entry_point="megamek_gym.env:MegaMekEnv",
)
