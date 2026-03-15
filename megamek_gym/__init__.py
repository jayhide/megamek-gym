from gymnasium.envs.registration import register


def register_envs():
    pass


register(
    id="MegaMekGym/MegaMek-v0",
    entry_point="megamek_gym.env:MegaMekEnv",
)
