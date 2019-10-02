from gym.envs.registration import register

register(
    id='CarlaGym-v0',
    entry_point='gym_carla.envs:CarlaEnv',
)

register(
    id='CarlaGym-Sync-v0',
    entry_point='gym_carla.envs:CarlaSyncEnv',
)

