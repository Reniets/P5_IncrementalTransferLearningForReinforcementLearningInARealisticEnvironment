from gym.envs.registration import register

register(
    id='CarlaGym-v0',
    entry_point='gym_carla.envs:CarlaEnv',
)
