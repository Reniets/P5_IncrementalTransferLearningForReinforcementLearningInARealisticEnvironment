import os, logging
import time

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
logging.getLogger("tensorflow").setLevel(logging.CRITICAL)
logging.getLogger("tensorflow_hub").setLevel(logging.CRITICAL)

import gym
from stable_baselines.common.policies import MlpPolicy
from stable_baselines.common.vec_env import SubprocVecEnv
from stable_baselines import *
from gym_carla.carla_utils import startCarlaSims, killCarlaSims, Action

from gym_carla.envs.carla_env import CarlaEnv


def runCarlaGymTraining():
    startCarlaSims()

    env = SubprocVecEnv([lambda: gym.make('CarlaGym-v0')])
    model = A2C(MlpPolicy, env, verbose=1)
    model.learn(total_timesteps=10000)
    model.save("a2c_carla1")

    input('Press enter to exit...')
    killCarlaSims()

runCarlaGymTraining()

