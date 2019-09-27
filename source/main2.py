import os, logging
from source.data_handler import makeVideoFromSensorFrames, clearFrameFolder
import time
from gym_carla import settings
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
logging.getLogger("tensorflow").setLevel(logging.CRITICAL)
logging.getLogger("tensorflow_hub").setLevel(logging.CRITICAL)

import gym
from stable_baselines.common.policies import MlpPolicy
from stable_baselines.common.vec_env import SubprocVecEnv
from stable_baselines import *
from stable_baselines.bench import Monitor
from gym_carla.carla_utils import startCarlaSims, killCarlaSims, Action

from gym_carla.envs.carla_env import CarlaEnv


def manuallyEvaluateAgent():
    startCarlaSims()

    env = SubprocVecEnv([lambda: gym.make('CarlaGym-v0')])

    # Load the trained agent
    model = PPO2.load("log/best_model_discrete_new_rewards_12", env, nminibatches=settings.CARLA_SIMS_NO)

    # Enjoy trained agent
    obs = env.reset()
    for i in range(100000):
        action, _states = model.predict(obs)
        obs, rewards, dones, info = env.step(action)
        time.sleep(0.1)


manuallyEvaluateAgent()