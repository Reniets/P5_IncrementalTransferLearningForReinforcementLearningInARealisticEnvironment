import os, logging
from source.data_handler import makeVideoFromSensorFrames, clearFrameFolder
import time

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
logging.getLogger("tensorflow").setLevel(logging.CRITICAL)
logging.getLogger("tensorflow_hub").setLevel(logging.CRITICAL)

import gym
from stable_baselines.common.policies import MlpPolicy, CnnLstmPolicy
from stable_baselines.common.vec_env import SubprocVecEnv
from stable_baselines import *
from stable_baselines.bench import Monitor
from gym_carla.carla_utils import startCarlaSims, killCarlaSims, Action

from gym_carla.envs.carla_env import CarlaEnv

n_steps = 0
agentNum = 0
name = "best_model_discrete_no_early_end"

def callback(_locals, _globals):
    global n_steps, agentNum
    if (n_steps + 1) % 20000 == 0:
        agentNum += 1

    # Print stats every 1000 calls
    if (n_steps + 1) % 20000 == 0:
        print("Saving new best model")
        _locals['self'].save(f"log/{name}_{str(agentNum)}.pkl")

    n_steps += 1
    return True


def runCarlaGymTraining():
    startCarlaSims()

    env = SubprocVecEnv([lambda: gym.make('CarlaGym-v0')])

    if os.path.isfile(f"log/{name}_0.pkl"):
        print("load")
        model = A2C.load(f"log/{name}_0", env, tensorboard_log='./tensorboard_log')
    else:
        print("New")
        model = A2C(CnnLstmPolicy, env, tensorboard_log='./tensorboard_log')

    model.learn(total_timesteps=1000000000, callback=callback)
    model.save("a2c_carla1")
    print("Done")

    killCarlaSims()


# clearFrameFolder()
runCarlaGymTraining()

# makeVideoFromSensorFrames()
