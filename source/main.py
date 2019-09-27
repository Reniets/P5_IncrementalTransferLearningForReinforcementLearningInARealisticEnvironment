import os, logging
from source.data_handler import makeVideoFromSensorFrames, clearFrameFolder
import time

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
logging.getLogger("tensorflow").setLevel(logging.CRITICAL)
logging.getLogger("tensorflow_hub").setLevel(logging.CRITICAL)

import gym
from gym_carla import settings
from stable_baselines.common.policies import MlpPolicy, CnnLstmPolicy
from stable_baselines.common.vec_env import SubprocVecEnv
from stable_baselines import *
from stable_baselines.bench import Monitor
from gym_carla.carla_utils import startCarlaSims, killCarlaSims, Action
from gym_carla.envs.carla_env import CarlaEnv

n_steps = 0

startCarlaSims()
# Setup environment

env = SubprocVecEnv([lambda i=i: gym.make('CarlaGym-v0', carlaInstance=i) for i in range(settings.CARLA_SIMS_NO)])

# Decide which RL module and policy
RL_MODULE = PPO2
POLICY = CnnLstmPolicy
# give model name
model_name = "best_model_discrete_new_rewards"
# Decide whether to start from a previous version and if so which previous version
previous_version = None  # Ex (model_name, 5)  # None if blank slate, tuple of (name, agentNum) if continued or transfer

# Automatically assigns the agentNum.
agentNum = previous_version[1] + 1 if previous_version is not None and previous_version[0] is model_name else 0


def callback(_locals, _globals):
    global n_steps, agentNum
    n_steps += 1

    # Print stats every 20000 calls
    if n_steps % 100 == 0:
        print(f"Saving new model: step {n_steps}")
        _locals['self'].save(f"log/{model_name}_{str(agentNum)}.pkl")
        agentNum += 1

    return True


def load_model_from_file(module, v):
    if os.path.isfile(f"log/{v[0]}_{v[1]}.pkl"):
        print("load")
        return module.load(f"log/{v[0]}_{v[1]}", env)
    else:
        print(f"Failed to load previous model. File does not exist: log/{v[0]}_{v[1]}.pkl")
        raise


# Load from previous model:
if previous_version is not None:
    model = load_model_from_file(RL_MODULE, previous_version)
else:
    print("NEW")
    model = RL_MODULE(POLICY, env, nminibatches=settings.CARLA_SIMS_NO)

# makeVideoFromSensorFrames()
clearFrameFolder()
print("Done")
# Perform learning
model.learn(total_timesteps=1000000000, callback=callback)
model.save(f"{model_name}_{n_steps}")

print("Done!")

killCarlaSims()

