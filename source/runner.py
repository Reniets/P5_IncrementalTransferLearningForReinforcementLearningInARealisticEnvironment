import sys
import time
from multiprocessing import Manager, Lock

import gym
import os

from os import path
import sys

from source.callback import Callback

sys.path.append(path.abspath('../../stable-baselines'))

from gym.spaces import Discrete, Box, Tuple, MultiDiscrete
from stable_baselines.bench import Monitor
from gym_carla import settings
from stable_baselines.common.policies import MlpPolicy, CnnLstmPolicy, CnnPolicy
from stable_baselines.common.vec_env import SubprocVecEnv, DummyVecEnv
from stable_baselines.ppo2 import PPO2
from stable_baselines.a2c import A2C
from gym_carla.carla_utils import startCarlaSims, killCarlaSims, Action, ActionType, TransferType
from source.reward import Reward
import numpy as np
import tensorflow as tf
from multiprocessing import Condition, Value
from ctypes import c_uint64


class Runner:
    def __init__(self):
        self.env = None
        self.rlModule = None
        self.policy = None
        self.newModel = None
        self.modelNum = None
        self.model = None
        self.lock = Condition(Lock())
        self.frameNumber = Value(c_uint64, 0)
        self.waiting_threads = Value(c_uint64, 0)
        self.modelName = None
        self.callback = Callback(self)

        self.parDict = {

        }

    def train(self, total_timesteps=1000000000):
        self._setup()
        self.model = self._getModel(strictLoad=False)  # Load the model

        # Perform learning
        #before = time.clock()
        self.model.learn(total_timesteps=total_timesteps, callback=self.callback.callback, tb_log_name=self.modelName)
        #self.model.learn(total_timesteps=100000, callback=self._callback)
        #duration = time.clock() - before
        #print(f"World ticks: {self.world_ticks.value}, Duration: {duration}", "")
        #print(f"WT/S: {self.world_ticks.value / duration}, CWT/S: {self.world_ticks.value / duration * settings.CARS_PER_SIM}")

        self.model.save(f"{self.modelName}_{self.modelNum}")

        print("Done Training")
        self.env.close()

        killCarlaSims()

    def _setup(self):
        # Setup environment
        frame = startCarlaSims()

        self.frameNumber = Value(c_uint64, frame)

        # self.env = SubprocVecEnv([lambda i=i: gym.make('CarlaGym-v0', name=self.modelName, carlaInstance=i) for i in range(settings.CARLA_SIMS_NO)])
        self.env = SubprocVecEnv([self.make_env(i) for i in range(settings.CARS_PER_SIM)])

        # Decide which RL module and policy
        self.rlModule = getattr(sys.modules[__name__], settings.MODEL_RL_MODULE)
        self.policy = getattr(sys.modules[__name__], settings.MODEL_POLICY)
        self.modelName = settings.MODEL_NAME
        self.modelNum = settings.MODEL_NUMBER if settings.MODEL_NUMBER is not None else 0

    def make_env(self, instance):
        # lock = Condition(Lock())
        # frameNumber = Value(c_int, 0)
        # waiting_threads = Value(c_int, 0)

        def _init():
            env = gym.make('CarlaGym-Sync-v0',
                           name=self.modelName,
                           carlaInstance=instance,
                           lock=self.lock,
                           frameNumber=self.frameNumber,
                           waiting_threads=self.waiting_threads,
                           thread_count = settings.CARS_PER_SIM
                           )

            env = Monitor(env, f'monitor/{instance}', allow_early_resets=True)

            return env

        return _init

    def _getModel(self, strictLoad=False):
        # Get model name
        modelName = self._getModelName()

        # Boolean features to determine if a model should be loaded
        model_number_is_set = settings.MODEL_NUMBER is not None
        model_exist = os.path.isfile(modelName)
        not_imitation_transfer = settings.TRANSFER_AGENT != TransferType.IMITATION.value

        should_load_model = model_number_is_set and model_exist and not_imitation_transfer

        # Create model
        if should_load_model:
            print("LOAD MODEL")
            model = self.rlModule.load(modelName, **self._getModelKwags(False))
            print(f"DONE LOADING: {modelName}")

        elif strictLoad:
            raise Exception(f"Expected strict load but no model found: {self.modelName}_{self.modelNum}. "
                            f"Try changing model name and number in settings or disable strictLoad")
        else:
            print("NEW MODEL")
            model = self.rlModule(**self._getModelKwags(True))
            print("DONE CREATING NEW MODEL")

        return model

    def _getModelKwags(self, new_model: bool):
        kwags = {
            "env": self.env,
            "tensorboard_log": "./ExperimentTensorboardLog" if settings.MODEL_USE_TENSORBOARD_LOG else None,
            "n_steps": settings.MODEL_N_STEPS,
            "nminibatches": settings.MODEL_MINI_BATCHES,
            "ent_coef": settings.MODEL_ENT_COEF,
            "learning_rate": lambda frac: settings.MODEL_LEARNING_RATE,
            "cliprange": lambda frac: settings.MODEL_CLIP_RANGE,
            "cliprange_vf": lambda frac: settings.MODEL_CLIP_RANGE
        }

        if new_model:
            kwags.update({"policy": self.policy})

        return kwags

    def _getModelName(self):
        return f"ExperimentLogsFinal/{self.modelName}_{self.modelNum}.zip" if not settings.TRANSFER_AGENT == TransferType.WEIGHTS.value else f"TransferAgentLogs/{self.modelName}.zip"

    def evaluate(self):
        self._setup()
        self.model = self._getModel(strictLoad=True)  # Load the trained agent

        # Evaluate agent in environment
        obs = self.env.reset()

        state = None
        # When using VecEnv, done is a vector
        done = [False for _ in range(self.env.num_envs)]
        for _ in range(100000):
            # We need to pass the previous state and a mask for recurrent policies
            # to reset lstm state when a new episode begin
            action, state = self.model.predict(obs, state=state, mask=done, deterministic=True)
            obs, reward, done, _ = self.env.step(action)