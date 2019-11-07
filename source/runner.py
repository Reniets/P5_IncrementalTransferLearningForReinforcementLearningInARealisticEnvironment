import sys
import time
from multiprocessing import Manager, Lock

import gym
import os

from os import path
import sys

from source.callback import Callback
from source.uncertainty_calculator import UncertaintyCalculator

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
from database.sql import Sql


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
        self.directorModelName = None
        self.callback = Callback(self)
        self.seenObservations = []
        self.maxStateCounter = 0
        self.loadFrom = None
        self.sessionId = None

        self.parDict = {

        }

    def train(self, loadFrom=None, directorLoadFrom=None, total_timesteps=1000000000):
        self._setup()
        self.directorModelName = directorLoadFrom
        self.loadFrom = loadFrom
        self.model = self._getModel(strictLoad=False)  # Load the model
        # Perform learning
        self.model.learn(total_timesteps=total_timesteps, callback=self.callback.callback, tb_log_name=self.modelName)

        self.model.save(f"TrainingLogs/FullyTrainedAgentLogs/{self.modelName}_{self.modelNum}")

        print("Done Training")
        self.env.close()

        killCarlaSims()

    def _setup(self):
        # Setup environment
        frame = startCarlaSims()
        self.modelName = settings.MODEL_NAME
        sql = Sql()
        self.sessionId = sql.INSERT_newSession(self.modelName)

        self.frameNumber = Value(c_uint64, frame)

        # self.env = SubprocVecEnv([lambda i=i: gym.make('CarlaGym-v0', name=self.modelName, carlaInstance=i) for i in range(settings.CARLA_SIMS_NO)])
        self.env = SubprocVecEnv([self.make_env(i) for i in range(settings.CARS_PER_SIM)])

        # Decide which RL module and policy
        self.rlModule = getattr(sys.modules[__name__], settings.MODEL_RL_MODULE)
        self.policy = getattr(sys.modules[__name__], settings.MODEL_POLICY)
        self.modelNum = settings.MODEL_NUMBER if settings.MODEL_NUMBER is not None else 0

    def make_env(self, instance):
        # lock = Condition(Lock())
        # frameNumber = Value(c_int, 0)
        # waiting_threads = Value(c_int, 0)

        def _init():
            env = gym.make('CarlaGym-Sync-v0',
                           name=self.modelName,
                           carlaInstance=instance,
                           sessionId=self.sessionId,
                           lock=self.lock,
                           frameNumber=self.frameNumber,
                           waiting_threads=self.waiting_threads,
                           thread_count=settings.CARS_PER_SIM
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

        should_load_model = model_number_is_set and model_exist# and not_imitation_transfer
        print(self.modelName, model_exist)
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

    def _getModelImitation(self):
        model_name = self.directorModelName

        if os.path.isfile(model_name):
            print(F"LOAD DIRECTOR: {model_name}")
            return self.rlModule.load(model_name, **self._getModelKwags(False, dismiss_director=True))

        else:
            raise Exception(f"Expected imitation agent, but no model found: {model_name}. "
                            f"Try changing model name and number in settings.")

    def _getModelKwags(self, new_model: bool, dismiss_director=False):
        # Always
        kwags = {
            "env": self.env,
            "tensorboard_log": "./TensorboardLogs" if settings.MODEL_USE_TENSORBOARD_LOG else None,
            "n_steps": settings.MODEL_N_STEPS,
            "nminibatches": settings.MODEL_MINI_BATCHES,
            "noptepochs": settings.MODEL_NOPTEPOCHS,
            "ent_coef": settings.MODEL_ENT_COEF,
            "learning_rate": lambda frac: settings.MODEL_LEARNING_RATE,
            "cliprange": lambda frac: settings.MODEL_CLIP_RANGE,
            "cliprange_vf": lambda frac: settings.MODEL_CLIP_RANGE_VF,
            "gamma": settings.MODEL_DISCOUNT_FACTOR,
            "vf_coef": settings.MODEL_VF_COEF
        }

        # Only new models
        if new_model:
            kwags.update({"policy": self.policy})

        # Only imitation agents
        if settings.TRANSFER_AGENT == TransferType.IMITATION.value and not dismiss_director:

            # uncertainty = lambda: None
            # uncertainty.get_uncertainty = lambda obs: self.calculateUncertainty(obs)
            # uncertainty.update_uncertainty = lambda indices: self.updateStateCounter(indices)

            kwags.update({
                "polling_rate": lambda: settings.TRANSFER_POLLING_RATE_START,
                "uncertainty": UncertaintyCalculator(self.modelName),
                #"uncertainty": lambda obs: self.calculateUncertainty(obs),
                "director": self._getModelImitation(),
            })

        return kwags

    def calculateUncertainty(self, obs):
        uncertainties = []
        ids = []

        # Get uncertainties
        for observation in obs:
            uncertainty, index = self._getObservationUncertaintyAndID(observation)

            uncertainties.append(uncertainty)
            ids.append(index)

            # uncertainties.append(self._getObservationUncertainty(observation))

        return uncertainties, ids

    def _getObservationUncertaintyAndID(self, observation):
        state_counter, index = self._getStateCounterAndIndex(observation)

        return 1/(1+(settings.UNCERTAINTY_RATE*state_counter)), index

    def _getObservationUncertainty(self, observation):
        state_counter = self._getStateCounterAndIncrement(observation)

        if state_counter == 0:
            pass#print(f"NEW STATE!! - TOTAL STATES: {len(self.seenObservations)}")

        return 1/(1+(settings.UNCERTAINTY_RATE*state_counter))

    def _getStateCounterAndIndex(self, obs):
        # Loop all seen images, and try to locate an image that is close to the current one
        for index, ob_tuple in enumerate(self.seenObservations):
            seen_ob = ob_tuple[1]
            equal_pixels = self._getImageEqualness(seen_ob, obs)

            if equal_pixels >= settings.TRANSFER_IMITATION_THRESHOLD:
                return ob_tuple[0], index

        # We have never seen anything like this new observation, so add it to the list,
        # and return 0 since it's the first time we see it
        self.seenObservations.append((0, obs))
        #print(f"NEW STATE!! - TOTAL STATES: {len(self.seenObservations)}")
        return 0, len(self.seenObservations)-1

    def updateStateCounter(self, indices):
        for index in indices:
            self.seenObservations[index][0] += 1
            #self.seenObservations[index] = (seen_counter + 1, obs)

    def _getStateCounterAndIncrement(self, new_ob):
        # Loop all seen images, and try to locate an image that is close to the current one
        for index, ob_tuple in enumerate(self.seenObservations):
            seen_ob = ob_tuple[1]
            equal_pixels = self._getImageEqualness(seen_ob, new_ob)

            if equal_pixels >= settings.TRANSFER_IMITATION_THRESHOLD:
                self._incrementStateCounter(index, new_ob)
                return ob_tuple[0]

        # We have never seen anything like this new observation, so add it to the list,
        # and return 0 since it's the first time we see it
        self.seenObservations.append((1, new_ob))
        return 0

    # Increments the counter, and returns the counter pre increment
    def _incrementStateCounter(self, seen_ob_index, seen_ob):
        ob_tuple = self.seenObservations[seen_ob_index]                     # Get the data tuple
        seen_counter = ob_tuple[0]                                          # Get the seen counter
        self.seenObservations[seen_ob_index] = (seen_counter + 1, seen_ob)  # Increment the counter

        if self.maxStateCounter < seen_counter + 1:
            self.maxStateCounter = seen_counter + 1

    def _getImageEqualness(self, img_a, img_b):
        return np.sum(np.equal(img_a, img_b))

    def _getModelName(self):
        if settings.TRANSFER_AGENT is 0:
            modelName = f"TrainingLogs/FullyTrainedAgentLogs/{self.modelName}.zip"
        elif settings.TRANSFER_AGENT is 1:
            modelName = self.loadFrom
        else:
            if self.loadFrom is not None:
                modelName = self.loadFrom
            else:
                modelName = f"TrainingLogs/FullyTrainedAgentLogs/{self.modelName}.zip"

        return modelName

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

            action, state = self.model.predict(obs, state=state, mask=done, deterministic=False)
            obs, reward, done, _ = self.env.step(action)
