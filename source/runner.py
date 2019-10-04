import sys
from multiprocessing import Manager, Lock

import gym
import os
from gym_carla import settings
from stable_baselines.common.policies import MlpPolicy, CnnLstmPolicy
from stable_baselines.common.vec_env import SubprocVecEnv, DummyVecEnv
from stable_baselines.ppo2 import PPO2
from stable_baselines.a2c import A2C
from gym_carla.carla_utils import startCarlaSims, killCarlaSims, Action, ActionType
from source.reward import Reward
import numpy as np
from multiprocessing import Condition, Value
from ctypes import c_int


class Runner:
    def __init__(self):
        self.env = None
        self.rlModule = None
        self.policy = None
        self.newModel = None
        self.modelNum = None
        self.nSteps = 0
        self.model = None
        self.maxRewardAchieved = float('-inf')

        self.modelName = f"{settings.MODEL_RL_MODULE}" \
                         f"_{ActionType(settings.MODEL_ACTION_TYPE).name}" \
                         f"_seconds-{settings.CARLA_SECONDS_PER_EPISODE}" \
                         f"_policy-{settings.MODEL_POLICY}" \
                         f"_Cars-{settings.CARS_PER_SIM}" \
                         f"_MoreGrassPenalty_MoreTurnSensi"



    def train(self):
        self._setup()
        self.model = self._getModel(strictLoad=False)  # Load the model

        # Perform learning
        self.model.learn(total_timesteps=1000000000, callback=self._callback)
        self.model.save(f"{self.modelName}_{self.modelNum}")

        print("Done Training")

        killCarlaSims()

    def evaluate(self):
        self._setup()
        self.model = self._getModel(strictLoad=True)  # Load the trained agent
        lock = Condition(Lock())
        frameNumber = Value(c_int, 0)
        waiting_threads = Value(c_int, 0)
        #self.env = SubprocVecEnv([lambda i=i: gym.make('CarlaGym-v0', name=self.modelName, carlaInstance=i) for i in range(1)])

        # Evaluate agent in environment
        obs = self.env.reset()

        # model.predict(test_obs) would through an error
        # because the number of test env is different from the number of training env
        # so we need to complete the observation with zeroes
        # zero_completed_obs = np.zeros((settings.CARS_PER_SIM,) + self.env.observation_space.shape)
        # zero_completed_obs[0, :] = test_obs
        # zero_completed_obs = test_obs
        # # IMPORTANT: with recurrent policies, don't forget the state
        # state = None
        # while True:
        #     action, state = self.model.predict(zero_completed_obs, state=state)
        #     # The test env is expecting only one action
        #     new_obs, reward, done, info = self.env.step([action[0]])
        #     # Update the obs
        #     zero_completed_obs[0, :] = new_obs


        for i in range(100000):
            action, _states = self.model.predict(obs)
            obs, rewards, done, info = self.env.step(action)

    def _setup(self):
        # Setup environment
        startCarlaSims()
        lock = Condition(Lock())
        frameNumber = Value(c_int, 0)
        waiting_threads = Value(c_int, 0)
        # self.env = SubprocVecEnv([lambda i=i: gym.make('CarlaGym-v0', name=self.modelName, carlaInstance=i) for i in range(settings.CARLA_SIMS_NO)])
        self.env = SubprocVecEnv([lambda i=i: gym.make('CarlaGym-Sync-v0', name=self.modelName, thread_count=settings.CARS_PER_SIM, frameNumber=frameNumber, waiting_threads=waiting_threads, carlaInstance=i, lock=lock) for i in range(settings.CARS_PER_SIM)])

        # Decide which RL module and policy
        self.rlModule = getattr(sys.modules[__name__], settings.MODEL_RL_MODULE)
        self.policy = getattr(sys.modules[__name__], settings.MODEL_POLICY)
        self.modelNum = settings.MODEL_NUMBER if settings.MODEL_NUMBER is not None else 0

    def _callback(self, _locals, _globals):
        self.nSteps += 1

        # info = _locals["ep_infos"]
        # print(f"{self.nSteps}: {info}")
        mean = np.sum(_locals['true_reward'])
        if self.maxRewardAchieved < mean:
            self.maxRewardAchieved = mean
            if self.nSteps > 10:
                print(f"Saving best model: step {self.nSteps} reward: {mean}")
                _locals['self'].save(f"log/{self.modelName}_{self.modelNum}_best.pkl")

        # Print stats every 100 calls
        if self.nSteps % settings.MODEL_EXPORT_RATE == 0:
            print(f"Saving new model: step {self.nSteps}")
            _locals['self'].save(f"log/{self.modelName}_{self.modelNum}.pkl")
            self.modelNum += 1

        return True

    def _getModel(self, strictLoad=False):
        tensorboard_log = "./tensorboard_log" if settings.MODEL_USE_TENSORBOARD_LOG else None

        # Load from previous model:
        if settings.MODEL_NUMBER is not None and os.path.isfile(f"log/{self.modelName}_{self.modelNum}_best.pkl"):
            print("LOAD MODEL")
            model = self.rlModule.load(f"log/{self.modelName}_{self.modelNum}_best.pkl", env=self.env, tensorboard_log=tensorboard_log, n_steps=settings.MODEL_N_STEPS, nminibatches=settings.MODEL_MINI_BATCHES)
            print("done loading")
            self.modelNum += 1  # Avoid overwriting the loaded model
        # Create new model
        elif strictLoad:
            raise Exception(f"Expected strict load but no model found: {self.modelName}_{self.modelNum}. "
                            f"Try changing model name and number in settings or disable strictLoad")
        else:
            print("NEW MODEL")
            model = self.rlModule(policy=self.policy, env=self.env, tensorboard_log=tensorboard_log, n_steps=settings.MODEL_N_STEPS, nminibatches=settings.MODEL_MINI_BATCHES)

        return model


