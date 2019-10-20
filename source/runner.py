import sys
import time
from multiprocessing import Manager, Lock

import gym
import os

from os import path
import sys
sys.path.append(path.abspath('../../stable-baselines'))

from gym.spaces import Discrete, Box, Tuple, MultiDiscrete
from stable_baselines.bench import Monitor
from gym_carla import settings
from stable_baselines.common.policies import MlpPolicy, CnnLstmPolicy, CnnPolicy
from stable_baselines.common.vec_env import SubprocVecEnv, DummyVecEnv
from stable_baselines.ppo2 import PPO2
from stable_baselines.a2c import A2C
from gym_carla.carla_utils import startCarlaSims, killCarlaSims, Action, ActionType
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
        self.nEpisodes = 0
        self.model = None
        self.maxRewardAchieved = float('-inf')
        self.lock = Condition(Lock())
        self.frameNumber = Value(c_uint64, 0)
        self.waiting_threads = Value(c_uint64, 0)
        self.prev_episode = 0
        self.modelName = None

        self.parDict = {

        }

    def train(self, total_timesteps=1000000000):
        self._setup()
        self.model = self._getModel(strictLoad=False)  # Load the model

        # Perform learning
        #before = time.clock()
        self.model.learn(total_timesteps=total_timesteps, callback=self._callback, tb_log_name=self.modelName)
        #self.model.learn(total_timesteps=100000, callback=self._callback)
        #duration = time.clock() - before
        #print(f"World ticks: {self.world_ticks.value}, Duration: {duration}", "")
        #print(f"WT/S: {self.world_ticks.value / duration}, CWT/S: {self.world_ticks.value / duration * settings.CARS_PER_SIM}")

        self.model.save(f"{self.modelName}_{self.modelNum}")

        print("Done Training")
        self.env.close()

        killCarlaSims()

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

    def _model_learning_rate(self, frac):
        print(type(self))
        n_episodes = self._getEpisodeCount()
        print(f'{frac}, {n_episodes}')

        return settings.MODEL_LEARNING_RATE

    def _getEpisodeCount(self):
        return len(self.env.get_attr('episode_rewards', 0)[0])

    def _callback(self, runner_locals, _locals):
        self.nEpisodes += 1

        self._updateLearningRate(_locals)
        self._updateClipRange(_locals)
        self._storeTensorBoardData(_locals)
        self._exportBestModel(runner_locals, _locals)
        self._printCallbackStats(_locals)

        if self.nEpisodes % settings.CARLA_EVALUATION_RATE == 0:
            self._testVehicles(_locals, runner_locals)

        return True

    def _updateLearningRate(self, _locals):
        new_learning_rate = self._calcluateNewLearningRate_Exponential()
        _locals['self'].learning_rate = lambda frac: new_learning_rate

    def _updateClipRange(self, _locals):
        new_clip_range = self._calculateNewClipRange_Linear()
        _locals['self'].cliprange = lambda frac: new_clip_range
        # _locals['self'].cliprange_vf = lambda frac: new_clip_range
        # _locals.update({'cliprange_vf': lambda frac: new_clip_range})

    def _calculateNewClipRange_Linear(self):
        scale = self._getEpisodeScaleTowardsZero()
        clip_diff = settings.MODEL_CLIP_RANGE - settings.MODEL_CLIP_RANGE_MIN
        newClip = max(settings.MODEL_CLIP_RANGE_MIN + (clip_diff*scale), 0)

        return newClip

    def _calcluateNewLearningRate_Exponential(self):
        n_episodes = self._getEpisodeCount()

        newLearningRate = settings.MODEL_LEARNING_RATE * (1 / (1 + (settings.MODEL_LEARNING_RATE * 20) * n_episodes))

        return max(newLearningRate, settings.MODEL_LEARNING_RATE_MIN)

    def _calculateNewLearningRate_Linear(self):
        scale = self._getEpisodeScaleTowardsZero()                                     # Calculate scale depending on max episode progress
        new_learning_rate = max(settings.MODEL_LEARNING_RATE * scale, 0)    # Calculate new learning rate

        return max(new_learning_rate, settings.MODEL_LEARNING_RATE_MIN)     # Return the learning rate, while respecting minimum learning rate

    def _getEpisodeScaleTowardsZero(self):
        n_episodes = self._getEpisodeCount()
        return max(1 - (n_episodes / settings.MODEL_MAX_EPISODES), 0)

    def _printCallbackStats(self, _locals):
        # Print stats every 100 calls
        if self.nEpisodes % settings.MODEL_EXPORT_RATE == 0:
            print(f"Saving new model: step {self.nEpisodes}")
            _locals['self'].save(f"ExperimentLogsFinal/{self.modelName}_{self.modelNum}.pkl")
            self.modelNum += 1

    def _exportBestModel(self, runner_locals, _locals):
        # info = _locals["ep_infos"]
        # print(f"{self.nSteps}: {info}")
        mean = np.sum(np.asarray(runner_locals['mb_rewards']))
        if self.maxRewardAchieved < mean:
            self.maxRewardAchieved = mean
            if self.nEpisodes > 10:
                print(f"Saving best model: step {self.nEpisodes} reward: {mean}")
                _locals['self'].save(f"ExperimentLogsFinal/{self.modelName}_{self.modelNum}_best.pkl")

    def _storeTensorBoardData(self, _locals):
        n_episodes = self._getEpisodeCount()

        if n_episodes > self.prev_episode:
            self.prev_episode = n_episodes

            allRewards = self.env.get_attr('episode_rewards', [i for i in range(settings.CARS_PER_SIM)])
            values = [array[-1] for array in allRewards]

            median = np.median(values)
            summary = tf.Summary(value=[tf.Summary.Value(tag='episodeRewardMedian', simple_value=median)])
            _locals['writer'].add_summary(summary, n_episodes)

            max = np.max(values)
            summary = tf.Summary(value=[tf.Summary.Value(tag='episodeRewardMax', simple_value=max)])
            _locals['writer'].add_summary(summary, n_episodes)

            mean = np.mean(values)
            summary = tf.Summary(value=[tf.Summary.Value(tag='episodeRewardMean', simple_value=mean)])
            _locals['writer'].add_summary(summary, n_episodes)

    def _getModel(self, strictLoad=False):
        tensorboard_log = "./ExperimentTensorboardLog" if settings.MODEL_USE_TENSORBOARD_LOG else None

        # Load from previous model:
        if settings.MODEL_NUMBER is not None and os.path.isfile(f"ExperimentLogsFinal/{self.modelName}_{self.modelNum}.pkl"):
            print("LOAD MODEL")
            model = self.rlModule.load(f"ExperimentLogsFinal/{self.modelName}_{self.modelNum}.pkl", env=self.env, tensorboard_log=tensorboard_log, n_steps=settings.MODEL_N_STEPS, nminibatches=settings.MODEL_MINI_BATCHES, ent_coef=settings.MODEL_ENT_COEF, learning_rate=lambda frac: settings.MODEL_LEARNING_RATE, cliprange=lambda frac: settings.MODEL_CLIP_RANGE, cliprange_vf=lambda frac: settings.MODEL_CLIP_RANGE)
            print("done loading")
            self.modelNum += 1  # Avoid overwriting the loaded model
        # Create new model
        elif strictLoad:
            raise Exception(f"Expected strict load but no model found: {self.modelName}_{self.modelNum}. "
                            f"Try changing model name and number in settings or disable strictLoad")
        else:
            print("NEW MODEL")
            model = self.rlModule(policy=self.policy, env=self.env, tensorboard_log=tensorboard_log, n_steps=settings.MODEL_N_STEPS, nminibatches=settings.MODEL_MINI_BATCHES, ent_coef=settings.MODEL_ENT_COEF, learning_rate=lambda frac: settings.MODEL_LEARNING_RATE, cliprange=lambda frac: settings.MODEL_CLIP_RANGE, cliprange_vf=lambda frac: settings.MODEL_CLIP_RANGE)

        return model

    def _testVehicles(self, _locals, runner_locals):
        print("Evaluating vehicles...")
        # Evaluate agent in environment

        # TODO: If we want to evaluate on 'alt' maps, load them here!!!

        # obs = self.env.reset()
        #
        # state = None
        # # When using VecEnv, done is a vector
        # done = [False for _ in range(self.env.num_envs)]
        # rewards_accum = np.zeros(settings.CARS_PER_SIM)
        # for _ in range(settings.CARLA_TICKS_PER_EPISODE_STATIC):
        #     # We need to pass the previous state and a mask for recurrent policies
        #     # to reset lstm state when a new episode begin
        #     action, state = self.model.predict(obs, state=state, mask=done, deterministic=False)
        #     obs, rewards, done, _ = self.env.step(action)
        #     rewards_accum += rewards

        rewards_accum = np.zeros(settings.CARS_PER_SIM)

        state = None
        done = False
        obs = self.env.reset()

        # Play some limited number of steps
        for i in range(settings.CARLA_TICKS_PER_EPISODE_STATIC):
            action, state = self.model.predict(obs, state=state, mask=done, deterministic=True)
            obs, rewards, done, info = self.env.step(action)
            rewards_accum += rewards

        self.env.reset()
        print("Done evaluating")
        mean = rewards_accum.mean()
        summary = tf.Summary(value=[tf.Summary.Value(tag='EvaluationMean', simple_value=mean)])
        _locals['writer'].add_summary(summary, self.nEpisodes)



