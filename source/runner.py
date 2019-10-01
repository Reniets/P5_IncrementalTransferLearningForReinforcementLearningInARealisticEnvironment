import sys

import gym
import os
from gym_carla import settings
from stable_baselines.common.policies import MlpPolicy, CnnLstmPolicy
from stable_baselines.common.vec_env import SubprocVecEnv
from stable_baselines.ppo2 import PPO2
from stable_baselines.a2c import A2C
from gym_carla.carla_utils import startCarlaSims, killCarlaSims, Action


class Runner:
    def __init__(self):
        self.env = None
        self.rlModule = None
        self.policy = None
        self.newModel = None
        self.modelName = None
        self.modelNum = None
        self.nSteps = 0
        self.model = None

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
        model = self._getModel(strictLoad=True)  # Load the trained agent

        # Evaluate agent in environment
        obs = self.env.reset()
        for i in range(100000):
            action, _states = model.predict(obs)
            obs, rewards, done, info = self.env.step(action)

    def _setup(self):
        # Setup environment
        startCarlaSims()
        self.env = SubprocVecEnv([lambda i=i: gym.make('CarlaGym-v0', model=self.model, carlaInstance=i) for i in range(settings.CARLA_SIMS_NO)])

        # Decide which RL module and policy
        self.rlModule = getattr(sys.modules[__name__], settings.MODEL_RL_MODULE)
        self.policy = getattr(sys.modules[__name__], settings.MODEL_POLICY)
        self.modelName = settings.MODEL_NAME
        self.modelNum = settings.MODEL_NUMBER if settings.MODEL_NUMBER is not None else 0

    def _callback(self, _locals, _globals):
        self.nSteps += 1

        # Print stats every 100 calls
        if self.nSteps % settings.MODEL_EXPORT_RATE == 0:
            print(f"Saving new model: step {self.nSteps}")
            _locals['self'].save(f"log/{self.modelName}_{self.modelNum}.pkl")
            self.modelNum += 1

        return True

    def _getModel(self, strictLoad=False):
        tensorboard_log = "./tensorboard_log" if settings.MODEL_USE_TENSORBOARD_LOG else None

        # Load from previous model:
        if settings.MODEL_NUMBER is not None and os.path.isfile(f"log/{self.modelName}_{self.modelNum}.pkl"):
            print("LOAD MODEL")
            model = self.rlModule.load(f"log/{self.modelName}_{self.modelNum}", env=self.env, tensorboard_log=tensorboard_log, n_steps=settings.MODEL_N_STEPS)
            self.modelNum += 1  # Avoid overwriting the loaded model
        # Create new model
        elif strictLoad:
            raise Exception(f"Expected strict load but no model found: {self.modelName}_{self.modelNum}. "
                            f"Try changing model name and number in settings or disable strictLoad")
        else:
            print("NEW MODEL")
            model = self.rlModule(policy=self.policy, env=self.env, tensorboard_log=tensorboard_log, n_steps=settings.MODEL_N_STEPS)

        return model


