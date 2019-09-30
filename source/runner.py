import gym
from gym_carla import settings
from stable_baselines.common.policies import MlpPolicy, CnnLstmPolicy
from stable_baselines.common.vec_env import SubprocVecEnv
from stable_baselines import *
from gym_carla.carla_utils import startCarlaSims, killCarlaSims, Action


class Runner:
    def __init__(self):
        self.env = None
        self.rlModule = None
        self.policy = None
        self.newModel = None
        self.modelName = None
        self.agentNum = None
        self.nSteps = 0

    def train(self):
        self._setup()
        model = self._getModel(strictLoad=False)  # Load the model

        # Perform learning
        model.learn(total_timesteps=1000000000, callback=self._callback)
        model.save(f"{self.modelName}_{self.nSteps}")

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
        self.env = SubprocVecEnv([lambda i=i: gym.make('CarlaGym-v0', carlaInstance=i) for i in range(settings.CARLA_SIMS_NO)])

        # Decide which RL module and policy
        self.rlModule = settings.MODEL_RL_MODULE
        self.policy = settings.MODEL_POLICY
        self.modelName = settings.MODEL_NAME
        self.modelNum = settings.MODEL_NUMBER

    def _callback(self, _locals, _globals):
        self.nSteps += 1

        # Print stats every 100 calls
        if self.nSteps % 100 == 0:
            print(f"Saving new model: step {self.nSteps}")
            _locals['self'].save(f"log/{self.modelName}_{str(self.agentNum)}.pkl")
            self.agentNum += 1

        return True

    def _getModel(self, strictLoad=False):
        # Load from previous model:
        if os.path.isfile(f"log/{self.modelName}_{self.modelNum}.pkl"):
            print("LOAD MODEL")
            model = self.rlModule.load(f"log/{self.modelName}_{self.modelName}", env=self.env)
        # Create new model
        elif strictLoad:
            raise Exception(f"Expected strict load but no model found: {self.modelName}_{self.modelNum}. "
                            f"Try changing model name and number in settings or disable strictLoad")
        else:
            print("NEW MODEL")
            model = self.rlModule(policy=self.policy, env=self.env, nminibatches=settings.CARLA_SIMS_NO, tensorboard_log="./tensorboard_log")

        return model


