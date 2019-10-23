# from gym_carla.carla_utils import startCarlaSims
# import os, logging
# os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
# logging.getLogger("tensorflow").setLevel(logging.CRITICAL)
# logging.getLogger("tensorflow_hub").setLevel(logging.CRITICAL)
from source.runner import Runner
import gym_carla.settings as settings
import sys
from source.componentTransfer import ComponentTransfer
ct = ComponentTransfer()
ct.transfer('Base_Level_1_Modified_14.zip', 'Transfer_Level_2_Modified_0.zip', [i for i in range(14)])

i = int(sys.argv[1])

maps = [
    'Straight_Slim',
    'Curve_Slim',
    'Square',
    'Niveau3',
    'Kryds',
    'Final1'
]

runner = Runner()
map = maps[i]

settings.MODEL_NAME = f"Transfer_FromLevel_1_ToLevel_2"
settings.CARLA_SIMS[0][2] = map
total_timesteps = settings.CARLA_TICKS_PER_EPISODE_STATIC * settings.EPISODES_PER_SESSION * (i+1) * settings.CARS_PER_SIM
runner.train(total_timesteps=total_timesteps)  # Train a model

# runner.evaluate()     # Evaluate model

