# from gym_carla.carla_utils import startCarlaSims
# import os, logging
# os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
# logging.getLogger("tensorflow").setLevel(logging.CRITICAL)
# logging.getLogger("tensorflow_hub").setLevel(logging.CRITICAL)

from source.runner import Runner
import gym_carla.settings as settings
import sys
from source.componentTransfer import ComponentTransfer

if __name__ == '__main__':

    mapNumber = int(sys.argv[1])

    ct = ComponentTransfer()
    ct.transfer(f'Base_Level_{mapNumber}_Final.zip', 'CleanAgent.zip', toLevel=mapNumber + 1, parameterIndicesToTransfer=[i for i in range(14)])  # Full transfer


    maps = [
        'Straight_Slim',
        'Curve_Slim',
        'Square',
        'Niveau3',
        'Kryds',
        'Final1'
    ]

    runner = Runner()
    map = maps[mapNumber]

    settings.MODEL_NAME = f"Transfer_FromLevel_{mapNumber}_ToLevel_{mapNumber + 1}"
    settings.CARLA_SIMS[0][2] = map
    total_timesteps = settings.CARLA_TICKS_PER_EPISODE_STATIC * settings.EPISODES_PER_SESSION * settings.CARS_PER_SIM
    runner.train(total_timesteps=total_timesteps)  # Train a model
    #
    # # runner.evaluate()     # Evaluate model
    #
