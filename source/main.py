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
    #fromName = f'TransferAgentLogs/Transfer_FromLevel_{mapNumber -2}_ToLevel_{mapNumber-1}_Continued.zip' if mapNumber != 2 else f'Base_Level_1_Final.zip'
    #ct.transfer(fromName, 'CleanAgent.zip', fromLevel=mapNumber-1, toLevel=mapNumber, parameterIndicesToTransfer=[i for i in range(14)])  # Full transfer
    #ct.transfer(f'Base_Level_{mapNumber}_Final.zip', 'CleanAgent.zip', toLevel=mapNumber + 1, fromLevel=mapNumber, parameterIndicesToTransfer=[i for i in range(14)])  # Full transfer
    ct.transfer(f'Base_Level_{mapNumber}_Final.zip', 'CleanAgent.zip', toLevel=mapNumber + 1, fromLevel=mapNumber, parameterIndicesToTransfer=[i for i in range(8)])  # Selective transfer


    maps = [
        'Straight_Slim',
        'Curve_Slim',
        'Square',
        'Niveau3',
        #'Niveau3_Obst',
        'Kryds',
        #'Kryds_Obst',
        'Final1'
        #'Final1_Obst'
    ]

    runner = Runner()
    map = maps[mapNumber]

    settings.MODEL_NAME = f"Transfer_FromLevel_{mapNumber}_ToLevel_{mapNumber+1}_Selective"
    settings.CARLA_SIMS[0][2] = map
    total_timesteps = settings.CARLA_TICKS_PER_EPISODE_STATIC * settings.EPISODES_PER_SESSION * settings.CARS_PER_SIM
    runner.train(total_timesteps=total_timesteps)  # Train a model
    #
    # # runner.evaluate()     # Evaluate model
    #
