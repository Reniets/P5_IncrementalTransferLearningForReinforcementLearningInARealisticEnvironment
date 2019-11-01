from source.runner import Runner
import gym_carla.settings as settings
import sys
from source.componentTransfer import ComponentTransfer

if __name__ == '__main__':

    mapNumber = int(sys.argv[1])
    settings.TRANSFER_AGENT = int(sys.argv[2])

    if settings.TRANSFER_AGENT == 0:
        logFolder = 'TrainingLogs/BaselineAgentLogs/'
    elif settings.TRANSFER_AGENT == 1:
        logFolder = 'TrainingLogs/TransferAgentLogs/'

    if settings.TRANSFER_AGENT == 1:
        settings.MODEL_NAME = f"Transfer_FromLevel_{mapNumber}_ToLevel_{mapNumber + 1}"
        settings.MODEL_NUMBER = 0
        loadFrom = f'TrainingLogs/FullyTrainedAgentLogs/Base_Level_{mapNumber}_0.zip'

        ct = ComponentTransfer()
        # fromName = f'TransferAgentLogs/Transfer_FromLevel_{mapNumber - 1}_ToLevel_{mapNumber}_0.zip' if mapNumber != 2 else f'Base_Level_1_Final.zip'
        # ct.transfer(fromName, 'CleanAgent.zip', fromLevel=mapNumber-1, toLevel=mapNumber, parameterIndicesToTransfer=[i for i in range(14)])  # Full transfer
        # ct.transfer(f'Base_Level_{mapNumber}_Final.zip', 'TrainingLogs/CleanAgent.zip', toLevel=mapNumber + 1, fromLevel=mapNumber, parameterIndicesToTransfer=[i for i in range(8)])  # Selective transfer
        ct.transfer(loadFrom, 'TrainingLogs/CleanAgent.zip', fromLevel=mapNumber, toLevel=mapNumber+1, parameterIndicesToTransfer=[i for i in range(14)])  # Full transfer
        mapNumber += 1
        settings.MODEL_LEARNING_RATE = 0.00025
    else:
        settings.MODEL_NAME = f"Base_Level_{mapNumber}"
        settings.MODEL_NUMBER = None
        settings.MODEL_LEARNING_RATE = 0.00075
        loadFrom = None

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
    map = maps[mapNumber-1]

    settings.CARLA_SIMS[0][2] = map

    episodeMultiplier = mapNumber if settings.TRANSFER_AGENT == 0 else 1
    print(f"loadfrom: {loadFrom}")
    total_timesteps = settings.CARLA_TICKS_PER_EPISODE_STATIC * settings.EPISODES_PER_SESSION * settings.CARS_PER_SIM * episodeMultiplier
    runner.train(
        loadFrom=loadFrom,
        total_timesteps=total_timesteps)  # Train a model
    #
    # # runner.evaluate()     # Evaluate model
    #
