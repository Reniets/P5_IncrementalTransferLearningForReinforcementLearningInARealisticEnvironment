import glob

from source.runner import Runner
import gym_carla.settings as settings
import sys
from source.componentTransfer import ComponentTransfer

def makeTransfer(mapNumberFrom, mapNumberTo):
    settings.MODEL_NAME = f"Transfer_FromLevel_{mapNumberFrom}_ToLevel_{mapNumberTo}"
    settings.MODEL_NUMBER = 0
    loadFrom = glob.glob(f'TrainingLogs/FullyTrainedAgentLogs/Base_Level_{mapNumberFrom}_*')[0]

    ct = ComponentTransfer()
    ct.transfer(loadFrom, 'TrainingLogs/CleanAgent.zip', fromLevel=mapNumberFrom, toLevel=mapNumberTo,
                parameterIndicesToTransfer=[i for i in range(16)])  # Full transfer

    settings.MODEL_LEARNING_RATE = 0.00025


if __name__ == '__main__':
    mapNumber = int(sys.argv[1])
    settings.TRANSFER_AGENT = int(sys.argv[2])
    loadFrom = None
    directorLoadFrom = None

    if settings.TRANSFER_AGENT == 0:  # No transfer
        logFolder = 'TrainingLogs/BaselineAgentLogs/'
        makeTransfer(mapNumberFrom=mapNumber, mapNumberTo=mapNumber+1)

    elif settings.TRANSFER_AGENT == 1:  # Transfer
        logFolder = 'TrainingLogs/TransferAgentLogs/'
        settings.MODEL_NAME = f"Base_Level_{mapNumber}"
        settings.MODEL_NUMBER = None
        settings.MODEL_LEARNING_RATE = 0.00075

    elif settings.TRANSFER_AGENT == 2:  # Imitation
        #loadFrom = 'TrainingLogs/Transfer_FromLevel_6_ToLevel_6_cnnForImitation.zip'
        settings.MODEL_NAME = f"Imitation_Lstm_Test"
        settings.MODEL_NUMBER = None
        directorLoadFrom = 'TrainingLogs/FullyTrainedAgentLogs/Base_Level_6_89.zip'

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

    episodeMultiplier = mapNumber# if settings.TRANSFER_AGENT == 0 else 1
    total_timeSteps = settings.CARLA_TICKS_PER_EPISODE_STATIC * settings.EPISODES_PER_SESSION * settings.CARS_PER_SIM * episodeMultiplier
    runner.train(
        loadFrom=loadFrom,
        directorLoadFrom=directorLoadFrom,
        total_timesteps=total_timeSteps)  # Train a model
    #
    # # runner.evaluate()     # Evaluate model
    #
