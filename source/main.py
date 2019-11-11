import glob

from source.runner import Runner
import gym_carla.settings as settings
import sys
from source.componentTransfer import ComponentTransfer

def makeTransfer(mapNumberFrom, mapNumberTo, is_continuous=False):
    print(f"Starting {'continuous' if is_continuous else 'single'} transfer from map {mapNumberFrom} to {mapNumberTo}!")
    settings.MODEL_NAME = f"Transfer_FromLevel_{mapNumberFrom}_ToLevel_{mapNumberTo}{f'_c' if is_continuous else ''}"
    settings.MODEL_NUMBER = 0
    if is_continuous and mapNumberFrom is not 1:
        loadFrom = glob.glob(f'TrainingLogs/FullyTrainedAgentLogs/Transfer_FromLevel_{mapNumberFrom-1}_ToLevel_{mapNumberTo-1}_c_*')[0]
    else:
        loadFrom = glob.glob(f'TrainingLogs/FullyTrainedAgentLogs/Base_Level_{mapNumberFrom}_*')[0]

    ct = ComponentTransfer()
    path = ct.transfer(loadFrom, 'TrainingLogs/CleanAgent.zip', fromLevel=mapNumberFrom, toLevel=mapNumberTo,
                parameterIndicesToTransfer=[i for i in range(14)], is_continuous=is_continuous)  # Full transfer

    settings.MODEL_LEARNING_RATE = 0.00025

    return path


if __name__ == '__main__':
    mapNumber = int(sys.argv[1])
    settings.TRANSFER_AGENT = int(sys.argv[2])
    loadFrom = None
    directorLoadFrom = None

    if settings.TRANSFER_AGENT == 0:  # No transfer
        logFolder = 'TrainingLogs/BaselineAgentLogs/'
        settings.MODEL_NAME = f"Base_Level_{mapNumber}"
        settings.MODEL_NUMBER = None
        settings.MODEL_LEARNING_RATE = 0.00075
        # TODO: Fix so we have a different argument for running selective transfer instead of using this

    elif settings.TRANSFER_AGENT == 1:  # Transfer
        is_continuous_transfer = bool(sys.argv[3]) if len(sys.argv) > 3 else False
        logFolder = 'TrainingLogs/TransferAgentLogs/'
        loadFrom = makeTransfer(mapNumber, mapNumber+1, is_continuous=is_continuous_transfer)

    elif settings.TRANSFER_AGENT == 2:  # Imitation
        #loadFrom = 'TrainingLogs/Transfer_FromLevel_6_ToLevel_6_cnnForImitation.zip'
        is_continuous_transfer = bool(sys.argv[3]) if len(sys.argv) > 3 else False
        settings.MODEL_NAME = f"Imitation_FromLevel_{mapNumber-1}_ToLevel_{mapNumber}{'_c' if is_continuous_transfer else ''}"
        settings.MODEL_NUMBER = None
        if is_continuous_transfer:
            assert mapNumber > 2, "It is expected that non-continuous was ran before this"
            directorLoadFrom = glob.glob(f'TrainingLogs/FullyTrainedAgentLogs/Imitation_FromLevel_{mapNumber-2}_ToLevel_{mapNumber-1}{"_c" if mapNumber is not 3 else ""}*')[0]
        else:
            directorLoadFrom = glob.glob(f'TrainingLogs/FullyTrainedAgentLogs/Base_Level_{mapNumber-1}_*')[0]
        settings.MODEL_LEARNING_RATE = 0.002
        if mapNumber is 2:  # Because there is only one spawn point
            settings.UNCERTAINTY_TIMES *= 10

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
    map = maps[mapNumber-1 if settings.TRANSFER_AGENT is not 1 else mapNumber]

    settings.CARLA_SIMS[0][2] = map

    episodeMultiplier = mapNumber if settings.TRANSFER_AGENT == 0 else 1
    total_timeSteps = settings.CARLA_TICKS_PER_EPISODE_STATIC * settings.EPISODES_PER_SESSION * settings.CARS_PER_SIM * episodeMultiplier
    runner.train(
        loadFrom=loadFrom,
        directorLoadFrom=directorLoadFrom,
        total_timesteps=total_timeSteps)  # Train a model
    #
    # # runner.evaluate()     # Evaluate model
    #
