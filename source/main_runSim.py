from gym_carla.carla_utils import startCarlaSims
import gym_carla.settings as s
from source import manual_control
from source.componentTransfer import ComponentTransfer
import glob

def runSingleSimAndEnterManualControl():
    s.AGENT_SYNCED = False
    s.CARLA_SIMS[0][2] = 'Square'
    startCarlaSims()

    manual_control.main()

def makeSingleTransfer(fromMapNumber, toMapNumber):
    loadFrom = glob.glob(f'TrainingLogs/FullyTrainedAgentLogs/Base_Level_{fromMapNumber}_*')[0]
    ct = ComponentTransfer()
    ct.transfer(loadFrom, 'TrainingLogs/CleanAgent.zip', fromLevel=fromMapNumber, toLevel=toMapNumber,
                parameterIndicesToTransfer=[i for i in range(8)])  # Full transfer


if __name__ == '__main__':
    makeSingleTransfer(6, 6)
