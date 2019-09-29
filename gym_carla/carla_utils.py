import sys
import glob
import subprocess
import time
from enum import Enum
import psutil as psutil

from gym_carla import settings


class Action(Enum):
    DO_NOTHING = 0
    TURN_LEFT = 1
    TURN_RIGHT = 2
    FORWARD = 3
    BRAKE = 4
    # TURN_LEFT_AND_FORWARD = 5
    # TURN_RIGHT_AND_FORWARD = 6
    # TURN_LEFT_AND_BRAKE = 7
    # TURN_RIGHT_AND_BRAKE = 8

DISCRETE_ACTIONS = {
    Action.DO_NOTHING: [0, 0, 0],
    Action.TURN_LEFT: [0.0, 0.0, -0.5],
    Action.TURN_RIGHT: [0.0, 0.0, 0.5],
    Action.FORWARD: [0.7, 0.0, 0.0],
    Action.BRAKE: [0.0, 1.0, 0.0]
    # Action.TURN_LEFT_AND_FORWARD: [1.0, 0.0, -0.5],
    # Action.TURN_RIGHT_AND_FORWARD: [1.0, 0.0, 0.5],
    # Action.TURN_LEFT_AND_BRAKE: [0, 1.0, -0.5],
    # Action.TURN_RIGHT_AND_BRAKE: [0, 1.0, 0.5]
}


# Functions
def makeCarlaImportable():
    try:
        sys.path.append(glob.glob(
            settings.CARLA_PATH + '/PythonAPI/carla/dist/carla-*%d.*-%s.egg' % (sys.version_info.major, 'linux-x86_64'))[0])
    except IndexError:
        pass

makeCarlaImportable()
import carla

resWidth = 400
resHeight = 400


# Starts Carla simulations and changes their maps
def startCarlaSims():
    print("Starting Carla...")
    killCarlaSims()

    for host in range(settings.CARLA_SIMS_NO):
        winX = (host*resWidth)+100
        winY = 1000-resHeight
        subprocess.Popen(['./CarlaUE4.sh' + f" -windowed -ResX={resWidth} -ResY={resHeight} -WinX={winX} -WinY={winY} -fps=1 -carla-rpc-port={str(settings.CARLA_SIMS[host][1])}"], cwd=settings.CARLA_PATH, shell=True)
        time.sleep(4)  # If DISPLAY is off, sleep longer

    for host in range(settings.CARLA_SIMS_NO):
        client = carla.Client(*settings.CARLA_SIMS[host][:2])
        client.set_timeout(5.0)
        curMap = client.get_world().get_map().name
        mapChoice = settings.CARLA_SIMS[host][2]

        # Ensure that map even needs to be loaded
        if curMap != mapChoice:
            carla.Client(*settings.CARLA_SIMS[host][:2]).load_world(mapChoice)  # Load chosen map

            # Wait for the map to be loaded in
            while carla.Client(*settings.CARLA_SIMS[host][:2]).get_world().get_map().name != mapChoice:
                print('Loop')
                time.sleep(0.05)

        if settings.AGENT_SYNCED:
            clientSettings = client.get_world().get_settings()
            clientSettings.fixed_delta_seconds = settings.TIME_STEP_SIZE
            clientSettings.synchronous_mode = True
            client.get_world().apply_settings(clientSettings)


def killCarlaSims():
    # Iterate processes and terminate carla ones
    for process in psutil.process_iter():
        if process.name().lower().startswith('carlaue4'):
            process.terminate()
    for process in psutil.process_iter():
        if process.name().lower().startswith('carlaue4'):
            process.terminate()


def restartCarlaSims():
    pass





























