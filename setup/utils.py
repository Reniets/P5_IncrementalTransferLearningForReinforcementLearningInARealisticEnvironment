import sys
import glob
import os

# Constants
CARLA_PATH = '/home/d506e19/carla/Dist/CARLA_Shipping_0.9.6-22-g2b120e37-dirty/LinuxNoEditor'


# Functions
def makeCarlaImportable():
    try:
        sys.path.append(glob.glob(CARLA_PATH + '/PythonAPI/carla/dist/carla-*%d.%d-%s.egg' % (
            sys.version_info.major,
            6,
            # sys.version_info.minor,
            'win-amd64' if os.name == 'nt' else 'linux-x86_64'))[0])
    except IndexError:
        pass
