# Agent settings
AGENT_SYNCED = True  # Synchronizes agent with frame updates from Carla
TIME_STEP_SIZE = 0.1  # Size of fixed time step

# CarlaEnv settings
CARLA_PATH = '/home/d506e19/carla/Dist/CARLA_Shipping_0.9.6-22-g2b120e37-dirty/LinuxNoEditor'  # Path to Carla root folder
CARLA_SIMS_NO = 4  # Number of simulations
CARLA_SIMS = [['localhost', 3000, 'Curve2'], ['localhost', 3003, 'Curve2'], ['localhost', 3006, 'Curve2'], ['localhost', 3009, 'Curve2'], ['localhost', 3012, 'Curve2'], ['localhost', 3015, 'Curve2']]  # Possible simulations that can be created in format [Host, port, mapname]
TICKS_PER_EPISODE = 60*(1/TIME_STEP_SIZE)*4
IMG_WIDTH = 50
IMG_HEIGHT = 50
VIDEO_EXPORT_RATE = 100
VIDEO_MAX_WIDTH = 200
VIDEO_MAX_HEIGHT = 200
VIDEO_ALWAYS_ON = True
