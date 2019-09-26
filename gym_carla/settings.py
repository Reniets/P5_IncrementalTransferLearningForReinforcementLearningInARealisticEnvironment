# CarlaEnv settings
CARLA_PATH = '/home/d506e19/carla/Dist/CARLA_Shipping_0.9.6-22-g2b120e37-dirty/LinuxNoEditor'  # Path to Carla root folder
CARLA_SIMS_NO = 1  # Number of simulations
CARLA_SIMS = [['localhost', 3000, 'Curve2'], ['localhost', 3003, 'Curve2'], ['localhost', 3006, 'Curve2'], ['localhost', 3009, 'Curve2']]  # Possible simulations that can be created in format [Host, port, mapname]
TICKS_PER_EPISODE = 400  # Number of seconds of each episode
IMG_WIDTH = 100
IMG_HEIGHT = 100
VIDEO_EXPORT_RATE = 5

# Agent settings
AGENT_SYNCED = True  # Synchronizes agent with frame updates from Carla
TIME_STEP_SIZE = 0.1  # Size of fixed time step
