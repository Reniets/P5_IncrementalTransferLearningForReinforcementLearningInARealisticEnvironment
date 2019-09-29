# Agent settings
AGENT_SYNCED = True  # Synchronizes agent with frame updates from Carla
TIME_STEP_SIZE = 0.1  # Size of fixed time step

# CarlaEnv settings
CARLA_PATH = '/home/simon/Desktop/Carla'  # Path to Carla root folder
CARLA_SIMS_NO = 1  # Number of simulations
CARLA_SIMS = [['localhost', 3000, 'Town03'], ['localhost', 3003, 'Curve2'], ['localhost', 3006, 'Curve2'], ['localhost', 3009, 'Curve2'], ['localhost', 3012, 'Curve2'], ['localhost', 3015, 'Curve2']]  # Possible simulations that can be created in format [Host, port, mapname]
TICKS_PER_EPISODE = 60*(1/TIME_STEP_SIZE)
IMG_WIDTH = 50
IMG_HEIGHT = 50
VIDEO_EXPORT_RATE = 50
VIDEO_MAX_WIDTH = 200
VIDEO_MAX_HEIGHT = 200
VIDEO_ALWAYS_ON = False

# Model settings
MODEL_RL_MODULE = PPO2
MODEL_POLICY = CnnLstmPolicy
MODEL_NAME = "box_new_rewards"
MODEL_NUMBER = 0
