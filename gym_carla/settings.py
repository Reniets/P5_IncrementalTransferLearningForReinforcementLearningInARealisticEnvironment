# CarlaEnv settings
CARLA_PATH = '/home/simon/Desktop/carla'  # Path to Carla root folder
CARLA_SIMS_NO = 1  # Number of simulations
CARLA_SIMS = [['localhost', 2000, 'Town01'], ['localhost', 2002, 'Straight3']]  # Possible simulations that can be created in format [Host, port, mapname]
SECONDS_PER_EPISODE = 10  # Number of seconds of each episode
EPISODE_FPS = 60  # Desired
IMG_WIDTH = 400
IMG_HEIGHT = 200

# Agent settings
AGENT_SYNCED = True  # Synchronizes agent with frame updates from Carla
