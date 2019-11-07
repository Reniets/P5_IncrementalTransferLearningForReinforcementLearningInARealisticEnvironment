# Agent settings
AGENT_SYNCED = True  # Synchronizes agent with frame updates from Carla
AGENT_TIME_STEP_SIZE = 0.1  # Size of fixed time step

# CarlaEnv settings
CARLA_PATH = '/home/d506e19/carla/Dist/CARLA_Shipping_0.9.6-32-g38e50751-dirty/LinuxNoEditor'  # Path to Carla root folder
CARLA_SIMS_NO = 1  # Number of simulations
CARS_PER_SIM = 21

CARLA_SIMS = [['localhost', 3000, 'Niveau3'], ['localhost', 3003, 'Square'], ['localhost', 3006, 'Curve_Wide'], ['localhost', 3009, 'Curve_Wide'], ['localhost', 3012, 'Curve_Wide'], ['localhost', 3015, 'Curve_Wide']]  # Possible simulations that can be created in format [Host, port, mapname]
CARLA_SECONDS_MODE_LINEAR = False
CARLA_SECONDS_PER_EPISODE_LINEAR_MIN = 15
CARLA_SECONDS_PER_EPISODE_LINEAR_MAX = 45
CARLA_SECONDS_PER_EPISODE_EPISODE_RANGE = 300
CARLA_SECONDS_PER_EPISODE_STATIC = 35
CARLA_TICKS_PER_EPISODE_STATIC = int(CARLA_SECONDS_PER_EPISODE_STATIC * (1 / AGENT_TIME_STEP_SIZE))  # Steps
CARLA_IMG_WIDTH = 50  # Pixels
CARLA_IMG_HEIGHT = 50  # Pixels
CARLA_IMG_MAX_SPEED = 80  # km/h
CARLA_EVALUATION_RATE = 5  # Number of training episodes before a validation episode
CARLA_EVALUATION_MAPS = {} # {'Kryds':'Kryds_alt', 'Final1':'Final1_alt', 'Niveau3':'Niveau3_alt', 'Curve_Slim':'Curve_Slim_alt', 'Curve_Wide':'Curve_Wide_alt'}

USE_RANDOM_SPAWN_POINTS = True
EPISODES_PER_SESSION = 300

# Video settings
VIDEO_EXPORT_RATE = 1  # Episodes
VIDEO_MAX_WIDTH = 200  # Pixels
VIDEO_MAX_HEIGHT = 200  # Pixels
VIDEO_ALWAYS_ON = False  # Bool

# Model settings
MODEL_RL_MODULE = "PPO2"
MODEL_MINI_BATCHES = 7
MODEL_NOPTEPOCHS = 10
MODEL_POLICY = "CnnLstmPolicy"
MODEL_NUMBER = 0  # int|None. If int carla will try to load model, if none it will never even try!
MODEL_USE_TENSORBOARD_LOG = True  # Bool
MODEL_EXPORT_RATE = 20 if MODEL_RL_MODULE is "PPO2" else 200000000  # Callback steps (Dependent on n_episodes)
MODEL_N_STEPS = 128
MODEL_ACTION_TYPE = 1  # 0: Discrete, 1: MultiDiscrete, 2: Box
MODEL_ENT_COEF = 0.01  # Default: 0.01
MODEL_VF_COEF = 0.5
MODEL_LEARNING_RATE = 0.002  # Default: 0.00025, was 0.00075
MODEL_LEARNING_RATE_MIN = 0.0001
MODEL_DISCOUNT_FACTOR = 0.99  # Default: 0.99
MODEL_MAX_EPISODES = 600
MODEL_CLIP_RANGE = 1.
MODEL_CLIP_RANGE_MIN = 0.2
MODEL_CLIP_RANGE_VF = 0.2
MODEL_NAME = "DefaultName"

TRANSFER_AGENT = 2  # 0: NONE, 1: WEIGHTS, 2: IMITATION
TRANSFER_POLLING_RATE_START = 1  # [1; 0]
TRANSFER_POLLING_RATE_MIN = 0.0  # [1; 0]
UNCERTAINTY_RATE = 0.0001
TRANSFER_IMITATION_THRESHOLD = 2000  # N pixels alike to be considered equals
LOG_SENSOR = False


