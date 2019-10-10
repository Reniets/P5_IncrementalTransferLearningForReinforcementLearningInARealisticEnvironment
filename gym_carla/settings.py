# Agent settings
AGENT_SYNCED = True  # Synchronizes agent with frame updates from Carla
AGENT_TIME_STEP_SIZE = 0.1  # Size of fixed time step

# CarlaEnv settings
CARLA_PATH = '/home/d506e19/carla/Dist/CARLA_Shipping_0.9.6-31-g4c377ce7-dirty/LinuxNoEditor'  # Path to Carla root folder
CARLA_SIMS_NO = 1  # Number of simulations
CARS_PER_SIM = 64

CARLA_SIMS = [['localhost', 3000, 'Kryds'], ['localhost', 3003, 'Curve_Wide'], ['localhost', 3006, 'Curve_Wide'], ['localhost', 3009, 'Curve_Wide'], ['localhost', 3012, 'Curve_Wide'], ['localhost', 3015, 'Curve_Wide']]  # Possible simulations that can be created in format [Host, port, mapname]
CARLA_SECONDS_PER_EPISODE = 35
CARLA_TICKS_PER_EPISODE = int(CARLA_SECONDS_PER_EPISODE * (1 / AGENT_TIME_STEP_SIZE))  # Steps
CARLA_IMG_WIDTH = 50  # Pixels
CARLA_IMG_HEIGHT = 50  # Pixels
CARLA_IMG_MAX_SPEED = 80  # km/h

# Video settings
VIDEO_EXPORT_RATE = 1  # Episodes
VIDEO_MAX_WIDTH = 200  # Pixels
VIDEO_MAX_HEIGHT = 200  # Pixels
VIDEO_ALWAYS_ON = False  # Bool

# Model settings
MODEL_RL_MODULE = "PPO2"
MODEL_MINI_BATCHES = 8  # Cars per sim must be multiple of this
MODEL_POLICY = "CnnLstmPolicy"
MODEL_NUMBER = 81  # int|None. If int carla will try to load model, if none it will never even try!
MODEL_USE_TENSORBOARD_LOG = True  # Bool
MODEL_EXPORT_RATE = 100 if MODEL_RL_MODULE is "PPO2" else 2000  # Callback steps (Dependent on n_steps of rl_module, ppo2=128, a2c=5)
MODEL_N_STEPS = 64
MODEL_ACTION_TYPE = 1  # 0: Discrete, 1: MultiDiscrete, 2: Box
MODEL_ENT_COEF = 0.01  # Default: 0.01
MODEL_LEARNING_RATE = 0.00025  # Default: 0.00025
