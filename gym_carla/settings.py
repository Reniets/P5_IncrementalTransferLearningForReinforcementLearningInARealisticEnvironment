# Agent settings
AGENT_SYNCED = True  # Synchronizes agent with frame updates from Carla
AGENT_TIME_STEP_SIZE = 0.1  # Size of fixed time step

# CarlaEnv settings
CARLA_PATH = '/home/d506e19/carla/Dist/CARLA_Shipping_0.9.6-22-g2b120e37-dirty/LinuxNoEditor'  # Path to Carla root folder
CARLA_SIMS_NO = 4  # Number of simulations
CARLA_SIMS = [['localhost', 3000, 'Curve2'], ['localhost', 3003, 'Curve2'], ['localhost', 3006, 'Curve2'], ['localhost', 3009, 'Curve2'], ['localhost', 3012, 'Curve2'], ['localhost', 3015, 'Curve2']]  # Possible simulations that can be created in format [Host, port, mapname]
CARLA_TICKS_PER_EPISODE = 60 * (1 / AGENT_TIME_STEP_SIZE)  # Steps
CARLA_IMG_WIDTH = 50  # Pixels
CARLA_IMG_HEIGHT = 50  # Pixels

# Video settings
VIDEO_EXPORT_RATE = 25  # Episodes
VIDEO_MAX_WIDTH = 200  # Pixels
VIDEO_MAX_HEIGHT = 200  # Pixels
VIDEO_ALWAYS_ON = False  # Bool

# Model settings
MODEL_RL_MODULE = "PPO2"
MODEL_POLICY = "CnnLstmPolicy"
MODEL_NAME = "PPO2_Reproduce_moreTimeSteps"
MODEL_NUMBER = 1  # int|None. If int carla will try to load model, if none it will never even try!
MODEL_USE_TENSORBOARD_LOG = True  # Bool
MODEL_EXPORT_RATE = 100 if MODEL_RL_MODULE is "PPO2" else 2000  # Callback steps (Dependent on n_steps of rl_module, ppo2=128, a2c=5)
MODEL_N_STEPS = 128
MODEL_ACTION_TYPE = 1  # 0: Discrete, 1: MultiDiscrete, 2: Box
