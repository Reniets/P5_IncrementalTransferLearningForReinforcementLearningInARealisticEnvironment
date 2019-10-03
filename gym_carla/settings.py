# Agent settings
AGENT_SYNCED = True  # Synchronizes agent with frame updates from Carla
AGENT_TIME_STEP_SIZE = 0.1  # Size of fixed time step

# CarlaEnv settings
CARLA_PATH = '/home/d506e19/carla/Dist/CARLA_Shipping_0.9.6-25-gf909d022-dirty/LinuxNoEditor'  # Path to Carla root folder
CARLA_SIMS_NO = 4  # Number of simulations
CARLA_SIMS = [['localhost', 3000, 'Curve2'], ['localhost', 3003, 'Curve2'], ['localhost', 3006, 'Curve2'], ['localhost', 3009, 'Curve2'], ['localhost', 3012, 'Curve2'], ['localhost', 3015, 'Curve2']]  # Possible simulations that can be created in format [Host, port, mapname]
CARLA_SECONDS_PER_EPISODE = 45
CARLA_TICKS_PER_EPISODE = CARLA_SECONDS_PER_EPISODE * (1 / AGENT_TIME_STEP_SIZE)  # Steps
CARLA_IMG_WIDTH = 50  # Pixels
CARLA_IMG_HEIGHT = 50  # Pixels
CARLA_IMG_MAX_SPEED = 80  # km/h

# Video settings
VIDEO_EXPORT_RATE = 10  # Episodes
VIDEO_MAX_WIDTH = 200  # Pixels
VIDEO_MAX_HEIGHT = 200  # Pixels
VIDEO_ALWAYS_ON = False  # Bool

# Model settings
MODEL_RL_MODULE = "PPO2"
MODEL_POLICY = "CnnLstmPolicy"
MODEL_NUMBER = 32  # int|None. If int carla will try to load model, if none it will never even try!
MODEL_USE_TENSORBOARD_LOG = True  # Bool
MODEL_EXPORT_RATE = 100 if MODEL_RL_MODULE is "PPO2" else 2000  # Callback steps (Dependent on n_steps of rl_module, ppo2=128, a2c=5)
MODEL_N_STEPS = 256
MODEL_ACTION_TYPE = 2  # 0: Discrete, 1: MultiDiscrete, 2: Box