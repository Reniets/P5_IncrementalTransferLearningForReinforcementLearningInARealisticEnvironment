from gym_carla.carla_utils import startCarlaSims
import gym_carla.settings as s
from source import manual_control

s.AGENT_SYNCED=False
s.CARLA_SIMS[0][2] = 'Square'
startCarlaSims()

manual_control.main()