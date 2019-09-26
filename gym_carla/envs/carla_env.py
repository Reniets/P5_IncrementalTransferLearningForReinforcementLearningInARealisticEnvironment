import math
from PIL import Image

import gym
from gym.spaces import Discrete, Box, Tuple
import numpy as np
from gym_carla.carla_utils import *

makeCarlaImportable()
import carla
stepsCountEpisode = 0

class CarlaEnv(gym.Env):

    def __init__(self, carlaInstance=0):
        # Connect a client
        self.client = carla.Client(*settings.CARLA_SIMS[carlaInstance][:2])
        self.client.set_timeout(2.0)

        # Set necessary instance variables related to client
        self.world = self.client.get_world()
        self.blueprintLibrary = self.world.get_blueprint_library()
        # Sensors and helper lists
        # self.collisionHist = []
        self.actorList = []
        self.imgWidth = settings.IMG_WIDTH
        self.imgHeight = settings.IMG_HEIGHT
        self.episodeLen = settings.SECONDS_PER_EPISODE

        # Declare variables for later use
        self.vehicle = None
        self.segSensor = None
        # self.colSensor = None
        self.grassSensor = None
        self.imgFrame = None
        self.wheelsOnGrass = None
        self.episodeStartTime = None
        self.episodeReward = None

        # Defines image space as a box which can look at standard rgb images of size imgWidth by imgHeight
        imageSpace = Box(low=0, high=255, shape=(self.imgHeight, self.imgWidth, 3), dtype=np.uint8)

        # Defines observation and action spaces
        self.observation_space = imageSpace
        self.action_space = Discrete(len(DISCRETE_ACTIONS))
        # self.action_space = Box(np.array([-0.5, 0, 0]), np.array([+0.5, +1, +1]), dtype=np.float32)    # Steer,

        if settings.AGENT_SYNCED: self.world.tick()


    ''':returns initial observation'''
    def reset(self):
        # global stepsCountEpisode
        # print(stepsCountEpisode)
        # stepsCountEpisode = 0

        # print('Reward: ' + str(self.episodeReward))
        self.episodeReward = 0

        # Destroy all previous actors, and clear actor list
        self._resetActorList()

        # Create new actors and add to actor list
        self._createActors()

        # TODO: Make some system that allows previewing episodes once in a while

        # Workaround to start episode as quickly as possible
        self._setActionDiscrete(Action.BRAKE.value)

        # Wait for camera to send first image
        self._waitForWorldToBeReady()

        # Disengage brakes from earlier workaround
        self._setActionDiscrete(Action.DO_NOTHING.value)

        # Start episode timer
        self.episodeStartTime = time.time()

        return self.imgFrame  # Returns initial observation (First image)

    def _createActors(self):
        # Spawn vehicle
        self.vehicle = self._createNewVehicle()
        self.actorList.append(self.vehicle)  # Add to list of actors which makes it easy to clean up later

        # Make segmentation sensor blueprint
        self.segSensor = self._createSegmentationSensor()
        self.actorList.append(self.segSensor)

        # Create grass sensor
        self.grassSensor = self._createGrassSensor()
        self.actorList.append(self.grassSensor)

    # Destroy all previous actors, and clear actor list
    def _resetActorList(self):
        # Destroy all actors from previous episode
        for actor in self.actorList:
            actor.destroy()

        # Clear all actors from the list from previous episode
        self.actorList = []

    # Waits until the world is ready for training
    def _waitForWorldToBeReady(self):
        while self._isWorldNotReady():
            if settings.AGENT_SYNCED: self.world.tick()

    # Returns true if the world is not yet ready for training
    def _isWorldNotReady(self):
        return self.imgFrame is None or self.wheelsOnGrass != 0

    # Creates a new vehicle and spawns it into the world as an actor
    # Returns the vehicle
    def _createNewVehicle(self):
        vehicle_blueprint = self.blueprintLibrary.filter('model3')[0]
        vehicle_spawn_transform = self.world.get_map().get_spawn_points()[0]  # Pick first (and probably only) spawn point
        return self.world.spawn_actor(vehicle_blueprint, vehicle_spawn_transform)  # Spawn vehicle

    # Creates a new segmentation sensor and spawns it into the world as an actor
    # Returns the sensor
    def _createSegmentationSensor(self):
        # Make segmentation sensor blueprint
        seg_sensor_blueprint = self.blueprintLibrary.find('sensor.camera.semantic_segmentation')
        seg_sensor_blueprint.set_attribute('image_size_x', str(self.imgWidth))
        seg_sensor_blueprint.set_attribute('image_size_y', str(self.imgHeight))
        seg_sensor_blueprint.set_attribute('fov', '110')
        relative_transform_sensor = carla.Transform(carla.Location(x=2.5, z=0.7))  # Place sensor on the front of car

        # Spawn semantic segmentation sensor, start listening for data and add to actorList
        seg_sensor = self.world.spawn_actor(seg_sensor_blueprint, relative_transform_sensor, attach_to=self.vehicle)
        seg_sensor.listen(self._processImage)

        return seg_sensor

    # Creates a new grass sensor and spawns it into the world as an actor
    # Returns the sensor
    def _createGrassSensor(self):
        # Sensor blueprint
        grass_blueprint = self.blueprintLibrary.find('sensor.other.safe_distance')
        grass_blueprint.set_attribute('safe_distance_z_height', '60')
        grass_blueprint.set_attribute('safe_distance_z_origin', '10')

        # Grass sensor actor
        grass_sensor = self.world.spawn_actor(grass_blueprint, carla.Transform(), attach_to=self.vehicle)
        grass_sensor.listen(self._grass_data)

        # Return created actor
        return grass_sensor

    ''':returns (obs, reward, done, extra)'''
    def step(self, action):
        # global stepsCountEpisode
        # stepsCountEpisode += 1

        # Do action
        self._setActionDiscrete(action)
        # self._setActionBox(action)

        if settings.AGENT_SYNCED: self.world.tick()

        # Update reward
        reward = self._calcReward()
        self.episodeReward += reward

        return self.imgFrame, reward, self._isDone(), {}

    # Applies a discrete action to the vehicle
    def _setActionDiscrete(self, action):
        if action != Action.DO_NOTHING.value:  # If action does something, apply action
            self.vehicle.apply_control(carla.VehicleControl(
                throttle=DISCRETE_ACTIONS[Action(action)][0],
                brake=DISCRETE_ACTIONS[Action(action)][1],
                steer=DISCRETE_ACTIONS[Action(action)][2])
            )

    # Applies a box action to the vehicle
    def _setActionBox(self, action):
        self.vehicle.apply_control(carla.VehicleControl(
            steer=float(action[0]),
            brake=float(action[1]),
            throttle=float(action[2])
        ))

    # Returns the reward for the current state
    def _calcReward(self):
        reward = 0
        speed = self._getCarVelocity()
        expected_speed = 20

        reward += (speed / expected_speed) * self._wheelsOnRoad()
        reward -= self.wheelsOnGrass

        return reward

    # Returns the amount of wheels on the road
    def _wheelsOnRoad(self):
        return 4 - self.wheelsOnGrass

    # Returns the cars current velocity in km/h
    def _getCarVelocity(self):
        vel_vec = self.vehicle.get_velocity()                               # The velocity vector
        mps = math.sqrt(vel_vec.x ** 2 + vel_vec.y ** 2 + vel_vec.z ** 2)   # Meter pr. second
        kph = mps * 3.6  # Speed in km/h (From m/s)                         # Km pr hour

        return kph

    # Returns true if the current episode should be stopped
    def _isDone(self):
        # If episode length is exceeded it is done
        episode_expired = self._isEpisodeExpired()
        car_on_grass = self._isCarOnGrass()
        max_negative_reward = self._isCarOnGrass()

        return episode_expired # or car_on_gras or max_negative_reward

    # Returns true if the current max episode time has elapsed
    def _isEpisodeExpired(self):
        return (self.episodeStartTime + self.episodeLen) < time.time()

    # Returns true if all four wheels are not on the road
    def _isCarOnGrass(self):
        return self.wheelsOnGrass == 4

    # Returns true if the maximum negative reward has been accumulated
    def _isMaxNegativeRewardAccumulated(self):
        return self.episodeReward < -500

    '''Each time step, model predicts and steps an action, after which render is called'''
    def render(self, mode='human'):
        pass

    def _processImage(self, data):
        cc = carla.ColorConverter.CityScapesPalette
        data.convert(cc)
        # Get image, reshape and remove alpha channel
        image = np.array(data.raw_data)
        image = image.reshape((self.imgHeight, self.imgWidth, 4))
        image = image[:, :, :3]

        # bgra
        self.imgFrame = image

        # Save images to disk (Output folder)
        # img = Image.fromarray(image, 'RGB')
        # img.save('my.png')
        # data.save_to_disk('../data/frames/%06d.png' % self.imgFrame*10, cc)

    def _grass_data(self, event):
        self.wheelsOnGrass = event[0] + event[1] + event[2] + event[3]
        # print(f"({event[0]},{event[1]},{event[2]},{event[3]})")










































