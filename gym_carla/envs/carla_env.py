import math

import gym
from gym.spaces import Discrete, Box, Tuple
import numpy as np
from gym_carla.carla_utils import *

makeCarlaImportable()
import carla


class CarlaEnv(gym.Env):

    def __init__(self, carlaInstance=0):

        # Connect a client
        self.client = carla.Client(*settings.CARLA_SIMS[carlaInstance][:2])
        self.client.set_timeout(2.0)

        # Set necessary instance variables related to client
        self.world = self.client.get_world()
        self.blueprintLibrary = self.world.get_blueprint_library()
        self.vehicleBlueprint = self.blueprintLibrary.filter('model3')[0]

        # Sensors and helper lists
        self.collisionHist = []
        self.actorList = []
        self.imgWidth = settings.IMG_WIDTH
        self.imgHeight = settings.IMG_HEIGHT
        self.episodeLen = settings.SECONDS_PER_EPISODE

        # Declare variables for later use
        self.vehicle = None
        self.segSensor = None
        self.colSensor = None
        self.imgFrame = None
        self.episodeStartTime = None

        # Defines image space as a box which can look at standard rgb images of size imgWidth by imgHeight
        imageSpace = Box(low=0, high=255, shape=(self.imgHeight, self.imgWidth, 3), dtype=np.uint8)

        # Defines observation and action spaces
        self.observation_space = imageSpace
        self.action_space = Discrete(len(DISCRETE_ACTIONS))

    ''':returns initial observation'''
    def reset(self):
        # Destroy all actors from previous episode
        for actor in self.actorList:
            actor.destroy()

        # Clear all actors from the list from previous episode
        self.actorList = []

        # Spawn vehicle
        vehicleSpawnTransform = self.world.get_map().get_spawn_points()[0]  # Pick first (and probably only) spawn point
        self.vehicle = self.world.spawn_actor(self.vehicleBlueprint, vehicleSpawnTransform)  # Spawn vehicle
        self.actorList.append(self.vehicle)  # Add to list of actors which makes it easy to clean up later

        # Make segmentation sensor blueprint
        segSensorBlueprint = self.blueprintLibrary.find('sensor.camera.semantic_segmentation')
        segSensorBlueprint.set_attribute('image_size_x', str(self.imgWidth))
        segSensorBlueprint.set_attribute('image_size_y', str(self.imgHeight))
        segSensorBlueprint.set_attribute('fov', '110')
        relativeTransformSensor = carla.Transform(carla.Location(x=2.5, z=0.7))  # Place sensor on the front of car

        # Spawn semantic segmentation sensor, start listening for data and add to actorList
        self.segSensor = self.world.spawn_actor(segSensorBlueprint, relativeTransformSensor, attach_to=self.vehicle)
        self.segSensor.listen(self._processImage)
        self.actorList.append(self.segSensor)

        # TODO: Make some system that allows previewing episodes once in a while

        # Workaround to start episode as quickly as possible
        self.vehicle.apply_control(carla.VehicleControl(throttle=1.0, brake=1.0))
        time.sleep(4)

        # Clear collision history
        self.collisionHist = []

        # Create collision sensor
        colSensorBlueprint = self.world.get_blueprint_library().find('sensor.other.collision')  # Get blueprint
        self.colSensor = self.world.spawn_actor(colSensorBlueprint, carla.Transform(), attach_to=self.vehicle)  # Create colSensor actor and attach to vehicle
        self.colSensor.listen(self._collision_data)  # Make it listen for collisions
        self.actorList.append(self.colSensor)  # Add to actorList

        # Wait for camera to send first image
        while self.imgFrame is None:
            time.sleep(0.05)

        # Disengage brakes from earlier workaround
        self.vehicle.apply_control(carla.VehicleControl(brake=0.0))

        # Start episode timer
        self.episodeStartTime = time.time()

        return self.imgFrame  # Returns initial observation (First image and speed of 0)

    ''':returns (obs, reward, done, extra)'''
    def step(self, action):
        # Initialize return information
        obs = self.imgFrame
        reward = 0
        done = False

        if action != Action.DO_NOTHING.value:  # If action does something, apply action
            self.vehicle.apply_control(carla.VehicleControl(throttle=DISCRETE_ACTIONS[Action(action)][0], brake=DISCRETE_ACTIONS[Action(action)][1], steer=DISCRETE_ACTIONS[Action(action)][2]))

        v = self.vehicle.get_velocity()

        # If car collided end episode and give a penalty
        if len(self.collisionHist) != 0:
            done = True
            reward = -1

        # If episode length is exceeded it is done
        if (self.episodeStartTime + self.episodeLen) < time.time():
            done = True

        # TODO: Calculate rewards based on speed and driving surface

        return obs, reward, done, {}


    '''Each time step, model predicts and steps an action, after which render is called'''
    def render(self, mode='human'):
        pass

    def _processImage(self, data):
        # Get image, reshape and remove alpha channel
        image = np.array(data.raw_data)
        image = image.reshape((self.imgHeight, self.imgWidth, 4))
        image = image[:, :, :3]

        self.imgFrame = image

    def _collision_data(self, event):
        # What we collided with and what was the impulse
        #collision_actor_id = event.other_actor.type_id
        #collision_impulse = math.sqrt(event.normal_impulse.x ** 2 + event.normal_impulse.y ** 2 + event.normal_impulse.z ** 2)

        # TODO: Register whether it was grass or wall and give reward based on it

        self.collisionHist.append(event)













































