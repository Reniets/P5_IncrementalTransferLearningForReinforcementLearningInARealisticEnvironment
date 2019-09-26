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
        self.vehicleBlueprint = self.blueprintLibrary.filter('model3')[0]
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
        global stepsCountEpisode
        # print(stepsCountEpisode)
        stepsCountEpisode = 0

        print('Reward: ' + str(self.episodeReward))
        self.episodeReward = 0

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

        # Create grass sensor
        grassBlueprint = self.blueprintLibrary.find('sensor.other.safe_distance')
        grassBlueprint.set_attribute('safe_distance_z_height', '60')
        grassBlueprint.set_attribute('safe_distance_z_origin', '10')
        self.grassSensor = self.world.spawn_actor(grassBlueprint, carla.Transform(), attach_to=self.vehicle)
        self.grassSensor.listen(self._grass_data)
        self.actorList.append(self.grassSensor)

        # Wait for camera to send first image
        while self.imgFrame is None or self.wheelsOnGrass != 0:
            if settings.AGENT_SYNCED: self.world.tick()

        # Disengage brakes from earlier workaround
        self.vehicle.apply_control(carla.VehicleControl(brake=0.0))

        # Start episode timer
        self.episodeStartTime = time.time()

        return self.imgFrame  # Returns initial observation (First image)

    ''':returns (obs, reward, done, extra)'''
    def step(self, action):
        global stepsCountEpisode
        stepsCountEpisode += 1

        # Initialize return information
        obsFrame = self.imgFrame
        wheelsOnGrass = self.wheelsOnGrass
        reward = 0
        done = False

        # Discrete
        if action != Action.DO_NOTHING.value:  # If action does something, apply action
            self.vehicle.apply_control(carla.VehicleControl(
                throttle=DISCRETE_ACTIONS[Action(action)][0],
                brake=DISCRETE_ACTIONS[Action(action)][1],
                steer=DISCRETE_ACTIONS[Action(action)][2])
            )

        # Box
        # self.vehicle.apply_control(carla.VehicleControl(
        #     steer=float(action[0]),
        #     brake=float(action[1]),
        #     throttle=float(action[2])
        # ))

        vel = self.vehicle.get_velocity()
        speed = 3.6 * math.sqrt(vel.x**2 + vel.y**2 + vel.z**2)  # Speed in km/h (From m/s)
        expectedSpeed = 20

        reward += (speed/expectedSpeed) * (4-wheelsOnGrass)
        reward -= wheelsOnGrass

        # If car collided end episode and give a penalty
        # if wheelsOnGrass == 4 or self.episodeReward < -500:
        #     done = True
        #     reward -= 100
        #     print("Wheels")

        # If episode length is exceeded it is done
        if (self.episodeStartTime + self.episodeLen) < time.time():
            print("Time")
            done = True

        self.episodeReward += reward

        if settings.AGENT_SYNCED: self.world.tick()
        return obsFrame, reward, done, {}

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










































