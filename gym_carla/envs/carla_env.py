import math
import os
import queue
import gym
from gym.spaces import Discrete, Box, Tuple, MultiDiscrete
from database.sql import Sql
import numpy as np
from gym_carla.carla_utils import *
import cv2

# Import classes
from source.reward import Reward
from source.media_handler import MediaHandler


makeCarlaImportable()
import carla
from PIL import Image


class CarlaEnv(gym.Env):
    """Sets up CARLA simulation and declares necessary instance variables"""
    def __init__(self, name="NoNameWasGiven", carlaInstance=0):

        # Connect a client
        self.client = carla.Client(*settings.CARLA_SIMS[carlaInstance][:2])
        self.client.set_timeout(2.0)
        self.modelName = name

        # Set necessary instance variables related to client
        self.world = self.client.get_world()
        self.blueprintLibrary = self.world.get_blueprint_library()
        self.carlaInstance = carlaInstance

        # Sensors and helper lists
        self.actorList = []
        self.imgWidth = settings.CARLA_IMG_WIDTH
        self.imgHeight = settings.CARLA_IMG_HEIGHT
        self.episodeTicks = 0
        self.totalTicks = 0

        # Video variables
        self.episodeNr = 0
        self.sql = Sql()
        self.sessionId = self.sql.INSERT_newSession(self.modelName) if (self.carlaInstance == 0) else None

        # Early stopping variables
        self.grassLocation = None
        self.grassStuckTick = 0

        # Declare variables for later use
        self.vehicle = None
        self.segSensor = None
        self.grassSensor = None
        self.imgFrame = None
        self.wheelsOnGrass = None
        self.episodeStartTime = 0
        self.episodeReward = None
        self.frameNumber = 0
        self.queues = []  # List of tuples (queue, dataProcessingFunction)

        # Declare reward dependent values
        self.car_last_tick_pos = None
        self.car_last_tick_transform = None
        self.car_last_tick_wheels_on_road = None
        self.car_last_episode_time = None

        # Declare classes
        self.reward = Reward(self)
        self.mediaHandler = MediaHandler(self)

        # Defines image space as a box which can look at standard rgb images of size imgWidth by imgHeight
        imageSpace = Box(low=0, high=255, shape=(self.imgHeight, self.imgWidth, 3), dtype=np.uint8)

        # Defines observation and action spaces
        self.observation_space = imageSpace

        if settings.MODEL_ACTION_TYPE == ActionType.DISCRETE.value:
            self.action_space = Discrete(len(DISCRETE_ACTIONS))
        elif settings.MODEL_ACTION_TYPE == ActionType.MULTI_DISCRETE.value:
            # 1) Throttle: Discrete 4 - [0]:0.0, [1]:0.3, [2]:0.6, [3]:1.0
            # 2) Brake: Discrete 3 - [0]:0.0, [1]:0.5, [2]:1
            # 3) Steer: Discrete 5 - [0]:-1.0, [1]:-0.5, [2]:0.0, [3]:0.5, [4]:1.0
            self.action_space = MultiDiscrete([4, 3, 5])
            self.throttleMapLen = float(self.action_space.nvec[0]-1)
            self.brakeMapLen = float(self.action_space.nvec[1]-1)
            self.steerMapLen = float(self.action_space.nvec[2]-1)/2
        elif settings.MODEL_ACTION_TYPE == ActionType.BOX.value:
            # [Throttle, Steer, brake]
            self.action_space = Box(np.array([0, 0, -1]), np.array([+1, +1, +1]), dtype=np.float32)
        else:
            raise Exception("No such action type, change settings")

        if settings.AGENT_SYNCED: self.tick(10)

    ''':returns initial observation'''
    def reset(self):
        self.episodeNr += 1  # Count episodes

        # Print episode and reward for that episode
        if self.carlaInstance == 0 and self.car_last_episode_time is not None:
            print(f"Episode: {self.episodeNr} - Reward: {self.episodeReward} \t - Time: {time.time() - self.car_last_episode_time}")

        # Frames are only added, if it's a video episode, so if there are frames it means that last episode
        # was a video episode, so we should export it, before we reset the frames list below
        if self.mediaHandler.episodeFrames:
            self.mediaHandler.exportAndUploadVideoToDB()

        # Reset actors, variables and rewards for next episode
        self._resetActorList()
        self._resetInstanceVariables()
        self.episodeReward = 0

        # Create new actors and add to actor list
        self._createActors()

        # Workaround to start episode as quickly as possible
        self._applyActionDiscrete(Action.BRAKE.value)

        # Wait for camera to send first image
        self._waitForWorldToBeReady()

        # Set last tick variables to equal starting pos information
        self.car_last_tick_pos = self.vehicle.get_location()
        self.car_last_tick_transform = self.vehicle.get_transform()
        self.car_last_tick_wheels_on_road = 4

        # Disengage brakes from earlier workaround
        self._applyActionDiscrete(Action.DO_NOTHING.value)

        return self.imgFrame  # Returns initial observation (First image)

    ''':returns (obs, reward, done, extra)'''
    def step(self, action):
        self.episodeTicks += 1
        #self.totalTicks += 1

        # Do action
        if settings.MODEL_ACTION_TYPE == ActionType.DISCRETE.value:
            self._applyActionDiscrete(action)
        elif settings.MODEL_ACTION_TYPE == ActionType.MULTI_DISCRETE.value:
            self._applyActionMultiDiscrete(action)
        elif settings.MODEL_ACTION_TYPE == ActionType.BOX.value:
            self._applyActionBox(action)
        else:
            raise Exception("No such action type, change settings")

        if settings.AGENT_SYNCED: self.tick(10)

        is_done = self._isDone()  # Must be calculated before rewards

        # Update reward
        reward = self.reward.calcReward()
        self.episodeReward += reward

        # if is_done and self.carlaInstance == 0 and self.mediaHandler.episodeFrames:
        #     extra = {"episode": {"episodeNr": self.episodeNr, "frames": self.mediaHandler.episodeFrames}}
        # else:
        #     extra = {}

        return self.imgFrame, reward, is_done, {}  # extra

    def tick(self, timeout):
        self.frameNumber = self.world.tick()
        data = [self._retrieve_data(queueTuple, timeout) for queueTuple in self.queues]
        assert all(x.frame == self.frameNumber for x in data)
        return data

    def _makeQueue(self, registerEvent, processData):
        q = queue.Queue()
        registerEvent(q.put)
        self.queues.append((q, processData))

    def _retrieve_data(self, queueTuple, timeout):
        while True:
            data = queueTuple[0].get(timeout=timeout)
            dataProcessFunction = queueTuple[1]

            if data.frame == self.frameNumber:
                dataProcessFunction(data)  # Process data
                return data

    def _resetInstanceVariables(self):
        # Declare variables for later use
        self.vehicle = None
        self.segSensor = None
        self.grassSensor = None
        self.imgFrame = None
        self.wheelsOnGrass = None
        self.episodeTicks = 0
        self.episodeReward = None
        self.queues = []

        # Early stopping
        self.grassLocation = None
        self.grassStuckTick = 0

        # Declare reward dependent values
        self.car_last_tick_pos = None
        self.car_last_tick_transform = None
        self.car_last_tick_wheels_on_road = None
        self.car_last_episode_time = time.time()

        # Video
        self.mediaHandler.episodeFrames = []

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
        self.world.tick()
        while self._isWorldNotReady():
            if settings.AGENT_SYNCED: self.tick(10)

    # Returns true if the world is not yet ready for training
    def _isWorldNotReady(self):
        # print(self.wheelsOnGrass)
        return self.imgFrame is None or self.wheelsOnGrass != 0

    # Creates a new vehicle and spawns it into the world as an actor
    # Returns the vehicle
    def _createNewVehicle(self):
        vehicle_blueprint = self.blueprintLibrary.filter('test')[0]
        vehicle_spawn_transform = self.world.get_map().get_spawn_points()[0]  # Pick first (and probably only) spawn point
        return self.world.spawn_actor(vehicle_blueprint, vehicle_spawn_transform)  # Spawn vehicle

    # Creates a new segmentation sensor and spawns it into the world as an actor
    # Returns the sensor
    def _createSegmentationSensor(self):
        # Make segmentation sensor blueprint
        seg_sensor_blueprint = self.blueprintLibrary.find('sensor.camera.modified_semantic_segmentation')
        seg_sensor_blueprint.set_attribute('image_size_x', str(self.imgWidth))
        seg_sensor_blueprint.set_attribute('image_size_y', str(self.imgHeight))
        seg_sensor_blueprint.set_attribute('fov', '110')
        relative_transform_sensor = carla.Transform(carla.Location(x=3, z=3), carla.Rotation(pitch=-45))  # Place sensor on the front of car

        # Spawn semantic segmentation sensor, start listening for data and add to actorList
        seg_sensor = self.world.spawn_actor(seg_sensor_blueprint, relative_transform_sensor, attach_to=self.vehicle)
        self._makeQueue(seg_sensor.listen, processData=self.mediaHandler.processImage)
        # seg_sensor.listen(self.mediaHandler.processImage)
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
        self._makeQueue(grass_sensor.listen, processData=self._grass_data)
        # grass_sensor.listen(self._grass_data)
        # Return created actor
        return grass_sensor

    # Applies a discrete action to the vehicle
    def _applyActionDiscrete(self, action):
        # If action does something, apply action
        self.vehicle.apply_control(carla.VehicleControl(
            throttle=DISCRETE_ACTIONS[Action(action)][0],
            brake=DISCRETE_ACTIONS[Action(action)][1],
            steer=DISCRETE_ACTIONS[Action(action)][2]
        ))

    def _applyActionMultiDiscrete(self, action):
        # If action does something, apply action
        self.vehicle.apply_control(carla.VehicleControl(
            throttle=action[0]/self.throttleMapLen,
            brake=action[1]/self.brakeMapLen,
            steer=(action[2]/self.steerMapLen)-1
        ))

    # Applies a box action to the vehicle
    def _applyActionBox(self, action):
        self.vehicle.apply_control(carla.VehicleControl(
            throttle=float(action[0]),
            brake=float(action[1]),
            steer=float(action[2]),
        ))

    # Returns the amount of meters traveled since last tick
    # and updated last pos to current pos
    def metersTraveledSinceLastTick(self):
        # Calculate meters driven
        last = self.car_last_tick_pos
        current = self.vehicle.get_location()

        x_diff = current.x - last.x
        y_diff = current.y - last.y
        z_diff = current.z - last.z

        distance_traveled = math.sqrt(x_diff**2 + y_diff**2 + z_diff**2)

        # Return distance traveled in meters
        return distance_traveled

    # Returns the amount of wheels on the road
    def wheelsOnRoad(self):
        return 4 - self.wheelsOnGrass

    # Returns the cars current velocity in km/h
    def getCarVelocity(self):
        if self.vehicle is None or not self.vehicle.is_alive:
            return 0

        vel_vec = self.vehicle.get_velocity()                               # The velocity vector
        mps = math.sqrt(vel_vec.x ** 2 + vel_vec.y ** 2 + vel_vec.z ** 2)   # Meter pr. second
        kph = mps * 3.6  # Speed in km/h (From m/s)                         # Km pr hour

        return kph

    # Returns true if the current episode should be stopped
    def _isDone(self):
        # If episode length is exceeded it is done
        episode_expired = self._isEpisodeExpired()
        is_stuck_on_grass = self._isStuckOnGrass()
        car_on_grass = self._isCarOnGrass()
        max_negative_reward = self._isMaxNegativeRewardAccumulated()

        return episode_expired or is_stuck_on_grass  # or car_on_grass or max_negative_reward

    # Returns true if the current max episode time has elapsed
    def _isEpisodeExpired(self):
        return self.episodeTicks > settings.CARLA_TICKS_PER_EPISODE

    # Returns true if all four wheels are not on the road
    def _isCarOnGrass(self):
        return self.wheelsOnGrass == 4

    # Returns true if the maximum negative reward has been accumulated
    def _isMaxNegativeRewardAccumulated(self):
        return self.episodeReward < -500

    def _isStuckOnGrass(self):
        if self.wheelsOnGrass == 4 and self.metersTraveledSinceLastTick() < 0.5:
            self.grassStuckTick += 1
            return self.grassStuckTick > 5
        else:
            self.grassStuckTick = 0
            return False

    '''Each time step, model predicts and steps an action, after which render is called'''
    def render(self, mode='human'):
        pass

    def _grass_data(self, event):
        self.wheelsOnGrass = event[0] + event[1] + event[2] + event[3]
        # print(f"({event[0]},{event[1]},{event[2]},{event[3]})")

