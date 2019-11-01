import math
import os
import queue
import random

import gym
from gym.spaces import Discrete, Box, Tuple, MultiDiscrete
from database.sql import Sql
import numpy as np
from gym_carla.carla_utils import *
import cv2
# Import classes
from source.reward import Reward
from source.media_handler import MediaHandler
from source.gps import Gps
from multiprocessing import Condition, Lock
makeCarlaImportable()
import carla
from PIL import Image


class CarlaSyncEnv(gym.Env):
    """Sets up CARLA simulation and declares necessary instance variables"""

    def __init__(self, thread_count, lock, frameNumber, waiting_threads, carlaInstance=0, sessionId=None, world_ticks=None, name="NoNameWasGiven", runner=None, serverIndex=0):
        # Connect a client
        self.client = carla.Client(*settings.CARLA_SIMS[serverIndex][:2])
        self.client.set_timeout(2.0)
        self.thread_count = thread_count
        self.tick_lock = lock
        self.modelName = name
        self.frameNumber = frameNumber
        self.waiting_threads = waiting_threads
        self.world_ticks = world_ticks

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
        self.episodeNr = 0  # TODO WARNING: be careful using this as it also counts validation episodes
        self.sql = Sql()
        self.sessionId = sessionId

        # Early stopping variables
        self.grassLocation = None
        self.grassStuckTick = 0

        # Declare variables for later use
        self.vehicle = None
        self.segSensor = None
        self.grassSensor = None
        self.splineSensor = None
        self.imgFrame = None
        self.wheelsOnGrass = None
        self.episodeStartTime = 0
        self.episodeReward = None
        self.distanceOnSpline = None
        self.splineMaxDistance = None
        self.previousDistanceOnSpline = None

        self.queues = []  # List of tuples (queue, dataProcessingFunction)
        self.logBuffer = []

        self.runner = runner

        # Declare reward dependent values
        self.car_last_tick_pos = None
        self.car_last_tick_transform = None
        self.car_last_tick_wheels_on_road = None
        self.car_last_episode_time = None

        # Declare classes
        self.reward = Reward(self)
        self.mediaHandler = MediaHandler(self)
        self.gps = Gps(self)

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
            self.action_space = MultiDiscrete([4, 3, 21])
            self.throttleMapLen = float(self.action_space.nvec[0]-1)
            self.brakeMapLen = float(self.action_space.nvec[1]-1)
            self.steerMapLen = float(self.action_space.nvec[2]-1)/2
        elif settings.MODEL_ACTION_TYPE == ActionType.BOX.value:
            # [Throttle, Steer, brake]
            self.action_space = Box(np.array([0, 0, -0.5]), np.array([+1, +1, +0.5]), dtype=np.float32)
        else:
            raise Exception("No such action type, change settings")

        if settings.AGENT_SYNCED: self.tick(10)

    def close(self):
        self._resetActorList()

    ''':returns initial observation'''
    def reset(self):
        self.episodeNr += 1  # Count episodes TODO WARNING: be careful using this as it also counts validation episodes

        #self.logSensor("reset()")

        # Print episode and reward for that episode
        if self.carlaInstance == 0 and self.car_last_episode_time is not None:
            print(f"Episode:  {self.episodeNr} - Reward: {self.episodeReward} \t - Time: {time.time() - self.car_last_episode_time}")

        self.sql.INSERT_newEpisode(self.sessionId, self.carlaEnv.carlaInstance, self.episodeNr, self.episodeReward, None, None)

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
        # self.totalTicks += 1

        # Do action
        if settings.MODEL_ACTION_TYPE == ActionType.DISCRETE.value:
            self._applyActionDiscrete(action)
        elif settings.MODEL_ACTION_TYPE == ActionType.MULTI_DISCRETE.value:
            self._applyActionMultiDiscrete(action)
        elif settings.MODEL_ACTION_TYPE == ActionType.BOX.value:
            self._applyActionBox(action)
        else:
            raise Exception("No such action type, change settings")

        self.gps.log_location()

        if settings.AGENT_SYNCED:
            self.tick(10)

        is_done = self._isDone()  # Must be calculated before rewards

        # Update reward
        reward = self.reward.calcReward()
        self.episodeReward += reward

        # if is_done and self.carlaInstance == 0 and self.mediaHandler.episodeFrames:
        #     extra = {"episode": {"episodeNr": self.episodeNr, "frames": self.mediaHandler.episodeFrames}}
        # else:
        #     extra = {}

        return self.imgFrame, reward, is_done, {}  # extra

    def synchronized_world_tick(self):
        #self.logSensor(f"synchronized_world_tick pre [{self.frameNumber.value}]")
        self.tick_lock.acquire()
        self.waiting_threads.value += 1
        if self.waiting_threads.value < self.thread_count:
            #self.logSensor("Waiting")
            self.tick_lock.wait()
            #self.logSensor("Done waiting")
            # Wait until someone notifies that the world has ticked
        else:
            if self.world_ticks is not None:
                self.world_ticks.value += 1
            self.frameNumber.value = self.world.tick()

            self.waiting_threads.value = 0
            self.tick_lock.notify_all()
        self.tick_lock.release()

    def tick(self, timeout):
        #self.logSensor(f"Tick pre [{self.frameNumber.value}]")
        self.synchronized_world_tick()
        #self.logSensor(f"Tick post [{self.frameNumber.value}]")

        data = [self._retrieve_data(queueTuple, timeout) for queueTuple in self.queues]
        #self.logSensor(f"-> Data: {data}")

        # assert all(x.frame == self.frameNumber.value for x in data)
        return data

    def tick_unsync(self, timeout):
        #self.logSensor(f"Tick_unsync pre [{self.frameNumber.value}]")
        old_frame = self.world.tick()
        self.frameNumber.value = self.world.get_snapshot().timestamp.frame
        #self.logSensor(f"Tick_unsync post [{self.frameNumber.value}]")
        #self.logSensor(f"old_frame_metod[{old_frame} vs new_frame_method{self.frameNumber.value}] - {'EXCEPTION' if old_frame != self.frameNumber.value else ''}")

        data = [self._retrieve_data(queueTuple, timeout) for queueTuple in self.queues]
        #self.logSensor(f"-> Data: {data}")

        return data

    def _makeQueue(self, registerEvent, processData):
        #self.logSensor(f"MakeQueue()")
        q = queue.Queue()
        registerEvent(lambda item, block=True, timeout=None: self._putData(q, item, block=block, timeout=timeout))
        self.queues.append((q, processData))

    def _putData(self, q, item, block, timeout):
        #self.logSensor(f"Put item({item.frame}) in queue({q})")
        q.put(item, block=block, timeout=timeout)

    def _retrieve_data(self, queueTuple, timeout):
        #self.logSensor(f"Retrive_data(): ")

        while True:
            try:
                data = queueTuple[0].get(block=True, timeout=timeout)
                #self.logSensor(f"-> Data: {data}")
            except queue.Empty as e:
                #self.logSensor("Retrive_data() Queue empty - EXCEPTION")
                return None
            except Exception as e:
                #self.logSensor("Retrive_data() other - EXCEPTION")
                return None

            if data.frame == self.frameNumber.value:
                #self.logSensor(f"--> Same frame number")
                dataProcessFunction = queueTuple[1]
                dataProcessFunction(data)  # Process data
                return data

    def _printLog(self):
        if not settings.LOG_SENSOR or not self.carlaInstance == 0:
            return

        log_path = "/data.log"
        append_write = self._getLogAccess(log_path)

        log_file = open(log_path, append_write)

        for line in self.logBuffer:
            self._logLn(log_file, line)

        # Reset buffer
        self.logBuffer = []

    def logSensor(self, line):
        if settings.LOG_SENSOR and self.carlaInstance == 0:
            self.logBuffer.append(line)

    def _logLn(self, file, line):
        file.write(f"{self.frameNumber.value}: {line}\n")

    def _getLogAccess(self, file_path):
        if os.path.exists(file_path):
            return 'a'  # append if already exists
        else:
            return 'w'  # make a new file if not

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
        self.previousDistanceOnSpline = None

        # Video
        self.mediaHandler.episodeFrames = []

        # # GPS
        # self.gps.reset()


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

        self.splineSensor = self._createSplineSensor()
        self.actorList.append(self.splineSensor)

    # Destroy all previous actors, and clear actor list
    def _resetActorList(self):
        # Destroy all actors from previous episode
        for actor in self.actorList:
            actor.destroy()

        # Clear all actors from the list from previous episode
        self.actorList = []

    # Waits until the world is ready for training
    def _waitForWorldToBeReady(self):
        #self.logSensor("_waitForWorldToBeReady()")

        self.tick_lock.acquire()
        while self._isWorldNotReady():
            if settings.AGENT_SYNCED:
                self.tick_unsync(10)
        self.tick_lock.release()

        self.tick(10)

        #self.logSensor("->world is ready!")

    # Returns true if the world is not yet ready for training
    def _isWorldNotReady(self):
        # print(self.wheelsOnGrass)
        return self.imgFrame is None or self.wheelsOnGrass != 0

    # Creates a new vehicle and spawns it into the world as an actor
    # Returns the vehicle
    def _createNewVehicle(self):
        vehicle_blueprint = self.blueprintLibrary.filter('test')[0]
        color = random.choice(vehicle_blueprint.get_attribute('color').recommended_values)
        vehicle_blueprint.set_attribute('color', '0,255,0')

        vehicle_spawn_transforms = self.world.get_map().get_spawn_points()
        if settings.USE_RANDOM_SPAWN_POINTS:
            index = self.carlaInstance % len(vehicle_spawn_transforms)
            #vehicle_spawn_transform = random.choice(vehicle_spawn_transforms)  # Pick a random spawn point
            vehicle_spawn_transform = vehicle_spawn_transforms[index]
        else:
            vehicle_spawn_transform = vehicle_spawn_transforms[0]  # Use the first spawn point
        return self.world.spawn_actor(vehicle_blueprint, vehicle_spawn_transform)  # Spawn vehicle

    # Creates a new segmentation sensor and spawns it into the world as an actor
    # Returns the sensor
    def _createSegmentationSensor(self):
        # Make segmentation sensor blueprint
        seg_sensor_blueprint = self.blueprintLibrary.find('sensor.camera.modified_semantic_segmentation')
        seg_sensor_blueprint.set_attribute('image_size_x', str(self.imgWidth))
        seg_sensor_blueprint.set_attribute('image_size_y', str(self.imgHeight))
        seg_sensor_blueprint.set_attribute('fov', '110')
        relative_transform_sensor = carla.Transform(carla.Location(x=2, z=3), carla.Rotation(pitch=-45))  # Place sensor on the front of car

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

    # Creates a new spline distance sensor and spawns it into the world as an actor
    # Returns the sensor
    def _createSplineSensor(self):
        # Sensor blueprint
        spline_blueprint = self.blueprintLibrary.find('sensor.other.spline_distance')

        # Grass sensor actor
        spline_sensor = self.world.spawn_actor(spline_blueprint, carla.Transform(), attach_to=self.vehicle)
        self._makeQueue(spline_sensor.listen, processData=self._spline_data)
        # Return created actor
        return spline_sensor

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

    def getDistanceMovedAlongSpline(self):
        distanceAlongSpline = self.distanceOnSpline - self.previousDistanceOnSpline if self.previousDistanceOnSpline is not None else 0
        if distanceAlongSpline < -0.8 * self.splineMaxDistance:  # If the car has completed an entire loop the distance moved will be a negative number close to the max spline distance
            distanceAlongSpline = self.splineMaxDistance - self.previousDistanceOnSpline + self.distanceOnSpline  # Should instead be the distance to the finish line + the distance past the finish line
        elif distanceAlongSpline > 0.8 * self.splineMaxDistance:  # If the car somehow reverses by the finish line it will have moved a distance close to max spline distance
            distanceAlongSpline = -(self.previousDistanceOnSpline + (self.splineMaxDistance - self.distanceOnSpline))  # Should instead be the negative distance that the vehicle moved backwards
        elif abs(distanceAlongSpline) > 1000:
            distanceAlongSpline = 0
        return distanceAlongSpline/100

    # Returns true if the current episode should be stopped
    def _isDone(self):
        # If episode length is exceeded it is done
        episode_expired = self._isEpisodeExpired()
        is_stuck_on_grass = self._isStuckOnGrass()
        car_on_grass = self._isCarOnGrass()
        max_negative_reward = self._isMaxNegativeRewardAccumulated()

        return episode_expired #or is_stuck_on_grass  # or car_on_grass or max_negative_reward

    # Returns true if the current max episode time has elapsed
    def _isEpisodeExpired(self):
        if settings.CARLA_SECONDS_MODE_LINEAR:
            scale = min((self.episodeNr / settings.CARLA_SECONDS_PER_EPISODE_EPISODE_RANGE), 1)                         # Calculate scale depending on episode nr
            range_diff = settings.CARLA_SECONDS_PER_EPISODE_LINEAR_MAX - settings.CARLA_SECONDS_PER_EPISODE_LINEAR_MIN  # Calculate min / max difference
            total_seconds = settings.CARLA_SECONDS_PER_EPISODE_LINEAR_MIN + (range_diff * scale)                        # Calculate current total episodes

            # if self.carlaInstance == 0:
            #     print(str(total_seconds))

            total_max_episode_ticks = int(total_seconds * (1 / settings.AGENT_TIME_STEP_SIZE))                          # Calculate new total_max_episode_ticks
        else:
            total_max_episode_ticks = settings.CARLA_TICKS_PER_EPISODE_STATIC

        return self.episodeTicks > total_max_episode_ticks

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
        #self.logSensor("_grass_data()")
        self.wheelsOnGrass = event[0] + event[1] + event[2] + event[3]
        # print(f"({event[0]},{event[1]},{event[2]},{event[3]})")

    def _spline_data(self, event):
        #self.logSensor("_spline_data()")
        self.distanceOnSpline = event[0]
        self.splineMaxDistance = event[1]

        # if self.carlaInstance is 0 and (self.episodeTicks + 1) % 100 == 0: print(f"Progress: {self.distanceOnSpline/self.splineMaxDistance*100:.2f}%")