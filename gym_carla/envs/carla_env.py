import math
import os
import gym
from gym.spaces import Discrete, Box, Tuple
from database.sql import Sql
import numpy as np
from gym_carla.carla_utils import *
import cv2

from source.numpyNumbers import NumpyNumbers

makeCarlaImportable()
import carla
from PIL import Image
stepsCountEpisode = 0

class CarlaEnv(gym.Env):

    def __init__(self, carlaInstance=0):

        self.carlaInstance = carlaInstance
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
        self.episodeTicks = 0

        # Video variables
        self.episodeNr = 0
        self.episodeFrames = []
        self.sql = Sql()
        self.sessionId = self.sql.INSERT_newSession(settings.MODEL_NAME) if (self.carlaInstance == 0) else None

        # Early stopping variables
        self.grassLocation = None
        self.grassStuckTick = 0

        # Declare variables for later use
        self.vehicle = None
        self.segSensor = None
        # self.colSensor = None
        self.grassSensor = None
        self.imgFrame = None
        self.wheelsOnGrass = None
        self.episodeStartTime = 0
        self.episodeReward = None

        # Declare reward dependent values
        self.car_last_tick_pos = None
        self.car_last_tick_wheels_on_road = None

        # Defines image space as a box which can look at standard rgb images of size imgWidth by imgHeight
        imageSpace = Box(low=0, high=255, shape=(self.imgHeight, self.imgWidth, 3), dtype=np.uint8)

        # Defines observation and action spaces
        self.observation_space = imageSpace
        # self.action_space = Discrete(len(DISCRETE_ACTIONS))
        # [Throttle, Steer, brake]
        self.action_space = Box(np.array([0, 0, -0.5]), np.array([+1, +1, +0.5]), dtype=np.float32)
        # OLD: self.action_space = Box(np.array([-0.5, 0, 0]), np.array([+0.5, +1, +1]), dtype=np.float32)

        if settings.AGENT_SYNCED: self.world.tick()


    ''':returns initial observation'''
    def reset(self):
        self.episodeNr += 1

        # Frames are only added, if it's a video episode, so if there are frames it means that last episode
        # was a video episode, so we should export it, before we reset the frames list below
        if self.episodeFrames:
            folder = "../data/videos"
            file_name = f"videoTest_{self.episodeNr}.avi"
            file_path = folder+"/"+file_name

            self._exportVideo(folder, file_name, self.episodeFrames)
            self._uploadVideoFileToDb(file_path, self.sessionId, self.episodeNr, self.episodeReward)
            # os.remove(file_path)

        if self.carlaInstance == 0:
            print(f"Episode: {self.episodeNr} - Reward: {self.episodeReward}")

        # global stepsCountEpisode
        # print(stepsCountEpisode)
        # stepsCountEpisode = 0

        # Destroy all previous actors, and clear actor list
        self._resetActorList()
        self._resetInstanceVariables()

        self.episodeReward = 0

        # Create new actors and add to actor list
        self._createActors()

        # TODO: Make some system that allows previewing episodes once in a while

        # Workaround to start episode as quickly as possible
        self._setActionDiscrete(Action.BRAKE.value)

        # Wait for camera to send first image
        self._waitForWorldToBeReady()

        # Set last tick variables to equal starting pos information
        self.car_last_tick_pos = self.vehicle.get_location()
        self.car_last_tick_wheels_on_road = 4

        # Disengage brakes from earlier workaround
        self._setActionDiscrete(Action.DO_NOTHING.value)

        return self.imgFrame  # Returns initial observation (First image)

    def _resetInstanceVariables(self):
        # Declare variables for later use
        self.vehicle = None
        self.segSensor = None
        # self.colSensor = None
        self.grassSensor = None
        self.imgFrame = None
        self.wheelsOnGrass = None
        self.episodeTicks = 0
        self.episodeReward = None

        # Early stopping
        self.grassLocation = None
        self.grassStuckTick = 0

        # Declare reward dependent values
        self.car_last_tick_pos = None
        self.car_last_tick_wheels_on_road = None

        # Video
        self.episodeFrames = []

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
        # if settings.AGENT_SYNCED: self.world.tick()

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
        relative_transform_sensor = carla.Transform(carla.Location(x=3, z=3), carla.Rotation(pitch=-45))  # Place sensor on the front of car

        # Spawn semantic segmentation sensor, start listening for data and add to actorList
        seg_sensor = self.world.spawn_actor(seg_sensor_blueprint, relative_transform_sensor, attach_to=self.vehicle)
        seg_sensor.listen(self._processImage)

        return seg_sensor

    # Returns true, if the current episode is a video episode
    def _isVideoEpisode(self):
        return (self.carlaInstance == 0) and (self.episodeNr % settings.VIDEO_EXPORT_RATE == 0)

    # Exports a video from numpy arrays to the file system
    def _exportVideo(self, folder, file_name, frames):
        if not os.path.isdir(folder):
            os.mkdir(folder)

        file_path = folder + "/" + file_name

        video_size = (self._getVideoHeight(), self._getVideoWidth())
        fps = 1/settings.TIME_STEP_SIZE

        out = cv2.VideoWriter(file_path, cv2.VideoWriter_fourcc(*'DIVX'), fps, video_size)

        for frame in frames:
            out.write(frame)

        out.release()

    def _uploadVideoFileToDb(self, file_path, session_id, episode_nr, episode_reward):
        with open(file_path, 'rb') as f:
            video_blob = f.read()

        self.sql.INSERT_newEpisode(session_id, episode_nr, episode_reward, video_blob)


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
        self.episodeTicks += 1

        # Do action
        # self._setActionDiscrete(action)
        self._setActionBox(action)

        if settings.AGENT_SYNCED: self.world.tick()

        is_done = self._isDone() # Must be calculated before rewards

        # Update reward
        reward = self._calcRewardNew()
        self.episodeReward += reward
        # print('Reward: \t' + str(self.episodeReward) + "\t - " + str(reward))
        return self.imgFrame, reward, is_done, {}

    # Applies a discrete action to the vehicle
    def _setActionDiscrete(self, action):
        # If action does something, apply action
        self.vehicle.apply_control(carla.VehicleControl(
            throttle=DISCRETE_ACTIONS[Action(action)][0],
            brake=DISCRETE_ACTIONS[Action(action)][1],
            steer=DISCRETE_ACTIONS[Action(action)][2])
        )

    # Applies a box action to the vehicle
    def _setActionBox(self, action):
        self.vehicle.apply_control(carla.VehicleControl(
            throttle=float(action[0]),
            brake=float(action[1]),
            steer=float(action[2]),
        ))

    # Returns the reward for the current state
    def _calcReward(self):
        reward = 0
        speed = self._getCarVelocity()
        expected_speed = 20

        reward += (speed / expected_speed) * self._wheelsOnRoad()
        reward -= self.wheelsOnGrass

        return reward

    def _calcRewardNew(self):
        reward = 0

      # reward += self._rewardSubGoal()             * weight
        reward += self._rewardDriveFarOnRoad()      * 1.00  # Reward
        # reward += self._rewardDriveShortOnGrass()   * 1.50  # Penalty
        # reward += self._rewardReturnToRoad()        * 1.00  # Reward / Penalty
        # reward += self._rewardStayOnRoad()          * 0.05  # Reward
        reward += self._rewardAvoidGrass()          * 1.00  # Penalty
        # reward += self._rewardDriveFast()         * 0.10

        self._updateLastTickVariables()  # MUST BE LAST THING IN REWARD FUNCTION

        return reward

    def _updateLastTickVariables(self):
        self.car_last_tick_pos = self.vehicle.get_location()
        self.car_last_tick_wheels_on_road = self._wheelsOnRoad()

    def _rewardStayOnRoad(self):
        return self._wheelsOnRoad() * 0.25

    def _rewardAvoidGrass(self):
        return self.wheelsOnGrass * (-1)

    def _rewardDriveFast(self):
        return (self._getCarVelocity() / 50) * self._rewardStayOnRoad()

    def _rewardDriveFar(self):
        return self._metersTraveledSinceLastTick()

    def _rewardDriveFarOnRoad(self):
        return self._rewardDriveFar() * self._wheelsOnRoad()

    def _rewardDriveShortOnGrass(self):
        return -(self._rewardDriveFar() * self.wheelsOnGrass)

    def _rewardReturnToRoad(self):
        wheel_diff = self._wheelsOnRoadDiffFromLastTick()

        if wheel_diff > 0:
            return wheel_diff * 25
        elif wheel_diff < 0:
            return wheel_diff * (-50)
        else:
            return 0

    # Returns the amount of meters traveled since last tick
    # and updated last pos to current pos
    def _metersTraveledSinceLastTick(self):
        # Calculate meters driven
        last = self.car_last_tick_pos
        current = self.vehicle.get_location()

        x_diff = current.x - last.x
        y_diff = current.y - last.y
        z_diff = current.z - last.z

        distance_traveled = math.sqrt(x_diff**2 + y_diff**2 + z_diff**2)

        # Return distance traveled in meters
        return distance_traveled

    # Returns the difference from current tick to last tick of how many wheels are currently on the road
    # Also updates last to current tick
    def _wheelsOnRoadDiffFromLastTick(self):
        last = self.car_last_tick_wheels_on_road
        current = self._wheelsOnRoad()
        diff = current - last

        return diff

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
        is_stuck_on_grass = self._isStuckOnGrass()
        car_on_grass = self._isCarOnGrass()
        max_negative_reward = self._isCarOnGrass()

        return episode_expired or is_stuck_on_grass  # or car_on_grass or max_negative_reward

    # Returns true if the current max episode time has elapsed
    def _isEpisodeExpired(self):
        return self.episodeTicks > settings.TICKS_PER_EPISODE

    # Returns true if all four wheels are not on the road
    def _isCarOnGrass(self):
        return self.wheelsOnGrass == 4

    # Returns true if the maximum negative reward has been accumulated
    def _isMaxNegativeRewardAccumulated(self):
        return self.episodeReward < -500

    def _isStuckOnGrass(self):
        if self.wheelsOnGrass == 4 and self._metersTraveledSinceLastTick() == 0.0:
            self.grassStuckTick += 1
            return self.grassStuckTick > 5
        else:
            self.grassStuckTick = 0
            return False

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

        # Save image in memory for later video export
        if self._isVideoEpisode():
            self._storeImageForVideoEpisode(image)

        # Save images to disk (Output folder)
        if settings.VIDEO_ALWAYS_ON and self.carlaInstance == 0:
            cv2.imwrite(f"../data/frames/frame_{data.frame}.png", self._getResizedImageWithOverlay(image))
            #  data.save_to_disk('../data/frames/%06d.png' % data.frame)

    def _storeImageForVideoEpisode(self, image):
        image = self._getResizedImageWithOverlay(image)

        self.episodeFrames.append(np.asarray(image))

    def _getResizedImageWithOverlay(self, image):
        image = self._resizeImage(image)
        self._addFrameDataOverlay(image)

        return image

    def _resizeImage(self, image):
        width = self._getVideoWidth()
        height = self._getVideoHeight()

        return cv2.resize(image, dsize=(height, width), interpolation=cv2.INTER_CUBIC)

    def _getVideoWidth(self):
        return max(settings.IMG_WIDTH, settings.VIDEO_MAX_WIDTH)

    def _getVideoHeight(self):
        return max(settings.IMG_HEIGHT, settings.VIDEO_MAX_HEIGHT)

    def _addFrameDataOverlay(self, frame):
        nn = NumpyNumbers()
        speed = self._getCarVelocity()
        overlay = nn.getOverlay(round(speed), self._getVideoWidth(), self._getVideoHeight())

        for a, aa in enumerate(frame):
            for b, bb in enumerate(aa):
                for c, frame_pixel in enumerate(bb):
                    overlay_pixel = overlay[a, b, c]

                    if overlay_pixel != 0:
                        frame[a, b, c] = overlay_pixel

    def _grass_data(self, event):
        self.wheelsOnGrass = event[0] + event[1] + event[2] + event[3]
        # print(f"({event[0]},{event[1]},{event[2]},{event[3]})")










































