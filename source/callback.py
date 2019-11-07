from gym_carla import settings
from source.gps_image import GpsImage
import numpy as np
import tensorflow as tf
import cv2
import os
from gym_carla.carla_utils import changeMap
from database.sql import Sql


class Callback:

    def __init__(self, runner):
        self.runner = runner
        self.nEpisodes = 0
        self.prev_episode = 0
        self.maxRewardAchieved = float('-inf')
        self.sql = Sql()

    def callback(self, runner_locals, _locals):
        self.nEpisodes += 1

        self._updateLearningRate(_locals)
        self._updateClipRange(_locals)
        self._updatePollingRate(_locals)
        self._storeTensorBoardData(_locals)
        self._exportBestModel(runner_locals, _locals)
        self._exportGpsData(_locals)
        self._printCallbackStats(_locals)
        self._exportRewardsToDB(self._getAllCarRewards())

        if self.nEpisodes % settings.CARLA_EVALUATION_RATE == 0:
            self._testVehicles(_locals, runner_locals)

        return True

    def _exportRewardsToDB(self, rewards, evaluation=False):
        sessionId = self.runner.sessionId

        for instance in range(len(rewards)):
            self.sql.INSERT_newEpisode(sessionId, instance, self._getEpisodeCount(), rewards[instance], evaluation=evaluation)


    def _getEpisodeCount(self):
        return len(self.runner.env.get_attr('episode_rewards', 0)[0])

    def _getAllCarRewards(self):
        all_rewards = self.runner.env.get_attr('episode_rewards', [i for i in range(settings.CARS_PER_SIM)])
        values = [array[-1] for array in all_rewards]
        return values

    def _getAllCarlaEnvironmentGpsDatas(self):
        return self.runner.env.env_method('get_location', indices=[i for i in range(settings.CARS_PER_SIM)])

    def _maxCarGpsData(self, gps_data):
        return gps_data[np.argmax(self._getAllCarRewards())]

    def _minCarGpsData(self, gps_data):
        return gps_data[np.argmin(self._getAllCarRewards())]

    def _medianCarGpsData(self, gps_data):
        rewards = self._getAllCarRewards()
        median_index = np.argsort(rewards)[len(rewards) // 2]
        return gps_data[median_index]

    def _exportGpsData(self, _locals):
        gps_data = self._getAllCarlaEnvironmentGpsDatas()

        self._exportGpsDataForEnvironment(self._maxCarGpsData(gps_data), "max")
        self._exportGpsDataForEnvironment(self._minCarGpsData(gps_data), "min")
        self._exportGpsDataForEnvironment(self._medianCarGpsData(gps_data), "median")

    def _exportGpsDataForEnvironment(self, gps_data, name_prefix=""):
        # Export the image
        image_dir = f"GpsData/{self.runner.modelName}"

        if not os.path.exists(image_dir):
            os.makedirs(image_dir)

        if False:
            # Prepare gps image
            map_name = settings.CARLA_SIMS[0][2]
            gps_image = GpsImage(self.getReferenceImagePath(map_name), self.getReferenceCoordinates(map_name))

            # Create track image from reference photo and gps data
            track_image = gps_image.applyCoordinates(gps_data)

            cv2.imwrite(f"{image_dir}/{name_prefix}_{self._getEpisodeCount()}.png", track_image)

        if True:
            file = open(f"{image_dir}/gps_{name_prefix}_{self._getEpisodeCount()}.txt", "w+")

            for data in gps_data:
                file.write(f"{data[0]};{data[1]}\n")

            file.close()

    def getReferenceImagePath(self, name):
        return f"{self._getGpsReferenceBaseFolder(name)}/map.png"

    def getReferenceCoordinates(self, name):
        reference_points_path = f"{self._getGpsReferenceBaseFolder(name)}/ref.txt"

        file = open(reference_points_path, "r")
        ref_data = []

        for line in file:
            data_split = line.split(";")
            data = np.array([float(data_split[i]) for i in range(4)]).reshape((2, 2))
            ref_data.append(data)

        return tuple(ref_data)

    def _getGpsReferenceBaseFolder(self, name):
        return f"GpsData/A_referenceData/{name}"

    def _exportBestModel(self, runner_locals, _locals):
        # info = _locals["ep_infos"]
        # print(f"{self.nSteps}: {info}")
        mean = np.sum(np.asarray(runner_locals['mb_rewards']))
        if self.maxRewardAchieved < mean:
            self.maxRewardAchieved = mean
            if self.nEpisodes > 10:
                print(f"Saving best model: step {self.nEpisodes} reward: {mean}")
                _locals['self'].save(f"TrainingLogs/BaselineAgentLogs/{self.runner.modelName}_{self.runner.modelNum}_best")

    def _storeTensorBoardData(self, _locals):
        n_episodes = self._getEpisodeCount()

        if n_episodes > self.prev_episode:
            self.prev_episode = n_episodes

            values = self._getAllCarRewards()

            median = np.median(values)
            summary = tf.Summary(value=[tf.Summary.Value(tag='episodeRewardMedian', simple_value=median)])
            _locals['writer'].add_summary(summary, n_episodes)

            max = np.max(values)
            summary = tf.Summary(value=[tf.Summary.Value(tag='episodeRewardMax', simple_value=max)])
            _locals['writer'].add_summary(summary, n_episodes)

            mean = np.mean(values)
            summary = tf.Summary(value=[tf.Summary.Value(tag='episodeRewardMean', simple_value=mean)])
            _locals['writer'].add_summary(summary, n_episodes)

    def _updateLearningRate(self, _locals):
        new_learning_rate = self._calcluateNewLearningRate_Exponential()
        _locals['self'].learning_rate = lambda frac: new_learning_rate

    def _updatePollingRate(self, _locals):
        new_polling_rate = self._calculateNewPollingRate_Linear()
        _locals['self'].polling_rate = lambda: new_polling_rate

    def _updateClipRange(self, _locals):
        new_clip_range = self._calculateNewClipRange_Linear()
        _locals['self'].cliprange = lambda frac: new_clip_range
        # _locals['self'].cliprange_vf = lambda frac: new_clip_range
        # _locals.update({'cliprange_vf': lambda frac: new_clip_range})

    def _calculateNewClipRange_Linear(self):
        scale = self._getEpisodeScaleTowardsZero()
        clip_diff = settings.MODEL_CLIP_RANGE - settings.MODEL_CLIP_RANGE_MIN
        newClip = max(settings.MODEL_CLIP_RANGE_MIN + (clip_diff*scale), 0)

        return newClip

    def _calculateNewPollingRate_Linear(self):
        scale = self._getEpisodeScaleTowardsZero()
        poll_diff = settings.TRANSFER_POLLING_RATE_START - settings.TRANSFER_POLLING_RATE_MIN
        newPoll = max(settings.TRANSFER_POLLING_RATE_MIN + (poll_diff*scale), 0)

        return newPoll

    def _calcluateNewLearningRate_Exponential(self):
        n_episodes = self._getEpisodeCount()

        newLearningRate = settings.MODEL_LEARNING_RATE * (1 / (1 + (settings.MODEL_LEARNING_RATE * 20) * n_episodes))

        return max(newLearningRate, settings.MODEL_LEARNING_RATE_MIN)

    def _calculateNewLearningRate_Linear(self):
        scale = self._getEpisodeScaleTowardsZero()                          # Calculate scale depending on max episode progress
        new_learning_rate = max(settings.MODEL_LEARNING_RATE * scale, 0)    # Calculate new learning rate

        return max(new_learning_rate, settings.MODEL_LEARNING_RATE_MIN)     # Return the learning rate, while respecting minimum learning rate

    def _getEpisodeScaleTowardsZero(self):
        n_episodes = self._getEpisodeCount()
        return max(1 - (n_episodes / settings.MODEL_MAX_EPISODES), 0)

    def _printCallbackStats(self, _locals):
        # Print stats every 100 calls
        if self.nEpisodes % settings.MODEL_EXPORT_RATE == 0:
            print(f"Saving new model: step {self.nEpisodes}")
            _locals['self'].save(f"TrainingLogs/BaselineAgentLogs/{self.runner.modelName}_{self.runner.modelNum}")
            self.runner.modelNum += 1

    def _testVehicles(self, _locals, runner_locals):
        print("Evaluating vehicles...")
        # Evaluate agent in environment
        #director = self.runner._getModelImitation()
        # TODO: If we want to evaluate on 'alt' maps, load them here!!!
        old_map = settings.CARLA_SIMS[0][2]
        load_new_map = old_map in settings.CARLA_EVALUATION_MAPS
        if load_new_map:
            self.runner.env.env_method('prepare_for_world_change', indices=[i for i in range(settings.CARS_PER_SIM)])
            changeMap(settings.CARLA_EVALUATION_MAPS[old_map])
            self.runner.env.env_method('reset_actors', indices=[i for i in range(settings.CARS_PER_SIM)])

        # obs = self.runner.env.reset()
        #
        # state = None
        # # When using VecEnv, done is a vector
        # done = [False for _ in range(self.runner.env.num_envs)]
        # rewards_accum = np.zeros(settings.CARS_PER_SIM)
        # for _ in range(settings.CARLA_TICKS_PER_EPISODE_STATIC):
        #     # We need to pass the previous state and a mask for recurrent policies
        #     # to reset lstm state when a new episode begin
        #     action, state = self.runner.model.predict(obs, state=state, mask=done, deterministic=False)
        #     obs, rewards, done, _ = self.runner.env.step(action)
        #     rewards_accum += rewards

        rewards_accum = np.zeros(settings.CARS_PER_SIM)

        state = np.zeros((21, 512))
        done = [False for _ in range(settings.CARS_PER_SIM)]
        observations = self.runner.env.reset()
        obs = observations[0]

        # Play some limited number of steps
        for i in range(settings.CARLA_TICKS_PER_EPISODE_STATIC):
            #action_prob = self.runner.model.proba_step(obs, state, done)
            #director_action_prob = director.proba_step(obs, state, done)
            action, value, state, neglog = self.runner.model.step(obs, state=state, mask=done, deterministic=False)

            # with open("evaluation_approx_kl", 'a+') as file:
            #     approx_kl = 0
            #     for i in range(len(action_prob)):
            #         np_director_action_prob = np.concatenate(director_action_prob[i]).ravel()
            #         np_action_prob = np.concatenate(action_prob[i]).ravel()
            #         approx_kl += np.mean(np.square(np.subtract(np_director_action_prob, np_action_prob)))
            #     approx_kl /= len(action_prob)
            #     file.write(f"{approx_kl}\n")

            observations, rewards, done, info = self.runner.env.step(action)
            obs = observations[0]
            rewards_accum += rewards

        if load_new_map:
            self.runner.env.env_method('prepare_for_world_change', indices=[i for i in range(settings.CARS_PER_SIM)])
            changeMap(old_map)
            self.runner.env.env_method('reset_actors', indices=[i for i in range(settings.CARS_PER_SIM)])

        self.runner.env.reset()

        self._exportRewardsToDB(rewards_accum, True)

        mean = rewards_accum.mean()
        print(f"Done evaluating: {mean}")
        summary = tf.Summary(value=[tf.Summary.Value(tag='EvaluationMean', simple_value=mean)])
        _locals['writer'].add_summary(summary, self.nEpisodes)