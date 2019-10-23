from gym_carla import settings
import numpy as np
import tensorflow as tf


class Callback:

    def __init__(self, runner):
        self.runner = runner
        self.nEpisodes = 0
        self.prev_episode = 0
        self.maxRewardAchieved = float('-inf')

    def callback(self, runner_locals, _locals):
        self.nEpisodes += 1

        self._updateLearningRate(_locals)
        self._updateClipRange(_locals)
        self._storeTensorBoardData(_locals)
        self._exportBestModel(runner_locals, _locals)
        self._printCallbackStats(_locals)

        if self.nEpisodes % settings.CARLA_EVALUATION_RATE == 0:
            self._testVehicles(_locals, runner_locals)

        return True

    def _getEpisodeCount(self):
        return len(self.runner.env.get_attr('episode_rewards', 0)[0])

    def _exportBestModel(self, runner_locals, _locals):
        # info = _locals["ep_infos"]
        # print(f"{self.nSteps}: {info}")
        mean = np.sum(np.asarray(runner_locals['mb_rewards']))
        if self.maxRewardAchieved < mean:
            self.maxRewardAchieved = mean
            if self.nEpisodes > 10:
                print(f"Saving best model: step {self.nEpisodes} reward: {mean}")
                _locals['self'].save(f"ExperimentLogsFinal/{self.runner.modelName}_{self.runner.modelNum}_best")

    def _storeTensorBoardData(self, _locals):
        n_episodes = self._getEpisodeCount()

        if n_episodes > self.prev_episode:
            self.prev_episode = n_episodes

            allRewards = self.runner.env.get_attr('episode_rewards', [i for i in range(settings.CARS_PER_SIM)])
            values = [array[-1] for array in allRewards]

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
            _locals['self'].save(f"ExperimentLogsFinal/{self.runner.modelName}_{self.runner.modelNum}")
            self.runner.modelNum += 1

    def _testVehicles(self, _locals, runner_locals):
        print("Evaluating vehicles...")
        # Evaluate agent in environment

        # TODO: If we want to evaluate on 'alt' maps, load them here!!!

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

        state = None
        done = False
        obs = self.runner.env.reset()

        # Play some limited number of steps
        for i in range(settings.CARLA_TICKS_PER_EPISODE_STATIC):
            action, state = self.runner.model.predict(obs, state=state, mask=done, deterministic=True)
            obs, rewards, done, info = self.runner.env.step(action)
            rewards_accum += rewards

        self.runner.env.reset()
        mean = rewards_accum.mean()
        print(f"Done evaluating: {mean}")
        summary = tf.Summary(value=[tf.Summary.Value(tag='EvaluationMean', simple_value=mean)])
        _locals['writer'].add_summary(summary, self.nEpisodes)