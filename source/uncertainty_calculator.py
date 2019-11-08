import cv2

from gym_carla import settings
from source.uncertaintyLogger import UncertaintyLogger
import numpy as np
import os

class UncertaintyCalculator:
    def __init__(self, modelName):
        self.counter = 0
        self.logger = UncertaintyLogger()
        self.modelName = modelName
        self.seenObservations = []
        self.index = 0
        self.indexPixelMap = { 0:(0, 0, 0), 1:(70, 70, 70), 2:(153, 153, 190), # RGB - #BGR
                               3:(160, 170, 250), 4:(60, 20, 220), 5:(153, 153, 153),
                               6:(50, 234, 157), 7:(128, 64, 128), 8:(232, 35, 244),
                               9:(35, 142, 107), 10:(142, 0, 0), 11:(156, 102, 102), 12:(0, 220, 220) }

    def _getObservationUncertaintyAndID(self, observation):
        state_counter, index = self._getStateCounterAndIndex(observation)

        return max(1 - (state_counter/settings.UNCERTAINTY_TIMES), 0), index
        #return 1/(1+(settings.UNCERTAINTY_RATE*state_counter)), index

    def _getStateCounterAndIndex(self, obs):
        # Loop all seen images, and try to locate an image that is close to the current one
        for index, ob_tuple in enumerate(self.seenObservations):
            seen_ob = ob_tuple[1]
            equal_pixels = self._getImageEqualness(seen_ob, obs)

            if equal_pixels >= settings.TRANSFER_IMITATION_THRESHOLD:
                return ob_tuple[0], index

        # We have never seen anything like this new observation, so add it to the list,
        # and return 0 since it's the first time we see it
        self.seenObservations.append((0, np.copy(obs)))
        self._printImage(obs, len(self.seenObservations))
        print(f"NEW STATE!! - TOTAL STATES: {len(self.seenObservations)}")
        return 0, len(self.seenObservations)-1

    def _getImageEqualness(self, img_a, img_b):
        return np.sum(np.equal(img_a, img_b))

    def calculateUncertainty(self, obs):
        uncertainties = []
        ids = []

        # Get uncertainties
        for observation in obs:
            uncertainty, index = self._getObservationUncertaintyAndID(observation)

            uncertainties.append(uncertainty)
            ids.append(index)

        self._logOrExportCounters()

        return uncertainties, ids

    def _logOrExportCounters(self):
        if (self.counter + 1) % 50 == 0:
            self.logger.log(self.seenObservations)

        if (self.counter + 1) % 350 == 0:
            self.logger.exportAsCsv("imitation_frames", f"bar_race_{self.counter}.csv")

    def updateStateCounter(self, indices):
        self.counter += 1
        for index in indices:
            seen_counter, obs = self.seenObservations[index]
            self.seenObservations[index] = (seen_counter + 1, obs)

    def _printImage(self, obs, index):
        pixels = self._convertCategoryToImg(obs)
        path = f"UncertaintyObservations/{self.modelName}"
        if not os.path.isdir(path):
            os.makedirs(path)
        cv2.imwrite(f"{path}/{index}.png", pixels)

    def _convertCategoryToImg(self, obs):
        pixels = []
        
        for row in obs:
            for pixel in row:
                pixels.append(self.indexPixelMap[pixel])

        return np.reshape(pixels, (50, 50, 3))



