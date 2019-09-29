from gym_carla import settings
import cv2
import os
from gym_carla.carla_utils import makeCarlaImportable
import numpy as np

from source.numpyNumbers import NumpyNumbers

makeCarlaImportable()
import carla


class MediaHandler:
    def __init__(self, carlaEnv):
        self.carlaEnv = carlaEnv
        self.episodeFrames = []

    def processImage(self, data):
        cc = carla.ColorConverter.CityScapesPalette
        data.convert(cc)
        # Get image, reshape and remove alpha channel
        image = np.array(data.raw_data)
        image = image.reshape((self.carlaEnv.imgHeight, self.carlaEnv.imgWidth, 4))
        image = image[:, :, :3]

        # bgra
        self.carlaEnv.imgFrame = image

        # Save image in memory for later video export
        if self._isVideoEpisode():
            self._storeImageForVideoEpisode(image)

        # Save images to disk (Output folder)
        if settings.VIDEO_ALWAYS_ON and self.carlaEnv.carlaInstance == 0:
            cv2.imwrite(f"../data/frames/frame_{data.frame}.png", self._getResizedImageWithOverlay(image))
            #  data.save_to_disk('../data/frames/%06d.png' % data.frame)

    def exportAndUploadVideoToDB(self):
        folder = "../data/videos"
        file_name = f"videoTest_{self.carlaEnv.episodeNr}.avi"
        file_path = folder + "/" + file_name

        self._exportVideo(folder, file_name, self.episodeFrames)
        self._uploadVideoFileToDb(file_path, self.carlaEnv.sessionId, self.carlaEnv.episodeNr, self.carlaEnv.episodeReward)
        # os.remove(file_path)

    # Returns true, if the current episode is a video episode
    def _isVideoEpisode(self):
        return (self.carlaEnv.carlaInstance == 0) and (self.carlaEnv.episodeNr % settings.VIDEO_EXPORT_RATE == 0)

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
        speed = self.carlaEnv.getCarVelocity()
        overlay = nn.getOverlay(round(speed), self._getVideoWidth(), self._getVideoHeight())

        for a, aa in enumerate(frame):
            for b, bb in enumerate(aa):
                for c, frame_pixel in enumerate(bb):
                    overlay_pixel = overlay[a, b, c]

                    if overlay_pixel != 0:
                        frame[a, b, c] = overlay_pixel

    # Exports a video from numpy arrays to the file system
    def _exportVideo(self, folder, file_name, frames):
        if not os.path.isdir(folder):
            os.mkdir(folder)

        file_path = folder + "/" + file_name

        video_size = (self._getVideoHeight(), self._getVideoWidth())
        fps = 1 / settings.TIME_STEP_SIZE

        out = cv2.VideoWriter(file_path, cv2.VideoWriter_fourcc(*'DIVX'), fps, video_size)

        for frame in frames:
            out.write(frame)

        out.release()

    def _uploadVideoFileToDb(self, file_path, session_id, episode_nr, episode_reward):
        with open(file_path, 'rb') as f:
            video_blob = f.read()

        self.carlaEnv.sql.INSERT_newEpisode(session_id, episode_nr, episode_reward, video_blob)

