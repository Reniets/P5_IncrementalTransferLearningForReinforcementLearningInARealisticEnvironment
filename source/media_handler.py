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
        # self.carlaEnv.logSensor("processImage")
        cc = carla.ColorConverter.CityScapesPalette
        data.convert(cc)
        # Get image, reshape and remove alpha channel
        image = np.array(data.raw_data)
        image = image.reshape((self.carlaEnv.imgHeight, self.carlaEnv.imgWidth, 4))
        image = image[:, :, :3]
        self.addSpeedOverlayToFrame(image, self.carlaEnv.getCarVelocity())

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
        file_name = f"videoTest_{self.carlaEnv.episodeNr}.mp4"
        video_path = folder + "/" + file_name
        model_path = f"temp/{self.carlaEnv.modelName}_{self.carlaEnv.episodeNr}.pkl"

        self._exportVideo(folder, file_name, self.episodeFrames)
        # self.carlaEnv.model.save(model_path)
        self._uploadVideoAndModelToDb(video_path, model_path, self.carlaEnv.sessionId, self.carlaEnv.episodeNr, self.carlaEnv.episodeReward)
        os.remove(video_path)
        # os.remove(model_path)

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
        return max(settings.CARLA_IMG_WIDTH, settings.VIDEO_MAX_WIDTH)

    def _getVideoHeight(self):
        return max(settings.CARLA_IMG_HEIGHT, settings.VIDEO_MAX_HEIGHT)

    def _addFrameDataOverlay(self, frame):
        nn = NumpyNumbers()

        y_offset = nn.number_size[0] + 2

        # Speed
        speed = self.carlaEnv.getCarVelocity()
        speed_overlay = nn.getOverlay(round(speed))

        # Reward
        total_reward = self.carlaEnv.episodeReward
        reward_overlay = nn.getOverlay(round(total_reward))

        # Actions
        controls = self.carlaEnv.vehicle.get_control()
        throttle_overlay = nn.getOverlay(round(controls.throttle*100))
        brake_overlay = nn.getOverlay(round(controls.brake*100))
        steer_overlay = nn.getOverlay(round(controls.steer*100))

        self.addOverlayToFrame(frame, speed_overlay,    (y_offset * 0, self._getXOffset(frame, speed_overlay)))
        self.addOverlayToFrame(frame, reward_overlay,   (y_offset * 1, self._getXOffset(frame, reward_overlay)))
        self.addOverlayToFrame(frame, throttle_overlay, (y_offset * 2, self._getXOffset(frame, throttle_overlay)))
        self.addOverlayToFrame(frame, brake_overlay,    (y_offset * 3, self._getXOffset(frame, brake_overlay)))
        self.addOverlayToFrame(frame, steer_overlay,    (y_offset * 4, self._getXOffset(frame, steer_overlay)))

    def _getXOffset(self, frame, overlay):
        return frame.shape[1] - overlay.shape[1]

    # Exports a video from numpy arrays to the file system
    def _exportVideo(self, folder, file_name, frames):
        if not os.path.isdir(folder):
            os.mkdir(folder)

        file_path = folder + "/" + file_name

        video_size = (self._getVideoHeight(), self._getVideoWidth())
        fps = 1 / settings.AGENT_TIME_STEP_SIZE

        out = cv2.VideoWriter(file_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, video_size)

        for frame in frames:
            out.write(frame)

        out.release()

    def _uploadVideoAndModelToDb(self, video_path, model_path, session_id, episode_nr, episode_reward):
        with open(video_path, 'rb') as f:
            video_blob = f.read()

        # with open(model_path, 'rb') as f:
        #     model_blob = f.read()

        self.carlaEnv.sql.INSERT_newEpisode(session_id, self.carlaEnv.carlaInstance, episode_nr, episode_reward, video_blob, None)

    def addSpeedOverlayToFrame(self, frame, speed):
        overlay = self.createSpeedBarOverlay(speed, 50, 50)
        self.addOverlayToFrame(frame, overlay, (0, 0))

    def addOverlayToFrame(self, image, overlay, offset):
        for a, aa in enumerate(overlay):
            for b, bb in enumerate(aa):
                for c, frame_pixel in enumerate(bb):
                    if frame_pixel != 0:
                        image[a + offset[0], b + offset[1], c] = frame_pixel

    def createSpeedBarOverlay(self, speed, height, width):
        max_speed = 110  # km/h
        scale = min(speed / max_speed, 1)

        speed_pixel_width = round(width * scale)
        speed_pixel_height = 5

        speed_bar = np.ones((speed_pixel_height, speed_pixel_width))

        return self.addColorChannels(speed_bar, (255, 50, 50))

    def addColorChannels(self, number: np, color):
        return np.dstack((
            number * color[2],  # B
            number * color[1],  # G
            number * color[0],  # R
        ))