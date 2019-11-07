import cv2
import numpy as np


class GpsImage2:
    # Reference point structure: tuple(np(img_cords), np(env_cords)).
    # We expect there to be two img cords and two related env cords
    def __init__(self, save_path, reference_points, mode=cv2.IMREAD_COLOR, isLoopTrack=True, padding=60, scale=1):
        self.track_color = (0, 95, 255)     # BGR
        self.start_color = (150, 0, 0)    # BGR
        self.end_color   = (0, 0, 255)      # BGR
        self.isLoopTrack = isLoopTrack

        self.run_width = 1               # Pixels
        self.start_width = 2 * scale              # Pixels
        self.end_width   = 2 * scale              # Pixels

        self.minX, self.minY, self.maxX, self.maxY = self._getBounds(reference_points)

        self.padding = padding

        self.xExtent = int(abs(self.minX - self.maxX))
        self.yExtent = int(abs(self.minY - self.maxY))

        self.scale = int(scale)

        self.image = np.zeros(shape=[
            (self.padding + self.xExtent)*self.scale,
            (self.padding + self.yExtent)*self.scale,
            3
        ], dtype=np.uint8)
        self.image[:] = (0, 50, 0)
        self._drawMainTrack(reference_points)

    def _getBounds(self, gps_data):
        minX, minY, maxX, maxY = float('inf'), float('inf'), float('-inf'), float('-inf')
        for data in gps_data:
            if data[0] < minX:
                minX = data[0]
            elif data[0] > maxX:
                maxX = data[0]
            if data[1] < minY:
                minY = data[1]
            elif data[1] > maxY:
                maxY = data[1]
        return minX, minY, maxX, maxY

    def _drawMainTrack(self, reference_points):
        image = self.image.copy()
        track_width = 9 * self.scale
        track_color = (100, 100, 100)  # BGR
        last_point = None

        for coordinate in reference_points:
            img_coord = self._calculateImgCoordinate(coordinate)

            if last_point is None:
                last_point = img_coord

            cv2.line(image, last_point, img_coord, track_color, track_width)

            last_point = img_coord

        if self.isLoopTrack:
            cv2.line(image, last_point, self._calculateImgCoordinate(reference_points[0]), track_color, track_width)

        self.image = image

    def applyCoordinatesAndUpdate(self, gps_data, color=(0, 95, 255)):
        self.image = self.applyCoordinates(gps_data, color=color)

    # Coordinates should be a list of tuples
    def applyCoordinates(self, gps_data, color=(0, 95, 255), alpha=0.5):
        image = self.image.copy()

        last_point = None

        for coordinate in gps_data:
            img_coord = self._calculateImgCoordinate(coordinate)

            if last_point is None:
                last_point = img_coord

            cv2.line(image, last_point, img_coord, color, self.run_width)

            last_point = img_coord

        # End point circle
        cv2.line(image, last_point, last_point, color, self.end_width)
        cv2.line(image, self._calculateImgCoordinate(gps_data[0]), self._calculateImgCoordinate(gps_data[0]), color, self.start_width)
        image = cv2.addWeighted(self.image, alpha, image, 1-alpha, 0)
        return image

    def _calculateImgCoordinate(self, env_coordinate):
        return int((env_coordinate[1] - self.minY + self.padding/2)*self.scale), \
               int((env_coordinate[0] - self.minX + self.padding/2)*self.scale)
