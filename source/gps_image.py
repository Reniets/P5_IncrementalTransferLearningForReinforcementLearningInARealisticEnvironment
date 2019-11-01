import cv2
import numpy as np


class GpsImage:
    # Reference point structure: tuple(np(img_cords), np(env_cords)).
    # We expect there to be two img cords and two related env cords
    def __init__(self, img_path, reference_points, mode=cv2.IMREAD_COLOR):
        self.track_color = (0, 95, 255)     # BGR
        self.start_color = (150, 0, 0)    # BGR
        self.end_color   = (0, 0, 255)      # BGR

        self.track_width = 3                # Pixels
        self.start_width = 15               # Pixels
        self.end_width   = 15               # Pixels

        self.image = cv2.imread(img_path, mode)

        self.img_base = reference_points[0][0]
        self.env_base = reference_points[1][0]
        self.scales = self._calculateScales(reference_points)

    def _calculateScales(self, reference_points):
        # Separate img and env reference points
        img_cords = reference_points[0]
        env_cords = reference_points[1]

        # Calculate x and y scales (Assumes linear dependency eg. no perspective warping)
        img_cord_rel_1 = self._getImgCordRelativeToBase(img_cords[0])
        img_cord_rel_2 = self._getImgCordRelativeToBase(img_cords[1])

        env_cord_rel_1 = self._getImgCordRelativeToBase(env_cords[0])
        env_cord_rel_2 = self._getImgCordRelativeToBase(env_cords[1])

        return (img_cord_rel_1 - img_cord_rel_2) / (env_cord_rel_1 - env_cord_rel_2)

    def _getImgCordRelativeToBase(self, img_cord):
        return self._getCordRelativeToBase(img_cord, self.img_base)

    def _getEnvCordRelativeToBase(self, env_cord):
        return self._getCordRelativeToBase(env_cord, self.env_base)

    def _getCordRelativeToBase(self, cord, base):
        return np.array([cord[i] - base[i] for i in range(2)])

    # Coordinates should be a list of tuples
    def applyCoordinates(self, gps_data):
        image = self.image.copy()

        last_point = None

        for coordinate in gps_data:
            img_coord = tuple(self._calculateImgCoordinate(coordinate))

            if last_point is None:
                last_point = img_coord

            line_width = self.start_width if last_point == img_coord else self.track_width
            track_color = self.start_color if last_point == img_coord else self.track_color

            cv2.line(image, last_point, img_coord, track_color, line_width)

            last_point = img_coord

        # End point circle
        cv2.line(image, last_point, last_point, self.end_color, self.end_width)

        return image

    def _calculateImgCoordinate(self, env_coordinate):
        env_rel = self._getEnvCordRelativeToBase(env_coordinate)    # Get Coordinates relative to the environment base
        env_trans = self.scales * env_rel                           # Calculate the new pixel values using the transform scales
        new_cords = self.img_base + env_trans                       # Get the new cords relative to the img base

        return (int(n) for n in new_cords)
