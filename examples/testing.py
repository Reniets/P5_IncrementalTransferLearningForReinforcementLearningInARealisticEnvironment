#!/usr/bin/env python

# Copyright (c) 2019 Computer Vision Center (CVC) at the Universitat Autonoma de
# Barcelona (UAB).
#
# This work is licensed under the terms of the MIT license.
# For a copy, see <https://opensource.org/licenses/MIT>.

import time
import os
from os.path import isfile, join
import cv2
import datetime
from examples.carla_environment_tensorforce import CarlaEnvironmentTensorforce


def makeVideoFromSensorFrames(fps=10, pathIn="../data/frames/", pathOut="../data/videos/" + str(datetime.datetime.now().strftime("%Y-%m-%d %H:%M")) + ".avi"):
    frame_array = []
    files = [f for f in os.listdir(pathIn) if isfile(join(pathIn, f))]
    files.sort()

    for i in range(len(files)):
        filename = pathIn + files[i]
        # reading each files
        img = cv2.imread(filename)

        height, width, layers = img.shape
        size = (width, height)

        # inserting the frames into an image array
        frame_array.append(img)

    out = cv2.VideoWriter(pathOut, cv2.VideoWriter_fourcc(*'DIVX'), fps, size)

    for i in range(len(frame_array)):
        # writing to a image array
        out.write(frame_array[i])
    out.release()


def clearFrameFolder():
    folder_path = '../data/frames'
    for frame_file in os.listdir(folder_path):
        file_path = os.path.join(folder_path, frame_file)
        try:
            if os.path.isfile(file_path):
                os.unlink(file_path)
        except Exception as e:
            print(e)


def runMultipleCarlaServers(num=2):
    for i in range(num):
        CarlaEnvironmentTensorforce(str(2000 + i * 3))


def runSingleCarlaServer():
    abc = CarlaEnvironmentTensorforce('2000')
    input("Press Enter to close...")
    abc.close()


# Actual execution
#clearFrameFolder()              # Clear so no previous frames ruin the video
runSingleCarlaServer()          # Run single car environment, and capture frames from sensor (exported to data/frames)
# makeVideoFromSensorFrames()     # Concat all images in data/frames into a video
