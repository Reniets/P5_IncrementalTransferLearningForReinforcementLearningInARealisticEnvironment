import os
from os.path import isfile, join
import cv2
import datetime

from gym_carla.carla_utils import makeCarlaImportable

makeCarlaImportable()  # Defines a path to your carla folder which makes it visible to import
import carla


def makeVideoFromSensorFrames(fps=10):
    pathIn = "../data/frames/"
    pathOut = "../data/videos/"
    fileName = str(datetime.datetime.now().strftime("%Y-%m-%d %H:%M")) + ".avi"

    # Ensure that data path is present, otherwise create it
    # TODO: Find a better way to do this
    if not os.path.isdir("../data"):
        os.mkdir("../data")
    if not os.path.isdir(pathIn):
        os.mkdir(pathIn)
    if not os.path.isdir(pathOut):
        os.mkdir(pathOut)

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

    out = cv2.VideoWriter(pathOut + fileName, cv2.VideoWriter_fourcc(*'DIVX'), fps, size)

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


def processData(data):
    cc = carla.ColorConverter.CityScapesPalette

    # Save images to disk (Output folder)
    data.save_to_disk('../data/frames/%06d.png' % data.frame, cc)

    # Get Specific segment values
    '''
    im_raw = data.raw_data
    im_raw = np.reshape(im_raw, (self.IM_HEIGHT, self.IM_WIDTH, 4))
    im_raw_seg = im_raw[:, :, 2]  #Select the R-value to get the segment value
    print(im_raw_seg) #View link to find enumeration categories: https://carla.readthedocs.io/en/latest/cameras_and_sensors/#sensorcamerasemantic_segmentation
    '''

    # Show image plots of segment images
    '''data.convert(cc)
    im_raw_segment = data.raw_data
    im_raw_segment = np.reshape(im_raw_segment, (self.IM_HEIGHT, self.IM_WIDTH, 4))
    im_raw_segment = im_raw_segment[:, :, :3]
    plt.imshow(im_raw_segment, interpolation='nearest')
    plt.interactive(False)
    plt.axis('off')
    plt.show(block=False)'''