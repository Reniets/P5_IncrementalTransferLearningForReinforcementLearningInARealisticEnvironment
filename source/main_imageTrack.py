from source.gps_image import GpsImage2
import os
import numpy as np
import cv2
import glob

def getGpsData(path):
    file = open(path, "r")
    gps_data = []
    line_nr = 0
    for line in file:
        line_nr += 1
        data_split = line.split(";")
        data = np.array([float(data_split[i]) for i in range(2)])
        gps_data.append(data)

    return gps_data[-350:]


def getUE4GpsData(path):
    file = open(path, "r")
    gps_data = []

    for line in file:
        line = line.split(']')[-1]
        data_split = line.split(";")
        data = np.array([float(data_split[i]) for i in range(2)])
        #data[1] += 240
        gps_data.append(data)
    return gps_data


def getBounds(gps_data):
    minX, minY, maxX, maxY = float('inf'), float('inf'), float('-inf'), float('-inf')
    for data in gps_data:
        if data[0] < minX:
            minX =data[0]
        elif data[0] > maxX:
            maxX = data[0]
        if data[1] < minY:
            minY =data[1]
        elif data[1] > maxY:
            maxY = data[1]
    return minX, minY, maxX, maxY


def lerp(val1, val2, d):
    return tuple([x*(1-d)+y*d for x, y in zip(val1, val2)])


def lerp3(val1, val2, val3, d):
    return lerp(val1, val2, d*2) if d<0.5 else lerp(val2, val3, (d-0.5)*2)

#callback = Callback(None)

maps = [f"SelectivePI_FromLevel_{i-1}_ToLevel_{i}_c" for i in range(1, 7)]

base_maps = [f"Level_{i}" for i in range(1,7)]

non_loop_maps = [f"Level_{i}" for i in range(1, 3)]

metrics = ['median', 'max', 'min']
scale = 10

for k in range(2, len(maps)):
    for metric in metrics:
        #print(i)
        map_name = maps[k]
        print(f"{map_name}_{metric}")

        # Export the image
        image_dir = f"GpsData/A_referenceData/{map_name}"
        gps_data = getUE4GpsData(f"GpsData/A_referenceData/{base_maps[k]}/UERef.txt")

        if not os.path.exists(image_dir):
            os.makedirs(image_dir)

        gps_image = GpsImage2(f"{image_dir}/map.png", gps_data, scale=scale,
                              isLoopTrack=False if map_name in non_loop_maps else True)

        start_color = (255, 0, 0)
        mid_color = (0, 255, 0)
        end_color = (0, 0, 255)
        start = 1
        end = len(glob.glob(f"GpsData/{map_name}/gps_{metric}_*"))

        for i in range(start, end+1):
            run_gps_data = getGpsData(f"GpsData/{map_name}/gps_{metric}_{i}.txt")
            gps_image.applyCoordinatesAndUpdate(run_gps_data, color=lerp3(start_color, mid_color, end_color, (i - start) / (end - start)))

        cv2.imwrite(f"{image_dir}/{metric}_{scale}.png", gps_image.image)