import os
import signal
import subprocess
import time
from subprocess import Popen
from setup.utils import CARLA_PATH, makeCarlaImportable
from tensorforce.environments import Environment

makeCarlaImportable()  # Defines a path to your carla folder which makes it visible to import
import carla


class CarlaEnvironmentTensorforce(Environment):
    def __init__(self, port: str):
        # Open simulation
        self.p = Popen(CARLA_PATH + '/./CarlaUE4.sh -carla-settings="../CarlaSettings.ini" -world-port=' + port, shell=True, preexec_fn=os.setsid)
        time.sleep(5)  # If DISPLAY is off, sleep longer (10 secs)
        subprocess.Popen("./config.py --weather ClearSunset", cwd=CARLA_PATH + "/PythonAPI/util/", shell=True)
        self.IM_WIDTH = 400
        self.IM_HEIGHT = 200

        self.actor_list = []

        client = carla.Client('localhost', int(port))
        client.set_timeout(10.0)
        world = client.get_world()
        blueprint_library = world.get_blueprint_library()

        settings = world.get_settings()
        settings.fixed_delta_seconds = 0.1  # 10 fps MAX, physics will break at lower fps (TODO: Look a synchronous)
        world.apply_settings(settings)

        vehicle = blueprint_library.filter("model3")[0]  # Choose tesla as vehicle actor

        transform = world.get_map().get_spawn_points()[2]  # Pick some predictable spawn point
        vehicle = world.spawn_actor(vehicle, transform)
        self.actor_list.append(vehicle)

        time.sleep(5)

        vehicle.apply_control(carla.VehicleControl(throttle=1.0))

        camera_bp = blueprint_library.find('sensor.camera.semantic_segmentation')
        camera_bp.set_attribute('image_size_x', str(self.IM_WIDTH))
        camera_bp.set_attribute('image_size_y', str(self.IM_HEIGHT))
        camera_bp.set_attribute('fov', '110')
        camera_bp.set_attribute('sensor_tick', '0.1')

        relative_transform_cam = carla.Transform(carla.Location(x=2.5, z=0.7))
        cam_sensor = world.spawn_actor(camera_bp, relative_transform_cam, attach_to=vehicle)
        self.actor_list.append(cam_sensor)

        cam_sensor.listen(lambda data: self.__process_data(data))

    def states(self):
        pass

    def execute(self, actions):
        pass

    def actions(self):
        pass

    def reset(self):
        for actor in self.actor_list:
            actor.destroy()

    def close(self):
        os.killpg(os.getpgid(self.p.pid), signal.SIGKILL)
        os.killpg(os.getpgid(self.p.pid), signal.SIGTERM)

    def __process_data(self, data):
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
