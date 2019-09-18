import os
import signal
import subprocess
import time
from subprocess import Popen
from setup.utils import CARLA_PATH, makeCarlaImportable
from source.data_handler import processData

makeCarlaImportable()  # Defines a path to your carla folder which makes it visible to import
import carla


class CarlaEnvironment:
    def __init__(self, port: str):
        self.IM_WIDTH = 400
        self.IM_HEIGHT = 200
        self.actorList = {}
        self.port = port
        self.process = None
        self.world = None
        self.blueprintLibrary = None

    def create(self):
        args = [CARLA_PATH + '/./CarlaUE4.sh', '-carla-settings="../CarlaSettings.ini"', '-world-port=' + self.port]

        # Open simulation
        self.process = Popen(args, shell=True, preexec_fn=os.setsid)
        time.sleep(4)  # If DISPLAY is off, sleep longer (10 secs)

        # Change weather to clear sunset
        subprocess.Popen("./config.py --weather ClearSunset", cwd=CARLA_PATH + "/PythonAPI/util/", shell=True)

    def close(self):
        os.killpg(os.getpgid(self.process.pid), signal.SIGKILL)
        os.killpg(os.getpgid(self.process.pid), signal.SIGTERM)

    def connectClient(self):
        client = carla.Client('localhost', int(self.port))
        client.set_timeout(10.0)
        self.world = client.get_world()
        self.blueprintLibrary = self.world.get_blueprint_library()

    def setFixedTimeSteps(self, timeStep=0.1):
        settings = self.world.get_settings()
        settings.fixed_delta_seconds = timeStep  # 10 fps MAX, physics will break at lower fps(TODO: Look a synchronous)
        self.world.apply_settings(settings)

    def spawnVehicle(self, spawnPoint=2):
        vehicleBlueprint = self.blueprintLibrary.filter("model3")[0]  # Choose tesla as vehicle actor
        transform = self.world.get_map().get_spawn_points()[spawnPoint]  # Pick some predictable spawn point
        vehicle = self.world.spawn_actor(vehicleBlueprint, transform)
        self.actorList['model3'] = vehicle

        return vehicle

    def spawnSemanticSegmentationSensor(self, vehicle):
        cameraBlueprint = self.blueprintLibrary.find('sensor.camera.semantic_segmentation')
        cameraBlueprint.set_attribute('image_size_x', str(self.IM_WIDTH))
        cameraBlueprint.set_attribute('image_size_y', str(self.IM_HEIGHT))
        cameraBlueprint.set_attribute('fov', '110')
        cameraBlueprint.set_attribute('sensor_tick', '0.1')

        relativeTransformSensor = carla.Transform(carla.Location(x=2.5, z=0.7))  # Place on the front of car

        camSensor = self.world.spawn_actor(cameraBlueprint, relativeTransformSensor, attach_to=vehicle)
        self.actorList['semanticSegmentationSensor'] = camSensor

        camSensor.listen(lambda data: processData(data))  # Collect sensor data

    def applyAction(self, vehicle, action=carla.VehicleControl(throttle=0.5, steer=0, brake=0)):
        vehicle.apply_control(action)

    def setupTestEnvironment(self):
        self.create()
        self.connectClient()
        vehicle = self.spawnVehicle()
        self.spawnSemanticSegmentationSensor(vehicle)
        self.applyAction(vehicle, action=carla.VehicleControl(throttle=1))
