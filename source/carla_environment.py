import os
import signal
import subprocess
import time
import weakref
from subprocess import Popen
from source.data_handler import processData
from gym_carla import settings
from gym_carla.carla_utils import makeCarlaImportable
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
        args = [settings.CARLA_PATH + '/./CarlaUE4.sh', '-carla-settings="../CarlaSettings.ini"', '-world-port=' + self.port]

        # Open simulation
        self.process = Popen(args, shell=True, preexec_fn=os.setsid)
        time.sleep(4)  # If DISPLAY is off, sleep longer (10 secs)

        # Change weather to clear sunset
        subprocess.Popen("./config.py --weather ClearSunset", cwd=settings.CARLA_PATH + "/PythonAPI/util/", shell=True)

    def close(self):
        os.killpg(os.getpgid(self.process.pid), signal.SIGKILL)
        os.killpg(os.getpgid(self.process.pid), signal.SIGTERM)

    def connectClient(self):
        client = carla.Client('localhost', int(self.port))
        client.set_timeout(10.0)
        client.load_world('Straight3')
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

    def spawnCollisionSensor(self, vehicle):
        collisionBlueprint = self.blueprintLibrary.find('sensor.other.safe_distance')
        collisionBlueprint.set_attribute('safe_distance_down', '1.0')
        collisionBlueprint.set_attribute('safe_distance_front', '0.0')
        collisionBlueprint.set_attribute('safe_distance_back', '0.0')
        collisionBlueprint.set_attribute('safe_distance_lateral', '0.0')

        collisionSensor = self.world.spawn_actor(collisionBlueprint, carla.Transform(), attach_to=vehicle)
        self.actorList['collisionSensor'] = collisionSensor
        #weak_self = weakref.ref(self)
        #self.sensor.listen(lambda event: CollisionSensor._on_collision(weak_self, event))
        collisionSensor.listen(lambda event: self.collisionCallback(event))

    @staticmethod
    def collisionCallback(event):
        for e in event:
            print(str(e))
        #    for actor_id in event:
        #    actor = self.world.get_actor(actor_id)
        #    print('Actor too close: %s' % actor.type_id)

    def applyAction(self, vehicle, action=carla.VehicleControl(throttle=0.5, steer=0, brake=0)):
        vehicle.apply_control(action)

    def setupTestEnvironment(self):
        self.create()
        self.connectClient()
        vehicle = self.spawnVehicle()
        self.spawnSemanticSegmentationSensor(vehicle)
        self.spawnCollisionSensor(vehicle)
        self.applyAction(vehicle, action=carla.VehicleControl(throttle=1, steer=0.5))
