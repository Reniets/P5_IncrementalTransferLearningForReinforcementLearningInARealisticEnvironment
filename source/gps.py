class Gps:
    def __init__(self, carla_vehicle):
        self.carla_environment = carla_vehicle
        self.location_log = []

    def log_location(self):
        location = self.get_location()
        self.location_log.append(location)

    def get_location(self):
        return self.carla_environment.vehicle.get_location()

    def export_location_log(self, file_path):
        file = open(file_path, "w+")

        for location in self.location_log:
            file.write(self._convertLocationToText(location) + "\n")

        file.close()

    def reset(self):
        self.location_log = []

    def _convertLocationToText(self, location):
        return f"{location.x};{location.y};{location.z}"
