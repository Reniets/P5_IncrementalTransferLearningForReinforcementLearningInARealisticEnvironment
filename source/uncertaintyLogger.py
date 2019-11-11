import os
import csv
from source.uncertaintyLog import UncertaintyLog


class UncertaintyLogger:
    def __init__(self):
        self.logs = []
        self.total_frames = 0

    def log(self, seen_observations):
        log = UncertaintyLog(seen_observations)

        if log.len() > self.total_frames:
            self.total_frames = log.len()

        self.logs.append(log)

    def exportAsCsv(self, path, filename):
        # Create the rows to print
        rows = self._createRows()

        # Create directory if it does not exist
        if not os.path.isdir(path):
            os.makedirs(path)

        # Print the rows
        with open(path+os.sep+filename, "w") as file:
            writer = csv.writer(file, seperator=";")
            writer.writelines(rows)

    def _createRows(self):
        rows = []

        for frame_nr in range(self.total_frames):
            frame_data = [f"#{frame_nr}", f"img_{frame_nr}.png"]

            for log in self.logs:
                frame_data.append(log.getCount(frame_nr))

            rows.append(frame_data)

        return rows

