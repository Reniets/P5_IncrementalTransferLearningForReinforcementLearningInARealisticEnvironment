import csv
from source.data_plotting.graphData import sessions
import numpy as np

for name, sessionId in sessions.items():
    with open(f'../../data/{name}.csv', mode='r') as fromFile:
        with open(f'../../data/{name}_accum.csv', mode='w', newline='') as toFile:
            hasHeader = csv.Sniffer().sniff(fromFile.read(1024))
            fromFile.seek(0)
            fromFileReader = csv.reader(fromFile, delimiter=',')
            toFileWriter = csv.writer(toFile, delimiter=',')
            toFileWriter.writerow(['instance', 'episode', 'reward'])

            if hasHeader:
                next(fromFileReader)

            prev = np.zeros(21)
            for row in fromFileReader:
                instance = int(row[0])
                reward = float(row[2])
                prev[instance] += reward
                toFileWriter.writerow([row[0], row[1], prev[instance]])
