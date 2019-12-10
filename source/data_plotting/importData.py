from database.sql import Sql
import csv
from source.data_plotting.graphData import sessions
import numpy as np

def importFromDataBase():
    sql = Sql()

    for sessionList in sessions.values():
        for name, sessionId in sessionList.items():
            for eval in range(2):
                fileName = name+'.csv' if eval == 0 else name+'_eval.csv'

                with open('../../data/' + fileName, mode='w', newline='') as csvFile:
                    dataWriter = csv.writer(csvFile, delimiter=',')

                    fieldnames = ['instance', 'episode', 'reward']
                    dataWriter.writerow(fieldnames)

                    data = sql.SELECT_trainingData(sessionId, eval)

                    for dataRow in data:
                        dataWriter.writerow(dataRow)


def accumulateData():
    for sessionList in sessions.values():
        for name in sessionList.keys():
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


if __name__ == '__main__':
    importFromDataBase()
    accumulateData()
