from database.sql import Sql
import csv
from source.data_plotting.graphData import sessions
# Sessionids:

# Instance, Episode, Reward, Evaluation
sql = Sql()

for name, sessionId in sessions.items():
    for eval in range(2):
        fileName = name+'.csv' if eval == 0 else name+'_eval.csv'

        with open('../../data/' + fileName, mode='w', newline='') as csvFile:
            dataWriter = csv.writer(csvFile, delimiter=',')

            fieldnames = ['instance', 'episode', 'reward']
            dataWriter.writerow(fieldnames)

            data = sql.SELECT_trainingData(sessionId, eval)

            for dataRow in data:
                dataWriter.writerow(dataRow)
