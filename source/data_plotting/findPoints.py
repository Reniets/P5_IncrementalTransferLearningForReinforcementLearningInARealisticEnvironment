import csv
from source.data_plotting.graphData import sessions
import numpy as np


def findSlope():
    slopesBase = np.array([])
    slopesTransfer = np.array([0.0])

    for name in sessions.keys():
        episodesAccumMean = getEpisodesAccumMean(name)
        y1 = episodesAccumMean[-10]
        y2 = episodesAccumMean[-1]
        x1 = len(episodesAccumMean)-10
        x2 = len(episodesAccumMean)

        slope = int((y2-y1)/(x2-x1))

        if name.find('Transfer') != -1:
            slopesTransfer = np.append(slopesTransfer, slope)
        else:
            slopesBase = np.append(slopesBase, slope)
        print(f'Slope: {slope}')

    print(f'Base: {slopesBase},\nTransfer{slopesTransfer},\nDiff: {slopesTransfer-slopesBase}')


def findZeroCrossing():
    for name in sessions.keys():
        episodesAccumMean = getEpisodesAccumMean(name)

        aboveZeros = []
        for k, v in enumerate(episodesAccumMean):
            if v > 0:
                aboveZeros.append((k, v))

        print(f'Zero crossings: {aboveZeros[0][0]+1}')


def findMinimum():
    for name in sessions.keys():
        episodesAccumMean = getEpisodesAccumMean(name)

        print(f'Name: {name} - Ep min: {np.argmin(episodesAccumMean)+1}')


def getEpisodesAccumMean(name):
    with open(f'../../data/{name}_accum.csv', mode='r') as fromFile:
        hasHeader = csv.Sniffer().sniff(fromFile.read(1024))
        fromFile.seek(0)
        fromFileReader = csv.reader(fromFile, delimiter=',')

        if hasHeader:
            next(fromFileReader)

        episodesAccumMean = []
        curEp = 0
        for row in fromFileReader:
            episode = int(row[1])
            reward = float(row[2])

            if curEp != episode:
                episodesAccumMean.append(reward)
                curEp = episode
            else:
                episodesAccumMean[episode-1] += reward

        return np.array(episodesAccumMean)/21


if __name__ == '__main__':
    findMinimum()
    findZeroCrossing()
    findSlope()
