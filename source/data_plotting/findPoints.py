import csv
from source.data_plotting.graphData import sessions
import numpy as np


def findJumpStart():
    jumpStartBase = None
    jumpStartTransfer = None

    for sessionType, sessionList in sessions.items():
        if sessionType == 'Base':
            jumpStartBase = np.array([])
        else:
            jumpStartTransfer = np.array([0.0])  # Initial element is just for level 1 which is not defined in transfer

        for name in sessionList.keys():
            episodesAccumMean = getEpisodesAccumMean(name)

            jump_start = episodesAccumMean[4]/5

            if sessionType == 'Base':
                jumpStartBase = np.append(jumpStartBase, jump_start)
            else:
                jumpStartTransfer = np.append(jumpStartTransfer, jump_start)

            #print(f'Jumpstart: {jump_start}')

        if sessionType != 'Base':
            print(f'{sessionType} - Jumpstart Diff: {(jumpStartTransfer-jumpStartBase)[1:]}')


def findSlope():
    slopesBase = None
    slopesTransfer = None

    for sessionType, sessionList in sessions.items():
        if sessionType == 'Base':
            slopesBase = np.array([])
        else:
            slopesTransfer = np.array([0.0])  # Initial element is just for level 1 which is not defined in transfer

        for name in sessionList.keys():
            episodesAccumMean = getEpisodesAccumMean(name)
            greatest_slope = float('-inf')
            for x1 in range(len(episodesAccumMean)-10):
                x2 = x1+10
                y1 = episodesAccumMean[x1]
                y2 = episodesAccumMean[x2]
                slope = int((y2-y1)/(x2-x1))
                if slope > greatest_slope:
                    greatest_slope = slope

            if sessionType == 'Base':
                slopesBase = np.append(slopesBase, greatest_slope)
            else:
                slopesTransfer = np.append(slopesTransfer, greatest_slope)
            #print(f'Slope: {slope}')

        if sessionType != 'Base':
            print(f'{sessionType} - Slope Diff: {(slopesTransfer-slopesBase)[1:]}, Percentages: {[f"{(ratio-1)*100}%" for ratio in slopesTransfer/slopesBase][1:]}')


def findZeroCrossing():
    for sessionType, sessionList in sessions.items():
        epZeroCrossings = []

        for name in sessionList.keys():
            episodesAccumMean = getEpisodesAccumMean(name)

            aboveZeros = []
            for k, v in enumerate(episodesAccumMean):
                if v > 0:
                    aboveZeros.append((k, v))

            epZeroCrossings.append(aboveZeros[0][0]+1)

        print(f'{sessionType} - Zero crossings: {epZeroCrossings}')


def findMinimum():
    for sessionType, sessionList in sessions.items():
        epMins = []

        for name in sessionList.keys():
            episodesAccumMean = getEpisodesAccumMean(name)
            epMins.append(np.argmin(episodesAccumMean)+1)

        print(f'{sessionType} - Mins: {epMins}')


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
    findSlope()
    findMinimum()
    findZeroCrossing()
    findJumpStart()
