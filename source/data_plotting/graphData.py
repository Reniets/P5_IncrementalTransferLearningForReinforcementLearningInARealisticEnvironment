import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.ticker import MultipleLocator, FormatStrFormatter
from pandas import read_csv

sessions = {
    'Base': {
        'Base_1':           1060,
        'Base_2':           1061,
        'Base_3':           1080,
        'Base_4':           1063,
        'Base_5':           1064,
        'Base_6':           1069
    },
    'Direct': {
        'Transfer_1-2d':    1071,
        'Transfer_2-3d':    1072,
        'Transfer_3-4d':    1073,
        'Transfer_4-5d':    1074,
        'Transfer_5-6d':    1081
    },
    'Continuous': {
        'Transfer_1-2c':    1278,
        'Transfer_2-3c':    1279,
        'Transfer_3-4c':    1280,
        'Transfer_4-5c':    1281,
        'Transfer_5-6c':    1282
    },
}

if __name__ == '__main__':
    sns.set()
    sns.set_context("paper")

    fig, axes = plt.subplots(3, 2)
    eval = 0

    for sessionType, sessionList in sessions.items():
        if sessionType != 'Base':
            index = 1
            level = 2
        else:
            index = 0
            level = 1

        for sessionName in sessionList.keys():
            print(sessionList.keys())
            name = sessionName if eval is 0 else sessionName + '_eval'
            data = read_csv(f'../../data/{name}.csv') #, nrows=100)

            sns.lineplot(x='episode', y='reward', data=data, ax=axes[index // 2, index % 2])

            axes[index // 2, index % 2].set_title(f'Level {level}')
            axes[index // 2, index % 2].xaxis.set_major_locator(MultipleLocator((index+1) * 100))
            axes[index // 2, index % 2].xaxis.set_major_formatter(FormatStrFormatter('%d'))
            axes[index // 2, index % 2].yaxis.set_major_locator(MultipleLocator(1000))
            axes[index // 2, index % 2].yaxis.set_major_formatter(FormatStrFormatter('%d'))
            fig.show()

            index += 1
            level += 1

    # Hide x labels and tick labels for top plots and y ticks for right plots.
    for ax in axes.flat:
        ax.set_ylabel('')
        ax.set_xlabel('')

    fig.show()
    fig.savefig('../../data/BaseResultsT')
