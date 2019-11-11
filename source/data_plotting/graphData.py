import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.ticker import MultipleLocator, FormatStrFormatter
from pandas import read_csv

sessions = {
        'Base_1':           1060,
        'Base_2':           1061,
        'Base_3':           1080,
        'Base_4':           1063,
        'Base_5':           1064,
        'Base_6':           1069,
        'Imitation_Transfer_1-2':     1286,
        'Imitation_Transfer_2-3':     1290,
        'Imitation_Transfer_3-4':     1291,
        'Imitation_Transfer_4-5':     1292,
        'Imitation_Transfer_5-6':     1293
        # 'Transfer_1-2':     1071,
        # 'Transfer_2-3':     1072,
        # 'Transfer_3-4':     1073,
        # 'Transfer_4-5':     1074,
        # 'Transfer_5-6':     1081
    }

if __name__ == '__main__':
    sns.set()
    sns.set_context("paper")

    fig, axes = plt.subplots(3, 2)
    index = 0
    firstTransfer = False
    eval = 0
    level = 1

    for sessionName in sessions.keys():
        if not firstTransfer and sessionName.find('Transfer') is not -1:
            firstTransfer = True
            index = 1
            level = 2

        name = sessionName if eval is 0 else sessionName + '_eval'
        data = read_csv(f'../../data/{name}.csv')

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
    fig.savefig('../../data/BaseResults')
