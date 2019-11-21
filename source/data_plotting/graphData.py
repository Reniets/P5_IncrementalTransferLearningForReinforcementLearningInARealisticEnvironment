import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.ticker import MultipleLocator, FormatStrFormatter
from pandas import read_csv

sessions = {
    'Base': {
        'Base_1': 1060,
        'Base_2': 1061,
        'Base_3': 1080,
        'Base_4': 1063,
        'Base_5': 1064,
        'Base_6': 1069
    },
    'DirectSingle': {
        'SingleDirect_1-2': 1071,
        'SingleDirect_2-3': 1072,
        'SingleDirect_3-4': 1073,
        'SingleDirect_4-5': 1074,
        'SingleDirect_5-6': 1081
    },
    'DirectContinuous': {
        'SingleContinuous_1-2': 1278,
        'SingleContinuous_2-3': 1279,
        'SingleContinuous_3-4': 1280,
        'SingleContinuous_4-5': 1281,
        'SingleContinuous_5-6': 1282
    },
    'ImitationSingle': {
        'ImitationDirect_1-2': 1286,
        'ImitationDirect_2-3': 1290,
        'ImitationDirect_3-4': 1291,
        'ImitationDirect_4-5': 1292,
        'ImitationDirect_5-6': 1293
    },
    'ImitationContinuous': {
        'ImitationDirect_1-2': 1286,
        'ImitationContinuous_2-3': 1303,
        'ImitationContinuous_3-4': 1304,
        'ImitationContinuous_4-5': 1299,
        'ImitationContinuous_5-6': 1301
    }
}

if __name__ == '__main__':
    sns.set()
    sns.set_context("paper")
    sessionIndexes = ['Base', 'ImitationSingle', 'ImitationContinuous']
    eval = 0

    for accum in range(2):
        fig, axes = plt.subplots(3, 2)

        for sessionType in sessionIndexes:
            sessionList = sessions[sessionType]

            if sessionType != 'Base':
                index = 1
                level = 2
            else:
                index = 0
                level = 1

            for sessionName in sessionList.keys():
                name = sessionName if eval is 0 else sessionName + '_eval'

                sessionPath = f'../../data/{name}.csv' if accum == 0 else f'../../data/{name}_accum.csv'
                data = read_csv(sessionPath)  # , nrows=100)

                sns.lineplot(x='episode', y='reward', data=data, ax=axes[index // 2, index % 2])

                axes[index // 2, index % 2].set_title(f'Level {level}')
                axes[index // 2, index % 2].xaxis.set_major_locator(MultipleLocator((index + 1) * 100))
                axes[index // 2, index % 2].xaxis.set_major_formatter(FormatStrFormatter('%d'))

                if accum == 0:
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
        savePath = '../../data/BaseResults' if accum == 0 else '../../data/BaseResultsAccum'
        fig.savefig(savePath)
