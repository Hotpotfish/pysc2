import numpy as np
import matplotlib.pyplot as plt

from pysc2.agents.myAgent.myAgent_6.config import config
from pysc2.lib import features


class plt_function():

    def __init__(self):
        self.score_cumulative = np.array([])
        self.steps = np.array([])
        self.fig = plt.figure(figsize=(20, 20))

    def add_summary(self, obs, steps):
        self.score_cumulative = np.append(self.score_cumulative, obs.observation['score_cumulative'])
        self.steps = np.append(self.steps, steps)

    def plt_score_cumulative(self):
        i = 0
        self.score_cumulative = self.score_cumulative.reshape((len(self.steps), -1))
        for temp in features.ScoreCumulative.__members__:
            ax = self.fig.add_subplot(config.ROW, config.COLUMN, i + 1)
            ax.set_xlabel('step')
            ax.set_ylabel('score')
            ax.plot(self.steps, self.score_cumulative[:, i],linewidth=2.0)
            ax.set_title(str(temp))
            i += 1

        plt.show()
        self.score_cumulative = np.array([])
        self.steps = np.array([])

        # print()
