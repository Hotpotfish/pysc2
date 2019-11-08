from pysc2.agents.myAgent.myAgent_3.decisionMaker.DQN import DQN
import pysc2.agents.myAgent.myAgent_3.smart_actions as sa
import pysc2.agents.myAgent.myAgent_3.macro_operation as mo
import numpy as np

mu = 0
sigma = 1
learning_rate = 1e-4


class decision_maker():

    def __init__(self, network):
        self.network = network
        self.previous_state = None
        self.previous_action = None
        self.previous_sorce = None
        self.current_state = None


class hierarchical_learning_structure():

    def __init__(self):
        self.dataShape = (None, mo.mapSzie, mo.mapSzie, 10)
        self.top_decision_maker = decision_maker(DQN(mu, sigma, learning_rate, len(sa.controllers), self.dataShape))
        self.controllers = []
        for i in range(len(sa.controllers)):
            self.controllers.append(decision_maker(DQN(mu, sigma, learning_rate, len(sa.controllers), self.dataShape)))

    def get_top_observation(self, obs):
        state = np.array(obs.observation['feature_minimap']).reshape(self.dataShape)
        return state

    def choose_controller(self, obs):
        self.topState = self.get_top_observation(obs)
        controller_select = self.top_decision_maker.network.egreedy_action(self.topState)

        current_socre = obs.observation['ScoreCumulative']
        if self.top_decision_maker.previous_action is not None:
            reward = current_socre - self.top_decision_maker.previous_sorce
            self.top_decision_maker.network.perceive

            self.top_decision_maker.network.egreedy_action






        # for i in range(len(sa.controllers)):
        #     self.controllers.append(DQN(mu, sigma, learning_rate, len(sa.controllers), self.dataShape))
