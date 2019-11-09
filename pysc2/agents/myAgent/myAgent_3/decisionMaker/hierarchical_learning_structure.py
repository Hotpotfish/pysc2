from pysc2.agents.myAgent.myAgent_3.decisionMaker.DQN import DQN
import pysc2.agents.myAgent.myAgent_3.smart_actions as sa
import pysc2.agents.myAgent.myAgent_3.macro_operation as mo
import numpy as np
from pysc2.lib import actions

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
        self.topDataShape = (None, mo.mapSzie, mo.mapSzie, 11)
        self.controllerDataShape = (None, mo.mapSzie, mo.mapSzie, 2)
        self.top_decision_maker = decision_maker(
            DQN(mu, sigma, learning_rate, len(sa.controllers), self.topDataShape, 'top_decision_maker'))
        self.controllers = []
        for i in range(len(sa.controllers)):
            self.controllers.append(decision_maker(
                DQN(mu, sigma, learning_rate, len(sa.controllers[i]), self.controllerDataShape, 'controller' + str(i))))
            print()

    def get_top_observation(self, obs):

        state = np.array(obs.observation['feature_minimap']).reshape((-1, mo.mapSzie, mo.mapSzie, 11))
        return state

    def choose_controller(self, obs):
        self.top_decision_maker.current_state = self.get_top_observation(obs)

        current_socre = obs.observation['score_cumulative'][0]
        if self.top_decision_maker.previous_action is not None:
            reward = current_socre - self.top_decision_maker.previous_sorce
            self.top_decision_maker.network.perceive(self.top_decision_maker.previous_state,
                                                     self.top_decision_maker.previous_action,
                                                     reward,
                                                     self.top_decision_maker.current_state,
                                                     obs.last())

        controller_number = self.top_decision_maker.network.egreedy_action(self.top_decision_maker.current_state)

        self.top_decision_maker.previous_sorce = current_socre
        self.top_decision_maker.previous_state = self.top_decision_maker.current_state
        self.top_decision_maker.previous_action = controller_number
        return controller_number

    def get_build_observation(self, obs):

        state = np.array(obs.observation['feature_minimap'][0:2]).reshape((-1, mo.mapSzie, mo.mapSzie, 2))
        return state

    def choose_build_macro(self, obs):
        self.controllers[0].current_state = self.get_build_observation(obs)

        current_socre = obs.observation['score_cumulative'][0]
        if self.controllers[0].previous_action is not None:
            reward = current_socre - self.controllers[0].previous_sorce
            self.controllers[0].network.perceive(self.controllers[0].previous_state,
                                                 self.controllers[0].previous_action,
                                                 reward,
                                                 self.controllers[0].current_state)

        macro_number = self.controllers[0].network.egreedy_action(self.controllers[0].current_state)

        self.top_decision_maker.previous_sorce = current_socre
        self.top_decision_maker.previous_state = self.top_decision_maker.current_state
        self.top_decision_maker.previous_action = macro_number
        return sa.controllers[0][macro_number]

    def get_train_observation(self, obs):

        state = np.array(obs.observation['feature_minimap'][2:4]).reshape((-1, mo.mapSzie, mo.mapSzie, 2))
        return state

    def choose_train_macro(self, obs):
        self.controllers[1].current_state = self.get_train_observation(obs)

        current_socre = obs.observation['score_cumulative'][1]
        if self.controllers[1].previous_action is not None:
            reward = current_socre - self.controllers[1].previous_sorce
            self.controllers[1].network.perceive(self.controllers[1].previous_state,
                                                 self.controllers[1].previous_action,
                                                 reward,
                                                 self.controllers[1].current_state)

        macro_number = self.controllers[1].network.egreedy_action(self.controllers[1].current_state)

        self.top_decision_maker.previous_sorce = current_socre
        self.top_decision_maker.previous_state = self.top_decision_maker.current_state
        self.top_decision_maker.previous_action = macro_number
        return sa.controllers[1][macro_number]

    def get_harvest_observation(self, obs):

        state = np.array(obs.observation['feature_minimap'][4:6]).reshape((-1, mo.mapSzie, mo.mapSzie, 2))
        return state

    def choose_harvest_macro(self, obs):
        self.controllers[2].current_state = self.get_harvest_observation(obs)

        current_socre = obs.observation['score_cumulative'][2]
        if self.controllers[2].previous_action is not None:
            reward = current_socre - self.controllers[2].previous_sorce
            self.controllers[2].network.perceive(self.controllers[2].previous_state,
                                                 self.controllers[2].previous_action,
                                                 reward,
                                                 self.controllers[2].current_state)

        macro_number = self.controllers[2].network.egreedy_action(self.controllers[2].current_state)

        self.top_decision_maker.previous_sorce = current_socre
        self.top_decision_maker.previous_state = self.top_decision_maker.current_state
        self.top_decision_maker.previous_action = macro_number
        return sa.controllers[2][macro_number]

    def get_attack_observation(self, obs):

        state = np.array(obs.observation['feature_minimap'][6:8]).reshape((-1, mo.mapSzie, mo.mapSzie, 2))
        return state

    def choose_attack_macro(self, obs):
        self.controllers[3].current_state = self.get_attack_observation(obs)

        current_socre = obs.observation['score_cumulative'][3]
        if self.controllers[3].previous_action is not None:
            reward = current_socre - self.controllers[1].previous_sorce
            self.controllers[3].network.perceive(self.controllers[3].previous_state,
                                                 self.controllers[3].previous_action,
                                                 reward,
                                                 self.controllers[3].current_state)

        macro_number = self.controllers[3].network.egreedy_action(self.controllers[3].current_state)

        self.top_decision_maker.previous_sorce = current_socre
        self.top_decision_maker.previous_state = self.top_decision_maker.current_state
        self.top_decision_maker.previous_action = macro_number
        return sa.controllers[3][macro_number]

    def make_choice(self, obs):

        controller_number = self.choose_controller(obs)

        if controller_number == 0:
            macro = self.choose_build_macro(obs)
            return macro(obs)
        if controller_number == 1:
            macro = self.choose_train_macro(obs)
            return macro(obs)
        if controller_number == 2:
            macro = self.choose_harvest_macro(obs)
            return macro(obs)
        if controller_number == 3:
            macro = self.choose_attack_macro(obs)
            return macro(obs)

        return actions.RAW_FUNCTIONS.no_op()
