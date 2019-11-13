from pysc2.agents.myAgent.myAgent_4.decisionMaker.DQN import DQN
import pysc2.agents.myAgent.myAgent_4.smart_actions as sa
import pysc2.agents.myAgent.myAgent_4.macro_operation as mo
import numpy as np
from pysc2.lib.named_array import NamedNumpyArray as NNA

from pysc2.env.environment import StepType
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
        self.DataShape = (None, mo.mapSzie, mo.mapSzie, 39)
        # self.controllerDataShape = (None, mo.mapSzie, mo.mapSzie, 2)
        self.top_decision_maker = decision_maker(
            DQN(mu, sigma, learning_rate, len(sa.controllers), self.DataShape, 'top_decision_maker'))
        self.controllers = []
        for i in range(len(sa.controllers)):
            self.controllers.append(decision_maker(
                DQN(mu, sigma, learning_rate, len(sa.controllers[i]), self.DataShape, 'controller' + str(i))))
            print()

    def my_flatten(self,input_list):
        output_list = []
        while True:
            if input_list == []:
                break
            for index, i in enumerate(input_list):

                if type(i) == list:
                    input_list = i + input_list[index + 1:]
                    break
                else:
                    output_list.append(i)
                    input_list.pop(index)
                    break

        return output_list
    def get_all_observation(self, obs):
        state_layers = []
        non_serial_layer = []
        for key, value in obs.observation.items():

            if len(value) == 0:
                continue

            if key == 'feature_minimap' or key == 'feature_screen':
                for i in range(len(value)):
                    state_layers.append(np.array(value[i]))
                continue
            if type(value) is not str:
                value = value.tolist()
                non_serial_layer.append(value)
                # value = (np.array(value)).flatten()
                # non_serial_layer.append(value)

        non_serial_layer = np.array(self.my_flatten(non_serial_layer))
        number = len(non_serial_layer)
        dataSize = pow(mo.mapSzie, 2)
        loop = int(number / dataSize) + 1
        for i in range(loop):
            layer = np.zeros(shape=(dataSize,))
            if i != loop - 1:
                start = i * dataSize
                end = (i + 1) * dataSize
                layer = non_serial_layer[start:end]
                layer = layer.reshape((mo.mapSzie, mo.mapSzie))
                state_layers.append(layer)
                continue
            layer[0:(number - i * dataSize)] = non_serial_layer[i * dataSize: number]
            layer = layer.reshape((mo.mapSzie, mo.mapSzie))
            state_layers.append(layer)
        return np.array(state_layers).reshape((-1, mo.mapSzie, mo.mapSzie, 39))


    def get_top_observation(self, obs):

        state = np.array(obs.observation['feature_minimap']).reshape((-1, mo.mapSzie, mo.mapSzie, 11))
        return state

    def choose_controller(self, obs, mark):
        self.top_decision_maker.current_state = self.get_all_observation(obs)
        if mark == 'train':
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
        elif mark == 'test':
            return self.top_decision_maker.network.action(self.top_decision_maker.current_state)

    def get_build_observation(self, obs):

        state = np.array(obs.observation['feature_minimap'][0:2]).reshape((-1, mo.mapSzie, mo.mapSzie, 2))
        return state

    def choose_build_macro(self, obs, mark):
        self.controllers[0].current_state = self.get_all_observation(obs)

        if mark == 'train':

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
        elif mark == 'test':
            return sa.controllers[0][self.controllers[0].network.action(self.controllers[0].current_state)]

    def get_train_observation(self, obs):

        state = np.array(obs.observation['feature_minimap'][2:4]).reshape((-1, mo.mapSzie, mo.mapSzie, 2))
        return state

    def choose_train_macro(self, obs, mark):
        self.controllers[1].current_state = self.get_all_observation(obs)

        if mark == 'train':
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
        elif mark == 'test':
            return sa.controllers[1][self.controllers[1].network.action(self.controllers[1].current_state)]

    def get_harvest_observation(self, obs):

        state = np.array(obs.observation['feature_minimap'][4:6]).reshape((-1, mo.mapSzie, mo.mapSzie, 2))
        return state

    def choose_harvest_macro(self, obs, mark):
        self.controllers[2].current_state = self.get_all_observation(obs)

        if mark == 'train':
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
        elif mark == 'test':
            return sa.controllers[2][self.controllers[2].network.action(self.controllers[2].current_state)]

    def get_attack_observation(self, obs):

        state = np.array(obs.observation['feature_minimap'][6:8]).reshape((-1, mo.mapSzie, mo.mapSzie, 2))
        return state

    def choose_attack_macro(self, obs, mark):
        self.controllers[3].current_state = self.get_all_observation(obs)

        if mark == 'train':

            current_socre = obs.observation['score_cumulative'][3]
            if self.controllers[3].previous_action is not None:
                reward = current_socre - self.controllers[3].previous_sorce
                self.controllers[3].network.perceive(self.controllers[3].previous_state,
                                                     self.controllers[3].previous_action,
                                                     reward,
                                                     self.controllers[3].current_state)

            macro_number = self.controllers[3].network.egreedy_action(self.controllers[3].current_state)

            self.top_decision_maker.previous_sorce = current_socre
            self.top_decision_maker.previous_state = self.top_decision_maker.current_state
            self.top_decision_maker.previous_action = macro_number
            return sa.controllers[3][macro_number]
        elif mark == 'test':
            return sa.controllers[3][self.controllers[3].network.action(self.controllers[3].current_state)]

    def make_choice(self, obs, mark):

        if obs[0] == StepType.FIRST:
            return actions.RAW_FUNCTIONS.raw_move_camera((mo.mapSzie / 2, mo.mapSzie / 2))

        controller_number = self.choose_controller(obs, mark)

        if controller_number == 0:
            macro = self.choose_build_macro(obs, mark)
            return macro(obs)
        if controller_number == 1:
            macro = self.choose_train_macro(obs, mark)
            return macro(obs)
        if controller_number == 2:
            macro = self.choose_harvest_macro(obs, mark)
            return macro(obs)
        if controller_number == 3:
            macro = self.choose_attack_macro(obs, mark)
            return macro(obs)

        return actions.RAW_FUNCTIONS.no_op()
