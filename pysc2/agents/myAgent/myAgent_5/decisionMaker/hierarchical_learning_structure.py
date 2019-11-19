import datetime
import inspect
import os

from pysc2.agents.myAgent.myAgent_5.decisionMaker.DQN import DQN
import pysc2.agents.myAgent.myAgent_5.smart_actions as sa
import pysc2.agents.myAgent.myAgent_5.macro_operation as mo
import numpy as np

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
        self.previous_reward = None
        self.current_state = None
        self.load_and_train = True


class hierarchical_learning_structure():

    def __init__(self):
        self.episode = -1
        self.time = datetime.datetime.now().strftime('%Y%m%d%H%M%S')

        self.DataShape = (None, mo.mapSzie, mo.mapSzie, 39)
        self.top_decision_maker = decision_maker(
            DQN(mu, sigma, learning_rate, len(sa.controllers), 0, self.DataShape, 'top_decision_maker'))
        self.controllers = []
        for i in range(len(sa.controllers)):
            # 5代表增加的参数槽 6个槽分别代表动作编号，RAW_TYPES.queued, RAW_TYPES.unit_tags, RAW_TYPES.target_unit_tag 和RAW_TYPES.world（占两位）
            self.controllers.append(decision_maker(
                DQN(mu, sigma, learning_rate, len(sa.controllers[i]), 5, self.DataShape, 'controller' + str(i))))

    def my_flatten(self, input_list):
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

    def reflect(self, obs, macro_and_parameter):
        # macro_and_parameter 分别代表：动作（一维），RAW_TYPES.queued, RAW_TYPES.unit_tags, RAW_TYPES.target_unit_tag 和RAW_TYPES.world（占两位）
        m_a_p = macro_and_parameter
        raw_units = obs.observation['raw_units']
        raw_units_len = len(raw_units) - 1

        if macro_and_parameter[1] > 0.5:
            macro_and_parameter[1] = '1'
        else:
            macro_and_parameter[1] = '0'

        macro_and_parameter[2] = int(macro_and_parameter[2] * raw_units_len)
        macro_and_parameter[3] = int(macro_and_parameter[3] * raw_units_len)
        macro_and_parameter[4] = int(macro_and_parameter[4] * (mo.mapSzie - 1))
        macro_and_parameter[5] = int(macro_and_parameter[5] * (mo.mapSzie - 1))
        return m_a_p

    def assembly_action(self, obs, controller_number, macro_and_parameter):
        raw_units = obs.observation['raw_units']
        action = sa.controllers[controller_number][macro_and_parameter[0]]
        parameter = []
        # 根据参数名字填内容
        for i in range(len(action[5])):

            if action[5][i].name == 'queued':
                parameter.append(int(macro_and_parameter[1]))
                continue
            if action[5][i].name == 'unit_tags':
                parameter.append(raw_units[int(macro_and_parameter[2])].tag)
                continue
            if action[5][i].name == 'target_unit_tag':
                parameter.append(raw_units[int(macro_and_parameter[3])].tag)
                continue
            if action[5][i].name == 'world':
                parameter.append((int(macro_and_parameter[4]), int(macro_and_parameter[5])))
                continue

        parameter = tuple(parameter)
        return action(*parameter)

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

    def choose_controller(self, obs, mark, modelLoadPath):
        self.top_decision_maker.current_state = self.get_all_observation(obs)
        if mark == 'TRAIN':
            if self.top_decision_maker.previous_action is not None:
                self.top_decision_maker.network.perceive(self.top_decision_maker.previous_state,
                                                         self.top_decision_maker.previous_action,
                                                         self.top_decision_maker.previous_reward,
                                                         self.top_decision_maker.current_state,
                                                         obs.last())
            if modelLoadPath is not None and self.top_decision_maker.load_and_train is True:
                self.top_decision_maker.load_and_train = False
                self.top_decision_maker.network.restoreModel(modelLoadPath)
                print('top')

            controller_number = self.top_decision_maker.network.egreedy_action(self.top_decision_maker.current_state)
            self.top_decision_maker.previous_reward = obs.reward
            self.top_decision_maker.previous_state = self.top_decision_maker.current_state
            self.top_decision_maker.previous_action = controller_number
            return controller_number
        elif mark == 'TEST':
            return self.top_decision_maker.network.action(self.top_decision_maker.current_state, modelLoadPath)

    def choose_macro(self, obs, controller_number, mark, modelLoadPath):
        self.controllers[controller_number].current_state = self.get_all_observation(obs)

        if mark == 'TRAIN':
            if self.controllers[controller_number].previous_action is not None:
                self.controllers[controller_number].network.perceive(self.controllers[controller_number].previous_state,
                                                                     self.controllers[controller_number].previous_action,
                                                                     self.controllers[controller_number].previous_reward,
                                                                     self.controllers[controller_number].current_state,
                                                                     obs.last())
            if modelLoadPath is not None and self.controllers[controller_number].load_and_train is True:
                self.controllers[controller_number].load_and_train = False
                self.top_decision_maker.network.restoreModel(modelLoadPath)
                print('con'+str(controller_number))

            action_and_parameter = self.controllers[controller_number].network.egreedy_action(self.controllers[controller_number].current_state)
            self.controllers[controller_number].previous_reward = obs.reward
            self.controllers[controller_number].previous_state = self.controllers[controller_number].current_state
            self.controllers[controller_number].previous_action = action_and_parameter
            action_and_parameter = self.reflect(obs, action_and_parameter)
            action = self.assembly_action(obs, controller_number, action_and_parameter)
            return action

        elif mark == 'TEST':
            state = self.controllers[controller_number].current_state
            action_and_parameter = self.controllers[controller_number].network.action(state, modelLoadPath)
            macro_and_parameter = self.reflect(obs, action_and_parameter)
            action = self.assembly_action(obs, controller_number, macro_and_parameter)
            return action

    def make_choice(self, obs, mark, modelSavePath, modelLoadPath):

        if obs[0] == StepType.FIRST:
            self.episode += 1
            time = str(self.time)
            if mark == 'TRAIN':
                self.modelSavePath = modelSavePath + '/' + time + '/'

            self.modelLoadPath = modelLoadPath
            return actions.RAW_FUNCTIONS.raw_move_camera((mo.mapSzie / 2, mo.mapSzie / 2))

        elif obs[0] == StepType.LAST:
            self.top_decision_maker.network.train_Q_network(self.modelSavePath, self.episode)
            for i in range(len(sa.controllers)):
                self.controllers[i].network.train_Q_network(self.modelSavePath, self.episode)

        else:
            controller_number = int(self.choose_controller(obs, mark, self.modelLoadPath)[0])
            action = self.choose_macro(obs, controller_number, mark, self.modelLoadPath)
            return action
