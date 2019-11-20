import datetime

import pysc2.agents.myAgent.myAgent_6.config.config as config
from pysc2.agents.myAgent.myAgent_6.decisionMaker.DQN import DQN
import pysc2.agents.myAgent.myAgent_6.smart_actions as sa
import pysc2.agents.myAgent.myAgent_6.tools.handcraft_function as handcraft_function

from pysc2.env.environment import StepType
from pysc2.lib import actions


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
        self.begin_time = datetime.datetime.now().strftime('%Y%m%d%H%M%S')

        self.DataShape = (None, config.MAP_SIZE, config.MAP_SIZE, 39)
        self.top_decision_maker = decision_maker(
            DQN(config.MU, config.SIGMA, config.LEARING_RATE, len(sa.controllers), 0, self.DataShape, 'top_decision_maker'))
        self.controllers = []
        for i in range(len(sa.controllers)):
            # 5代表增加的参数槽 6个槽分别代表动作编号，RAW_TYPES.queued, RAW_TYPES.unit_tags, RAW_TYPES.target_unit_tag 和RAW_TYPES.world（占两位）
            self.controllers.append(
                decision_maker(DQN(config.MU, config.SIGMA, config.LEARING_RATE, len(sa.controllers[i]), 5, self.DataShape, 'controller' + str(i))))

    def top_decision_maker_train_model(self, obs, modelLoadPath):
        # 数据是否记录
        if self.top_decision_maker.previous_action is not None:
            self.top_decision_maker.network.perceive(self.top_decision_maker.previous_state,
                                                     self.top_decision_maker.previous_action,
                                                     self.top_decision_maker.previous_reward,
                                                     self.top_decision_maker.current_state,
                                                     obs.last())
        # 是否为继续训练模式
        if modelLoadPath is not None and self.top_decision_maker.load_and_train is True:
            self.top_decision_maker.load_and_train = False
            self.top_decision_maker.network.restoreModel(modelLoadPath)
            print('top')

        controller_number = self.top_decision_maker.network.egreedy_action(self.top_decision_maker.current_state)
        self.top_decision_maker.previous_reward = obs.reward
        self.top_decision_maker.previous_state = self.top_decision_maker.current_state
        self.top_decision_maker.previous_action = controller_number
        return controller_number

    def top_decision_maker_test_model(self, modelLoadPath):
        return self.top_decision_maker.network.action(self.top_decision_maker.current_state, modelLoadPath)

    def choose_controller(self, obs, mark, modelLoadPath):
        self.top_decision_maker.current_state = handcraft_function.get_all_observation(obs)
        if mark == 'TRAIN':
            controller_number = self.top_decision_maker_train_model(obs, modelLoadPath)
            return controller_number

        elif mark == 'TEST':
            controller_number = self.top_decision_maker_test_model(modelLoadPath)
            return controller_number

    def controller_train_model(self, obs, controller_number, modelLoadPath):
        if self.controllers[controller_number].previous_action is not None:
            self.controllers[controller_number].network.perceive(self.controllers[controller_number].previous_state,
                                                                 self.controllers[controller_number].previous_action,
                                                                 self.controllers[controller_number].previous_reward,
                                                                 self.controllers[controller_number].current_state,
                                                                 obs.last())
        if modelLoadPath is not None and self.controllers[controller_number].load_and_train is True:
            self.controllers[controller_number].load_and_train = False
            self.top_decision_maker.network.restoreModel(modelLoadPath)
            print('con' + str(controller_number))

        action_and_parameter = self.controllers[controller_number].network.egreedy_action(self.controllers[controller_number].current_state)
        self.controllers[controller_number].previous_reward = obs.reward
        self.controllers[controller_number].previous_state = self.controllers[controller_number].current_state
        self.controllers[controller_number].previous_action = action_and_parameter
        action_and_parameter = handcraft_function.reflect(obs, action_and_parameter)
        action = handcraft_function.assembly_action(obs, controller_number, action_and_parameter)
        return action

    def controller_test_model(self, obs, controller_number, modelLoadPath):
        state = self.controllers[controller_number].current_state
        action_and_parameter = self.controllers[controller_number].network.action(state, modelLoadPath)
        macro_and_parameter = handcraft_function.reflect(obs, action_and_parameter)
        action = handcraft_function.assembly_action(obs, controller_number, macro_and_parameter)
        return action

    def choose_macro(self, obs, controller_number, mark, modelLoadPath):
        self.controllers[controller_number].current_state = handcraft_function.get_all_observation(obs)

        if mark == 'TRAIN':
            action = self.controller_train_model(obs, controller_number, modelLoadPath)
            return action


        elif mark == 'TEST':
            action = self.controller_test_model(obs, controller_number, modelLoadPath)
            return action

    def get_save_and_loadPath(self, mark, modelSavePath, modelLoadPath):
        self.episode += 1
        time = str(self.begin_time)
        if mark == 'TRAIN':
            self.modelSavePath = modelSavePath + '/' + time + '/'

        self.modelLoadPath = modelLoadPath

    def train_all_neural_network(self):
        self.top_decision_maker.network.train_Q_network(self.modelSavePath, self.episode)
        for i in range(len(sa.controllers)):
            self.controllers[i].network.train_Q_network(self.modelSavePath, self.episode)

    def make_choice(self, obs, mark, modelSavePath, modelLoadPath):

        if obs[0] == StepType.FIRST:
            # 更新读取和保存路径
            self.get_save_and_loadPath(mark, modelSavePath, modelLoadPath)
            return actions.RAW_FUNCTIONS.raw_move_camera((config.MAP_SIZE / 2, config.MAP_SIZE / 2))

        elif obs[0] == StepType.LAST:
            self.train_all_neural_network()

        else:
            controller_number = int(self.choose_controller(obs, mark, self.modelLoadPath)[0])
            action = self.choose_macro(obs, controller_number, mark, self.modelLoadPath)
            print(action)
            return action
