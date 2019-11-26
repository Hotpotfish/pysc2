from pysc2.agents.myAgent.myAgent_7.config import config
from pysc2.agents.myAgent.myAgent_7.decisionMaker.DQN import DQN
from pysc2.agents.myAgent.myAgent_7.decisionMaker.decision_maker import decision_maker
import pysc2.agents.myAgent.myAgent_7.smart_actions as sa
from pysc2.agents.myAgent.myAgent_7.tools import handcraft_function


class level_2():
    def __init__(self):
        self.DataShape = (None, config.MAP_SIZE, config.MAP_SIZE, 39)
        self.controllers = []
        for key in sa.controllers.keys():
            # 5代表增加的参数槽 6个槽分别代表动作编号，RAW_TYPES.queued, RAW_TYPES.unit_tags, RAW_TYPES.target_unit_tag 和RAW_TYPES.world（占两位）
            self.controllers.append(decision_maker(DQN(config.MU, config.SIGMA, config.LEARING_RATE, len(sa.controllers[int(key)]), 5, self.DataShape, 'controller' + str(key))))

    # 重训练模式 无需读取外部模型
    def train_action(self, obs, controller_number):
        self.controllers[controller_number].current_state = handcraft_function.get_all_observation(obs)
        if self.controllers[controller_number].previous_action is not None:
            self.controllers[controller_number].network.perceive(self.controllers[controller_number].previous_state,
                                                                 self.controllers[controller_number].previous_action,
                                                                 self.controllers[controller_number].previous_reward,
                                                                 self.controllers[controller_number].current_state,
                                                                 obs.last())
        action_and_parameter = self.controllers[controller_number].network.egreedy_action(self.controllers[controller_number].current_state)
        self.controllers[controller_number].previous_reward = obs.reward
        self.controllers[controller_number].previous_state = self.controllers[controller_number].current_state
        self.controllers[controller_number].previous_action = action_and_parameter
        action_and_parameter = handcraft_function.reflect(obs, action_and_parameter)
        action = handcraft_function.assembly_action(obs, controller_number, action_and_parameter)
        return action

    def test_action(self, obs, controller_number):
        self.controllers[controller_number].current_state = handcraft_function.get_all_observation(obs)
        state = self.controllers[controller_number].current_state
        action_and_parameter = self.controllers[controller_number].network.action(state)
        macro_and_parameter = handcraft_function.reflect(obs, action_and_parameter)
        action = handcraft_function.assembly_action(obs, controller_number, macro_and_parameter)
        return action

    def train_network(self,modelSavePath):
        for i in range(len(sa.controllers)):
            self.controllers[i].network.train_Q_network(modelSavePath)

    def load_model(self, modelLoadPath):
        for i in range(len(sa.controllers)):
            self.controllers[i].network.restoreModel(modelLoadPath)

    def save_model(self, modelSavePath, episode):
        for i in range(len(sa.controllers)):
            self.controllers[i].network.saveModel(modelSavePath, episode)
