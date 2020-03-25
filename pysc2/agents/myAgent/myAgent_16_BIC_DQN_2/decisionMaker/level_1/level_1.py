from pysc2.agents.myAgent.myAgent_16_BIC_DQN_2.config import config
from pysc2.agents.myAgent.myAgent_16_BIC_DQN_2.decisionMaker.level_1.DQN_for_level_1 import DQN
from pysc2.agents.myAgent.myAgent_16_BIC_DQN_2.decisionMaker.decision_maker import decision_maker
import pysc2.agents.myAgent.myAgent_16_BIC_DQN_2.smart_actions as sa
from pysc2.agents.myAgent.myAgent_16_BIC_DQN_2.tools import handcraft_function


class level_1():
    def __init__(self):
        self.DataShape = (None, config.MAP_SIZE, config.MAP_SIZE, 39)
        self.top_decision_maker = decision_maker(
            DQN(config.MU, config.SIGMA, config.LEARING_RATE, len(sa.controllers), 0, self.DataShape, 'top_decision_maker'))

    # 重训练模式 无需读取外部模型
    def train_action(self, obs):
        self.top_decision_maker.current_state = handcraft_function.get_all_observation(obs)
        if self.top_decision_maker.previous_action is not None:
            self.top_decision_maker.network.perceive(self.top_decision_maker.previous_state,
                                                     self.top_decision_maker.previous_action,
                                                     self.top_decision_maker.previous_reward,
                                                     self.top_decision_maker.current_state,
                                                     obs.last())
        controller_number = self.top_decision_maker.network.egreedy_action(self.top_decision_maker.current_state)
        self.top_decision_maker.previous_reward = obs.reward
        self.top_decision_maker.previous_state = self.top_decision_maker.current_state
        self.top_decision_maker.previous_action = controller_number
        return controller_number

    def test_action(self, obs):
        self.top_decision_maker.current_state = handcraft_function.get_all_observation(obs)
        return self.top_decision_maker.network.action(self.top_decision_maker.current_state)

    def train_network(self, modelSavePath):
        self.top_decision_maker.network.train_Q_network(modelSavePath)

    def load_model(self, modelLoadPath):
        self.top_decision_maker.network.restoreModel(modelLoadPath)
        print('level_1 load complete!')

    def save_model(self, modelSavePath, episode):
        self.top_decision_maker.network.saveModel(modelSavePath, episode)
        print('level_1 episode %d save complete!' % (episode))
