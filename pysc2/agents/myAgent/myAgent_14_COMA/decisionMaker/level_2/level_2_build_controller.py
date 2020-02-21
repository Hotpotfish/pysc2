from pysc2.agents.myAgent.myAgent_11_DQN.config import config
from pysc2.agents.myAgent.myAgent_11_DQN.decisionMaker.level_1.DQN_for_level_1 import DQN
from pysc2.agents.myAgent.myAgent_8.decisionMaker.decision_maker import decision_maker
import pysc2.agents.myAgent.myAgent_8.smart_actions as sa
from pysc2.agents.myAgent.myAgent_8.tools import handcraft_function


class level_2_build_controller:
    def __init__(self):
        self.DataShape = (None, config.MAP_SIZE, config.MAP_SIZE, 39)
        self.controller = decision_maker(DQN(config.MU, config.SIGMA, config.LEARING_RATE, len(sa.build_controller), 5, self.DataShape, 'build_controller'))
        self.index = handcraft_function.find_controller_index(sa.build_controller)

    # 重训练模式 无需读取外部模型

    def train_action(self, obs):
        self.controller.current_state = handcraft_function.get_all_observation(obs)
        if self.controller.previous_action is not None:
            self.controller.network.perceive(self.controller.previous_state,
                                             self.controller.previous_action,
                                             self.controller.previous_reward,
                                             self.controller.current_state,
                                             obs.last())
        action_and_parameter = self.controller.network.egreedy_action(self.controller.current_state)
        self.controller.previous_reward = obs.reward
        self.controller.previous_state = self.controller.current_state
        self.controller.previous_action = action_and_parameter
        action_and_parameter = handcraft_function.reflect(obs, action_and_parameter)
        action = handcraft_function.assembly_action(obs, self.index, action_and_parameter)
        return action

    def test_action(self, obs):
        self.controller.current_state = handcraft_function.get_all_observation(obs)
        state = self.controller.current_state
        action_and_parameter = self.controller.network.action(state)
        macro_and_parameter = handcraft_function.reflect(obs, action_and_parameter)
        action = handcraft_function.assembly_action(obs, self.index, macro_and_parameter)
        return action
