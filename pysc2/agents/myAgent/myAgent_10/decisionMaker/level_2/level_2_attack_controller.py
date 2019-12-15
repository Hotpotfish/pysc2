from pysc2.agents.myAgent.myAgent_10.config import config

from pysc2.agents.myAgent.myAgent_10.decisionMaker.decision_maker import decision_maker
import pysc2.agents.myAgent.myAgent_10.smart_actions as sa
from pysc2.agents.myAgent.myAgent_10.decisionMaker.level_2.Bicnet_for_level_2 import Bicnet
from pysc2.agents.myAgent.myAgent_10.tools import handcraft_function, handcraft_function_for_level_2_attack_controller

from pysc2.agents.myAgent.myAgent_10.tools.handcraft_function_for_level_2_attack_controller import reward_compute_1, get_raw_units_observation


class level_2_attack_controller:
    def __init__(self):
        # self.DataShape = (None, 200, config.FEATURE_UNITS_LENGTH, 1)
        # self.DataShape = (None, config.MAP_SIZE, config.MAP_SIZE, 39)
        self.DataShape = (None, config.MAP_SIZE, config.MAP_SIZE, 1)
        self.controller = decision_maker(
            Bicnet(config.MU, config.SIGMA, config.LEARING_RATE, len(sa.attack_controller), config.ATTACT_CONTROLLER_PARAMETERDIM, config.COOP_AGENTS_NUMBER, config.ENEMY_UNIT_NUMBER,
                   self.DataShape, 'attack_controller'))
        self.index = handcraft_function.find_controller_index(sa.attack_controller)

    def train_action(self, obs):
        self.controller.current_state = get_raw_units_observation(obs)
        # self.controller.current_state = handcraft_function.get_all_observation(obs)
        self.controller.current_reward = reward_compute_1(obs)
        if self.controller.previous_action is not None and self.controller.current_reward:
            self.controller.network.perceive(self.controller.current_state,
                                             self.controller.previous_action,
                                             self.controller.current_reward,
                                             obs.last())
        action_and_parameter = self.controller.network.egreedy_action(self.controller.current_state)
        action_and_parameter = handcraft_function.reflect(len(sa.attack_controller), action_and_parameter)
        self.controller.previous_action = action_and_parameter
        action = handcraft_function.assembly_action(obs, self.index, action_and_parameter)
        return action

    def test_action(self, obs):
        self.controller.current_state = handcraft_function_for_level_2_attack_controller.get_raw_units_observation(obs)
        # self.controller.current_state = handcraft_function.get_all_observation(obs)
        state = self.controller.current_state
        action_and_parameter = self.controller.network.action(state)
        macro_and_parameter = handcraft_function.reflect(len(sa.attack_controller), action_and_parameter)
        action = handcraft_function.assembly_action(obs, self.index, macro_and_parameter)
        return action
