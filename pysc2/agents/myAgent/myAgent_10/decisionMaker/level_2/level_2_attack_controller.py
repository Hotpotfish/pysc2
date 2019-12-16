from pysc2.agents.myAgent.myAgent_10.config import config

from pysc2.agents.myAgent.myAgent_10.decisionMaker.decision_maker import decision_maker
import pysc2.agents.myAgent.myAgent_10.smart_actions as sa
from pysc2.agents.myAgent.myAgent_10.decisionMaker.level_2.Bicnet_for_level_2 import Bicnet
from pysc2.agents.myAgent.myAgent_10.tools import handcraft_function, handcraft_function_for_level_2_attack_controller

from pysc2.agents.myAgent.myAgent_10.tools.handcraft_function_for_level_2_attack_controller import get_agents_local_observation, reward_compute_2


class level_2_attack_controller:
    def __init__(self):
        self.state_data_shape = (None, config.MAP_SIZE, config.MAP_SIZE, 1)
        self.controller = decision_maker(
            Bicnet(config.MU, config.SIGMA, config.LEARING_RATE, len(sa.attack_controller), config.ATTACT_CONTROLLER_PARAMETERDIM, self.state_data_shape, config.COOP_AGENTS_NUMBER,
                   config.ENEMY_UNIT_NUMBER,
                   'attack_controller'))
        self.index = handcraft_function.find_controller_index(sa.attack_controller)

    def train_action(self, obs):
        self.controller.current_state = [obs.observation['feature_screen'][5], get_agents_local_observation(obs)]

        # self.controller.current_state = handcraft_function.get_all_observation(obs)
        # self.controller.current_reward = reward_compute_2(,self.controller.current_state)
        if self.controller.previous_action is not None:
            self.controller.previous_reward = reward_compute_2(self.controller.previous_state, self.controller.current_state)
            self.controller.network.perceive(self.controller.previous_state,
                                             self.controller.previous_action,
                                             self.controller.previous_reward,
                                             self.controller.current_state,
                                             obs.last())
        action_and_parameter = self.controller.network.egreedy_action(self.controller.current_state)
        action_and_parameter = handcraft_function.reflect(len(sa.attack_controller), action_and_parameter)
        self.controller.previous_state = self.controller.current_state
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
