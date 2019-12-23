from pysc2.agents.myAgent.myAgent_10.config import config
import numpy as np
from pysc2.agents.myAgent.myAgent_10.decisionMaker.decision_maker import decision_maker
import pysc2.agents.myAgent.myAgent_10.smart_actions as sa
from pysc2.agents.myAgent.myAgent_10.decisionMaker.level_2.Bicnet_for_level_2 import Bicnet
from pysc2.agents.myAgent.myAgent_10.tools import handcraft_function, handcraft_function_for_level_2_attack_controller

from pysc2.agents.myAgent.myAgent_10.tools.handcraft_function_for_level_2_attack_controller import get_agents_local_observation, reward_compute_2


class level_2_attack_controller:
    def __init__(self):
        self.state_data_shape = (None, config.MAP_SIZE, config.MAP_SIZE, 1)
        self.controller = decision_maker(
            Bicnet(config.MU, config.SIGMA, config.LEARING_RATE, config.ATTACT_CONTROLLER_ACTIONDIM, self.state_data_shape, config.COOP_AGENTS_NUMBER,
                   config.ENEMY_UNIT_NUMBER, 'attack_controller'))
        self.index = handcraft_function.find_controller_index(sa.attack_controller)

    def train_action(self, obs, save_path):

        if obs.first():
            self.controller.previous_state = None
            self.controller.previous_action = None
            self.controller.previous_reward = None
            self.controller.current_state = None

        self.controller.current_state = [np.array(obs.observation['feature_screen'][5][:, :, np.newaxis]), np.array(get_agents_local_observation(obs))]

        if self.controller.previous_action is not None:
            self.controller.previous_reward = reward_compute_2(self.controller.previous_state, self.controller.current_state)
            self.controller.network.perceive(self.controller.previous_state,
                                             self.controller.previous_action,
                                             self.controller.previous_reward,
                                             self.controller.current_state,
                                             obs.last(),
                                             save_path)
        action_prob = self.controller.network.egreedy_action(self.controller.current_state)
        action = handcraft_function_for_level_2_attack_controller.assembly_action(obs, action_prob)
        action = handcraft_function.reflect(len(sa.attack_controller), action)
        self.controller.previous_state = self.controller.current_state
        self.controller.previous_action = action

        return action

    def test_action(self, obs):
        self.controller.current_state = [np.array(obs.observation['feature_screen'][5][:, :, np.newaxis]), np.array(get_agents_local_observation(obs))]

        action= self.controller.network.action(self.controller.current_state)
        action = handcraft_function.reflect(len(sa.attack_controller), action)
        action = handcraft_function_for_level_2_attack_controller.assembly_action(obs, action)
        return action
