from pysc2.agents.myAgent.myAgent_11.config import config
import numpy as np
from pysc2.agents.myAgent.myAgent_11.decisionMaker.decision_maker import decision_maker
import pysc2.agents.myAgent.myAgent_11.smart_actions as sa
from pysc2.agents.myAgent.myAgent_11.decisionMaker.level_2.Bicnet_for_level_2 import Bicnet
from pysc2.agents.myAgent.myAgent_11.tools import handcraft_function, handcraft_function_for_level_2_attack_controller

from pysc2.agents.myAgent.myAgent_11.tools.handcraft_function_for_level_2_attack_controller import get_agents_state, \
    get_agents_obs, get_reward, win_or_loss
from pysc2.lib.actions import RAW_FUNCTIONS


class level_2_attack_controller:
    def __init__(self):
        self.state_data_shape = (None, config.MY_UNIT_NUMBER, config.COOP_AGENTS_OBDIM)
        self.controller = decision_maker(
            Bicnet(config.MU, config.SIGMA, config.LEARING_RATE, config.ATTACT_CONTROLLER_ACTIONDIM,
                   self.state_data_shape, config.MY_UNIT_NUMBER,
                   config.ENEMY_UNIT_NUMBER, 'attack_controller'))
        self.index = handcraft_function.find_controller_index(sa.attack_controller)
        self.init_obs = None
        self.pre_obs = None
        self.current_obs = None

    def train_action(self, obs, save_path):
        if obs.first():
            self.init_obs = obs

        self.controller.current_state = [np.array(get_agents_state(self.init_obs, obs)), np.array(get_agents_obs(self.init_obs, obs))]
        self.current_obs = obs

        if self.controller.previous_action is not None:
            self.controller.previous_reward = get_reward(self.current_obs, self.pre_obs) / 100  # reward_compute_2(self.controller.previous_state, self.controller.current_state, obs)
            self.controller.network.perceive(self.controller.previous_state,
                                             self.controller.previous_action,
                                             self.controller.previous_reward,
                                             self.controller.current_state,
                                             win_or_loss(obs),
                                             save_path)

        if obs.last():
            self.controller.previous_state = None
            self.controller.previous_action = None
            self.controller.previous_reward = None
            self.init_obs = None
            self.pre_obs = None
            self.current_obs = None
            return RAW_FUNCTIONS.no_op()
        else:
            action_prob = self.controller.network.action(self.controller.current_state)
            actions, action_numbers = handcraft_function_for_level_2_attack_controller.assembly_action(self.init_obs, obs, action_prob, 'train')
            self.controller.previous_state = self.controller.current_state
            self.controller.previous_action = action_numbers
            self.pre_obs = self.current_obs

        return actions

    def test_action(self, obs):
        self.controller.current_state = [np.array(obs.observation['feature_screen'][5][:, :, np.newaxis]),
                                         np.array(get_agents_obs(obs))]

        action_prob = self.controller.network.action(self.controller.current_state)
        actions, action_numbers = handcraft_function_for_level_2_attack_controller.assembly_action(obs, action_prob,
                                                                                                   'test')
        return actions
