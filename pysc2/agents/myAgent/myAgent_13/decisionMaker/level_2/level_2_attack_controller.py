from pysc2.agents.myAgent.myAgent_13.config import config
import numpy as np
from pysc2.agents.myAgent.myAgent_13.decisionMaker.decision_maker import decision_maker
import pysc2.agents.myAgent.myAgent_13.smart_actions as sa
from pysc2.agents.myAgent.myAgent_13.decisionMaker.level_2.net_for_level_2 import net
from pysc2.agents.myAgent.myAgent_13.tools import handcraft_function, handcraft_function_for_level_2_attack_controller

from pysc2.agents.myAgent.myAgent_13.tools.handcraft_function_for_level_2_attack_controller import get_reward, get_bound, get_state, win_or_loss, get_clusters_test, get_bounds_and_states, assembly_action_test
from pysc2.lib.actions import RAW_FUNCTIONS


class level_2_attack_controller:
    def __init__(self):
        self.state_data_shape = (None, config.MY_UNIT_NUMBER, config.COOP_AGENTS_OBDIM)
        self.controller = decision_maker(
            net(config.MU, config.SIGMA, config.LEARING_RATE, config.ATTACT_CONTROLLER_ACTIONDIM,
                self.state_data_shape, config.MY_UNIT_NUMBER,
                config.ENEMY_UNIT_NUMBER, 'attack_controller'))
        self.index = handcraft_function.find_controller_index(sa.attack_controller)
        self.init_obs = None
        self.pre_obs = None
        self.current_obs = None

    def train_action(self, obs, save_path):
        if obs.first():
            self.init_obs = obs

        self.controller.current_state = [np.array(get_bound(self.init_obs, obs)), np.array(get_state(self.init_obs, obs))]
        self.current_obs = obs

        if self.controller.previous_action is not None:
            self.controller.previous_reward = get_reward(self.current_obs, self.pre_obs)
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
            action = self.controller.network.egreedy_action(self.controller.current_state)
            actions = handcraft_function_for_level_2_attack_controller.assembly_action(self.init_obs, action)
            self.controller.previous_state = self.controller.current_state
            self.controller.previous_action = action
            self.pre_obs = self.current_obs

        return actions

    def test_action(self, obs):
        obs_new = get_clusters_test(obs)
        if len(obs_new) == 0:
            return RAW_FUNCTIONS.no_op()
        else:
            actions = []
            bounds_and_states, my_units_and_enemy_units_pack = get_bounds_and_states(obs_new)
            for i in range(len(bounds_and_states)):
                action_number = self.controller.network.action(bounds_and_states[i])
                action = assembly_action_test(my_units_and_enemy_units_pack[i][0], my_units_and_enemy_units_pack[i][1], action_number)
                actions += action
        return actions

        # self.controller.current_state = [np.array(get_bound(self.init_obs, obs)), np.array(get_state(self.init_obs, obs))]
        # action = self.controller.network.action(self.controller.current_state)
        # actions = handcraft_function_for_level_2_attack_controller.assembly_action(self.init_obs, action)
        # return actions
