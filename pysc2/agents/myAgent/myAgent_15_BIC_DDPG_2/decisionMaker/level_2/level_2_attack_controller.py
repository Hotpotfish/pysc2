from pysc2.agents.myAgent.myAgent_15_BIC_DDPG_2.config import config
import numpy as np
from pysc2.agents.myAgent.myAgent_15_BIC_DDPG_2.decisionMaker.decision_maker import decision_maker
import pysc2.agents.myAgent.myAgent_15_BIC_DDPG_2.smart_actions as sa
from pysc2.agents.myAgent.myAgent_15_BIC_DDPG_2.decisionMaker.level_2.net_for_level_2_ma import net
from pysc2.agents.myAgent.myAgent_15_BIC_DDPG_2.tools import handcraft_function, handcraft_function_for_level_2_attack_controller

from pysc2.agents.myAgent.myAgent_15_BIC_DDPG_2.tools.handcraft_function_for_level_2_attack_controller import get_reward, get_state, win_or_loss, get_clusters_test, get_bounds_and_states, assembly_action_test, get_agents_obs, \
    get_all_vaild_action, get_init_static_agent_type, get_specified_agent_all_valid_action, get_init_obs
from pysc2.lib.actions import RAW_FUNCTIONS


class level_2_attack_controller:
    def __init__(self):
        self.state_data_shape = (None, config.COOP_AGENTS_OBDIM)
        self.vaild_action = None
        self.controller = decision_maker(
            net(config.MU, config.SIGMA, config.LEARING_RATE, config.ACTION_DIM,
                self.state_data_shape, config.MY_UNIT_NUMBER,
                config.ENEMY_UNIT_NUMBER, 'attack_controller'))

        self.index = handcraft_function.find_controller_index(sa.attack_controller)

        self.init_obs = None
        self.init_static_agent_type = None

        self.pre_obs = None
        self.current_obs = None

    def train_action(self, obs, save_path):
        if self.init_static_agent_type is None:
            self.init_static_agent_type = get_init_static_agent_type(obs)
            self.vaild_action, bound = get_specified_agent_all_valid_action(self.init_static_agent_type)
            self.controller.network.valid_action = self.vaild_action
            self.controller.network.bound = (bound - 1) / 2

        if obs.first():
            self.init_obs = get_init_obs(obs, self.init_static_agent_type)

        self.controller.current_state = [np.array(get_state(self.init_obs, obs)), np.array(get_agents_obs(self.init_obs, obs))]
        self.current_obs = obs

        if self.controller.previous_action is not None:
            self.controller.previous_reward = get_reward(self.controller.current_state[1], self.controller.previous_state[1])
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
            actions = handcraft_function_for_level_2_attack_controller.assembly_action(self.init_obs, obs, action, self.vaild_action)
            self.controller.previous_state = self.controller.current_state
            self.controller.previous_action = action
            self.pre_obs = self.current_obs

        return actions

    def test_action(self, obs):
        if obs.first():
            self.init_obs = obs

        self.controller.current_state = [np.array(get_state(self.init_obs, obs)), np.array(get_agents_obs(self.init_obs, obs))]
        self.current_obs = obs
        action_out, action = self.controller.network.action(self.controller.current_state)
        actions = handcraft_function_for_level_2_attack_controller.assembly_action(self.init_obs, obs, action, self.vaild_action)
        return actions
