from pysc2.agents.myAgent.myAgent_12.config import config
import numpy as np

from pysc2.agents.myAgent.myAgent_12.config.config_for_level_2_attack_controller import ACTION_DIM, STATE_DIM
from pysc2.agents.myAgent.myAgent_12.decisionMaker.decision_maker import decision_maker
import pysc2.agents.myAgent.myAgent_12.smart_actions as sa
from pysc2.agents.myAgent.myAgent_12.decisionMaker.level_2.ddpg_for_level_2 import DDPG
from pysc2.agents.myAgent.myAgent_12.tools import handcraft_function, handcraft_function_for_level_2_attack_controller

from pysc2.agents.myAgent.myAgent_12.tools.handcraft_function_for_level_2_attack_controller import get_state, get_reward


class level_2_attack_controller:
    def __init__(self):
        self.state_data_shape = (None, STATE_DIM)
        self.controller = decision_maker(
            DDPG(config.MU, config.SIGMA, config.LEARING_RATE, ACTION_DIM, self.state_data_shape, 'attack_controller'))
        self.index = handcraft_function.find_controller_index(sa.attack_controller)

    def train_action(self, obs, save_path):
        self.controller.current_state = get_state(obs)

        if self.controller.previous_action is not None:
            self.controller.previous_reward = get_reward(self.controller.previous_state,
                                                         self.controller.current_state)  # reward_compute_2(self.controller.previous_state, self.controller.current_state, obs)
            self.controller.network.perceive(self.controller.previous_state,
                                             self.controller.previous_action,
                                             self.controller.previous_reward,
                                             self.controller.current_state,
                                             obs.last(),
                                             save_path)

        action_prob = self.controller.network.action(self.controller.current_state)
        action = np.random.choice(range(ACTION_DIM), p=action_prob)
        if obs.last():
            self.controller.previous_state = None
            self.controller.previous_action = None
            self.controller.previous_reward = None
        else:
            self.controller.previous_state = self.controller.current_state
            self.controller.previous_action = action

        return action

    def test_action(self, obs):
        self.controller.current_state = [np.array(obs.observation['feature_screen'][5][:, :, np.newaxis]), np.array(get_agents_obs(obs))]

        action_prob = self.controller.network.action(self.controller.current_state)
        actions, action_numbers = handcraft_function_for_level_2_attack_controller.assembly_action(obs, action_prob, 'test')
        return actions
