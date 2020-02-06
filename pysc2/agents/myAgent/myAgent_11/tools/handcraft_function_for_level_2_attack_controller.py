import math
import random

from pysc2.lib import actions as Action
import numpy as np
import pysc2.agents.myAgent.myAgent_11.smart_actions as sa
# 获得全局的观察

from pysc2.agents.myAgent.myAgent_11.config import config
from pysc2.lib import features


def computeDistance(unit, enemy_unit):
    x_difference = math.pow(unit.x - enemy_unit.x, 2)
    y_difference = math.pow(unit.y - enemy_unit.y, 2)

    distance = math.sqrt(x_difference + y_difference)

    return distance


def computeDistance_center(unit):
    x_difference = math.pow(unit.x - config.MAP_SIZE / 2, 2)
    y_difference = math.pow(unit.y - config.MAP_SIZE / 2, 2)

    distance = math.sqrt(x_difference + y_difference)

    return distance


def actionSelect(unit, obs, init_enemy_units_tag, action, var):
    action = int(np.clip(np.random.normal(action, var), 2, config.ATTACT_CONTROLLER_ACTIONDIM - 0.01))

    mask = []
    mask.append(0)
    for i in range(config.ENEMY_UNIT_NUMBER):
        enemy = find_unit_by_tag(obs, init_enemy_units_tag[i])
        if enemy is None:
            mask.append(0)
        # elif computeDistance(unit, enemy) >= config.ATTACK_RANGE:
        #     mask.append(0)
        else:
            mask.append(1)

    #
    mask_nozero = np.nonzero(mask)
    if action in mask_nozero[0]:
        return action
    else:
        return 0

    #
    # action_porb_real = np.multiply(np.array(mask), np.array(action_porb))
    #
    # action_porb_real = action_porb_real / np.sum(action_porb_real)

    # if mark == 'test':
    #     return np.argmax(action_porb_real)
    # if mark == 'train':
    #     if random.random() >= ep:
    #         return np.argmax(action_porb_real)
    #     else:
    #         avail_actions_ind = np.nonzero(action_porb_real)[0]
    #         action = np.random.choice(avail_actions_ind)
    # action = np.random.choice(range(config.ATTACT_CONTROLLER_ACTIONDIM), p=action_porb_real)
    # return action


def find_unit_by_tag(obs, tag):
    for unit in obs.observation['raw_units']:
        if unit.tag == tag:
            return unit
    return None


############################################
global var
var = 3


def assembly_action(init_obs, obs, action):
    actions = []
    action_numbers = []

    init_my_units_tag = [unit.tag for unit in init_obs.observation['raw_units'] if unit.alliance == features.PlayerRelative.SELF]
    init_enemy_units_tag = [unit.tag for unit in init_obs.observation['raw_units'] if unit.alliance == features.PlayerRelative.ENEMY]
    controller = sa.attack_controller

    global var
    var *= 0.9995
    # epsilon -= (config.INITIAL_EPSILON - config.FINAL_EPSILON) / 25000
    # if epsilon <= config.FINAL_EPSILON:
    #     epsilon = config.FINAL_EPSILON

    for i in range(config.MY_UNIT_NUMBER):
        my_unit = find_unit_by_tag(obs, init_my_units_tag[i])
        if my_unit is None:
            action_numbers.append(0)
            continue
        else:
            action_number = actionSelect(my_unit, obs, init_enemy_units_tag, action[i], var)
            # action_number = int(action[i])

            action_numbers.append(action_number)
            if action_number == 1:
                continue
            parameter = []

            if 1 < action_number <= 1+config.ENEMY_UNIT_NUMBER:
                a = controller[2]
                enemy = action_number - 1 -1
                parameter.append(0)
                parameter.append(my_unit.tag)
                parameter.append(init_enemy_units_tag[enemy])
                parameter = tuple(parameter)
                actions.append(a(*parameter))
    return actions, action_numbers


def get_agent_state(unit):
    states = np.array([])

    states = np.append(states, computeDistance_center(unit) / (config.MAP_SIZE * 1.41))
    states = np.append(states, unit.alliance / 4)
    states = np.append(states, unit.unit_type / 10000)
    states = np.append(states, unit.x / config.MAP_SIZE)
    states = np.append(states, unit.y / config.MAP_SIZE)
    states = np.append(states, unit.health / 100)
    states = np.append(states, unit.shield / 100)
    return states


def get_state(init_obs, obs):
    state = np.array([])
    init_my_units_tag = [unit.tag for unit in init_obs.observation['raw_units'] if unit.alliance == features.PlayerRelative.SELF]
    init_enemy_units_tag = [unit.tag for unit in init_obs.observation['raw_units'] if unit.alliance == features.PlayerRelative.ENEMY]

    for i in range(config.MY_UNIT_NUMBER):
        my_unit = find_unit_by_tag(obs, init_my_units_tag[i])
        if my_unit is not None:
            my_unit_state = get_agent_state(my_unit)
            state = np.append(state, my_unit_state)
        else:
            state = np.append(state, np.zeros(7))

    for i in range(config.ENEMY_UNIT_NUMBER):
        enemy_unit = find_unit_by_tag(obs, init_enemy_units_tag[i])
        if enemy_unit is not None:
            my_unit_state = get_agent_state(enemy_unit)
            state = np.append(state, my_unit_state)
        else:
            state = np.append(state, np.zeros(7))
    return state


def get_agents_state(init_obs, obs):
    states = []

    # init_my_units_tag = [unit.tag for unit in init_obs.observation['raw_units'] if unit.alliance == features.PlayerRelative.SELF]

    state_copy = get_state(init_obs, obs)

    for i in range(config.MY_UNIT_NUMBER):
        # my_unit = find_unit_by_tag(obs, init_my_units_tag[i])
        # if my_unit is None:
        #     states.append(np.zeros(config.COOP_AGENTS_OBDIM))
        # else:
        states.append(state_copy)

    return states


def get_agents_obs(init_obs, obs):
    agents_obs = []

    init_my_units_tag = [unit.tag for unit in init_obs.observation['raw_units'] if unit.alliance == features.PlayerRelative.SELF]
    init_enemy_units_tag = [unit.tag for unit in init_obs.observation['raw_units'] if unit.alliance == features.PlayerRelative.ENEMY]

    for i in range(config.MY_UNIT_NUMBER):
        # 一次查找己方单位的信息
        agent_obs = np.array([])
        my_unit = find_unit_by_tag(obs, init_my_units_tag[i])

        if my_unit is None:
            # 此时己方单位已死亡，所以观察值全为0
            agent_obs = np.zeros(config.COOP_AGENTS_OBDIM)
            agents_obs.append(agent_obs)
            continue

        for j in range(config.MY_UNIT_NUMBER):
            my_target_unit = find_unit_by_tag(obs, init_my_units_tag[j])
            # 按顺序遍历每个己方单位的信息
            if my_target_unit is None or computeDistance(my_unit, my_target_unit) >= config.OB_RANGE:
                agent_obs = np.append(agent_obs, np.zeros(7))
            else:
                agent_obs = np.append(agent_obs, computeDistance(my_unit, my_target_unit) / (config.MAP_SIZE * 1.41))
                agent_obs = np.append(agent_obs, my_target_unit.alliance / 4)
                agent_obs = np.append(agent_obs, my_target_unit.unit_type / 10000)
                agent_obs = np.append(agent_obs, my_target_unit.x / config.MAP_SIZE)
                agent_obs = np.append(agent_obs, my_target_unit.y / config.MAP_SIZE)
                agent_obs = np.append(agent_obs, my_target_unit.health / 100)
                agent_obs = np.append(agent_obs, my_target_unit.shield / 100)
        for j in range(config.ENEMY_UNIT_NUMBER):
            enemy_target_unit = find_unit_by_tag(obs, init_enemy_units_tag[j])
            # 按顺序遍历每个己方单位的信息
            if enemy_target_unit is None or computeDistance(my_unit, enemy_target_unit) >= config.OB_RANGE:
                agent_obs = np.append(agent_obs, np.zeros(7))
            else:
                agent_obs = np.append(agent_obs, computeDistance(my_unit, enemy_target_unit) / (config.MAP_SIZE * 1.41))
                agent_obs = np.append(agent_obs, enemy_target_unit.alliance / 4)
                agent_obs = np.append(agent_obs, enemy_target_unit.unit_type / 10000)
                agent_obs = np.append(agent_obs, enemy_target_unit.x / config.MAP_SIZE)
                agent_obs = np.append(agent_obs, enemy_target_unit.y / config.MAP_SIZE)
                agent_obs = np.append(agent_obs, enemy_target_unit.health / 100)
                agent_obs = np.append(agent_obs, enemy_target_unit.shield / 100)

        agents_obs.append(agent_obs)
    return agents_obs


def get_reward(obs, pre_obs):
    reward = 0
    my_units_health = [unit.health for unit in obs.observation['raw_units'] if unit.alliance == features.PlayerRelative.SELF]
    enemy_units_health = [unit.health for unit in obs.observation['raw_units'] if unit.alliance == features.PlayerRelative.ENEMY]

    my_units_health_pre = [unit.health for unit in pre_obs.observation['raw_units'] if unit.alliance == features.PlayerRelative.SELF]
    enemy_units_health_pre = [unit.health for unit in pre_obs.observation['raw_units'] if unit.alliance == features.PlayerRelative.ENEMY]

    # if len(enemy_units_health) == 0:
    #     reward += 1000

    if len(my_units_health) < len(my_units_health_pre):
        reward -= (len(my_units_health_pre) - len(my_units_health)) * 100

    if len(enemy_units_health) < len(enemy_units_health_pre):
        reward += (len(enemy_units_health_pre) - len(enemy_units_health)) * 100

    reward += (sum(my_units_health) - sum(my_units_health_pre)) - (sum(enemy_units_health) - sum(enemy_units_health_pre))

    return reward


def win_or_loss(obs):
    if obs.last():

        my_units = [unit for unit in obs.observation['raw_units'] if unit.alliance == features.PlayerRelative.SELF]

        if len(my_units) == 0:
            return -1
        else:
            return 1
    else:
        return 0
