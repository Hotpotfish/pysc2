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


def actionSelect(unit, obs, init_enemy_units_tag, action_porb, mark, ep):
    mask = []
    action_porb = np.exp(action_porb) / sum(np.exp(action_porb))

    for i in range(config.STATIC_ACTION_DIM):
        mask.append(1)

    for i in range(config.ENEMY_UNIT_NUMBER):
        enemy = find_unit_by_tag(obs, init_enemy_units_tag[i])
        if enemy is None:
            mask.append(0)
        elif computeDistance(unit, enemy) >= config.ATTACK_RANGE:
            mask.append(0)
        else:
            mask.append(1)

    action_porb_real = np.multiply(np.array(mask), np.array(action_porb))

    action_porb_real = action_porb_real / np.sum(action_porb_real)

    if mark == 'test':
        return np.argmax(action_porb_real)
    if mark == 'train':
        if random.random() >= ep:
            return np.argmax(action_porb_real)
        else:
            avail_actions_ind = np.nonzero(action_porb_real)[0]
            action = np.random.choice(avail_actions_ind)
        # action = np.random.choice(range(config.ATTACT_CONTROLLER_ACTIONDIM), p=action_porb_real)
        return action


def find_unit_by_tag(obs, tag):
    for unit in obs.observation['raw_units']:
        if unit.tag == tag:
            return unit
    return None


############################################
global epsilon
epsilon = config.INITIAL_EPSILON


def assembly_action(init_obs, obs, action_probs, mark):
    actions = []
    action_numbers = []

    init_my_units_tag = [unit.tag for unit in init_obs.observation['raw_units'] if unit.alliance == features.PlayerRelative.SELF]
    init_enemy_units_tag = [unit.tag for unit in init_obs.observation['raw_units'] if unit.alliance == features.PlayerRelative.ENEMY]
    controller = sa.attack_controller

    global epsilon
    epsilon -= (config.INITIAL_EPSILON - config.FINAL_EPSILON) / 25000
    if epsilon <= config.FINAL_EPSILON:
        epsilon = config.FINAL_EPSILON

    for i in range(config.MY_UNIT_NUMBER):
        my_unit = find_unit_by_tag(obs, init_my_units_tag[i])
        if my_unit is None:
            action_numbers.append(0)
            continue
        else:
            action_number = actionSelect(my_unit, obs, init_enemy_units_tag, action_probs[i], mark, epsilon)
            action_numbers.append(action_number)
            parameter = []

            if action_number == 0:
                actions.append(Action.RAW_FUNCTIONS.no_op())
                continue

            elif 0 < action_number <= 4:
                a = controller[1]

                dir = action_number - 1

                parameter.append(0)
                parameter.append(my_unit.tag)
                if dir == 0:
                    parameter.append((my_unit.x + 1, my_unit.y + 1))
                elif dir == 1:
                    parameter.append((my_unit.x - 1, my_unit.y - 1))
                elif dir == 2:
                    parameter.append((my_unit.x + 1, my_unit.y - 1))
                elif dir == 3:
                    parameter.append((my_unit.x - 1, my_unit.y + 1))

                parameter = tuple(parameter)
                actions.append(a(*parameter))

            elif 4 < action_number <= 4 + config.ENEMY_UNIT_NUMBER:
                a = controller[2]
                enemy = action_number - 1 - 4
                parameter.append(0)
                parameter.append(my_unit.tag)
                # print(str(len(enemy_units))+':'+enemy)
                parameter.append(init_enemy_units_tag[enemy])
                parameter = tuple(parameter)
                actions.append(a(*parameter))
    return actions, action_numbers

    #
    # my_raw_units = [unit for unit in obs.observation['raw_units'] if unit.alliance == features.PlayerRelative.SELF]
    # enemy_units = [unit for unit in obs.observation['raw_units'] if unit.alliance == features.PlayerRelative.ENEMY]
    #
    # my_raw_units_lenth = len(my_raw_units)
    #
    # actions = []
    # action_numbers = []
    #
    # controller = sa.attack_controller
    #
    # global epsilon
    # epsilon -= (config.INITIAL_EPSILON - config.FINAL_EPSILON) / 25000
    # if epsilon <= config.FINAL_EPSILON:
    #     epsilon = config.FINAL_EPSILON

    # # 根据参数名字填内容
    # if my_raw_units_lenth > config.MY_UNIT_NUMBER:
    #     for i in range(config.MY_UNIT_NUMBER):
    #         action_number = actionSelect(my_raw_units[i], enemy_units, action_probs[i], mark, epsilon)
    #         action_numbers.append(action_number)
    #         parameter = []
    #
    #         if action_number == 0:
    #             actions.append(Action.RAW_FUNCTIONS.no_op())
    #             continue
    #
    #         elif 0 < action_number <= 4:
    #             a = controller[1]
    #
    #             dir = action_number - 1
    #
    #             parameter.append(0)
    #             parameter.append(my_raw_units[i].tag)
    #             if dir == 0:
    #                 parameter.append((my_raw_units[i].x + 1, my_raw_units[i].y + 1))
    #             elif dir == 1:
    #                 parameter.append((my_raw_units[i].x - 1, my_raw_units[i].y - 1))
    #             elif dir == 2:
    #                 parameter.append((my_raw_units[i].x + 1, my_raw_units[i].y - 1))
    #             elif dir == 3:
    #                 parameter.append((my_raw_units[i].x - 1, my_raw_units[i].y + 1))
    #
    #             parameter = tuple(parameter)
    #             actions.append(a(*parameter))
    #
    #         elif 4 < action_number <= 4 + config.ENEMY_UNIT_NUMBER:
    #             a = controller[2]
    #             enemy = action_number - 1 - 4
    #             parameter.append(0)
    #             parameter.append(my_raw_units[i].tag)
    #             # print(str(len(enemy_units))+':'+enemy)
    #             parameter.append(enemy_units[enemy].tag)
    #             parameter = tuple(parameter)
    #             actions.append(a(*parameter))
    #
    # else:
    #
    #     for i in range(my_raw_units_lenth):
    #         action_number = actionSelect(my_raw_units[i], enemy_units, action_probs[i], mark, epsilon)
    #         action_numbers.append(action_number)
    #         parameter = []
    #         if action_number == 0:
    #             actions.append(Action.RAW_FUNCTIONS.no_op())
    #             continue
    #         elif 0 < action_number <= 4:
    #             a = controller[1]
    #             dir = action_number - 1
    #             parameter.append(0)
    #             parameter.append(my_raw_units[i].tag)
    #             if dir == 0:
    #                 parameter.append((my_raw_units[i].x + 1, my_raw_units[i].y + 1))
    #             elif dir == 1:
    #                 parameter.append((my_raw_units[i].x - 1, my_raw_units[i].y - 1))
    #             elif dir == 2:
    #                 parameter.append((my_raw_units[i].x + 1, my_raw_units[i].y - 1))
    #             elif dir == 3:
    #                 parameter.append((my_raw_units[i].x - 1, my_raw_units[i].y + 1))
    #             parameter = tuple(parameter)
    #             actions.append(a(*parameter))
    #
    #         elif 4 < action_number <= 4 + config.ENEMY_UNIT_NUMBER:
    #             a = controller[2]
    #             enemy = action_number - 1 - 4
    #             parameter.append(0)
    #             parameter.append(my_raw_units[i].tag)
    #             if len(enemy_units) == enemy:
    #                 print(str(len(enemy_units)) + ':' + str(enemy))
    #             parameter.append(enemy_units[enemy].tag)
    #             parameter = tuple(parameter)
    #             actions.append(a(*parameter))
    #
    #     for i in range(config.MY_UNIT_NUMBER - my_raw_units_lenth):
    #         action_numbers.append(0)
    #
    # return actions, action_numbers


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

    init_my_units_tag = [unit.tag for unit in init_obs.observation['raw_units'] if unit.alliance == features.PlayerRelative.SELF]

    state_copy = get_state(init_obs, obs)

    for i in range(config.MY_UNIT_NUMBER):
        my_unit = find_unit_by_tag(obs, init_my_units_tag[i])
        if my_unit is None:
            states.append(np.zeros(config.COOP_AGENTS_OBDIM))
        else:
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
