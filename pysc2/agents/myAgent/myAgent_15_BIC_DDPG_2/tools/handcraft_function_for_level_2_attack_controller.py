import math

import itertools

from scipy.spatial import KDTree
from sklearn.cluster import DBSCAN
import numpy as np
import pysc2.agents.myAgent.myAgent_15_BIC_DDPG_2.smart_actions as sa

from pysc2.agents.myAgent.myAgent_15_BIC_DDPG_2.config import config
from pysc2.agents.myAgent.myAgent_15_BIC_DDPG_2.tools import unit_list
from pysc2.lib import features
from pysc2.lib import actions as a
from pysc2.agents.myAgent.myAgent_15_BIC_DDPG_2.tools.unit_actions import inquire_action as inquire_action


def get_init_obs(obs, init_static_agent_type):
    init_obs = []
    sorted_tag = []
    for i in range(config.MY_UNIT_NUMBER + config.ENEMY_UNIT_NUMBER):
        if i < config.MY_UNIT_NUMBER:
            for unit in obs.observation['raw_units']:
                if unit.alliance == features.PlayerRelative.SELF and unit.unit_type == init_static_agent_type[i] and unit.tag not in sorted_tag:
                    init_obs.append(unit)
                    sorted_tag.append(unit.tag)
                    break
                    # np.delete(obs.observation['raw_units'], np.argwhere(obs.observation['raw_units'] == unit))
        else:
            for unit in obs.observation['raw_units']:
                if unit.alliance == features.PlayerRelative.ENEMY and unit.unit_type == init_static_agent_type[i] and unit.tag not in sorted_tag:
                    init_obs.append(unit)
                    sorted_tag.append(unit.tag)
                    break
                    # np.delete(obs.observation['raw_units'], np.argwhere(obs.observation['raw_units'] == unit))
    obs.observation['raw_units'] = init_obs
    return obs

    # init_obs += [unit for unit in obs.observation['raw_units'] if unit.alliance == features.PlayerRelative.SELF and unit.unit_type == init_static_agent_type[i]]
    # init_my_units = [unit for unit in obs.observation['raw_units'] if unit.alliance == features.PlayerRelative.SELF]
    # init_enemy_units = [unit for unit in obs.observation['raw_units'] if unit.alliance == features.PlayerRelative.ENEMY]


def get_init_static_agent_type(obs):
    init_my_units_type = [unit.unit_type for unit in obs.observation['raw_units'] if unit.alliance == features.PlayerRelative.SELF]
    init_enemy_units_type = [unit.unit_type for unit in obs.observation['raw_units'] if unit.alliance == features.PlayerRelative.ENEMY]
    all_agent_type = init_my_units_type + init_enemy_units_type
    return all_agent_type


def get_specified_agent_all_valid_action(all_agent_type):
    all_agent_valid_action = []
    bound = []

    for i in range(config.MY_UNIT_NUMBER):
        action_tpye_len = len(inquire_action(all_agent_type[i]))
        agent_valid_action = []
        for j in range(action_tpye_len):
            action = inquire_action(all_agent_type[i])[j]
            if len(action.args) == 0:
                function_id_1 = [int(action.id)]

                function_id_2 = [1e-10]
                x_2 = [1e-10]
                y_2 = [1e-10]

                function_id_3 = [1e-10]
                target_3 = [1e-10]

                for item in itertools.product(function_id_1, function_id_2, x_2, y_2, function_id_3, target_3):
                    agent_valid_action.append(item)

            elif len(action.args) == 2 and action.args[0].name == 'queued' and action.args[1].name == 'unit_tags':
                function_id_1 = [int(action.id)]

                function_id_2 = [1e-10]
                x_2 = [1e-10]
                y_2 = [1e-10]

                function_id_3 = [1e-10]
                target_3 = [1e-10]

                for item in itertools.product(function_id_1, function_id_2, x_2, y_2, function_id_3, target_3):
                    agent_valid_action.append(item)

            elif len(action.args) == 3 and action.args[0].name == 'queued' and action.args[1].name == 'unit_tags' and action.args[2].name == 'world':
                function_id_1 = [1e-10]

                function_id_2 = [int(action.id)]
                x_2 = [-1, 1]
                y_2 = [-1, 1]

                function_id_3 = [1e-10]
                target_3 = [1e-10]
                for item in itertools.product(function_id_1, function_id_2, x_2, y_2, function_id_3, target_3):
                    agent_valid_action.append(item)

            elif len(action.args) == 3 and action.args[0].name == 'queued' and action.args[1].name == 'unit_tags' and action.args[2].name == 'target_unit_tag':
                function_id_1 = [1e-10]
                function_id_2 = [1e-10]
                x_2 = [1e-10]
                y_2 = [1e-10]

                function_id_3 = [int(int(action.id))]
                if int(action.id) == 311:
                    target_3 = range(config.MY_UNIT_NUMBER)
                else:
                    target_3 = range(config.MY_UNIT_NUMBER, config.MY_UNIT_NUMBER + config.ENEMY_UNIT_NUMBER)
                for item in itertools.product(function_id_1, function_id_2, x_2, y_2, function_id_3, target_3):
                    agent_valid_action.append(item)
        bound.append(len(agent_valid_action))
        all_agent_valid_action.append(agent_valid_action)
    return all_agent_valid_action, np.array(bound)


def get_single_agent_closest_action(agent_number, agent_local_observation, all_valid_action):
    # all_valid_action = np.array(all_valid_action)
    all_valid_action = list(all_valid_action)
    if np.all(agent_local_observation[(agent_number * 8):(agent_number * 8 + 8)] == 0):
        actions = np.array([0])
        return actions
    agent_tpye = int(agent_local_observation[8 * agent_number + 2] * 2000)
    # if  agent_tpye == 54:
    #     print()
    agent_valid_actions = inquire_action(agent_tpye)
    # if  agent_valid_actions is None:
    #     print()
    actions = []
    action_tpye_len = len(agent_valid_actions)

    for i in range(action_tpye_len):
        action = agent_valid_actions[i]
        if len(action.args) == 0:
            # function_id_1 = [int(action.id)]
            #
            # function_id_2 = [1e-10]
            # x_2 = [1e-10]
            # y_2 = [1e-10]
            #
            # function_id_3 = [1e-10]
            # target_3 = [1e-10]
            #
            # for item in itertools.product(function_id_1, function_id_2, x_2, y_2, function_id_3, target_3):
            #     actions.append(all_valid_action.index(item))
            continue
        if len(action.args) == 2 and action.args[0].name == 'queued' and action.args[1].name == 'unit_tags':
            function_id_1 = [int(action.id)]

            function_id_2 = [1e-10]
            x_2 = [1e-10]
            y_2 = [1e-10]

            function_id_3 = [1e-10]
            target_3 = [1e-10]

            for item in itertools.product(function_id_1, function_id_2, x_2, y_2, function_id_3, target_3):
                actions.append(all_valid_action.index(item))
            continue
        elif len(action.args) == 3 and action.args[0].name == 'queued' and action.args[1].name == 'unit_tags' and action.args[2].name == 'world':
            function_id_1 = [1e-10]

            function_id_2 = [int(action.id)]
            x_2 = [-1, 1]
            y_2 = [-1, 1]

            function_id_3 = [1e-10]
            target_3 = [1e-10]
            for item in itertools.product(function_id_1, function_id_2, x_2, y_2, function_id_3, target_3):
                actions.append(all_valid_action.index(item))
            continue

        elif len(action.args) == 3 and action.args[0].name == 'queued' and action.args[1].name == 'unit_tags' and action.args[2].name == 'target_unit_tag':
            function_id_1 = [1e-10]
            function_id_2 = [1e-10]
            x_2 = [1e-10]
            y_2 = [1e-10]

            function_id_3 = [int(action.id)]
            target_3 = []
            if int(action.id) == 311:
                for j in range(config.MY_UNIT_NUMBER):
                    if j == agent_number or np.all(agent_local_observation[(j * 8):(j * 8 + 8)] == 0) or agent_local_observation[j * 8] * (config.MAP_SIZE * 1.41) >= config.ATTACK_RANGE:
                        continue
                    else:
                        target_3.append(j)
            else:
                for j in range(config.MY_UNIT_NUMBER, config.MY_UNIT_NUMBER + config.ENEMY_UNIT_NUMBER):
                    if j == agent_number or np.all(agent_local_observation[(j * 8):(j * 8 + 8)] == 0) or agent_local_observation[j * 8] * (config.MAP_SIZE * 1.41) >= config.ATTACK_RANGE:
                        continue
                    else:
                        target_3.append(j)
            if len(target_3) == 0:
                continue

            # target_3 = range(config.MY_UNIT_NUMBER + config.ENEMY_UNIT_NUMBER)
            for item in itertools.product(function_id_1, function_id_2, x_2, y_2, function_id_3, target_3):
                actions.append(all_valid_action.index(item))
            continue

    actions = np.array(actions)
    # agent_valid_actions_number = np.where((all_valid_action == actions[None]).all(-1))[1]
    # kdtree = KDTree(range(len(actions))[:, np.newaxis])

    # print()

    return actions


def agent_k_closest_action(agent_valid_actions, proto_action):
    kdtree = KDTree(agent_valid_actions[:, np.newaxis])
    action = []
    if config.K >= kdtree.n:

        temp_r = kdtree.query(proto_action, k=kdtree.n)
    else:
        temp_r = kdtree.query(proto_action, k=config.K)
    if config.K == 1:
        action.append(agent_valid_actions[kdtree.query(proto_action, k=config.K)[1]])
    else:
        action = temp_r[1]
    # if len(action) > 1:
    #     print()

    return action


def get_all_vaild_action():
    actions = []
    action_tpye_len = len(sa.attack_controller)
    for i in range(action_tpye_len):
        action = sa.attack_controller[i]
        if len(action.args) == 0:
            function_id_1 = [int(action.id)]

            function_id_2 = [1e-10]
            x_2 = [1e-10]
            y_2 = [1e-10]

            function_id_3 = [1e-10]
            target_3 = [1e-10]

            for item in itertools.product(function_id_1, function_id_2, x_2, y_2, function_id_3, target_3):
                actions.append(item)
            continue

        elif len(action.args) == 2 and action.args[0].name == 'queued' and action.args[1].name == 'unit_tags':
            function_id_1 = [int(action.id)]

            function_id_2 = [1e-10]
            x_2 = [1e-10]
            y_2 = [1e-10]

            function_id_3 = [1e-10]
            target_3 = [1e-10]

            for item in itertools.product(function_id_1, function_id_2, x_2, y_2, function_id_3, target_3):
                actions.append(item)
            continue
        elif len(action.args) == 3 and action.args[0].name == 'queued' and action.args[1].name == 'unit_tags' and action.args[2].name == 'world':
            function_id_1 = [1e-10]

            function_id_2 = [int(action.id)]
            x_2 = [-1, 1]
            y_2 = [-1, 1]

            function_id_3 = [1e-10]
            target_3 = [1e-10]
            for item in itertools.product(function_id_1, function_id_2, x_2, y_2, function_id_3, target_3):
                actions.append(item)
            continue

        elif len(action.args) == 3 and action.args[0].name == 'queued' and action.args[1].name == 'unit_tags' and action.args[2].name == 'target_unit_tag':
            function_id_1 = [1e-10]
            function_id_2 = [1e-10]
            x_2 = [1e-10]
            y_2 = [1e-10]

            function_id_3 = [int(int(action.id))]
            if int(action.id) == 311:
                target_3 = range(config.MY_UNIT_NUMBER)
            else:
                target_3 = range(config.MY_UNIT_NUMBER, config.MY_UNIT_NUMBER + config.ENEMY_UNIT_NUMBER)
            for item in itertools.product(function_id_1, function_id_2, x_2, y_2, function_id_3, target_3):
                actions.append(item)
            continue
    return actions


def get_k_closest_action(KDTree, proto_action):
    actions = []

    for i in range(config.MY_UNIT_NUMBER):
        action = []
        if config.K >= KDTree.n:

            temp_r = KDTree.query(proto_action[i], k=KDTree.n)
        else:
            temp_r = KDTree.query(proto_action[i], k=config.K)
        if config.K == 1:
            action.append(temp_r[1])
        else:
            action = temp_r[1]
        actions.append(action)
    return actions


def get_action_combination(KDTree, proto_action):
    # proto_action = proto_action + (KDTree.n - 1) / 2
    # print(proto_action)
    k_closest_action = get_k_closest_action(KDTree, proto_action)
    action_combination = []
    for item in itertools.product(*k_closest_action):
        action_combination.append(np.array(list(item)))
    return action_combination


# 十进制转任意进制 用于解析动作列表
def transport(action_number, action_dim):
    result = np.zeros(config.MY_UNIT_NUMBER)
    an = action_number
    ad = action_dim

    for i in range(config.MY_UNIT_NUMBER):
        if an / ad != 0:
            result[config.MY_UNIT_NUMBER - i - 1] = int(an % ad)
            an = int(an / ad)
        else:
            break
    return result


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


def find_unit_by_tag(obs, tag):
    for unit in obs.observation['raw_units']:
        if unit.tag == tag:
            return unit
    return None


def find_unit_pos(obs, tag):
    for i in range(len(obs.observation['raw_units'])):
        if obs.observation['raw_units'][i].tag == tag:
            return i
    else:
        return None


############################################

def assembly_action(init_obs, obs, action_numbers, vaild_action):
    actions = []

    init_my_units = [unit for unit in init_obs.observation['raw_units'] if unit.alliance == features.PlayerRelative.SELF]
    init_enemy_units = [unit for unit in init_obs.observation['raw_units'] if unit.alliance == features.PlayerRelative.ENEMY]

    for i in range(config.MY_UNIT_NUMBER):
        my_unit_pos = find_unit_pos(obs, init_my_units[i].tag)
        if my_unit_pos is None:
            continue
        my_unit = find_unit_by_tag(obs, init_my_units[i].tag)
        # if my_unit.unit_type == 54:
        #     print()
        parameter = []
        queued = 0
        parameter.append([queued])
        # print(action_numbers[i])
        if np.all(np.array(vaild_action[i][action_numbers[i]])[1:] == 1e-10):

            function_id = int(vaild_action[i][action_numbers[i]][0])
            parameter.append([my_unit_pos])
            # parameter.append([action_numbers[i][1], action_numbers[i][2]])

            if function_id == 0:
                actions.append(a.FunctionCall(function_id, []))
            else:
                actions.append(a.FunctionCall(function_id, parameter))

        elif np.all(np.array(vaild_action[i][action_numbers[i]])[[0, 4, 5]] == 1e-10):
            function_id = int(vaild_action[i][action_numbers[i]][1])
            parameter.append([my_unit_pos])

            parameter.append([vaild_action[i][action_numbers[i]][2] + my_unit.x, vaild_action[i][action_numbers[i]][3] + my_unit.y])
            # print()
            actions.append(a.FunctionCall(function_id, parameter))

        elif np.all(np.array(vaild_action[i][action_numbers[i]])[0:4] == 1e-10):
            function_id = int(vaild_action[i][action_numbers[i]][4])
            # if function_id == 311:
            #     print()
            parameter.append([my_unit_pos])
            if np.array(vaild_action[i][action_numbers[i]])[5] < config.MY_UNIT_NUMBER:
                target_unit_pos = find_unit_pos(obs, init_my_units[vaild_action[i][action_numbers[i]][5]].tag)
            else:
                target_unit_pos = find_unit_pos(obs, init_enemy_units[vaild_action[i][action_numbers[i]][5] - config.MY_UNIT_NUMBER].tag)
            if target_unit_pos is None:
                continue
            else:
                parameter.append([target_unit_pos])

            actions.append(a.FunctionCall(function_id, parameter))
    # print(actions)

    return actions


def get_agent_state(unit):
    states = np.array([])

    states = np.append(states, computeDistance_center(unit) / (config.MAP_SIZE * 1.41))
    states = np.append(states, unit.alliance / 4)
    states = np.append(states, unit.unit_type / 2000)
    states = np.append(states, unit.x / config.MAP_SIZE)
    states = np.append(states, unit.y / config.MAP_SIZE)
    states = np.append(states, unit.health / 200)
    states = np.append(states, unit.shield / 100)
    states = np.append(states, unit.weapon_cooldown / 10)
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
            state = np.append(state, np.zeros(config.COOP_AGENT_OBDIM))

    for i in range(config.ENEMY_UNIT_NUMBER):
        # if i >= len(init_enemy_units_tag):
        #     print()
        enemy_unit = find_unit_by_tag(obs, init_enemy_units_tag[i])
        if enemy_unit is not None:
            my_unit_state = get_agent_state(enemy_unit)
            state = np.append(state, my_unit_state)
        else:
            state = np.append(state, np.zeros(config.COOP_AGENT_OBDIM))
    return state


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
                agent_obs = np.append(agent_obs, np.zeros(8))
            else:
                agent_obs = np.append(agent_obs, computeDistance(my_unit, my_target_unit) / (config.MAP_SIZE * 1.41))
                agent_obs = np.append(agent_obs, my_target_unit.alliance / 4)
                agent_obs = np.append(agent_obs, my_target_unit.unit_type / 2000)
                agent_obs = np.append(agent_obs, my_target_unit.x / config.MAP_SIZE)
                agent_obs = np.append(agent_obs, my_target_unit.y / config.MAP_SIZE)
                agent_obs = np.append(agent_obs, my_target_unit.health / 200)
                agent_obs = np.append(agent_obs, my_target_unit.shield / 100)
                agent_obs = np.append(agent_obs, my_target_unit.weapon_cooldown / 10)
        for j in range(config.ENEMY_UNIT_NUMBER):
            enemy_target_unit = find_unit_by_tag(obs, init_enemy_units_tag[j])
            # 按顺序遍历每个己方单位的信息
            if enemy_target_unit is None or computeDistance(my_unit, enemy_target_unit) >= config.OB_RANGE:
                agent_obs = np.append(agent_obs, np.zeros(8))
            else:
                agent_obs = np.append(agent_obs, computeDistance(my_unit, enemy_target_unit) / (config.MAP_SIZE * 1.41))
                agent_obs = np.append(agent_obs, enemy_target_unit.alliance / 4)
                agent_obs = np.append(agent_obs, enemy_target_unit.unit_type / 2000)
                agent_obs = np.append(agent_obs, enemy_target_unit.x / config.MAP_SIZE)
                agent_obs = np.append(agent_obs, enemy_target_unit.y / config.MAP_SIZE)
                agent_obs = np.append(agent_obs, enemy_target_unit.health / 100)
                agent_obs = np.append(agent_obs, enemy_target_unit.shield / 100)
                agent_obs = np.append(agent_obs, enemy_target_unit.weapon_cooldown / 10)

        agents_obs.append(agent_obs)
    return agents_obs


# def get_reward(obs, pre_obs):
#     reward = 0
#     my_units = np.array([unit for unit in obs.observation['raw_units'] if unit.alliance == features.PlayerRelative.SELF])
#     enemy_units = np.array([unit for unit in obs.observation['raw_units'] if unit.alliance == features.PlayerRelative.ENEMY])
#
#     my_units_health_pre = np.array([unit for unit in pre_obs.observation['raw_units'] if unit.alliance == features.PlayerRelative.SELF])
#     enemy_units_health_pre = np.array([unit for unit in pre_obs.observation['raw_units'] if unit.alliance == features.PlayerRelative.ENEMY])
#     # 是否胜利
#     if obs.last():
#         if len(enemy_units) == 0:
#             reward += sum(my_units[:, 2]) + sum(my_units[:, 3]) + 200
#             return float(reward) / 200
#         elif len(my_units) == 0:
#             # reward = -sum(enemy_units[:, 2]) - sum(enemy_units[:, 3]) - 200
#             return 0
#
#     # 距离变化
#     # my_coord = np.array(list(zip(my_units[:, 12], my_units[:, 13])))
#     # emey_coord = np.array(list(zip(enemy_units[:, 12], enemy_units[:, 13])))
#     # kdtree = KDTree(emey_coord)
#     # distance_avg = kdtree.query(my_coord)
#     # reward -= (abs(sum(distance_avg[0]) / len(my_units) - 4) / (config.MAP_SIZE * 1.41)) * 5
#
#     # 人数变化
#     # if len(my_units) < len(my_units_health_pre):
#     #     reward -= ((len(my_units_health_pre) - len(my_units)) * 10) / 200
#     if len(enemy_units) < len(enemy_units_health_pre):
#         reward += ((len(enemy_units_health_pre) - len(enemy_units)) * 10) / 200
#
#     # 血量与护盾变化
#     reward += ((sum(my_units[:, 2]) - sum(my_units_health_pre[:, 2])) / 2 - (sum(enemy_units[:, 2]) - sum(enemy_units_health_pre[:, 2]))) / 200
#     reward += ((sum(my_units[:, 3]) - sum(my_units_health_pre[:, 3])) / 2 - (sum(enemy_units[:, 3]) - sum(enemy_units_health_pre[:, 3]))) / 200
#
#     return reward
def get_reward(obs, pre_obs):
    reward = 0
    my_units = np.array([unit for unit in obs.observation['raw_units'] if unit.alliance == features.PlayerRelative.SELF])
    enemy_units = np.array([unit for unit in obs.observation['raw_units'] if unit.alliance == features.PlayerRelative.ENEMY])

    my_units_health_pre = np.array([unit for unit in pre_obs.observation['raw_units'] if unit.alliance == features.PlayerRelative.SELF])
    enemy_units_health_pre = np.array([unit for unit in pre_obs.observation['raw_units'] if unit.alliance == features.PlayerRelative.ENEMY])
    # 是否胜利
    if obs.last():
        if len(enemy_units) == 0:
            reward += sum(my_units[:, 2]) + sum(my_units[:, 3]) + 200
            return float(reward) / 200
        elif len(my_units) == 0:
            reward = -sum(enemy_units[:, 2]) - sum(enemy_units[:, 3]) - 200
            return float(reward) / 200

    # 距离变化
    # my_coord = np.array(list(zip(my_units[:, 12], my_units[:, 13])))
    # emey_coord = np.array(list(zip(enemy_units[:, 12], enemy_units[:, 13])))
    # kdtree = KDTree(emey_coord)
    # distance_avg = kdtree.query(my_coord)
    # reward -= (abs(sum(distance_avg[0]) / len(my_units) - 4) / (config.MAP_SIZE * 1.41)) * 5

    # 人数变化
    if len(my_units) < len(my_units_health_pre):
        reward -= ((len(my_units_health_pre) - len(my_units)) * 10) / 200
    if len(enemy_units) < len(enemy_units_health_pre):
        reward += ((len(enemy_units_health_pre) - len(enemy_units)) * 10) / 200

    # 血量与护盾变化
    reward += ((sum(my_units[:, 2]) - sum(my_units_health_pre[:, 2])) / 2 - (sum(enemy_units[:, 2]) - sum(enemy_units_health_pre[:, 2]))) / 200
    reward += ((sum(my_units[:, 3]) - sum(my_units_health_pre[:, 3])) / 2 - (sum(enemy_units[:, 3]) - sum(enemy_units_health_pre[:, 3]))) / 200

    return reward


def win_or_loss(obs):
    if obs.last():

        # my_units = [unit for unit in obs.observation['raw_units'] if unit.alliance == features.PlayerRelative.SELF]
        enemy_units = [unit for unit in obs.observation['raw_units'] if unit.alliance == features.PlayerRelative.ENEMY]

        if len(enemy_units) == 0:
            return 1
        else:
            return -1
    else:
        return 0


############test############

def get_bound_test(my_units, enemy_units):
    bound = np.zeros(np.power(config.ATTACT_CONTROLLER_ACTIONDIM, config.MY_UNIT_NUMBER))
    leagal_actions = []

    for i in range(config.MY_UNIT_NUMBER):

        if i >= len(my_units):
            leagal_actions.append([0])
            continue
        else:
            action = []
            action.append(0)
            for j in range(config.ENEMY_UNIT_NUMBER):

                if j >= len(enemy_units):
                    action.append(0)
                else:
                    action.append(1)
            leagal_actions.append(list(np.nonzero(action)[0]))

    for i in itertools.product(*leagal_actions):
        number = 0
        for j in range(config.MY_UNIT_NUMBER):
            number += i[j] * np.power(config.ATTACT_CONTROLLER_ACTIONDIM, config.MY_UNIT_NUMBER - 1 - j)
        bound[number] = 1
    return bound


def get_state_test(my_units, enemy_units):
    state = np.array([])
    for i in range(config.MY_UNIT_NUMBER):
        if i < len(my_units):
            my_unit_state = get_agent_state(my_units[i])
            state = np.append(state, my_unit_state)
        else:
            state = np.append(state, np.zeros(config.COOP_AGENT_OBDIM))

    for i in range(config.ENEMY_UNIT_NUMBER):
        if i < len(enemy_units):
            my_unit_state = get_agent_state(enemy_units[i])
            state = np.append(state, my_unit_state)
        else:
            state = np.append(state, np.zeros(config.COOP_AGENT_OBDIM))
    return state


def compute_distance_test(x, y, units):
    temp = []
    for i in range(len(units)):
        x_difference = math.pow(units[i].x - x, 2)
        y_difference = math.pow(units[i].y - y, 2)
        distance = math.sqrt(x_difference + y_difference)
        temp.append((units[i], distance))

    return sorted(temp, key=lambda y: y[1])


def get_bounds_and_states(obs_new):
    bounds_and_states = []
    my_units_and_enemy_units_pack = []

    for i in range(len(obs_new)):

        sort_my_units = compute_distance_test(0, 0, obs_new[i][0])

        while len(sort_my_units) != 0:
            my_units = []
            enemy_units = []
            x = 0
            y = 0
            for j in range(config.MY_UNIT_NUMBER):
                if len(sort_my_units) == 0:
                    continue
                my_unit = sort_my_units.pop()[0]
                x += my_unit.x
                y += my_unit.y
                my_units.append(my_unit)
            x = x / config.MY_UNIT_NUMBER
            y = y / config.MY_UNIT_NUMBER

            sort_enemy_units = compute_distance_test(x, y, obs_new[i][1])
            for j in range(config.ENEMY_UNIT_NUMBER):
                if j >= len(sort_enemy_units):
                    break
                enemy_units.append(sort_enemy_units[j][0])

            my_units_and_enemy_units_pack.append([my_units, enemy_units])
            bounds_and_states.append([get_bound_test(my_units, enemy_units), get_state_test(my_units, enemy_units)])

    return bounds_and_states, my_units_and_enemy_units_pack


def assembly_action_test(my_units, enemy_units, action_number):
    actions = []

    controller = sa.attack_controller

    action_nmbers = transport(action_number, config.ATTACT_CONTROLLER_ACTIONDIM)

    for i in range(config.MY_UNIT_NUMBER):
        if action_nmbers[i] == 0:
            continue
        else:
            parameter = []
            my_unit = my_units[i]

            if 0 < action_nmbers[i] <= config.ENEMY_UNIT_NUMBER:
                a = controller[2]
                enemy = int(action_nmbers[i] - config.DEATH_ACTION_DIM)
                parameter.append(0)
                parameter.append(my_unit.tag)
                parameter.append(enemy_units[enemy].tag)
                parameter = tuple(parameter)
                actions.append(a(*parameter))
    return actions


# 获得聚类的战场划分，并且进行第一次打包返回
def get_clusters_test(obs):
    obs_new = []
    enemies = [unit for unit in obs.observation['raw_units'] if unit.alliance == features.PlayerRelative.ENEMY]
    if len(enemies) == 0:
        return obs_new
    else:
        # 对于目前的观察进行聚类
        coordinate = list(zip(obs.observation['raw_units'][:, 12], obs.observation['raw_units'][:, 13]))
        results = DBSCAN(eps=30, min_samples=2, metric='euclidean').fit_predict(coordinate)

        # 将聚类结果打上标记
        cluster = {}
        for i in range(len(results)):
            cluster[str(obs.observation.raw_units[i])] = results[i]

        # 此时进行战区状态的划分
        i = 0
        while 1:

            my_units = [unit for unit in obs.observation['raw_units'] if unit.alliance == features.PlayerRelative.SELF and
                        unit.unit_type in unit_list.combat_unit and
                        cluster[str(unit)] == i
                        ]
            enemy_units = [unit for unit in obs.observation['raw_units'] if unit.alliance == features.PlayerRelative.ENEMY and cluster[str(unit)] == i]

            if (len(my_units) == 0 and len(enemy_units) != 0) or (len(my_units) != 0 and len(enemy_units) == 0):
                i += 1
                continue
            elif len(my_units) == 0 and len(enemy_units) == 0:
                break
            else:
                obs_new.append([my_units, enemy_units])
                i += 1

        return obs_new
