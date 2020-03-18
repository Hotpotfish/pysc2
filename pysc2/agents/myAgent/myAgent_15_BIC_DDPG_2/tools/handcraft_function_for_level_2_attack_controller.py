import math

import itertools
from sklearn.cluster import DBSCAN
import numpy as np
import pysc2.agents.myAgent.myAgent_15_BIC_DDPG_2.smart_actions as sa

from pysc2.agents.myAgent.myAgent_15_BIC_DDPG_2.config import config
from pysc2.agents.myAgent.myAgent_15_BIC_DDPG_2.tools import unit_list
from pysc2.lib import features
from pysc2.lib import actions as a


def get_all_vaild_action():
    action_tpye_len = len(sa.attack_controller)
    action_dicts = {}
    raw_cmd_action = []
    raw_cmd_pt_action = []
    raw_cmd_unit_action = []
    raw_cmd_action_real = []
    raw_cmd_pt_action_real = []
    raw_cmd_unit_action_real = []
    i1, i2, i3 = 0, 0, 0
    for i in range(action_tpye_len):
        action = sa.attack_controller[i]

        if len(action.args) == 2 and action.args[0].name == 'queued' and action.args[1].name == 'unit_tags':
            function_id_1 = [i1]

            function_id_2 = [1e-10]
            x_2 = [1e-10]
            y_2 = [1e-10]

            function_id_3 = [1e-10]
            target_3 = [1e-10]

            for item in itertools.product(function_id_1, function_id_2, x_2, y_2, function_id_3, target_3):
                raw_cmd_action.append(item)
            raw_cmd_action_real.append(action)
            i1 += 1
        elif len(action.args) == 3 and action.args[0].name == 'queued' and action.args[1].name == 'unit_tags' and action.args[2].name == 'world':
            function_id_1 = [1e-10]

            function_id_2 = [i2]
            x_2 = range(config.MAP_SIZE)
            y_2 = range(config.MAP_SIZE)

            function_id_3 = [1e-10]
            target_3 = [1e-10]
            for item in itertools.product(function_id_1, function_id_2, x_2, y_2, function_id_3, target_3):
                raw_cmd_pt_action.append(item)
            raw_cmd_pt_action_real.append(action)
            i2 += 1

        elif len(action.args) == 3 and action.args[0].name == 'queued' and action.args[1].name == 'unit_tags' and action.args[2].name == 'target_unit_tag':
            function_id_1 = [1e-10]
            function_id_2 = [1e-10]
            x_2 = [1e-10]
            y_2 = [1e-10]

            function_id_3 = [i3]
            target_3 = range(config.MY_UNIT_NUMBER + config.ENEMY_UNIT_NUMBER)
            for item in itertools.product(function_id_1, function_id_2, x_2, y_2, function_id_3, target_3):
                raw_cmd_unit_action.append(item)
            raw_cmd_unit_action_real.append(action)
            i3 += 1

    action_dicts.update({'raw_cmd_action': raw_cmd_action, 'raw_cmd_pt_action': raw_cmd_pt_action, 'raw_cmd_unit_action': raw_cmd_unit_action,
                         'raw_cmd_action_real': raw_cmd_action_real, 'raw_cmd_pt_action_real': raw_cmd_pt_action_real, 'raw_cmd_unit_action_real': raw_cmd_unit_action_real})
    return action_dicts


def get_max_vaild_action_distance(vaild_action):
    raw_cmd_action_len = len(vaild_action['raw_cmd_action'])
    raw_cmd_pt_action_len = len(vaild_action['raw_cmd_pt_action'])
    raw_cmd_unit_action_len = len(vaild_action['raw_cmd_unit_action'])
    max_distance = [raw_cmd_action_len,
                    np.sqrt(np.power(raw_cmd_pt_action_len, 2) + np.power(config.MAP_SIZE, 2) + np.power(config.MAP_SIZE, 2)),
                    np.sqrt(np.power(raw_cmd_unit_action_len, 2) + np.power(config.MY_UNIT_NUMBER + config.ENEMY_UNIT_NUMBER, 2))]
    return max_distance

    # 找出最接近每个智能体输出的动作


def get_k_closest_action(vaild_action, max_vaild_action_distance, KDTrees, proto_action):
    raw_cmd_tree = KDTrees[0]
    raw_cmd_pt_tree = KDTrees[1]
    raw_cmd_unit_tree = KDTrees[2]
    if config.K > len(vaild_action['raw_cmd_action']):
        raw_cmd_temp = raw_cmd_tree.query(proto_action[:, 0][:, np.newaxis], k=len(vaild_action['raw_cmd_action']))
    else:
        raw_cmd_temp = raw_cmd_tree.query(proto_action[:, 0][:, np.newaxis], k=config.K)
    if config.K > len(vaild_action['raw_cmd_pt_action']):
        raw_cmd_pt_temp = raw_cmd_pt_tree.query(proto_action[:, 1:4], k=len(vaild_action['raw_cmd_pt_action']))
    else:
        raw_cmd_pt_temp = raw_cmd_pt_tree.query(proto_action[:, 1:4], k=config.K)

    if config.K > len(vaild_action['raw_cmd_unit_action']):
        raw_cmd_unit_temp = raw_cmd_unit_tree.query(proto_action[:, 4:6], k=len(vaild_action['raw_cmd_unit_action']))
    else:
        raw_cmd_unit_temp = raw_cmd_unit_tree.query(proto_action[:, 4:6], k=config.K)

    actions = []

    for i in range(config.MY_UNIT_NUMBER):
        action = []

        if len(vaild_action['raw_cmd_action']) == 1:
            raw_cmd_action = list(zip([raw_cmd_temp[0][i] / max_vaild_action_distance[0]], np.array(vaild_action['raw_cmd_action'])[raw_cmd_temp[1][i]][np.newaxis]))
        else:
            raw_cmd_action = list(zip(raw_cmd_temp[0][i] / max_vaild_action_distance[0], np.array(vaild_action['raw_cmd_action'])[raw_cmd_temp[1][i]]))
        if len(vaild_action['raw_cmd_pt_action']) == 1:
            raw_cmd_pt_action = list(zip([raw_cmd_pt_temp[0][i] / max_vaild_action_distance[1]], np.array(vaild_action['raw_cmd_pt_action'])[raw_cmd_pt_temp[1][i]][np.newaxis]))
        else:
            raw_cmd_pt_action = list(zip(raw_cmd_pt_temp[0][i] / max_vaild_action_distance[1], np.array(vaild_action['raw_cmd_pt_action'])[raw_cmd_pt_temp[1][i]]))
        if len(vaild_action['raw_cmd_unit_action']) == 1:
            raw_cmd_unit_action = list(zip([raw_cmd_unit_temp[0][i] / max_vaild_action_distance[2]], np.array(vaild_action['raw_cmd_unit_action'])[raw_cmd_unit_temp[1][i]][np.newaxis]))
        else:
            raw_cmd_unit_action = list(zip(raw_cmd_unit_temp[0][i] / max_vaild_action_distance[2], np.array(vaild_action['raw_cmd_unit_action'])[raw_cmd_unit_temp[1][i]]))

        # if config.K == 1:
        #     raw_cmd_action = list(zip([raw_cmd_temp[0][i] / max_vaild_action_distance[0]], np.array(vaild_action['raw_cmd_action'])[raw_cmd_temp[1][i]][np.newaxis]))
        #     raw_cmd_pt_action = list(zip([raw_cmd_pt_temp[0][i] / max_vaild_action_distance[1]], np.array(vaild_action['raw_cmd_pt_action'])[raw_cmd_pt_temp[1][i]][np.newaxis]))
        #     raw_cmd_unit_action = list(zip([raw_cmd_unit_temp[0][i] / max_vaild_action_distance[2]], np.array(vaild_action['raw_cmd_unit_action'])[raw_cmd_unit_temp[1][i]][np.newaxis]))
        # else:
        #     raw_cmd_action = list(zip(raw_cmd_temp[0][i] / max_vaild_action_distance[0], np.array(vaild_action['raw_cmd_action'])[raw_cmd_temp[1][i]]))
        #     raw_cmd_pt_action = list(zip(raw_cmd_pt_temp[0][i] / max_vaild_action_distance[1], np.array(vaild_action['raw_cmd_pt_action'])[raw_cmd_pt_temp[1][i]]))
        #     raw_cmd_unit_action = list(zip(raw_cmd_unit_temp[0][i] / max_vaild_action_distance[2], np.array(vaild_action['raw_cmd_unit_action'])[raw_cmd_unit_temp[1][i]]))
        action += raw_cmd_action
        action += raw_cmd_pt_action
        action += raw_cmd_unit_action
        action = sorted(action, key=(lambda x: x[0]))
        actions.append(np.array(action)[0:config.K, 1])

    return actions


def get_action_combination(vaild_action, max_vaild_action_distance, KDTrees, proto_action):
    k_closest_action = get_k_closest_action(vaild_action, max_vaild_action_distance, KDTrees, proto_action)
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


def get_bound(vaild_action):
    bound = [len(vaild_action['raw_cmd_action']),
             len(vaild_action['raw_cmd_pt_action']),
             config.MAP_SIZE,
             config.MAP_SIZE,
             len(vaild_action['raw_cmd_unit_action']),
             config.MY_UNIT_NUMBER + config.ENEMY_UNIT_NUMBER
             ]
    return bound


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

    init_my_units_tag = [unit.tag for unit in init_obs.observation['raw_units'] if unit.alliance == features.PlayerRelative.SELF]
    init_enemy_units_tag = [unit.tag for unit in init_obs.observation['raw_units'] if unit.alliance == features.PlayerRelative.ENEMY]

    for i in range(config.MY_UNIT_NUMBER):
        my_unit_pos = find_unit_pos(obs, init_my_units_tag[i])
        if my_unit_pos is None:
            continue
        parameter = []
        queued = 0
        parameter.append([queued])
        if np.all(action_numbers[i][1:] == 1e-10):

            function_id = int(vaild_action['raw_cmd_action_real'][int(action_numbers[i][0])].id)
            parameter.append([my_unit_pos])
            # parameter.append([action_numbers[i][1], action_numbers[i][2]])
            actions.append(a.FunctionCall(function_id, parameter))

        elif np.all(action_numbers[i][[0, 4, 5]] == 1e-10):
            function_id = int(vaild_action['raw_cmd_pt_action_real'][int(action_numbers[i][1])].id)
            parameter.append([my_unit_pos])
            parameter.append([int(action_numbers[i][2]), int(action_numbers[i][3])])
            actions.append(a.FunctionCall(function_id, parameter))

        elif np.all(action_numbers[i][0:4] == 1e-10):
            function_id = int(vaild_action['raw_cmd_unit_action_real'][int(action_numbers[i][4])].id)
            parameter.append([my_unit_pos])
            if int(action_numbers[i][5]) < config.MY_UNIT_NUMBER:
                target_unit_pos = find_unit_pos(obs, init_my_units_tag[int(action_numbers[i][5])])
            else:
                target_unit_pos = find_unit_pos(obs, init_enemy_units_tag[int(action_numbers[i][5]) - config.MY_UNIT_NUMBER])
            if target_unit_pos is None:
                continue
            else:
                parameter.append([target_unit_pos])
            actions.append(a.FunctionCall(function_id, parameter))

    return actions


def get_agent_state(unit):
    states = np.array([])

    states = np.append(states, computeDistance_center(unit) / (config.MAP_SIZE * 1.41))
    states = np.append(states, unit.alliance / 4)
    states = np.append(states, unit.unit_type / 2000)
    states = np.append(states, unit.x / config.MAP_SIZE)
    states = np.append(states, unit.y / config.MAP_SIZE)
    states = np.append(states, unit.health / 100)
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
        if i >= len(init_enemy_units_tag):
            print()
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
                agent_obs = np.append(agent_obs, my_target_unit.health / 100)
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


def get_reward(obs, pre_obs):
    reward = 0
    my_units = np.array([unit for unit in obs.observation['raw_units'] if unit.alliance == features.PlayerRelative.SELF])
    enemy_units = np.array([unit for unit in obs.observation['raw_units'] if unit.alliance == features.PlayerRelative.ENEMY])

    my_units_health_pre = np.array([unit for unit in pre_obs.observation['raw_units'] if unit.alliance == features.PlayerRelative.SELF])
    enemy_units_health_pre = np.array([unit for unit in pre_obs.observation['raw_units'] if unit.alliance == features.PlayerRelative.ENEMY])
    # 是否胜利
    if len(enemy_units) == 0:
        reward = sum(my_units[:, 2]) + sum(my_units[:, 3]) + 200
        return float(reward) / 200
    elif len(my_units) == 0:
        # reward = -sum(enemy_units[:, 2]) - sum(enemy_units[:, 3]) - 200
        return 0

    # 距离变化
    # my_coord = np.array(list(zip(my_units[:, 12], my_units[:, 13])))
    # emey_coord = np.array(list(zip(enemy_units[:, 12], enemy_units[:, 13])))
    # kdtree = KDTree(emey_coord)
    # distance_avg = kdtree.query(my_coord)
    # reward -= (abs(sum(distance_avg[0]) / len(my_units) - 4) / (config.MAP_SIZE * 1.41)) * 5

    # 人数变化
    # if len(my_units) < len(my_units_health_pre):
    #     reward -= ((len(my_units_health_pre) - len(my_units)) * 10) / 200
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
