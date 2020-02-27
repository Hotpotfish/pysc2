import math

import itertools
from sklearn.cluster import DBSCAN
import numpy as np
import pysc2.agents.myAgent.myAgent_15_BIC_DDPG_2.smart_actions as sa

from pysc2.agents.myAgent.myAgent_15_BIC_DDPG_2.config import config
from pysc2.agents.myAgent.myAgent_15_BIC_DDPG_2.tools import unit_list
from pysc2.lib import features
from pysc2.lib import actions as a
from scipy.spatial import KDTree


def get_all_vaild_action():
    action_tpye_len = len(sa.attack_controller)
    actions = []

    for i in range(action_tpye_len):
        temp_function = sa.attack_controller[i]
        function_id = [i]  #
        # queued = [0]
        target = [0]  #
        x = [0]  #
        y = [0]  #

        for arg in temp_function.args:
            if arg.name == 'target_unit_tag':
                target = range(config.MY_UNIT_NUMBER + config.ENEMY_UNIT_NUMBER)
            if arg.name == 'world':
                x = range(config.MAP_SIZE)
                y = range(config.MAP_SIZE)
        for item in itertools.product(function_id, target, x, y):
            actions.append(list(item))
    return actions


# 找出最接近每个智能体输出的动作
def get_k_closest_action(vaild_action, proto_action):
    tree = KDTree(vaild_action)
    temp_r = tree.query(proto_action, k=config.K)
    k_closest_action = []
    for i in range(config.MY_UNIT_NUMBER):
        action = []
        if config.K == 1:
            action.append(vaild_action[temp_r[1][i]])
        else:
            for j in range(config.K):
                action.append(vaild_action[temp_r[1][i][j]])
        k_closest_action.append(action)

    return k_closest_action

    # agents_action_dicts = []
    # for i in range(config.MY_UNIT_NUMBER):
    #     agents_action_dicts.append({})
    #
    # for i in range(len(vaild_action)):
    #     for j in range(config.MY_UNIT_NUMBER):
    #         distance = np.linalg.norm(np.array(vaild_action[i]) - np.array(proto_action[j]))
    #         agents_action_dicts[j].update({(str(vaild_action[i])): distance})
    #
    # action_distance_sorted = []
    # for i in range(config.MY_UNIT_NUMBER):
    #     r = sorted(agents_action_dicts[i].items(), key=lambda x: x[1])
    #     action_distance_sorted.append(r)
    #
    # k_closest_action = []
    # for i in range(config.MY_UNIT_NUMBER):
    #     actions = []
    #     for j in range(config.K):
    #         action = action_distance_sorted[i][j][0].replace('[', '')
    #         action = action.replace(']', '')
    #         action = action.split(',')
    #         actions.append(np.array(list(map(int, action))))
    #     k_closest_action.append(actions)
    # return np.array(k_closest_action)


def get_action_combination(vaild_action, proto_action):
    k_closest_action = get_k_closest_action(vaild_action, proto_action)
    action_combination = []
    for item in itertools.product(*k_closest_action):
        action_combination.append(np.array(list(item)))
    return np.array(action_combination)


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


def get_bound():
    bound = [len(sa.attack_controller), config.ENEMY_UNIT_NUMBER, config.MAP_SIZE, config.MAP_SIZE]
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

def assembly_action(init_obs, obs, action_numbers):
    actions = []

    init_my_units_tag = [unit.tag for unit in init_obs.observation['raw_units'] if unit.alliance == features.PlayerRelative.SELF]
    init_enemy_units_tag = [unit.tag for unit in init_obs.observation['raw_units'] if unit.alliance == features.PlayerRelative.ENEMY]
    # controller = sa.attack_controller

    for i in range(config.MY_UNIT_NUMBER):
        my_unit_pos = find_unit_pos(obs, init_my_units_tag[i])
        if my_unit_pos == None:
            continue
        parameter = []
        function_id = int(sa.attack_controller[action_numbers[i][0]].id)
        queued = 0
        action = sa.attack_controller[action_numbers[i][0]]
        for arg in action.args:
            if arg.name == 'queued':
                parameter.append([queued])
                continue
            if arg.name == 'unit_tags':
                parameter.append([my_unit_pos])
                continue
            if arg.name == 'target_unit_tag':
                if action_numbers[i][1] < config.MY_UNIT_NUMBER:
                    target_unit_pos = find_unit_pos(obs, init_my_units_tag[action_numbers[i][1]])
                else:
                    target_unit_pos = find_unit_pos(obs, init_enemy_units_tag[action_numbers[i][1] - config.MY_UNIT_NUMBER])

                if target_unit_pos == None:
                    parameter.clear()
                    break
                else:
                    parameter.append([target_unit_pos])
                    continue
            if arg.name == 'world':
                # coordinate = [action_numbers[i][2],action_numbers[i][3]]
                parameter.append([action_numbers[i][2], action_numbers[i][3]])
        if len(parameter) == 0:
            continue
        else:
            actions.append(a.FunctionCall(function_id, parameter))
    return actions


def get_agent_state(unit):
    states = np.array([])

    states = np.append(states, computeDistance_center(unit) / (config.MAP_SIZE * 1.41))
    states = np.append(states, unit.alliance / 4)
    states = np.append(states, unit.unit_type / 10000)
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
                agent_obs = np.append(agent_obs, my_target_unit.unit_type / 10000)
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
                agent_obs = np.append(agent_obs, enemy_target_unit.unit_type / 10000)
                agent_obs = np.append(agent_obs, enemy_target_unit.x / config.MAP_SIZE)
                agent_obs = np.append(agent_obs, enemy_target_unit.y / config.MAP_SIZE)
                agent_obs = np.append(agent_obs, enemy_target_unit.health / 100)
                agent_obs = np.append(agent_obs, enemy_target_unit.shield / 100)
                agent_obs = np.append(agent_obs, enemy_target_unit.weapon_cooldown / 10)

        agents_obs.append(agent_obs)
    return agents_obs


def get_reward(obs, pre_obs):
    reward = 0
    my_units_health = [unit.health for unit in obs.observation['raw_units'] if unit.alliance == features.PlayerRelative.SELF]
    enemy_units_health = [unit.health for unit in obs.observation['raw_units'] if unit.alliance == features.PlayerRelative.ENEMY]
    # reward = len(my_units_health) / (len(my_units_health) + len(enemy_units_health))

    my_units_health_pre = [unit.health for unit in pre_obs.observation['raw_units'] if unit.alliance == features.PlayerRelative.SELF]
    enemy_units_health_pre = [unit.health for unit in pre_obs.observation['raw_units'] if unit.alliance == features.PlayerRelative.ENEMY]

    if len(enemy_units_health) == 0:
        reward = sum(my_units_health) + 200
        return reward / 200
    if len(my_units_health) == 0:
        reward = -sum(my_units_health) - 200
        return reward / 200

    if len(my_units_health) < len(my_units_health_pre):
        reward -= (len(my_units_health_pre) - len(my_units_health)) * 10

    if len(enemy_units_health) < len(enemy_units_health_pre):
        reward += (len(enemy_units_health_pre) - len(enemy_units_health)) * 10

    reward += (sum(my_units_health) - sum(my_units_health_pre)) / 2 - (sum(enemy_units_health) - sum(enemy_units_health_pre))

    return float(reward) / 200


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