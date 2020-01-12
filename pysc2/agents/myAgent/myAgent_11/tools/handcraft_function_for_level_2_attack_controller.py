import math
import random

from pysc2.lib import actions as Action
import numpy as np
import pysc2.agents.myAgent.myAgent_11.smart_actions as sa
# 获得全局的观察
from pysc2.agents.myAgent.myAgent_11.tools.local_unit import soldier
from pysc2.agents.myAgent.myAgent_11.config import config
from pysc2.lib import features


# def int2bin(n, count=24):
#     """returns the binary of integer n, using count number of digits"""
#     return "".join([str((n >> y) & 1) for y in range(count-1, -1, -1)])


def normalization(data):
    # _range = np.max(data) - np.min(data)
    return data / 1000


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


def actionSelect(unit, enemy_units, action_porb, mark, ep):
    mask = []
    enemy_units_length = len(enemy_units)
    action_porb = np.exp(action_porb) / sum(np.exp(action_porb))

    for i in range(config.STATIC_ACTION_DIM):
        mask.append(1)

    if enemy_units_length > config.ENEMY_UNIT_NUMBER:
        for i in range(config.ENEMY_UNIT_NUMBER):
            distance = computeDistance(unit, enemy_units[i])
            if distance <= config.ATTACK_RANGE:
                mask.append(1)
            else:
                mask.append(0)
    else:
        for i in range(enemy_units_length):
            distance = computeDistance(unit, enemy_units[i])
            if distance <= config.ATTACK_RANGE:
                mask.append(1)
            else:
                mask.append(0)
        for i in range(config.ENEMY_UNIT_NUMBER - enemy_units_length):
            mask.append(0)
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
            return action


############################################
global epsilon
epsilon = config.INITIAL_EPSILON


def assembly_action(obs, action_probs, mark):
    # head = '{:0' + str(config.ATTACT_CONTROLLER_ACTIONDIM_BIN) + 'b}'
    my_raw_units = [unit for unit in obs.observation['raw_units'] if unit.alliance == features.PlayerRelative.SELF]
    enemy_units = [unit for unit in obs.observation['raw_units'] if unit.alliance == features.PlayerRelative.ENEMY]
    # handcraft_function.reflect

    my_raw_units_lenth = len(my_raw_units)

    actions = []
    action_numbers = []

    controller = sa.attack_controller

    global epsilon
    epsilon -= (config.INITIAL_EPSILON - config.FINAL_EPSILON) / 100000
    if epsilon <= config.FINAL_EPSILON:
        epsilon = config.FINAL_EPSILON

    # 根据参数名字填内容
    if my_raw_units_lenth > config.MY_UNIT_NUMBER:
        for i in range(config.MY_UNIT_NUMBER):
            action_number = actionSelect(my_raw_units[i], enemy_units, action_probs[i], mark, epsilon)
            action_numbers.append(action_number)
            parameter = []

            if action_number == 0:
                actions.append(Action.RAW_FUNCTIONS.no_op())
                continue

            elif 0 < action_number <= 4:
                a = controller[1]

                dir = action_number - 1

                parameter.append(0)
                parameter.append(my_raw_units[i].tag)
                if dir == 0:
                    parameter.append((my_raw_units[i].x + 1, my_raw_units[i].y + 1))
                elif dir == 1:
                    parameter.append((my_raw_units[i].x - 1, my_raw_units[i].y - 1))
                elif dir == 2:
                    parameter.append((my_raw_units[i].x + 1, my_raw_units[i].y - 1))
                elif dir == 3:
                    parameter.append((my_raw_units[i].x - 1, my_raw_units[i].y + 1))

                parameter = tuple(parameter)
                actions.append(a(*parameter))

            elif 4 < action_number <= 4 + config.ENEMY_UNIT_NUMBER:
                a = controller[2]
                enemy = action_number - 1 - 4
                parameter.append(0)
                parameter.append(my_raw_units[i].tag)
                # print(str(len(enemy_units))+':'+enemy)
                parameter.append(enemy_units[enemy].tag)
                parameter = tuple(parameter)
                actions.append(a(*parameter))

    else:

        for i in range(my_raw_units_lenth):
            action_number = actionSelect(my_raw_units[i], enemy_units, action_probs[i], mark, epsilon)
            action_numbers.append(action_number)
            parameter = []
            if action_number == 0:
                actions.append(Action.RAW_FUNCTIONS.no_op())
                continue
            elif 0 < action_number <= 4:
                a = controller[1]
                dir = action_number - 1
                parameter.append(0)
                parameter.append(my_raw_units[i].tag)
                if dir == 0:
                    parameter.append((my_raw_units[i].x + 1, my_raw_units[i].y + 1))
                elif dir == 1:
                    parameter.append((my_raw_units[i].x - 1, my_raw_units[i].y - 1))
                elif dir == 2:
                    parameter.append((my_raw_units[i].x + 1, my_raw_units[i].y - 1))
                elif dir == 3:
                    parameter.append((my_raw_units[i].x - 1, my_raw_units[i].y + 1))
                parameter = tuple(parameter)
                actions.append(a(*parameter))

            elif 4 < action_number <= 4 + config.ENEMY_UNIT_NUMBER:
                a = controller[2]
                enemy = action_number - 1 - 4
                parameter.append(0)
                parameter.append(my_raw_units[i].tag)
                if len(enemy_units) == enemy:
                    print(str(len(enemy_units)) + ':' + str(enemy))
                parameter.append(enemy_units[enemy].tag)
                parameter = tuple(parameter)
                actions.append(a(*parameter))

        for i in range(config.MY_UNIT_NUMBER - my_raw_units_lenth):
            action_numbers.append(0)

    return actions, action_numbers


def get_friend_and_enemy_health(unit, obs, my_unit_number, enemy_unit_number):
    friend = []
    enemy = []

    friend_k = np.zeros((my_unit_number,))
    enemy_k = np.zeros((enemy_unit_number,))

    for other_unit in obs.observation.raw_units:

        if unit.tag == other_unit.tag:
            continue

        x_difference = math.pow(unit.x - other_unit.x, 2)
        y_difference = math.pow(unit.y - other_unit.y, 2)

        distance = math.sqrt(x_difference + y_difference)
        # if distance <= config.OB_RANGE:
        #     continue

        if other_unit.alliance == features.PlayerRelative.SELF:
            friend.append([distance, other_unit.health])
            continue

        if other_unit.alliance == features.PlayerRelative.ENEMY:
            enemy.append((distance, other_unit.health))

    friend = np.array(sorted(friend, key=lambda f: f[0]))
    enemy = np.array(sorted(enemy, key=lambda e: e[0]))

    if len(friend) >= my_unit_number:
        friend_k = friend[:my_unit_number, 1]
    elif 1 <= len(friend) < my_unit_number:
        friend_k[:len(friend)] = friend[:, 1]
    else:
        friend_k = np.zeros(my_unit_number)

    if len(enemy) >= enemy_unit_number:
        enemy_k = enemy[:enemy_unit_number, 1]
    elif 1 <= len(enemy) < enemy_unit_number:
        enemy_k[:len(enemy)] = enemy[:, 1]
    else:
        enemy_k = np.zeros(enemy_unit_number)

    return friend_k, enemy_k

    #     enemy_K = enemy[:K, 1]


def get_agent_state(obs):
    states = np.array([])
    units = obs.observation['raw_units']
    units_len = len(units)

    for i in range(config.MY_UNIT_NUMBER + config.ENEMY_UNIT_NUMBER):
        if i >= units_len:
            states = np.append(states, np.zeros(7))

        else:
            states = np.append(states, computeDistance_center(units[i]) / 90.5)
            states = np.append(states, units[i].alliance / 4)
            states = np.append(states, units[i].unit_type / 1000)
            states = np.append(states, units[i].x / config.MAP_SIZE)
            states = np.append(states, units[i].y / config.MAP_SIZE)
            states = np.append(states, units[i].health / 1000)
            states = np.append(states, units[i].shield / 1000)
    return states


def get_agents_state(obs):
    state = []
    my_units = [unit for unit in obs.observation.raw_units if unit.alliance == features.PlayerRelative.SELF]
    my_units_lenth = len(my_units)

    for i in range(config.MY_UNIT_NUMBER):
        if i >= my_units_lenth:
            state.append(np.zeros(config.COOP_AGENTS_OBDIM))
        else:
            state.append(get_agent_state(obs))
    return state


def get_agents_obs(obs):
    agents_obs = []

    my_raw_units = [unit for unit in obs.observation['raw_units'] if unit.alliance == features.PlayerRelative.SELF]
    enemy_units = [unit for unit in obs.observation['raw_units'] if unit.alliance == features.PlayerRelative.ENEMY]

    my_raw_units_lenth = len(my_raw_units)
    enemy_units_lenth = len(enemy_units)

    for i in range(config.MY_UNIT_NUMBER):
        agent_obs = np.array([])

        if i >= my_raw_units_lenth:
            agent_obs = np.zeros((config.MY_UNIT_NUMBER + config.ENEMY_UNIT_NUMBER) * 7)
            agents_obs.append(agent_obs)
            continue

        for j in range(config.MY_UNIT_NUMBER):

            if j >= my_raw_units_lenth or computeDistance(my_raw_units[i], my_raw_units[j]) >= config.OB_RANGE:
                agent_obs = np.append(agent_obs, np.zeros(7))
            else:
                agent_obs = np.append(agent_obs, computeDistance(my_raw_units[i], my_raw_units[j]) / 90.5)
                agent_obs = np.append(agent_obs, my_raw_units[j].alliance / 4)
                agent_obs = np.append(agent_obs, my_raw_units[j].unit_type / 1000)
                agent_obs = np.append(agent_obs, my_raw_units[j].x / config.MAP_SIZE)
                agent_obs = np.append(agent_obs, my_raw_units[j].y / config.MAP_SIZE)
                agent_obs = np.append(agent_obs, my_raw_units[j].health / 1000)
                agent_obs = np.append(agent_obs, my_raw_units[j].shield / 1000)

        for j in range(config.ENEMY_UNIT_NUMBER):
            if j >= enemy_units_lenth or computeDistance(my_raw_units[i], enemy_units[j]) >= config.OB_RANGE:
                agent_obs = np.append(agent_obs, np.zeros(7))
            else:
                agent_obs = np.append(agent_obs, computeDistance(my_raw_units[i], enemy_units[j]) / 90.5)
                agent_obs = np.append(agent_obs, enemy_units[j].alliance / 4)
                agent_obs = np.append(agent_obs, enemy_units[j].unit_type / 1000)
                agent_obs = np.append(agent_obs, enemy_units[j].x / config.MAP_SIZE)
                agent_obs = np.append(agent_obs, enemy_units[j].y / config.MAP_SIZE)
                agent_obs = np.append(agent_obs, enemy_units[j].health / 1000)
                agent_obs = np.append(agent_obs, enemy_units[j].shield / 1000)

        agents_obs.append(agent_obs)

    return agents_obs
