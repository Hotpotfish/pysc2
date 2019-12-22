import math
from pysc2.lib import actions as Action
import numpy as np
import pysc2.agents.myAgent.myAgent_10.smart_actions as sa
# 获得全局的观察
from pysc2.agents.myAgent.myAgent_10.tools.local_unit import soldier
from pysc2.agents.myAgent.myAgent_10.config import config
from pysc2.lib import features


# def int2bin(n, count=24):
#     """returns the binary of integer n, using count number of digits"""
#     return "".join([str((n >> y) & 1) for y in range(count-1, -1, -1)])

def assembly_action(obs, action):
    head = '{:0' + str(config.ATTACT_CONTROLLER_ACTIONDIM_BIN) + 'b}'
    my_raw_units = [unit for unit in obs.observation['raw_units'] if unit.alliance == features.PlayerRelative.SELF]
    enemy_units = [unit for unit in obs.observation['raw_units'] if unit.alliance == features.PlayerRelative.ENEMY]

    my_raw_units_lenth = len(my_raw_units)
    enemy_units_lenth = len(enemy_units)

    # action = sa.controllers[controller_number][int(action_and_parameter[0])]

    # if macro_and_parameter[2] >= raw_units_lenth or macro_and_parameter[3] >= raw_units_lenth:
    #     return actions.RAW_FUNCTIONS.no_op()
    actions = []
    # 根据参数名字填内容
    if my_raw_units_lenth > config.COOP_AGENTS_NUMBER:
        for i in range(config.COOP_AGENTS_NUMBER):
            controller = sa.attack_controller
            parameter = []

            aciton_bin = head.format(action[i])

            action_number = int(aciton_bin[0:1], base=2)
            a = controller[action_number]

            parameter.append(0)
            parameter.append(my_raw_units[i].tag)
            enemy_or_dire = int(aciton_bin[1:], base=2)

            if a == Action.RAW_FUNCTIONS.Attack_unit:
                parameter.append(enemy_units[enemy_or_dire % enemy_units_lenth].tag)
                # parameter.append([queued, my_raw_units[i].tag, enemy_units[enemy_or_dire % enemy_units_lenth]])
                # parameter = parameter.flatten()
            elif a == Action.RAW_FUNCTIONS.Move_pt:
                if enemy_or_dire == 0:
                    parameter.append((my_raw_units[i].x + 1, my_raw_units[i].y + 1))
                elif enemy_or_dire == 1:
                    parameter.append((my_raw_units[i].x - 1, my_raw_units[i].y - 1))
                elif enemy_or_dire == 2:
                    parameter.append((my_raw_units[i].x + 1, my_raw_units[i].y - 1))
                else:
                    parameter.append((my_raw_units[i].x - 1, my_raw_units[i].y + 1))

            parameter = tuple(parameter)
            actions.append(a(*parameter))

    else:

        for i in range(my_raw_units_lenth):

            controller = sa.attack_controller

            parameter = []

            aciton_bin = head.format(action[i])

            action_number = int(aciton_bin[0:1], base=2)

            a = controller[action_number]

            # queued = int(aciton_bin[1:2], base=2)

            parameter.append(0)

            parameter.append(my_raw_units[i].tag)

            enemy_or_dire = int(aciton_bin[1:], base=2)

            if a == Action.RAW_FUNCTIONS.Attack_unit:
                if enemy_units_lenth == 0:
                    return Action.RAW_FUNCTIONS.no_op()

                parameter.append(enemy_units[enemy_or_dire % enemy_units_lenth].tag)

                # parameter.append([queued, my_raw_units[i].tag, enemy_units[enemy_or_dire % enemy_units_lenth]])

                # parameter = parameter.flatten()

            elif a == Action.RAW_FUNCTIONS.Move_pt:

                if enemy_or_dire == 0:

                    parameter.append((my_raw_units[i].x + 1, my_raw_units[i].y + 1))

                elif enemy_or_dire == 1:

                    parameter.append((my_raw_units[i].x - 1, my_raw_units[i].y - 1))

                elif enemy_or_dire == 2:

                    parameter.append((my_raw_units[i].x + 1, my_raw_units[i].y - 1))

                else:

                    parameter.append((my_raw_units[i].x - 1, my_raw_units[i].y + 1))

            parameter = tuple(parameter)

            actions.append(a(*parameter))
    return actions


def get_friend_and_enemy_health(unit, obs, k):
    friend = []
    enemy = []

    friend_k = np.zeros((k,))
    enemy_k = np.zeros((k,))

    for other_unit in obs.observation.raw_units:

        if unit.x == other_unit.x and unit.y == other_unit.y:
            continue

        x_difference = math.pow(unit.x - other_unit.x, 2)
        y_difference = math.pow(unit.y - other_unit.y, 2)

        distance = math.sqrt(x_difference + y_difference)

        if other_unit.alliance == features.PlayerRelative.SELF:
            friend.append([distance, other_unit.health])
            continue

        if other_unit.alliance == features.PlayerRelative.ENEMY:
            enemy.append((distance, other_unit.health))

    friend = np.array(sorted(friend, key=lambda f: f[0]))
    enemy = np.array(sorted(enemy, key=lambda e: e[0]))

    if len(friend) >= k:
        friend_k = friend[:k, 1]
    elif len(friend) >= 1 and len(friend) < k:
        friend_k[:len(friend)] = friend[:, 1]
    else:
        friend_k = np.zeros(k)

    if len(enemy) >= k:
        enemy_k = enemy[:k, 1]
    elif len(enemy) >= 1 and len(enemy) < k:
        enemy_k[:len(enemy)] = enemy[:, 1]
    else:
        enemy_k = np.zeros(k)

    return friend_k, enemy_k

    #     enemy_K = enemy[:K, 1]


def get_agents_local_observation(obs):
    # my_unit = [unit for unit in obs.observation.raw_units if unit.alliance == features.PlayerRelative.SELF]
    # 设计失误
    agents_local_observation = []

    my_units = [unit for unit in obs.observation.raw_units if unit.alliance == features.PlayerRelative.SELF]
    my_units_lenth = len(my_units)
    if my_units_lenth == 0:
        print()

    for i in range(config.COOP_AGENTS_NUMBER):
        if i >= my_units_lenth:
            unit_local = soldier()
            unit_local.unit_type = 0
            unit_local.health = 0
            unit_local.energy = 0
            unit_local.x = 0
            unit_local.y = 0
            unit_local.order_length = 0
            friend_k, enemy_k = get_friend_and_enemy_health(unit_local, obs, config.K)
            unit_local.frend_health = friend_k
            unit_local.enemy_health = enemy_k
            agents_local_observation.append(unit_local.get_list())
        else:
            unit_local = soldier()
            unit_local.unit_type = my_units[i].unit_type
            unit_local.health = my_units[i].health
            unit_local.energy = my_units[i].energy
            unit_local.x = my_units[i].x
            unit_local.y = my_units[i].y
            unit_local.order_length = my_units[i].order_length

            friend_k, enemy_k = get_friend_and_enemy_health(my_units[i], obs, config.K)

            unit_local.frend_health = friend_k
            unit_local.enemy_health = enemy_k
            agents_local_observation.append(unit_local.get_list())

    return agents_local_observation

    # output = np.zeros((config.COOP_AGENTS_NUMBER, config.COOP_AGENTS_OBDIM))

    # for unit in obs.observation.raw_units:
    #
    #     if unit.alliance == features.PlayerRelative.SELF:
    #         unit_local = soldier()
    #         unit_local.unit_type = unit.unit_type
    #         unit_local.health = unit.health
    #         unit_local.energy = unit.energy
    #         unit_local.x = unit.x
    #         unit_local.y = unit.y
    #         unit_local.order_length = unit.order_length
    #
    #         friend_k, enemy_k = get_friend_and_enemy_health(unit, obs, config.K)
    #
    #         unit_local.frend_health = friend_k
    #         unit_local.enemy_health = enemy_k
    #         agents_local_observation.append(unit_local.get_list())
    #
    # if len(agents_local_observation) >= config.COOP_AGENTS_NUMBER:
    #     output = agents_local_observation[:config.COOP_AGENTS_NUMBER]
    # elif 1 <= len(agents_local_observation) < config.COOP_AGENTS_NUMBER:
    #     output[:len(agents_local_observation)] = agents_local_observation
    # else:
    #     unit_local = soldier()
    #     unit_local.unit_type = 0
    #     unit_local.health = 0
    #     unit_local.energy = 0
    #     unit_local.x = 0
    #     unit_local.y = 0
    #     unit_local.order_length = 0
    #
    #     friend_k, enemy_k = get_friend_and_enemy_health(unit_local, obs, config.K)
    #
    #     unit_local.frend_health = friend_k
    #     unit_local.enemy_health = enemy_k
    #     output[0] = unit_local.get_list()

    return output


def get_raw_units_observation(obs):
    fin_list = np.zeros((200, 29))
    raw_units = obs.observation['raw_units'][:, :29]

    fin_list[0:len(raw_units)] = np.array(raw_units[:])

    return np.array(fin_list.reshape((-1, 200, 29, 1)))


def reward_compute_0(obs):
    # raw_units = obs.observation['raw_units']
    my_units = [unit for unit in obs.observation.raw_units
                if unit.alliance == features.PlayerRelative.SELF]
    enemy = [unit for unit in obs.observation.raw_units
             if unit.alliance == features.PlayerRelative.ENEMY]
    my_units_number = len(my_units)
    enemy_number = len(enemy)

    if my_units_number + enemy_number == 0:
        return 0
    reward = my_units_number / (my_units_number + enemy_number)
    return reward


def reward_compute_1(obs):
    my_fighting_capacity = 0
    enemy_fighting_capacity = 0

    for unit in obs.observation.raw_units:
        if unit.alliance == features.PlayerRelative.SELF:
            my_fighting_capacity += unit.health * 0.2 + 5 * 0.8

        elif unit.alliance == features.PlayerRelative.ENEMY:
            enemy_fighting_capacity += unit.health * 0.2 + 5 * 0.8

    all_fighting_capacity = my_fighting_capacity + enemy_fighting_capacity

    if all_fighting_capacity == 0:
        return 0

    win_rate_reward = (my_fighting_capacity / all_fighting_capacity) * 10
    reward = obs.reward * 5 + win_rate_reward - (obs.observation.game_loop / 1000)
    # print("step: %d  reward : %f" % (obs.observation.game_loop, reward))
    return reward


def reward_compute_2(previous_state, current_state):
    rward_all = np.array(current_state[1][:, 6:]) - np.array(previous_state[1][:, 6:])
    rward_all = np.sum(rward_all[:, 0:config.K] - rward_all[:, config.K:config.K + config.K], axis=1)
    return rward_all
