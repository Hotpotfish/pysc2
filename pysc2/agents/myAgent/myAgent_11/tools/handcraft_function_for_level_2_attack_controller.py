import math
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


def actionSelect(unit, enemy_units, action_porb):
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
    # action_porb_real = np.exp(action_porb_real) / sum(np.exp(action_porb_real))
    # action_porb_real = action_porb_real / np.sum(action_porb_real)

    # if mark == 'test':
    # if np.argmax(action_porb_real) - 5 >= enemy_units_length:
    #     print()

    return np.argmax(action_porb_real)
    # if mark == 'train':
    #     action_number = np.random.choice(range(len(action_porb_real)), p=action_porb_real.ravel())
    #
    #     # if action_number - 5 >= enemy_units_length:
    #     #     print()
    #
    #     return action_number


def assembly_action(obs, action_probs):
    # head = '{:0' + str(config.ATTACT_CONTROLLER_ACTIONDIM_BIN) + 'b}'
    my_raw_units = [unit for unit in obs.observation['raw_units'] if unit.alliance == features.PlayerRelative.SELF]
    enemy_units = [unit for unit in obs.observation['raw_units'] if unit.alliance == features.PlayerRelative.ENEMY]
    # handcraft_function.reflect

    my_raw_units_lenth = len(my_raw_units)

    actions = []
    action_numbers = []

    controller = sa.attack_controller

    # 根据参数名字填内容
    if my_raw_units_lenth > config.MY_UNIT_NUMBER:
        for i in range(config.MY_UNIT_NUMBER):
            action_number = actionSelect(my_raw_units[i], enemy_units, action_probs[i])

            action_numbers.append(action_number)

            parameter = []

            # aciton_bin = head.format(action_number)
            #
            # action_type = int(aciton_bin[0:2], base=2)

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
            action_number = actionSelect(my_raw_units[i], enemy_units, action_probs[i])

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


def get_agents_state(obs):
    state = []
    my_units = [unit for unit in obs.observation.raw_units if unit.alliance == features.PlayerRelative.SELF]
    my_units_lenth = len(my_units)
    for i in range(config.MY_UNIT_NUMBER):
        if i >= my_units_lenth:
            state.append(np.zeros(config.MAP_SIZE * config.MAP_SIZE))
        else:
            state.append(np.array(obs.observation['feature_screen'][5].flatten()))
    return state


def get_agents_local_observation(obs):
    agents_local_observation = []

    my_units = [unit for unit in obs.observation.raw_units if unit.alliance == features.PlayerRelative.SELF]
    my_units_lenth = len(my_units)
    # if my_units_lenth == 0:
    #     print()

    for i in range(config.MY_UNIT_NUMBER):
        if i >= my_units_lenth:
            empty = np.zeros(6 + config.MY_UNIT_NUMBER + config.ENEMY_UNIT_NUMBER)
            agents_local_observation.append(empty)
        else:
            unit_local = soldier()
            unit_local.unit_type = my_units[i].unit_type
            unit_local.health = my_units[i].health
            unit_local.energy = my_units[i].energy
            unit_local.x = my_units[i].x
            unit_local.y = my_units[i].y
            unit_local.order_length = my_units[i].order_length

            friend_k, enemy_k = get_friend_and_enemy_health(my_units[i], obs, config.MY_UNIT_NUMBER, config.ENEMY_UNIT_NUMBER)

            unit_local.frend_health = friend_k
            unit_local.enemy_health = enemy_k
            agents_local_observation.append(unit_local.get_list())

    return agents_local_observation


def get_raw_units_observation(obs):
    fin_list = np.zeros((200, 29))
    raw_units = obs.observation['raw_units'][:, :29]

    fin_list[0:len(raw_units)] = np.array(raw_units[:])

    return np.array(fin_list.reshape((-1, 200, 29, 1)))


def reward_compute_2(previous_state, current_state, obs):
    rward_all = []
    for i in range(config.MY_UNIT_NUMBER):
        if current_state[1][i][0] == 0:
            rward_all.append(0)
        else:
            temp = current_state[1][i][6:] - previous_state[1][i][6:]
            temp = np.sum(temp[0:config.MY_UNIT_NUMBER]) - np.sum(temp[config.MY_UNIT_NUMBER:config.MY_UNIT_NUMBER + config.ENEMY_UNIT_NUMBER]) + obs.reward
            rward_all.append(temp)

    # rward_all = np.array(current_state[1][:, 6:]) - np.array(previous_state[1][:, 6:])
    # # my_health_change =  rward_all[:config.]
    # rward_all = np.sum(rward_all[:, 0:config.K] - rward_all[:, config.K:config.K + config.K], axis=1)
    return np.array(rward_all)
