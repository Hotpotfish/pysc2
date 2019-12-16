import math

import numpy as np

# 获得全局的观察
from pysc2.agents.myAgent.myAgent_10.tools.local_unit import soldier
from pysc2.agents.myAgent.myAgent_10.config import config
from pysc2.lib import features


def get_friend_and_enemy_health(unit, obs, k):
    friend = []
    enemy = []

    friend_k = np.zeros((k,))
    enemy_k = np.zeros((k,))

    for other_unit in obs.observation.raw_units:

        if unit.tage == other_unit.tag:
            continue

        x_difference = math.pow(unit.x - other_unit.x, 2)
        y_difference = math.pow(unit.y - other_unit.y, 2)

        distance = math.sqrt(x_difference + y_difference)

        if other_unit.alliance == features.PlayerRelative.SELF:
            friend.append([distance, other_unit.health])
            continue

        if other_unit.alliance == features.PlayerRelative.ENEMY:
            enemy.append((distance, other_unit.health))

    friend = sorted(friend, key=lambda f: f[0])
    enemy = sorted(enemy, key=lambda e: e[0])

    if len(friend) >= k:
        friend_k = friend[:k, 1]
    else:
        friend_k[:len(friend)] = friend[:, 1]

    if len(enemy) >= k:
        enemy_k = enemy[:k, 1]
    else:
        enemy_k[:len(enemy)] = enemy[:, 1]

    return friend_k, enemy_k

    #     enemy_K = enemy[:K, 1]


def get_agents_local_observation(obs):
    # my_unit = [unit for unit in obs.observation.raw_units if unit.alliance == features.PlayerRelative.SELF]

    agents_local_observation = []
    output = np.zeros((config.COOP_AGENTS_NUMBER, config.COOP_AGENTS_OBDIM))

    for unit in obs.observation.raw_units:

        if unit.alliance == features.PlayerRelative.SELF:
            unit_local = soldier()
            unit_local.unit_type = unit.unit_type
            unit_local.health = unit.health
            unit_local.energy = unit.energy
            unit_local.x = unit.x
            unit_local.y = unit.y
            unit_local.order_length = unit.order_length

            friend_k, enemy_k = get_friend_and_enemy_health(unit, obs, config.K)

            unit_local.frend_health = friend_k
            unit_local.enemy_health = enemy_k
            agents_local_observation.append(unit_local.get_list)

    if len(agents_local_observation) >= config.COOP_AGENTS_NUMBER:
        output = agents_local_observation[:config.COOP_AGENTS_NUMBER]
    else:
        output[:len(agents_local_observation)] = agents_local_observation[:]

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
    rward_all = np.sum(rward_all[:, 0:5] - rward_all[:, 5:10], axis=1)
    return rward_all
