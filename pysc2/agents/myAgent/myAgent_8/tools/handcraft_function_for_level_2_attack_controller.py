import numpy as np

# 获得全局的观察
from pysc2.agents.myAgent.myAgent_8.config import config
from pysc2.lib import features


def get_raw_units_observation(obs):
    fin_list = np.zeros((200, 29))
    raw_units = obs.observation['raw_units'][:, :29]

    fin_list[0:len(raw_units)] = np.array(raw_units[:])

    return np.array(fin_list.reshape((-1, 200, 29, 1)))


# def get_enemy_units_by_type(obs, unit_type):
#     return [unit for unit in obs.observation.raw_units
#             if unit.unit_type == unit_type
#             and unit.alliance == features.PlayerRelative.ENEMY]

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
            my_fighting_capacity += unit.health * 0.1 + 5 * 0.9

        elif unit.alliance == features.PlayerRelative.ENEMY:
            enemy_fighting_capacity += unit.health * 0.1 + 5 * 0.9

    all_fighting_capacity = my_fighting_capacity + enemy_fighting_capacity

    if all_fighting_capacity == 0:
        return 0

    win_rate_reward = (my_fighting_capacity / all_fighting_capacity) * 10
    reward = obs.reward * 5 + win_rate_reward - (obs.observation.game_loop / 1000)
    # print("step: %d  reward : %f" % (obs.observation.game_loop, reward))
    return reward
