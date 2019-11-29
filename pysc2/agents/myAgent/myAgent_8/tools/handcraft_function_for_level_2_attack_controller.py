import numpy as np

# 获得全局的观察
from pysc2.lib import features


def get_raw_units_observation(obs):
    fin_list = np.zeros((200, 46))
    raw_units = obs.observation['raw_units']

    fin_list[0:len(raw_units)] = np.array(raw_units[:])

    return np.array(fin_list.reshape((-1, 200, 46, 1)))


# def get_enemy_units_by_type(obs, unit_type):
#     return [unit for unit in obs.observation.raw_units
#             if unit.unit_type == unit_type
#             and unit.alliance == features.PlayerRelative.ENEMY]

def reward_compute(obs):
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
