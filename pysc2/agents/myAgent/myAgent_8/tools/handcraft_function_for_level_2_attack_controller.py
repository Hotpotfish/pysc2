import numpy as np


# 获得全局的观察
def get_raw_units_observation(obs):
    fin_list = np.zeros((200, 46))
    raw_units = obs.observation['raw_units']

    fin_list[0:len(raw_units)] = np.array(raw_units[:])

    return np.array(fin_list.reshape((-1, 200, 46, 1)))
