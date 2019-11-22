import numpy as np
import pysc2.agents.myAgent.myAgent_7.config.config as config
import pysc2.agents.myAgent.myAgent_7.smart_actions as sa

import matplotlib.pyplot as plt

from pysc2.env.environment import StepType


# def plt_score(obs, steps):
#     plt.subplot(221)
#     plt.plot(steps, obs.observation['score_cumulative'][0])
#
#     if obs[0] == StepType.LAST:
#         plt.show()


# 平铺嵌套数组
def my_flatten(input_list):
    output_list = []
    while True:
        if input_list == []:
            break
        for index, i in enumerate(input_list):

            if type(i) == list:
                input_list = i + input_list[index + 1:]
                break
            else:
                output_list.append(i)
                input_list.pop(index)
                break

    return output_list


# 参数映射
# 将神经网络的参数映射到可以执行的程度
def reflect(obs, macro_and_parameter):
    # macro_and_parameter 分别代表：动作（一维），RAW_TYPES.queued, RAW_TYPES.unit_tags, RAW_TYPES.target_unit_tag 和RAW_TYPES.world（占两位）
    m_a_p = macro_and_parameter
    raw_units = obs.observation['raw_units']
    raw_units_len = len(raw_units) - 1

    if macro_and_parameter[1] > 0.5:
        macro_and_parameter[1] = '1'
    else:
        macro_and_parameter[1] = '0'

    macro_and_parameter[2] = int(macro_and_parameter[2] * raw_units_len)
    macro_and_parameter[3] = int(macro_and_parameter[3] * raw_units_len)
    macro_and_parameter[4] = int(macro_and_parameter[4] * (config.MAP_SIZE - 1))
    macro_and_parameter[5] = int(macro_and_parameter[5] * (config.MAP_SIZE - 1))
    return m_a_p


# 动作组装
# 将动作组装成可执行的结果
def assembly_action(obs, controller_number, macro_and_parameter):
    raw_units = obs.observation['raw_units']
    action = sa.controllers[controller_number][macro_and_parameter[0]]
    parameter = []
    # 根据参数名字填内容
    for i in range(len(action[5])):
        if action[5][i].name == 'queued':
            parameter.append(int(macro_and_parameter[1]))
            continue
        if action[5][i].name == 'unit_tags':
            parameter.append(raw_units[int(macro_and_parameter[2])].tag)
            continue
        if action[5][i].name == 'target_unit_tag':
            parameter.append(raw_units[int(macro_and_parameter[3])].tag)
            continue
        if action[5][i].name == 'world':
            parameter.append((int(macro_and_parameter[4]), int(macro_and_parameter[5])))
            continue

    parameter = tuple(parameter)
    return action(*parameter)


# 获得全局的观察
def get_all_observation(obs):
    state_layers = []
    non_serial_layer = []
    for key, value in obs.observation.items():

        if len(value) == 0:
            continue

        if key == 'feature_minimap' or key == 'feature_screen':
            for i in range(len(value)):
                state_layers.append(np.array(value[i]))
            continue
        if type(value) is not str:
            value = value.tolist()
            non_serial_layer.append(value)
    non_serial_layer = np.array(my_flatten(non_serial_layer))
    number = len(non_serial_layer)
    dataSize = pow(config.MAP_SIZE, 2)
    loop = int(number / dataSize) + 1
    for i in range(loop):
        layer = np.zeros(shape=(dataSize,))
        if i != loop - 1:
            start = i * dataSize
            end = (i + 1) * dataSize
            layer = non_serial_layer[start:end]
            layer = layer.reshape((config.MAP_SIZE, config.MAP_SIZE))
            state_layers.append(layer)
            continue
        layer[0:(number - i * dataSize)] = non_serial_layer[i * dataSize: number]
        layer = layer.reshape((config.MAP_SIZE, config.MAP_SIZE))
        state_layers.append(layer)
    return np.array(state_layers).reshape((-1, config.MAP_SIZE, config.MAP_SIZE, 39))
