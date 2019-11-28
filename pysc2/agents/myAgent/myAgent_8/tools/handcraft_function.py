import numpy as np
import pysc2.agents.myAgent.myAgent_8.config.config as config
import pysc2.agents.myAgent.myAgent_8.smart_actions as sa

from pysc2.agents.myAgent.myAgent_8.decisionMaker.level_2.level_2_attack_controller import level_2_attack_controller
from pysc2.agents.myAgent.myAgent_8.decisionMaker.level_2.level_2_build_controller import level_2_build_controller
from pysc2.agents.myAgent.myAgent_8.decisionMaker.level_2.level_2_harvest_controller import level_2_harvest_controller
from pysc2.agents.myAgent.myAgent_8.decisionMaker.level_2.level_2_research_controller import level_2_research_controller
from pysc2.agents.myAgent.myAgent_8.decisionMaker.level_2.level_2_train_controller import level_2_train_controller

# 平铺嵌套数组
from pysc2.env.environment import StepType
from pysc2.lib import actions


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
def reflect(actiondim, action_and_parameter):
    # macro_and_parameter 分别代表：动作（一维），RAW_TYPES.queued, RAW_TYPES.unit_tags, RAW_TYPES.target_unit_tag 和RAW_TYPES.world（占两位）
    m_a_p = np.array([])
    start = 0
    end = actiondim
    m_a_p = np.append(m_a_p, np.argmax(action_and_parameter[start: end]))

    start = end
    end += config.QUEUED
    m_a_p = np.append(m_a_p, np.argmax(action_and_parameter[start: end]))

    start = end
    end += config.MY_UNIT_NUMBER
    m_a_p = np.append(m_a_p, np.argmax(action_and_parameter[start: end]))

    start += end
    end += config.ENEMY_UNIT_NUMBER
    m_a_p = np.append(m_a_p, np.argmax(action_and_parameter[start: end]))

    start = end
    m_a_p = np.append(m_a_p, np.argmax(action_and_parameter[start:]))

    return m_a_p


# 动作组装
# 将动作组装成可执行的结果
def assembly_action(obs, controller_number, macro_and_parameter):
    raw_units = obs.observation['raw_units']
    raw_units_lenth = len(raw_units)
    action = sa.controllers[controller_number][int(macro_and_parameter[0])]
    parameter = []

    # if macro_and_parameter[2] >= raw_units_lenth or macro_and_parameter[3] >= raw_units_lenth:
    #     return actions.RAW_FUNCTIONS.no_op()
    # 根据参数名字填内容
    for i in range(len(action[5])):
        if action[5][i].name == 'queued':
            parameter.append(int(macro_and_parameter[1]))
            continue
        if action[5][i].name == 'unit_tags':
            parameter.append(raw_units[int(macro_and_parameter[2]) % raw_units_lenth].tag)
            continue
        if action[5][i].name == 'target_unit_tag':
            parameter.append(raw_units[int(macro_and_parameter[3]) % raw_units_lenth].tag)
            continue
        if action[5][i].name == 'world':
            number = macro_and_parameter[4]
            y = int(number / config.MAP_SIZE)
            x = int(number % config.MAP_SIZE)
            parameter.append((x, y))
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


# 向分层框架添加2层控制器的信息
def append_controllers():
    controllers = []
    for i in range(len(sa.controllers)):
        if sa.controllers[i] == sa.build_controller:
            controllers.append(level_2_build_controller())
        elif sa.controllers[i] == sa.attack_controller:
            controllers.append(level_2_attack_controller())
        elif sa.controllers[i] == sa.harvest_controller:
            controllers.append(level_2_harvest_controller())
        elif sa.controllers[i] == sa.research_controller:
            controllers.append(level_2_research_controller())
        elif sa.controllers[i] == sa.train_controller:
            controllers.append(level_2_train_controller())
    return controllers


# 找到控制器在列表的索引
def find_controller_index(controller):
    for i in range(len(sa.controllers)):
        if controller == sa.controllers[i]:
            return i


def win_or_loss(obs):
    if obs[0] == StepType.LAST:
        units = np.array(obs.observation['raw_units'])
        enemies = np.where(units[:, 1] == 4)
        if not len(enemies):
            return 1

    return -1


def one_hot_encoding(number, dim):
    one_hot = np.zeros((dim,))
    one_hot[number] = 1
    return one_hot
