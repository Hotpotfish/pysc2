import numpy as np

import pysc2.agents.myAgent.myAgent_12.smart_actions as sa
from pysc2.agents.myAgent.myAgent_12.config import config
from pysc2.agents.myAgent.myAgent_12.decisionMaker.level_2.level_2_attack_controller import level_2_attack_controller
from pysc2.agents.myAgent.myAgent_12.decisionMaker.level_2.level_2_build_controller import level_2_build_controller
from pysc2.agents.myAgent.myAgent_12.decisionMaker.level_2.level_2_harvest_controller import level_2_harvest_controller
from pysc2.agents.myAgent.myAgent_12.decisionMaker.level_2.level_2_research_controller import level_2_research_controller
from pysc2.agents.myAgent.myAgent_12.decisionMaker.level_2.level_2_train_controller import level_2_train_controller


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
