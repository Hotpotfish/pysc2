import numpy as np

import pysc2.agents.myAgent.myAgent_12.smart_actions as sa
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
