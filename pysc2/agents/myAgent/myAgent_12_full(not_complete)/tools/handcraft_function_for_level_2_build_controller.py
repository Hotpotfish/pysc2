import random

from pysc2.agents.myAgent.myAgent_12.config import config, config_for_level_2_build_controller
from pysc2.agents.myAgent.myAgent_12.tools.unit_list import combat_unit, building
from pysc2.lib import features, actions, units
import numpy as np


hard_code = [  # 28种
    0,  # 军械库
    5,  # 兵营
    0,  # 飞行的兵营
    0,  # 重定向兵营
    0,  # 兵营实验室
    0,  # #碉堡
    1,  # 指挥中心
    0,  # 飞行的指挥中心
    0,  # 工程湾
    0,  # 工厂
    0,  # 飞起来的工厂
    0,  # 工厂重定向
    0,  # 工厂实验室
    0,  # 聚变芯体
    0,  # 幽灵军校
    0,  # 导弹塔
    0,  # 轨道指挥中心
    0,  # 飞行的指挥中心
    0,  # 行星要塞
    0,  # 反应堆
    0,  # 感应塔
    0,  # 星港
    0,  # 星港飞起来
    0,  # 星港重定向
    0,  # 星港实验室
    15,  # 补给站
    0,  # 补给站下
    0,  # 科技实验室
]


#######################动作集合
def chooseARandomPlace(input_x, input_y):
    offset = 20
    add_y = random.randint(-offset, offset)
    add_x = random.randint(-offset, offset)

    if input_x + add_x >= config.MAP_SIZE:

        outx = config.MAP_SIZE

    elif input_x + add_x < 0:
        outx = 0

    else:
        outx = input_x + add_x

    if input_y + add_y >= config.MAP_SIZE:

        outy = config.MAP_SIZE

    elif input_y + add_y < 0:
        outy = 0

    else:
        outy = input_y + add_y

    return (outx, outy)


def get_distances(units, xy):
    units_xy = [(unit.x, unit.y) for unit in units]
    return np.linalg.norm(np.array(units_xy) - np.array(xy), axis=1)


def build_supply_depot(obs):
    commandCenters = get_my_units_by_type(obs, units.Terran.CommandCenter)
    if len(commandCenters) > 0:
        commandCenter = commandCenters[random.randint(0, len(commandCenters) - 1)]
        scvs = get_my_units_by_type(obs, units.Terran.SCV)
        if (obs.observation.player.minerals >= 100 and len(scvs) > 0 and obs.observation.player.food_cap < 200):
            supply_depot_xy = chooseARandomPlace(commandCenter.x, commandCenter.y)
            distances = get_distances(scvs, supply_depot_xy)
            scv = scvs[np.argmin(distances)]
            return actions.RAW_FUNCTIONS.Build_SupplyDepot_pt("now", scv.tag, supply_depot_xy)
    return actions.RAW_FUNCTIONS.no_op()


def get_my_completed_units_by_type(obs, unit_type):
    return [unit for unit in obs.observation.raw_units
            if unit.unit_type == unit_type
            and unit.build_progress == 100
            and unit.alliance == features.PlayerRelative.SELF]


def build_barracks(obs):
    commandCenters = get_my_units_by_type(obs, units.Terran.CommandCenter)
    if len(commandCenters) > 0:
        completed_supply_depots = get_my_completed_units_by_type(
            obs, units.Terran.SupplyDepot)

        commandCenter = commandCenters[random.randint(0, len(commandCenters) - 1)]
        scvs = get_my_units_by_type(obs, units.Terran.SCV)
        if (len(completed_supply_depots) > 0 and
                obs.observation.player.minerals >= 150 and len(scvs) > 0):
            barracks_xy = chooseARandomPlace(commandCenter.x, commandCenter.y)
            distances = get_distances(scvs, barracks_xy)
            scv = scvs[np.argmin(distances)]
            return actions.RAW_FUNCTIONS.Build_Barracks_pt(
                "now", scv.tag, barracks_xy)
    return actions.RAW_FUNCTIONS.no_op()


def do_nothing(obs):
    return actions.RAW_FUNCTIONS.no_op()


def assembly_action(obs, action_prob, controller, mark):
    if mark == 'train':
        action_number = np.random.choice(range(config_for_level_2_build_controller.ACTION_DIM), p=action_prob)
    else:
        action_number = np.argmax(action_prob)
    action = controller[action_number](obs)
    return action, action_number


####################################### 观察集合
def get_my_units_by_type(obs, unit_type):
    return [unit for unit in obs.observation.raw_units
            if unit.unit_type == unit_type
            and unit.alliance == features.PlayerRelative.SELF]


def get_enemy_units_by_type(obs, unit_type):
    return [unit for unit in obs.observation.raw_units
            if unit.unit_type == unit_type
            and unit.alliance == features.PlayerRelative.ENEMY]


def get_state(obs):
    state = []  # 52维
    state.append(obs.observation['game_loop'] / 10000)  # step 1
    for unit_type in combat_unit:
        state.append(len(get_my_units_by_type(obs, unit_type)) / 200)  # 18
    for building_type in building:
        state.append(len(get_my_units_by_type(obs, building_type)) / 20)  # 28

    state.append(obs.observation.player.minerals)  # 1
    state.append(obs.observation.player.vespene)  # 1
    state.append(len(get_my_units_by_type(obs, 45)) / 200)  # 1
    state.append(((len(get_enemy_units_by_type(obs, 18))) +  # 敌方指挥中心数量 1
                  len(get_enemy_units_by_type(obs, 36)) +
                  len(get_enemy_units_by_type(obs, 132)) +
                  len(get_enemy_units_by_type(obs, 134)) +
                  len(get_enemy_units_by_type(obs, 130))) / 20
                 )
    state.append(len(get_enemy_units_by_type(obs, 45)) / 200)  # 1

    return np.array(state)


def get_reward(obs):
    building_state = []
    for unit_type in building:
        building_state.append(len(get_my_units_by_type(obs, unit_type)))  # 18

    reward = (200 - np.linalg.norm(np.array(hard_code) - np.array(building_state))) / 200

    return reward
