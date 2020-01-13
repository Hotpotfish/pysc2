import math
import random

from pysc2.agents.myAgent.myAgent_12.tools.unit_list import combat_unit, building
from pysc2.lib import features, actions
import numpy as np
import pysc2.agents.myAgent.myAgent_12.config.config_for_level_2_attack_controller as config_for_level_2_attack_controller


#######################动作集合
# 寻找指定编号的士兵
def find_my_combat_unit_in_zone(obs, row_x, colum_x):
    bottom_x = colum_x * config_for_level_2_attack_controller.COLUMN_DISTANCE
    top_x = (colum_x + 1) * config_for_level_2_attack_controller.COLUMN_DISTANCE

    bottom_y = row_x * config_for_level_2_attack_controller.ROW_DISTANCE
    top_y = (row_x + 1) * config_for_level_2_attack_controller.ROW_DISTANCE

    my_match_unit = []
    for unit in obs.observation.raw_units:
        if unit.alliance == features.PlayerRelative.SELF \
                and unit.unit_type in combat_unit \
                and bottom_x <= unit.x < top_x \
                and bottom_y <= unit.y < top_y:
            my_match_unit.append(unit)
    return my_match_unit


# 目标区域内随机点
def attacked_zone_random_point(row_y, colum_y):
    bottom_x = colum_y * config_for_level_2_attack_controller.COLUMN_DISTANCE
    top_x = (colum_y + 1) * config_for_level_2_attack_controller.COLUMN_DISTANCE

    bottom_y = row_y * config_for_level_2_attack_controller.ROW_DISTANCE
    top_y = (row_y + 1) * config_for_level_2_attack_controller.ROW_DISTANCE

    random_x = np.random.randint(bottom_x, top_x)
    random_y = np.random.randint(bottom_y, top_y)

    return (random_x, random_y)


# 第X块攻击第Y块 攻击动作集
def attackZone(obs, x, y):
    action = []
    # 计算自己哪一块的兵要攻击了
    row_x = x / config_for_level_2_attack_controller.ROW
    colum_x = x % config_for_level_2_attack_controller.COLUMN

    row_y = y / config_for_level_2_attack_controller.ROW
    colum_y = y % config_for_level_2_attack_controller.COLUMN

    my_match_unit = find_my_combat_unit_in_zone(obs, row_x, colum_x)
    target_zone_random_point = attacked_zone_random_point(row_y, colum_y)

    if len(my_match_unit) == 0:
        action.append(actions.RAW_FUNCTIONS.no_op())
        return action
    else:
        for i in range(len(my_match_unit)):
            action.append(actions.RAW_FUNCTIONS.Attack_pt("now", my_match_unit[i].tag, target_zone_random_point))
        return action


def assembly_action(obs, action_prob, controller):
    action_number = np.random.choice(range(config_for_level_2_attack_controller.ACTION_DIM), p=action_prob)
    order = controller[action_number]
    action = attackZone(obs, order[0], order[1])
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


def get_reward(pre_obs, obs):
    # reward = np.sum(pre_obs.observation['ScoreByCategory'][1]) + np.sum(pre_obs.observation['ScoreByCategory'][2]) - \
    #          np.sum(pre_obs.observation['ScoreByCategory'][3]) + np.sum(pre_obs.observation['ScoreByCategory'][4])
    kill_reward = np.sum(obs.observation['score_by_category'][1]) + np.sum(obs.observation['score_by_category'][2]) - \
                  np.sum(pre_obs.observation['score_by_category'][1]) - np.sum(pre_obs.observation['score_by_category'][2])
    loss_reward = np.sum(obs.observation['score_by_category'][3]) + np.sum(obs.observation['score_by_category'][4]) - \
                  np.sum(pre_obs.observation['score_by_category'][3]) - np.sum(pre_obs.observation['score_by_category'][4])
    reward = (kill_reward - loss_reward) / 1000
    return reward
