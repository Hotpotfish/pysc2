import random
from pysc2.lib import actions
from pysc2.lib import features
from pysc2.lib import units
from queue import Queue
import numpy as np

_NOT_QUEUED = [0]
_QUEUED = [1]

mapSzie = 256


def chooseARandomPlace(input_x, input_y):
    add_y = random.randint(-10, 10)
    add_x = random.randint(-10, 10)

    if input_x + add_x >= mapSzie:

        outx = mapSzie

    elif input_x + add_x < 0:
        outx = 0

    else:
        outx = input_x + add_x

    if input_y + add_y >= mapSzie:

        outy = mapSzie

    elif input_y + add_y < 0:
        outy = 0

    else:
        outy = input_y + add_y

    return (outx, outy)


def get_my_units_by_type(obs, unit_type):
    return [unit for unit in obs.observation.raw_units
            if unit.unit_type == unit_type
            and unit.alliance == features.PlayerRelative.SELF]


def get_enemy_units_by_type(obs, unit_type):
    return [unit for unit in obs.observation.raw_units
            if unit.unit_type == unit_type
            and unit.alliance == features.PlayerRelative.ENEMY]


def get_my_completed_units_by_type(obs, unit_type):
    return [unit for unit in obs.observation.raw_units
            if unit.unit_type == unit_type
            and unit.build_progress == 100
            and unit.alliance == features.PlayerRelative.SELF]


def get_enemy_completed_units_by_type(obs, unit_type):
    return [unit for unit in obs.observation.raw_units
            if unit.unit_type == unit_type
            and unit.build_progress == 100
            and unit.alliance == features.PlayerRelative.ENEMY]


def find_any_enemy(obs):
    return [unit for unit in obs.observation.raw_units
            if unit.alliance == features.PlayerRelative.ENEMY]


def get_distances(obs, units, xy):
    units_xy = [(unit.x, unit.y) for unit in units]
    return np.linalg.norm(np.array(units_xy) - np.array(xy), axis=1)


def harvest_minerals(obs):
    scvs = get_my_units_by_type(obs, units.Terran.SCV)
    idle_scvs = [scv for scv in scvs if scv.order_length == 0]
    if len(idle_scvs) > 0:
        mineral_patches = [unit for unit in obs.observation.raw_units
                           if unit.unit_type in [
                               units.Neutral.BattleStationMineralField,
                               units.Neutral.BattleStationMineralField750,
                               units.Neutral.LabMineralField,
                               units.Neutral.LabMineralField750,
                               units.Neutral.MineralField,
                               units.Neutral.MineralField750,
                               units.Neutral.PurifierMineralField,
                               units.Neutral.PurifierMineralField750,
                               units.Neutral.PurifierRichMineralField,
                               units.Neutral.PurifierRichMineralField750,
                               units.Neutral.RichMineralField,
                               units.Neutral.RichMineralField750
                           ]]
        scv = random.choice(idle_scvs)
        distances = get_distances(obs, mineral_patches, (scv.x, scv.y))
        mineral_patch = mineral_patches[np.argmin(distances)]
        return actions.RAW_FUNCTIONS.Harvest_Gather_unit(
            "queued", scv.tag, mineral_patch.tag)
    return actions.RAW_FUNCTIONS.no_op()


def build_supply_depot(obs):
    commandCenters = get_my_units_by_type(obs, units.Terran.CommandCenter)
    if len(commandCenters) > 0:
        commandCenter = commandCenters[random.randint(0, len(commandCenters) - 1)]
        scvs = get_my_units_by_type(obs, units.Terran.SCV)
        if (obs.observation.player.minerals >= 100 and len(scvs) > 0 and obs.observation.player.food_cap < 200):
            supply_depot_xy = chooseARandomPlace(commandCenter.x, commandCenter.y)
            distances = get_distances(obs, scvs, supply_depot_xy)
            scv = scvs[np.argmin(distances)]
            return actions.RAW_FUNCTIONS.Build_SupplyDepot_pt(
                "queued", scv.tag, supply_depot_xy)
    return actions.RAW_FUNCTIONS.no_op()


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
            distances = get_distances(obs, scvs, barracks_xy)
            scv = scvs[np.argmin(distances)]
            return actions.RAW_FUNCTIONS.Build_Barracks_pt(
                "queued", scv.tag, barracks_xy)
    return actions.RAW_FUNCTIONS.no_op()


def train_marine(obs):
    completed_barrackses = get_my_completed_units_by_type(
        obs, units.Terran.Barracks)
    free_supply = (obs.observation.player.food_cap -
                   obs.observation.player.food_used)
    if (len(completed_barrackses) > 0 and obs.observation.player.minerals >= 100
            and free_supply > 0):
        barracks = get_my_units_by_type(obs, units.Terran.Barracks)
        barrack = barracks[random.randint(0, len(barracks) - 1)]
        if barrack.order_length < 5:
            return actions.RAW_FUNCTIONS.Train_Marine_quick("now", barrack.tag)
    return actions.RAW_FUNCTIONS.no_op()


def attack(obs):
    marines = get_my_units_by_type(obs, units.Terran.Marine)
    if len(marines) > 0:
        enmies = find_any_enemy(obs)
        attack_orders = []
        if len(enmies) > 0:
            for i in range(len(marines)):
                marine_xy = (marines[i].x, marines[i].y)
                distances = get_distances(obs, enmies, marine_xy)
                enmy = enmies[np.argmax(distances)]
                attack_orders.append(actions.RAW_FUNCTIONS.Attack_pt("now", marines[i].tag, (enmy.x, enmy.y)))
            return attack_orders

        else:
            for i in range(len(marines)):
                random_x = random.randint(0, mapSzie - 1)
                random_y = random.randint(0, mapSzie - 1)
                attack_orders.append(actions.RAW_FUNCTIONS.Move_pt("queued", marines[i].tag, (random_x, random_y)))
            return attack_orders

    return actions.RAW_FUNCTIONS.no_op()
