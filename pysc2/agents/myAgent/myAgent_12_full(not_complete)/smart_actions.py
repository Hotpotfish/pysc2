from pysc2.agents.myAgent.myAgent_12.config.config_for_level_2_attack_controller import MAP_ZONE
from pysc2.agents.myAgent.myAgent_12.tools.handcraft_function_for_level_2_build_controller import build_supply_depot, build_barracks, do_nothing
from pysc2.lib import actions as action

build_controller = [
    build_supply_depot,
    build_barracks,
    do_nothing,
]

train_controller = [
    action.RAW_FUNCTIONS.Train_Marine_quick,
    action.RAW_FUNCTIONS.Train_SCV_quick,
    action.RAW_FUNCTIONS.Train_Banshee_quick,
    action.RAW_FUNCTIONS.Train_Battlecruiser_quick,
    action.RAW_FUNCTIONS.Train_Ghost_quick,
    action.RAW_FUNCTIONS.Train_Hellbat_quick,
    action.RAW_FUNCTIONS.Train_Hellion_quick,
    action.RAW_FUNCTIONS.Train_Marauder_quick,
    action.RAW_FUNCTIONS.Train_Medivac_quick,
    action.RAW_FUNCTIONS.Train_SiegeTank_quick,
    action.RAW_FUNCTIONS.Train_Thor_quick,
    action.RAW_FUNCTIONS.Train_WidowMine_quick,
]

harvest_controller = [
    action.RAW_FUNCTIONS.Harvest_Gather_unit,
    action.RAW_FUNCTIONS.Harvest_Gather_SCV_unit,
    action.RAW_FUNCTIONS.Harvest_Return_quick,
    action.RAW_FUNCTIONS.Harvest_Return_SCV_quick,

]

attack_controller = [(i, j) for i in range(MAP_ZONE) for j in range(MAP_ZONE)]

research_controller = [
    action.RAW_FUNCTIONS.Research_TerranInfantryArmorLevel1_quick,
    action.RAW_FUNCTIONS.Research_TerranInfantryArmorLevel2_quick,
    action.RAW_FUNCTIONS.Research_TerranInfantryArmorLevel3_quick,
    action.RAW_FUNCTIONS.Research_TerranInfantryArmor_quick,
    action.RAW_FUNCTIONS.Research_TerranInfantryWeaponsLevel1_quick,
    action.RAW_FUNCTIONS.Research_TerranInfantryWeaponsLevel2_quick,
    action.RAW_FUNCTIONS.Research_TerranInfantryWeaponsLevel3_quick
]

controllers = [
    build_controller,
    # train_controller,
    # harvest_controller,
    # research_controller,
    attack_controller,

]

# controllers = {
#     0 attack_controller,
# }
