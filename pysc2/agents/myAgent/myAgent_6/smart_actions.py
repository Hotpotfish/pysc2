from pysc2.lib import actions as action

build_controller = {
    0: action.RAW_FUNCTIONS.Build_SupplyDepot_pt,
    1: action.RAW_FUNCTIONS.Build_Barracks_pt,
    2: action.RAW_FUNCTIONS.Build_Refinery_pt,
    3: action.RAW_FUNCTIONS.Build_Armory_pt,
    4: action.RAW_FUNCTIONS.Build_Bunker_pt,
    5: action.RAW_FUNCTIONS.Build_CommandCenter_pt,
    6: action.RAW_FUNCTIONS.Build_EngineeringBay_pt,
    7: action.RAW_FUNCTIONS.Build_Factory_pt,
    8: action.RAW_FUNCTIONS.Build_FusionCore_pt,
    9: action.RAW_FUNCTIONS.Build_GhostAcademy_pt,
    10: action.RAW_FUNCTIONS.Build_MissileTurret_pt,
    11: action.RAW_FUNCTIONS.Build_Nuke_quick,
    12: action.RAW_FUNCTIONS.Build_Reactor_pt,
    13: action.RAW_FUNCTIONS.Build_Reactor_quick,
    14: action.RAW_FUNCTIONS.Build_Reactor_Barracks_pt,
    15: action.RAW_FUNCTIONS.Build_Reactor_Barracks_quick,
    16: action.RAW_FUNCTIONS.Build_Reactor_Factory_pt,
    17: action.RAW_FUNCTIONS.Build_Reactor_Factory_quick,
    18: action.RAW_FUNCTIONS.Build_Reactor_Starport_pt,
    19: action.RAW_FUNCTIONS.Build_Reactor_Starport_quick,
    20: action.RAW_FUNCTIONS.Build_SensorTower_pt,
    21: action.RAW_FUNCTIONS.Build_Starport_pt,
    22: action.RAW_FUNCTIONS.Build_TechLab_pt,
    23: action.RAW_FUNCTIONS.Build_TechLab_quick,
    24: action.RAW_FUNCTIONS.Build_TechLab_Barracks_pt,




}

train_controller = {
    0: action.RAW_FUNCTIONS.Train_Marine_quick,
    1: action.RAW_FUNCTIONS.Train_SCV_quick,
    2: action.RAW_FUNCTIONS.Train_Banshee_quick,
    3: action.RAW_FUNCTIONS.Train_Battlecruiser_quick,
    4: action.RAW_FUNCTIONS.Train_Ghost_quick,
    5: action.RAW_FUNCTIONS.Train_Hellbat_quick,
    6: action.RAW_FUNCTIONS.Train_Hellion_quick,
    7: action.RAW_FUNCTIONS.Train_Marauder_quick,
    8: action.RAW_FUNCTIONS.Train_Medivac_quick,
    9: action.RAW_FUNCTIONS.Train_SiegeTank_quick,
    10: action.RAW_FUNCTIONS.Train_Thor_quick,
    11: action.RAW_FUNCTIONS.Train_WidowMine_quick,
}

harvest_controller = {
    0: action.RAW_FUNCTIONS.Harvest_Gather_unit,
    1: action.RAW_FUNCTIONS.Harvest_Gather_SCV_unit,
    2: action.RAW_FUNCTIONS.Harvest_Return_quick,
    3: action.RAW_FUNCTIONS.Harvest_Return_SCV_quick,

}

attack_controller = {
    0: action.RAW_FUNCTIONS.Attack_pt,
    1: action.RAW_FUNCTIONS.Attack_unit,
    2: action.RAW_FUNCTIONS.Attack_Attack_pt,
    3: action.RAW_FUNCTIONS.Attack_AttackBuilding_pt,
    4: action.RAW_FUNCTIONS.Attack_Attack_unit,
    5: action.RAW_FUNCTIONS.Attack_AttackBuilding_unit,
    6: action.RAW_FUNCTIONS.Attack_Battlecruiser_pt,
    7: action.RAW_FUNCTIONS.Attack_Battlecruiser_unit,
    8: action.RAW_FUNCTIONS.Attack_Redirect_pt,
    9: action.RAW_FUNCTIONS.Attack_Redirect_unit,
}

Research_controller = {
    0: action.RAW_FUNCTIONS.Research_TerranInfantryArmorLevel1_quick,
    1: action.RAW_FUNCTIONS.Research_TerranInfantryArmorLevel2_quick,
    2: action.RAW_FUNCTIONS.Research_TerranInfantryArmorLevel3_quick,
    3: action.RAW_FUNCTIONS.Research_TerranInfantryArmor_quick,
    4: action.RAW_FUNCTIONS.Research_TerranInfantryWeaponsLevel1_quick,
    5: action.RAW_FUNCTIONS.Research_TerranInfantryWeaponsLevel2_quick,
    6: action.RAW_FUNCTIONS.Research_TerranInfantryWeaponsLevel3_quick
}

controllers = {
    0: build_controller,
    1: train_controller,
    2: harvest_controller,
    3: attack_controller,
    4: Research_controller,
}
