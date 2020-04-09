from pysc2.lib import actions as action

build_controller = [
    action.RAW_FUNCTIONS.Build_SupplyDepot_pt,
    action.RAW_FUNCTIONS.Build_Barracks_pt,
    action.RAW_FUNCTIONS.Build_Refinery_pt,
    action.RAW_FUNCTIONS.Build_Armory_pt,
    action.RAW_FUNCTIONS.Build_Bunker_pt,
    action.RAW_FUNCTIONS.Build_CommandCenter_pt,
    action.RAW_FUNCTIONS.Build_EngineeringBay_pt,
    action.RAW_FUNCTIONS.Build_Factory_pt,
    action.RAW_FUNCTIONS.Build_FusionCore_pt,
    action.RAW_FUNCTIONS.Build_GhostAcademy_pt,
    action.RAW_FUNCTIONS.Build_MissileTurret_pt,
    action.RAW_FUNCTIONS.Build_Nuke_quick,
    action.RAW_FUNCTIONS.Build_Reactor_pt,
    action.RAW_FUNCTIONS.Build_Reactor_quick,
    action.RAW_FUNCTIONS.Build_Reactor_Barracks_pt,
    action.RAW_FUNCTIONS.Build_Reactor_Barracks_quick,
    action.RAW_FUNCTIONS.Build_Reactor_Factory_pt,
    action.RAW_FUNCTIONS.Build_Reactor_Factory_quick,
    action.RAW_FUNCTIONS.Build_Reactor_Starport_pt,
    action.RAW_FUNCTIONS.Build_Reactor_Starport_quick,
    action.RAW_FUNCTIONS.Build_SensorTower_pt,
    action.RAW_FUNCTIONS.Build_Starport_pt,
    action.RAW_FUNCTIONS.Build_TechLab_pt,
    action.RAW_FUNCTIONS.Build_TechLab_quick,
    action.RAW_FUNCTIONS.Build_TechLab_Barracks_pt,
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

attack_controller = [
    action.RAW_FUNCTIONS.no_op,
    # action.RAW_FUNCTIONS.Effect_Stim_quick,  # 兴奋剂
    action.RAW_FUNCTIONS.Move_pt,
    action.RAW_FUNCTIONS.Attack_unit,


    # action.RAW_FUNCTIONS.Morph_Hellion_quick,  # 恶火变身
    # action.RAW_FUNCTIONS.Morph_Hellbat_quick,
    # action.RAW_FUNCTIONS.Morph_SiegeMode_quick,  # 坦克展开
    # action.RAW_FUNCTIONS.Morph_Unsiege_quick,  # 坦克收束

    # action.RAW_FUNCTIONS.Effect_Stim_quick,  # 兴奋剂
    # action.RAW_FUNCTIONS.Morph_ThorExplosiveMode_quick,  # 雷神爆破模式
    # action.RAW_FUNCTIONS.Morph_ThorHighImpactMode_quick,  # 雷神冲击模式

    # action.RAW_FUNCTIONS.Behavior_CloakOff_quick,  # 影身模式
    # action.RAW_FUNCTIONS.Behavior_CloakOn_quick,  # 解除隐身
    #
    # action.RAW_FUNCTIONS.Effect_GhostSnipe_unit,
    # action.RAW_FUNCTIONS.Effect_EMP_unit,

    action.RAW_FUNCTIONS.Effect_KD8Charge_unit,

    # action.RAW_FUNCTIONS.Effect_InterferenceMatrix_unit,  # 干扰矩阵
    # action.RAW_FUNCTIONS.Effect_AutoTurret_pt,  # 自动炮台
    # action.RAW_FUNCTIONS.Effect_AntiArmorMissile_unit,  # 反装甲导弹
    #
    # action.RAW_FUNCTIONS.Morph_VikingAssaultMode_quick,  # 维京机甲模式
    # action.RAW_FUNCTIONS.Morph_VikingFighterMode_quick,  # 维京战斗机模式

    # action.RAW_FUNCTIONS.Load_unit,  # 载人
    # action.RAW_FUNCTIONS.UnloadAllAt_pt,  # 卸人
    action.RAW_FUNCTIONS.Effect_Heal_unit,

    # action.RAW_FUNCTIONS.Effect_NukeCalldown_pt,  # 叫核弹

    # action.RAW_FUNCTIONS.Effect_YamatoGun_unit,
    # action.RAW_FUNCTIONS.Effect_TacticalJump_pt,

]

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
    # build_controller,
    # train_controller,
    # harvest_controller,
    # research_controller,
    attack_controller,

]

# controllers = {
#     0 attack_controller,
# }
