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
    # action.RAW_FUNCTIONS.Effect_Stim_quick,  # �˷ܼ�
    action.RAW_FUNCTIONS.Move_pt,
    action.RAW_FUNCTIONS.Attack_unit,


    # action.RAW_FUNCTIONS.Morph_Hellion_quick,  # ������
    # action.RAW_FUNCTIONS.Morph_Hellbat_quick,
    # action.RAW_FUNCTIONS.Morph_SiegeMode_quick,  # ̹��չ��
    # action.RAW_FUNCTIONS.Morph_Unsiege_quick,  # ̹������

    # action.RAW_FUNCTIONS.Effect_Stim_quick,  # �˷ܼ�
    # action.RAW_FUNCTIONS.Morph_ThorExplosiveMode_quick,  # ������ģʽ
    # action.RAW_FUNCTIONS.Morph_ThorHighImpactMode_quick,  # ������ģʽ

    # action.RAW_FUNCTIONS.Behavior_CloakOff_quick,  # Ӱ��ģʽ
    # action.RAW_FUNCTIONS.Behavior_CloakOn_quick,  # �������
    #
    # action.RAW_FUNCTIONS.Effect_GhostSnipe_unit,
    # action.RAW_FUNCTIONS.Effect_EMP_unit,

    action.RAW_FUNCTIONS.Effect_KD8Charge_unit,

    # action.RAW_FUNCTIONS.Effect_InterferenceMatrix_unit,  # ���ž���
    # action.RAW_FUNCTIONS.Effect_AutoTurret_pt,  # �Զ���̨
    # action.RAW_FUNCTIONS.Effect_AntiArmorMissile_unit,  # ��װ�׵���
    #
    # action.RAW_FUNCTIONS.Morph_VikingAssaultMode_quick,  # ά������ģʽ
    # action.RAW_FUNCTIONS.Morph_VikingFighterMode_quick,  # ά��ս����ģʽ

    # action.RAW_FUNCTIONS.Load_unit,  # ����
    # action.RAW_FUNCTIONS.UnloadAllAt_pt,  # ж��
    action.RAW_FUNCTIONS.Effect_Heal_unit,

    # action.RAW_FUNCTIONS.Effect_NukeCalldown_pt,  # �к˵�

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
