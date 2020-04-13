from pysc2.lib import actions as action

marine_action = [
    action.RAW_FUNCTIONS.no_op,
    action.RAW_FUNCTIONS.Attack_unit,
    action.RAW_FUNCTIONS.Move_pt,
    action.RAW_FUNCTIONS.Effect_Stim_quick
]

reaper_action = [
    action.RAW_FUNCTIONS.no_op,
    action.RAW_FUNCTIONS.Effect_KD8Charge_unit,
    action.RAW_FUNCTIONS.Attack_unit,

    action.RAW_FUNCTIONS.Move_pt,
]

medivac_action = [
    action.RAW_FUNCTIONS.no_op,
    action.RAW_FUNCTIONS.Effect_Heal_unit,
    action.RAW_FUNCTIONS.Move_pt,
]

marauder_action = [
    action.RAW_FUNCTIONS.no_op,
    action.RAW_FUNCTIONS.Attack_unit,
    action.RAW_FUNCTIONS.Move_pt,
    action.RAW_FUNCTIONS.Effect_Stim_quick
]


def inquire_action(agent_type):
    if agent_type == 48:
        return marine_action
    elif agent_type == 49:
        return reaper_action
    elif agent_type == 54:
        return medivac_action
    elif agent_type == 51:
        return marauder_action
