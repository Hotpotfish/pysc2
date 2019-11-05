import pysc2.agents.myAgent.myAgent_2.macro_operation as mc

controllers = {
    0: 'build',
    1: 'train',
    2: 'harvest',
    3: 'attack',
}

build_controller = {
    0: mc.build_supply_depot,
    1: mc.build_barracks,
    2: mc.build_refinery,
}

train_controller = {
    0: mc.train_marine,
    1: mc.train_scv,
}

harvest_controller = {
    0: mc.harvest_minerals,
    1: mc.harvest_VespeneGeyser,
}

attack_controller = {
    0: mc.attack,

}

smart_actions = {
    0: mc.harvest_minerals,
    1: mc.build_supply_depot,
    2: mc.build_barracks,
    3: mc.train_marine,
    4: mc.attack,
    5: mc.train_scv,
    6: mc.harvest_VespeneGeyser,
    7: mc.build_refinery,
}
