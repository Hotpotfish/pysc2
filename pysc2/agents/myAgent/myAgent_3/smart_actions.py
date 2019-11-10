import pysc2.agents.myAgent.myAgent_3.macro_operation as mc



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

controllers = {
    0: build_controller,
    1: train_controller,
    2: harvest_controller,
    3: attack_controller,
}
