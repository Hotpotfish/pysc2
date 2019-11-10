from queue import Queue

import pysc2.agents.myAgent.myAgent_3.macro_operation as macro_operation

from absl import app
from pysc2.agents import base_agent
from pysc2.agents.myAgent.myAgent_3.decisionMaker.hierarchical_learning_structure import hierarchical_learning_structure
from pysc2.lib import actions, features
from pysc2.env import sc2_env, run_loop
import pysc2.agents.myAgent.myAgent_3.smart_actions as sa


KILL_UNIT_REWARD = 0.2
KILL_BUILDING_REWARD = 0.5


class myAgent(base_agent.BaseAgent):

    def __init__(self):
        super(myAgent, self).__init__()
        # self.hierarchical_learning_structure = hierarchical_learning_structure()


    def step(self, obs):
        super(myAgent, self).step(obs)
        # action = self.hierarchical_learning_structure.make_choice(obs)
        #
        # return action
        return sa.attack_controller[0](obs)


def main(unused_argv):
    agent1 = myAgent()

    try:
        with sc2_env.SC2Env(
                map_name="DefeatRoaches",
                players=[sc2_env.Agent(sc2_env.Race.terran),],
                         # sc2_env.Bot(sc2_env.Race.protoss,
                         #             sc2_env.Difficulty.very_easy)],
                agent_interface_format=features.AgentInterfaceFormat(
                    feature_dimensions=features.Dimensions(screen=macro_operation.mapSzie,
                                                           minimap=macro_operation.mapSzie),

                    action_space=actions.ActionSpace.RAW,
                    use_raw_units=True,
                    raw_resolution=macro_operation.mapSzie,
                    use_unit_counts=True
                ),
                step_mul=8,
                disable_fog=False,
                visualize=True,
                realtime=True

        ) as env:
            run_loop.run_loop([agent1], env)

    except KeyboardInterrupt:
        pass

# def main(unused_argv):
#     agent1 = myAgent()
#
#     try:
#         with sc2_env.SC2Env(
#                 map_name="Simple96",
#                 players=[sc2_env.Agent(sc2_env.Race.terran),
#                          sc2_env.Bot(sc2_env.Race.protoss,
#                                      sc2_env.Difficulty.very_easy)],
#                 agent_interface_format=features.AgentInterfaceFormat(
#                     feature_dimensions=features.Dimensions(screen=macro_operation.mapSzie,
#                                                            minimap=macro_operation.mapSzie),
#
#                     action_space=actions.ActionSpace.RAW,
#                     use_raw_units=True,
#                     raw_resolution=macro_operation.mapSzie,
#                     use_unit_counts=True
#                 ),
#                 step_mul=8,
#                 disable_fog=False,
#                 visualize=True,
#                 realtime=False
#
#         ) as env:
#             run_loop.run_loop([agent1], env)
#
#     except KeyboardInterrupt:
#         pass


if __name__ == "__main__":
    app.run(main)
