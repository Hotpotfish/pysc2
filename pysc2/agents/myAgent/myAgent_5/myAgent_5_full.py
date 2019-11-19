from queue import Queue

import pysc2.agents.myAgent.myAgent_4.macro_operation as macro_operation

from absl import app
from pysc2.agents import base_agent
from pysc2.agents.myAgent.myAgent_5.decisionMaker.hierarchical_learning_structure import hierarchical_learning_structure

from pysc2.lib import actions, features
from pysc2.env import sc2_env, run_loop


class myAgent(base_agent.BaseAgent):

    def __init__(self):
        super(myAgent, self).__init__()
        self.hierarchical_learning_structure = hierarchical_learning_structure()

    def step(self, obs):
        super(myAgent, self).step(obs)
        action = self.hierarchical_learning_structure.make_choice(obs, 'TRAIN', 'e:/model/', None)
        # action = self.hierarchical_learning_structure.make_choice(obs, 'TRAIN', 'e:/model/', 'model/20191118211813/episode_20')
        # action = self.hierarchical_learning_structure.make_choice(obs, 'TEST', None, 'e:/model/20191119112956/episode_0')
        # action = self.hierarchical_learning_structure.make_choice(obs, 'TEST', None, None)

        return action


def main(unused_argv):
    agent1 = myAgent()

    try:
        with sc2_env.SC2Env(
                map_name="Flat96",
                players=[sc2_env.Agent(sc2_env.Race.terran),
                         sc2_env.Bot(sc2_env.Race.protoss,
                                     sc2_env.Difficulty.very_easy)],
                agent_interface_format=features.AgentInterfaceFormat(
                    feature_dimensions=features.Dimensions(screen=macro_operation.mapSzie,
                                                           minimap=macro_operation.mapSzie),
                    camera_width_world_units=macro_operation.mapSzie * 1,

                    action_space=actions.ActionSpace.RAW,
                    use_raw_units=True,
                    raw_resolution=macro_operation.mapSzie,
                    use_unit_counts=True
                ),
                score_index = 0,

                step_mul=16,
                disable_fog=False,
                visualize=True,
                realtime=False

        ) as env:
            run_loop.run_loop([agent1], env)

    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    app.run(main)
