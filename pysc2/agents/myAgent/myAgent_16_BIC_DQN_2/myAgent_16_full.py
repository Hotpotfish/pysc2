from absl import app
from pysc2.agents import base_agent
from pysc2.agents.myAgent.myAgent_16_BIC_DQN_2.decisionMaker.hierarchical_learning_structure import hierarchical_learning_structure
import pysc2.agents.myAgent.myAgent_16_BIC_DQN_2.config.config as config
from pysc2.agents.myAgent.myAgent_16_BIC_DQN_2.tools.plt_function import plt_function

from pysc2.env.environment import StepType
from pysc2.lib import actions, features
from pysc2.env import sc2_env, run_loop


class myAgent(base_agent.BaseAgent):

    def __init__(self):
        super(myAgent, self).__init__()
        self.plt_function = plt_function()
        self.hierarchical_learning_structure = hierarchical_learning_structure()

    def add_or_plt(self, obs, steps):
        self.plt_function.add_summary(obs, steps)
        if obs[0] == StepType.LAST:
            self.plt_function.plt_score_cumulative()

    def step(self, obs):
        self.add_or_plt(obs, self.steps)
        super(myAgent, self).step(obs)
        action = self.hierarchical_learning_structure.execute(obs, 'TRAIN', 'e:/model/', None)
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
                    feature_dimensions=features.Dimensions(screen=config.MAP_SIZE,
                                                           minimap=config.MAP_SIZE),
                    camera_width_world_units=config.MAP_SIZE * 1,

                    action_space=actions.ActionSpace.RAW,
                    use_raw_units=True,
                    raw_resolution=config.MAP_SIZE,
                    use_unit_counts=True
                ),
                score_index=-1,

                step_mul=32,
                disable_fog=False,
                visualize=True,
                realtime=False

        ) as env:
            run_loop.run_loop([agent1], env)

    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    app.run(main)
