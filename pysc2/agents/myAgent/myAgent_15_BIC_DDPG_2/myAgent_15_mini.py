import pysc2.agents.myAgent.myAgent_15_BIC_DDPG_2.config.config as config
from absl import app
from pysc2.agents import base_agent
from pysc2.agents.myAgent.myAgent_15_BIC_DDPG_2.decisionMaker.hierarchical_learning_structure import \
    hierarchical_learning_structure
from pysc2.agents.myAgent.myAgent_15_BIC_DDPG_2.tools.plt_function import plt_function
from pysc2.env.environment import StepType

from pysc2.lib import actions, features
from pysc2.env import sc2_env, run_loop, available_actions_printer


class myAgent(base_agent.BaseAgent):

    def __init__(self):
        super(myAgent, self).__init__()
        self.hierarchical_learning_structure = hierarchical_learning_structure()
        self.plt_function = plt_function()

    def add_or_plt(self, obs, steps):
        self.plt_function.add_summary(obs, steps)
        if obs[0] == StepType.LAST:
            self.plt_function.plt_score_cumulative()

    def step(self, obs):
        # self.add_or_plt(obs, self.steps)

        super(myAgent, self).step(obs)
        action = self.hierarchical_learning_structure.execute(obs, 'TRAIN', 'd:/model/', None)
        # action = self.hierarchical_learning_structure.execute(obs, 'TRAIN', 'e:/model/',  'D:/model/20191230172144/episode_300')
        # action = self.hierarchical_learning_structure.execute(obs, 'TEST', None, 'D:/model/20200316151111/episode_2000')
        # print(action)
        return action


def main(unused_argv):
    agent1 = myAgent()

    try:
        with sc2_env.SC2Env(
                map_name="8t_vs_7z",
                players=[sc2_env.Agent(sc2_env.Race.terran), ],
                agent_interface_format=features.AgentInterfaceFormat(
                    feature_dimensions=features.Dimensions(screen=config.MAP_SIZE,
                                                           minimap=config.MAP_SIZE),

                    camera_width_world_units=config.MAP_SIZE * 1,
                    action_space=actions.ActionSpace.RAW,
                    use_raw_units=True,
                    raw_resolution=config.MAP_SIZE,
                ),
                score_index=0,
                step_mul=8,
                disable_fog=False,
                visualize=False,
                realtime=False,

        ) as env:
            # env=available_actions_printer.AvailableActionsPrinter(env)
            run_loop.run_loop([agent1], env)

    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    app.run(main)
