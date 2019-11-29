import pysc2.agents.myAgent.myAgent_8.config.config as config
from absl import app
from pysc2.agents import base_agent
from pysc2.agents.myAgent.myAgent_8.decisionMaker.hierarchical_learning_structure import hierarchical_learning_structure
from pysc2.agents.myAgent.myAgent_8.tools.plt_function import plt_function
from pysc2.env.environment import StepType

from pysc2.lib import actions, features
from pysc2.env import sc2_env, run_loop


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
        action = self.hierarchical_learning_structure.execute(obs, 'TRAIN', 'e:/model/', None)
        # action = self.hierarchical_learning_structure.execute(obs, 'TRAIN', 'e:/model/', 'E:/model//20191128235435/episode_320')
        # action = self.hierarchical_learning_structure.execute(obs, 'TEST', None, 'E:/model/20191128232219/episode_80')
        # print(action)
        return action


def main(unused_argv):
    agent1 = myAgent()

    try:
        with sc2_env.SC2Env(
                map_name="DefeatZerglingsAndBanelings",
                players=[sc2_env.Agent(sc2_env.Race.terran), ],
                agent_interface_format=features.AgentInterfaceFormat(
                    feature_dimensions=features.Dimensions(screen=config.MAP_SIZE,
                                                           minimap=config.MAP_SIZE),
                    camera_width_world_units=config.MAP_SIZE * 1,
                    action_space=actions.ActionSpace.RAW,
                    use_raw_units=True,
                    raw_resolution=config.MAP_SIZE,
                    use_unit_counts=True
                ),
                score_index=0,
                step_mul=0.0001,
                disable_fog=False,
                visualize=True,
                realtime=True

        ) as env:
            run_loop.run_loop([agent1], env)

    except KeyboardInterrupt:
        pass




if __name__ == "__main__":
    app.run(main)
