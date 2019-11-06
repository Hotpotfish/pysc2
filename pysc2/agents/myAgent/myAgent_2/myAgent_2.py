from queue import Queue

import pysc2.agents.myAgent.myAgent_2.macro_operation as macro_operation

from absl import app
from pysc2.agents import base_agent
from pysc2.lib import actions, features
from pysc2.env import sc2_env, run_loop
import pysc2.agents.myAgent.myAgent_2.smart_actions as sa
import pysc2.agents.myAgent.myAgent_2.q_learing_table as q_learing_table


KILL_UNIT_REWARD = 0.2
KILL_BUILDING_REWARD = 0.5


class myAgent(base_agent.BaseAgent):

    def __init__(self):
        super(myAgent, self).__init__()

        self.qlearn = q_learing_table.QLearningTable(actions=list(range(len(sa.smart_actions))))

        self.previous_killed_unit_score = 0
        self.previous_killed_building_score = 0

        self.previous_action = None
        self.previous_state = None

        self.figureData = []

    def step(self, obs):
        super(myAgent, self).step(obs)
        current_state = q_learing_table.currentState(obs).getData()

        killed_unit_score = obs.observation['score_cumulative'][5]
        killed_building_score = obs.observation['score_cumulative'][6]

        self.figureData.append([self.steps, killed_unit_score, killed_building_score])

        if self.previous_action is not None:
            reward = 0

            if killed_unit_score > self.previous_killed_unit_score:
                reward += KILL_UNIT_REWARD

            if killed_building_score > self.previous_killed_building_score:
                reward += KILL_BUILDING_REWARD

            self.qlearn.learn(str(self.previous_state), self.previous_action, reward, str(current_state))
        rl_action = self.qlearn.choose_action(str(current_state))

        self.previous_killed_unit_score = killed_unit_score
        self.previous_killed_building_score = killed_building_score
        self.previous_state = current_state
        self.previous_action = rl_action

        return sa.smart_actions[4](obs)


def main(unused_argv):
    agent1 = myAgent()

    try:
        with sc2_env.SC2Env(
                map_name="DefeatZerglingsAndBanelings",
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
                step_mul=0.000000001,
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
