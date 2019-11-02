from queue import Queue

import pysc2.agents.myAgent.myAgent_2.macro_operation as macro_operation

from absl import app
from pysc2.agents import base_agent
from pysc2.lib import actions, features
from pysc2.env import sc2_env, run_loop
import pysc2.agents.myAgent.myAgent_2.smart_actions as sa
import pysc2.agents.myAgent.myAgent_2.q_learing_table as q_learing_table

_NO_OP = actions.FUNCTIONS.no_op.id
_NOT_QUEUED = [0]
_QUEUED = [1]

KILL_UNIT_REWARD = 0.2
KILL_BUILDING_REWARD = 0.5


class myAgent(base_agent.BaseAgent):

    def __init__(self):
        super(myAgent, self).__init__()

        # 宏操作的大队列
        self.macroOpQueue = Queue()

        # 宏操作中原子的小队列
        self.tempMarcoOp = None

        self.tempMarcoOp_step = 0

        self.qlearn = q_learing_table.QLearningTable(actions=list(range(len(sa.smart_actions))))

        self.previous_killed_unit_score = 0
        self.previous_killed_building_score = 0

        self.previous_action = None
        self.previous_state = None

        self.figureData = []

    def inQueue(self, functionNumber):

        self.macroOpQueue.put(functionNumber)

        return 0

    def opperation(self, obs):

        # 宏操作队列判空
        if self.macroOpQueue.empty() and self.tempMarcoOp is None:
            return actions.FunctionCall(_NO_OP, [])

        if self.tempMarcoOp is None:
            self.tempMarcoOp = self.macroOpQueue.get()

        if self.tempMarcoOp_step < sa.smart_actions[self.tempMarcoOp][1]:

            atomicOp = sa.smart_actions[self.tempMarcoOp][0](obs, self.tempMarcoOp_step)

            if atomicOp is not None and int(atomicOp[0]) in obs.observation['available_actions']:
                self.tempMarcoOp_step += 1

                return atomicOp

        self.tempMarcoOp = None

        self.tempMarcoOp_step = 0

        return actions.FunctionCall(_NO_OP, [])

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
        self.inQueue(rl_action)

        self.previous_killed_unit_score = killed_unit_score
        self.previous_killed_building_score = killed_building_score
        self.previous_state = current_state
        self.previous_action = rl_action

        f = self.opperation(obs)

        return f


# def main(unused_argv):
#     agent = myAgent()
#     try:
#         while True:
#             with sc2_env.SC2Env(
#                     map_name="Flat96",
#                     players=[sc2_env.Agent(race=sc2_env.Race.terran, name='agent'),
#                              sc2_env.Bot(sc2_env.Race.random,
#                                          sc2_env.Difficulty.very_easy)],
#                     agent_interface_format=features.AgentInterfaceFormat(
#                         feature_dimensions=features.Dimensions(screen=macro_operation.screenSize,
#                                                                minimap=macro_operation.minimapSize),
#                         camera_width_world_units=macro_operation.screenSize,
#                         use_unit_counts=True,
#
#                     ),
#
#                     step_mul=8,
#                     game_steps_per_episode=0,
#                     realtime=True,
#                     visualize=True,
#
#             ) as env:
#
#                 agent.setup(env.observation_spec(), env.action_spec())
#                 timesteps = env.reset()
#                 agent.reset()
#
#                 while True:
#                     step_actions = [agent.step(timesteps[0]),]
#                     if timesteps[0].last():
#                         agent.figureData = np.array(agent.figureData)
#                         plt.plot(agent.figureData[:, 0], agent.figureData[:, 1])
#                         plt.plot(agent.figureData[:, 0], agent.figureData[:, 2])
#
#                         plt.show()
#
#                         break
#                     timesteps = env.step(step_actions, step_mul=0.00000000000000001)
#
#     except KeyboardInterrupt:
#         pass
#
#
# if __name__ == "__main__":
#     app.run(main)

def main(unused_argv):
    agent1 = myAgent()


    try:
        while True:
            with sc2_env.SC2Env(
                    map_name="Flat96",
                    players=[sc2_env.Agent(race=sc2_env.Race.terran, name='agent1'),
                             sc2_env.Bot(sc2_env.Race.protoss,
                                         sc2_env.Difficulty.very_easy)],

                    agent_interface_format=features.AgentInterfaceFormat(
                        feature_dimensions=features.Dimensions(screen=macro_operation.screenSize,
                                                               minimap=macro_operation.minimapSize),
                        action_space=actions.ActionSpace.RAW,
                        camera_width_world_units=macro_operation.screenSize,
                        use_unit_counts=True,
                        use_raw_units=True,
                        raw_resolution=macro_operation.screenSize,

                    ),

                    step_mul=8,
                    game_steps_per_episode=0,
                    realtime=False,
                    visualize=True,

            ) as env:
                run_loop.run_loop([agent1], env)

    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    app.run(main)
