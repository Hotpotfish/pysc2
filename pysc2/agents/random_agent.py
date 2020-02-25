# Copyright 2017 Google Inc. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS-IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""A random agent for starcraft."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy

import pysc2.agents.myAgent.myAgent_13_BIC_DQN.config.config as config
from absl import app
from pysc2.agents import base_agent

from pysc2.lib import actions, features
from pysc2.env import sc2_env, run_loop


class RandomAgent(base_agent.BaseAgent):
    """A random agent for starcraft."""

    def step(self, obs):
        # super(RandomAgent, self).step(obs)
        # function_id = numpy.random.choice(563)
        # args = [[numpy.random.randint(0, size) for size in arg.sizes]
        #         for arg in self.action_spec.functions[function_id].args]
        super(RandomAgent, self).step(obs)
        function_id = 3
        args = [[0], [0], [-1]]
        args2 = [[0], [1], [-1]]
        print(actions.FunctionCall(function_id, args))

        return [actions.FunctionCall(function_id, args)]#actions.FunctionCall(function_id, args2)]


def main(unused_argv):
    agent1 = RandomAgent()

    try:
        with sc2_env.SC2Env(
                map_name="DefeatRoaches",
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
            run_loop.run_loop([agent1], env, max_episodes=config.EPISODES)

    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    app.run(main)
