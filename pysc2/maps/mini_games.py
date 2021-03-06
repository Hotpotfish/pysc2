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
"""Define the mini game map configs. These are maps made by Deepmind."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from pysc2.maps import lib


class MiniGame(lib.Map):
    directory = "mini_games"
    download = "https://github.com/deepmind/pysc2#get-the-maps"
    players = 1
    score_index = 0
    game_steps_per_episode = 0
    step_mul = 8


mini_games = [
    "BuildMarines",  # 900s
    "CollectMineralsAndGas",  # 420s
    "CollectMineralShards",  # 120s
    "DefeatRoaches",  # 120s
    "DefeatZerglingsAndBanelings",  # 120s
    "FindAndDefeatZerglings",  # 180s
    "MoveToBeacon",  # 120s
    '3m_vs_3m',
    '3r_vs_3r',
    '3r_vs_4r',
    '30m_vs_30m',
    '30m_vs_60z',
    '2m_vs_1zlot',
    'mix_vs_mix',
    'mix_vs_mix_2',
    'mix_vs_mix_3',
    '8t_vs_7z',
    '1r_vs_1z',
    '8m_vs_8m',
    '3m_vs_3m',
    '3t_vs_3z',
    '8t_vs_7z_1',
    'test',
    '9t_vs_10z'

]

for name in mini_games:
    globals()[name] = type(name, (MiniGame,), dict(filename=name))
