# import random
# import numpy as np
# import pandas as pd
# import os
# from absl import app
# from pysc2.agents import base_agent
# from pysc2.lib import actions, features, units
# from pysc2.env import sc2_env, run_loop
#
#
# class QLearningTable:
#     def __init__(self, actions, learning_rate=0.01, reward_decay=0.9):
#         self.actions = actions
#         self.learning_rate = learning_rate
#         self.reward_decay = reward_decay
#         self.q_table = pd.DataFrame(columns=self.actions, dtype=np.float64)
#
#     def choose_action(self, observation, e_greedy=0.9):
#         self.check_state_exist(observation)
#         if np.random.uniform() < e_greedy:
#             state_action = self.q_table.loc[observation, :]
#             action = np.random.choice(
#                 state_action[state_action == np.max(state_action)].index)
#         else:
#             action = np.random.choice(self.actions)
#         return action
#
#     def learn(self, s, a, r, s_):
#         self.check_state_exist(s_)
#         q_predict = self.q_table.loc[s, a]
#         if s_ != 'terminal':
#             q_target = r + self.reward_decay * self.q_table.loc[s_, :].max()
#         else:
#             q_target = r
#         self.q_table.loc[s, a] += self.learning_rate * (q_target - q_predict)
#
#     def check_state_exist(self, state):
#         if state not in self.q_table.index:
#             self.q_table = self.q_table.append(pd.Series([0] * len(self.actions),
#                                                          index=self.q_table.columns,
#                                                          name=state))
#
#
# class Agent(base_agent.BaseAgent):
#     actions = ("do_nothing",
#                "harvest_minerals",
#                "build_supply_depot",
#                "build_barracks",
#                "train_marine",
#                "attack")
#
#     def get_my_units_by_type(self, obs, unit_type):
#         return [unit for unit in obs.observation.raw_units
#                 if unit.unit_type == unit_type
#                 and unit.alliance == features.PlayerRelative.SELF]
#
#     def get_enemy_units_by_type(self, obs, unit_type):
#         return [unit for unit in obs.observation.raw_units
#                 if unit.unit_type == unit_type
#                 and unit.alliance == features.PlayerRelative.ENEMY]
#
#     def get_my_completed_units_by_type(self, obs, unit_type):
#         return [unit for unit in obs.observation.raw_units
#                 if unit.unit_type == unit_type
#                 and unit.build_progress == 100
#                 and unit.alliance == features.PlayerRelative.SELF]
#
#     def get_enemy_completed_units_by_type(self, obs, unit_type):
#         return [unit for unit in obs.observation.raw_units
#                 if unit.unit_type == unit_type
#                 and unit.build_progress == 100
#                 and unit.alliance == features.PlayerRelative.ENEMY]
#
#     def get_distances(self, obs, units, xy):
#         units_xy = [(unit.x, unit.y) for unit in units]
#         return np.linalg.norm(np.array(units_xy) - np.array(xy), axis=1)
#
#     def step(self, obs):
#         super(Agent, self).step(obs)
#         if obs.first():
#             command_center = self.get_my_units_by_type(
#                 obs, units.Terran.CommandCenter)[0]
#             self.base_top_left = (command_center.x < 32)
#
#     def do_nothing(self, obs):
#         return actions.RAW_FUNCTIONS.no_op()
#
#     def harvest_minerals(self, obs):
#         scvs = self.get_my_units_by_type(obs, units.Terran.SCV)
#         idle_scvs = [scv for scv in scvs if scv.order_length == 0]
#         if len(idle_scvs) > 0:
#             mineral_patches = [unit for unit in obs.observation.raw_units
#                                if unit.unit_type in [
#                                    units.Neutral.BattleStationMineralField,
#                                    units.Neutral.BattleStationMineralField750,
#                                    units.Neutral.LabMineralField,
#                                    units.Neutral.LabMineralField750,
#                                    units.Neutral.MineralField,
#                                    units.Neutral.MineralField750,
#                                    units.Neutral.PurifierMineralField,
#                                    units.Neutral.PurifierMineralField750,
#                                    units.Neutral.PurifierRichMineralField,
#                                    units.Neutral.PurifierRichMineralField750,
#                                    units.Neutral.RichMineralField,
#                                    units.Neutral.RichMineralField750
#                                ]]
#             scv = random.choice(idle_scvs)
#             distances = self.get_distances(obs, mineral_patches, (scv.x, scv.y))
#             mineral_patch = mineral_patches[np.argmin(distances)]
#             return actions.RAW_FUNCTIONS.Harvest_Gather_unit(
#                 "now", scv.tag, mineral_patch.tag)
#         return actions.RAW_FUNCTIONS.no_op()
#
#     def build_supply_depot(self, obs):
#         supply_depots = self.get_my_units_by_type(obs, units.Terran.SupplyDepot)
#         scvs = self.get_my_units_by_type(obs, units.Terran.SCV)
#         if (len(supply_depots) == 0 and obs.observation.player.minerals >= 100 and
#                 len(scvs) > 0):
#             supply_depot_xy = (22, 26) if self.base_top_left else (35, 42)
#             distances = self.get_distances(obs, scvs, supply_depot_xy)
#             scv = scvs[np.argmin(distances)]
#             return actions.RAW_FUNCTIONS.Build_SupplyDepot_pt(
#                 "now", scv.tag, supply_depot_xy)
#         return actions.RAW_FUNCTIONS.no_op()
#
#     def build_barracks(self, obs):
#         completed_supply_depots = self.get_my_completed_units_by_type(
#             obs, units.Terran.SupplyDepot)
#         barrackses = self.get_my_units_by_type(obs, units.Terran.Barracks)
#         scvs = self.get_my_units_by_type(obs, units.Terran.SCV)
#         if (len(completed_supply_depots) > 0 and len(barrackses) == 0 and
#                 obs.observation.player.minerals >= 150 and len(scvs) > 0):
#             barracks_xy = (22, 21) if self.base_top_left else (35, 45)
#             distances = self.get_distances(obs, scvs, barracks_xy)
#             scv = scvs[np.argmin(distances)]
#             return actions.RAW_FUNCTIONS.Build_Barracks_pt(
#                 "now", scv.tag, barracks_xy)
#         return actions.RAW_FUNCTIONS.no_op()
#
#     def train_marine(self, obs):
#         completed_barrackses = self.get_my_completed_units_by_type(
#             obs, units.Terran.Barracks)
#         free_supply = (obs.observation.player.food_cap -
#                        obs.observation.player.food_used)
#         if (len(completed_barrackses) > 0 and obs.observation.player.minerals >= 100
#                 and free_supply > 0):
#             barracks = self.get_my_units_by_type(obs, units.Terran.Barracks)[0]
#             if barracks.order_length < 5:
#                 return actions.RAW_FUNCTIONS.Train_Marine_quick("now", barracks.tag)
#         return actions.RAW_FUNCTIONS.no_op()
#
#     def attack(self, obs):
#         marines = self.get_my_units_by_type(obs, units.Terran.Marine)
#         if len(marines) > 0:
#             attack_xy = (38, 44) if self.base_top_left else (19, 23)
#             distances = self.get_distances(obs, marines, attack_xy)
#             marine = marines[np.argmax(distances)]
#             x_offset = random.randint(-4, 4)
#             y_offset = random.randint(-4, 4)
#             return actions.RAW_FUNCTIONS.Attack_pt(
#                 "now", marine.tag, (attack_xy[0] + x_offset, attack_xy[1] + y_offset))
#         return actions.RAW_FUNCTIONS.no_op()
#
#
# class RandomAgent(Agent):
#     def step(self, obs):
#         super(RandomAgent, self).step(obs)
#         action = random.choice(self.actions)
#         return getattr(self, action)(obs)
#
#
# class SmartAgent(Agent):
#     def __init__(self):
#         super(SmartAgent, self).__init__()
#         self.qtable = QLearningTable(self.actions)
#         self.new_game()
#
#     def reset(self):
#         super(SmartAgent, self).reset()
#         self.new_game()
#
#     def new_game(self):
#         self.base_top_left = None
#         self.previous_state = None
#         self.previous_action = None
#
#     def get_state(self, obs):
#         scvs = self.get_my_units_by_type(obs, units.Terran.SCV)
#         idle_scvs = [scv for scv in scvs if scv.order_length == 0]
#         command_centers = self.get_my_units_by_type(obs, units.Terran.CommandCenter)
#         supply_depots = self.get_my_units_by_type(obs, units.Terran.SupplyDepot)
#         completed_supply_depots = self.get_my_completed_units_by_type(
#             obs, units.Terran.SupplyDepot)
#         barrackses = self.get_my_units_by_type(obs, units.Terran.Barracks)
#         completed_barrackses = self.get_my_completed_units_by_type(
#             obs, units.Terran.Barracks)
#         marines = self.get_my_units_by_type(obs, units.Terran.Marine)
#
#         queued_marines = (completed_barrackses[0].order_length
#                           if len(completed_barrackses) > 0 else 0)
#
#         free_supply = (obs.observation.player.food_cap -
#                        obs.observation.player.food_used)
#         can_afford_supply_depot = obs.observation.player.minerals >= 100
#         can_afford_barracks = obs.observation.player.minerals >= 150
#         can_afford_marine = obs.observation.player.minerals >= 100
#
#         enemy_scvs = self.get_enemy_units_by_type(obs, units.Terran.SCV)
#         enemy_idle_scvs = [scv for scv in enemy_scvs if scv.order_length == 0]
#         enemy_command_centers = self.get_enemy_units_by_type(
#             obs, units.Terran.CommandCenter)
#         enemy_supply_depots = self.get_enemy_units_by_type(
#             obs, units.Terran.SupplyDepot)
#         enemy_completed_supply_depots = self.get_enemy_completed_units_by_type(
#             obs, units.Terran.SupplyDepot)
#         enemy_barrackses = self.get_enemy_units_by_type(obs, units.Terran.Barracks)
#         enemy_completed_barrackses = self.get_enemy_completed_units_by_type(
#             obs, units.Terran.Barracks)
#         enemy_marines = self.get_enemy_units_by_type(obs, units.Terran.Marine)
#
#         return (len(command_centers),
#                 len(scvs),
#                 len(idle_scvs),
#                 len(supply_depots),
#                 len(completed_supply_depots),
#                 len(barrackses),
#                 len(completed_barrackses),
#                 len(marines),
#                 queued_marines,
#                 free_supply,
#                 can_afford_supply_depot,
#                 can_afford_barracks,
#                 can_afford_marine,
#                 len(enemy_command_centers),
#                 len(enemy_scvs),
#                 len(enemy_idle_scvs),
#                 len(enemy_supply_depots),
#                 len(enemy_completed_supply_depots),
#                 len(enemy_barrackses),
#                 len(enemy_completed_barrackses),
#                 len(enemy_marines))
#
#     def step(self, obs):
#         super(SmartAgent, self).step(obs)
#         state = str(self.get_state(obs))
#         action = self.qtable.choose_action(state)
#         if self.previous_action is not None:
#             self.qtable.learn(self.previous_state,
#                               self.previous_action,
#                               obs.reward,
#                               'terminal' if obs.last() else state)
#         self.previous_state = state
#         self.previous_action = action
#         return getattr(self, action)(obs)
#
#
# def main(unused_argv):
#     agent1 = SmartAgent()
#     agent2 = RandomAgent()
#     try:
#         with sc2_env.SC2Env(
#                 map_name="Simple64",
#                 players=[sc2_env.Agent(sc2_env.Race.terran),
#                          sc2_env.Agent(sc2_env.Race.terran)],
#                 agent_interface_format=features.AgentInterfaceFormat(
#                     action_space=actions.ActionSpace.RAW,
#                     use_raw_units=True,
#                     raw_resolution=64,
#                 ),
#                 step_mul=8,
#                 disable_fog=True,
#         ) as env:
#             run_loop.run_loop([agent1, agent2], env, max_episodes=1000)
#     except KeyboardInterrupt:
#         pass
#
#
# if __name__ == "__main__":
#     app.run(main)

# Copyright 2019 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Self-diagnosis script for TensorBoard.

Instructions: Save this script to your local machine, then execute it in
the same environment (virtualenv, Conda, etc.) from which you normally
run TensorBoard. Read the output and follow the directions.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# This script may only depend on the Python standard library. It is not
# built with Bazel and should not assume any third-party dependencies.
import collections
import errno
import functools
import hashlib
import inspect
import logging
import os
import pipes
import shlex
import socket
import subprocess
import sys
import tempfile
import textwrap
import traceback


# A *check* is a function (of no arguments) that performs a diagnostic,
# writes log messages, and optionally yields suggestions. Each check
# runs in isolation; exceptions will be caught and reported.
CHECKS = []


# A suggestion to the end user.
#   headline (str): A short description, like "Turn it off and on
#     again". Should be imperative with no trailing punctuation. May
#     contain inline Markdown.
#   description (str): A full enumeration of the steps that the user
#     should take to accept the suggestion. Within this string, prose
#     should be formatted with `reflow`. May contain Markdown.
Suggestion = collections.namedtuple("Suggestion", ("headline", "description"))


def check(fn):
    """Decorator to register a function as a check.

    Checks are run in the order in which they are registered.

    Args:
      fn: A function that takes no arguments and either returns `None` or
        returns a generator of `Suggestion`s. (The ability to return
        `None` is to work around the awkwardness of defining empty
        generator functions in Python.)

    Returns:
      A wrapped version of `fn` that returns a generator of `Suggestion`s.
    """

    @functools.wraps(fn)
    def wrapper():
        result = fn()
        return iter(()) if result is None else result

    CHECKS.append(wrapper)
    return wrapper


def reflow(paragraph):
    return textwrap.fill(textwrap.dedent(paragraph).strip())


def pip(args):
    """Invoke command-line Pip with the specified args.

    Returns:
      A bytestring containing the output of Pip.
    """
    # Suppress the Python 2.7 deprecation warning.
    PYTHONWARNINGS_KEY = "PYTHONWARNINGS"
    old_pythonwarnings = os.environ.get(PYTHONWARNINGS_KEY)
    new_pythonwarnings = "%s%s" % (
        "ignore:DEPRECATION",
        ",%s" % old_pythonwarnings if old_pythonwarnings else "",
    )
    command = [sys.executable, "-m", "pip", "--disable-pip-version-check"]
    command.extend(args)
    try:
        os.environ[PYTHONWARNINGS_KEY] = new_pythonwarnings
        return subprocess.check_output(command)
    finally:
        if old_pythonwarnings is None:
            del os.environ[PYTHONWARNINGS_KEY]
        else:
            os.environ[PYTHONWARNINGS_KEY] = old_pythonwarnings


def which(name):
    """Return the path to a binary, or `None` if it's not on the path.

    Returns:
      A bytestring.
    """
    binary = "where" if os.name == "nt" else "which"
    try:
        return subprocess.check_output([binary, name])
    except subprocess.CalledProcessError:
        return None


def sgetattr(attr, default):
    """Get an attribute off the `socket` module, or use a default."""
    sentinel = object()
    result = getattr(socket, attr, sentinel)
    if result is sentinel:
        print("socket.%s does not exist" % attr)
        return default
    else:
        print("socket.%s = %r" % (attr, result))
        return result


@check
def autoidentify():
    """Print the Git hash of this version of `diagnose_tensorboard.py`.

    Given this hash, use `git cat-file blob HASH` to recover the
    relevant version of the script.
    """
    module = sys.modules[__name__]
    try:
        source = inspect.getsource(module).encode("utf-8")
    except TypeError:
        logging.info("diagnose_tensorboard.py source unavailable")
    else:
        # Git inserts a length-prefix before hashing; cf. `git-hash-object`.
        blob = b"blob %d\0%s" % (len(source), source)
        hash = hashlib.sha1(blob).hexdigest()
        logging.info("diagnose_tensorboard.py version %s", hash)


@check
def general():
    logging.info("sys.version_info: %s", sys.version_info)
    logging.info("os.name: %s", os.name)
    na = type("N/A", (object,), {"__repr__": lambda self: "N/A"})
    logging.info(
        "os.uname(): %r", getattr(os, "uname", na)(),
    )
    logging.info(
        "sys.getwindowsversion(): %r", getattr(sys, "getwindowsversion", na)(),
    )


@check
def package_management():
    conda_meta = os.path.join(sys.prefix, "conda-meta")
    logging.info("has conda-meta: %s", os.path.exists(conda_meta))
    logging.info("$VIRTUAL_ENV: %r", os.environ.get("VIRTUAL_ENV"))


@check
def installed_packages():
    freeze = pip(["freeze", "--all"]).decode("utf-8").splitlines()
    packages = {line.split(u"==")[0]: line for line in freeze}
    packages_set = frozenset(packages)

    # For each of the following families, expect exactly one package to be
    # installed.
    expect_unique = [
        frozenset([u"tensorboard", u"tb-nightly", u"tensorflow-tensorboard",]),
        frozenset(
            [
                u"tensorflow",
                u"tensorflow-gpu",
                u"tf-nightly",
                u"tf-nightly-2.0-preview",
                u"tf-nightly-gpu",
                u"tf-nightly-gpu-2.0-preview",
            ]
        ),
        frozenset(
            [
                u"tensorflow-estimator",
                u"tensorflow-estimator-2.0-preview",
                u"tf-estimator-nightly",
            ]
        ),
    ]

    found_conflict = False
    for family in expect_unique:
        actual = family & packages_set
        for package in actual:
            logging.info("installed: %s", packages[package])
        if len(actual) == 0:
            logging.warning("no installation among: %s", sorted(family))
        elif len(actual) > 1:
            logging.warning("conflicting installations: %s", sorted(actual))
            found_conflict = True

    if found_conflict:
        preamble = reflow(
            """
            Conflicting package installations found. Depending on the order
            of installations and uninstallations, behavior may be undefined.
            Please uninstall ALL versions of TensorFlow and TensorBoard,
            then reinstall ONLY the desired version of TensorFlow, which
            will transitively pull in the proper version of TensorBoard. (If
            you use TensorBoard without TensorFlow, just reinstall the
            appropriate version of TensorBoard directly.)
            """
        )
        packages_to_uninstall = sorted(
            frozenset().union(*expect_unique) & packages_set
        )
        commands = [
            "pip uninstall %s" % " ".join(packages_to_uninstall),
            "pip install tensorflow  # or `tensorflow-gpu`, or `tf-nightly`, ...",
        ]
        message = "%s\n\nNamely:\n\n%s" % (
            preamble,
            "\n".join("\t%s" % c for c in commands),
        )
        yield Suggestion("Fix conflicting installations", message)


@check
def tensorboard_python_version():
    from tensorboard import version

    logging.info("tensorboard.version.VERSION: %r", version.VERSION)


@check
def tensorflow_python_version():
    import tensorflow as tf

    logging.info("tensorflow.__version__: %r", tf.__version__)
    logging.info("tensorflow.__git_version__: %r", tf.__git_version__)


@check
def tensorboard_binary_path():
    logging.info("which tensorboard: %r", which("tensorboard"))


@check
def addrinfos():
    sgetattr("has_ipv6", None)
    family = sgetattr("AF_UNSPEC", 0)
    socktype = sgetattr("SOCK_STREAM", 0)
    protocol = 0
    flags_loopback = sgetattr("AI_ADDRCONFIG", 0)
    flags_wildcard = sgetattr("AI_PASSIVE", 0)

    hints_loopback = (family, socktype, protocol, flags_loopback)
    infos_loopback = socket.getaddrinfo(None, 0, *hints_loopback)
    print("Loopback flags: %r" % (flags_loopback,))
    print("Loopback infos: %r" % (infos_loopback,))

    hints_wildcard = (family, socktype, protocol, flags_wildcard)
    infos_wildcard = socket.getaddrinfo(None, 0, *hints_wildcard)
    print("Wildcard flags: %r" % (flags_wildcard,))
    print("Wildcard infos: %r" % (infos_wildcard,))


@check
def readable_fqdn():
    # May raise `UnicodeDecodeError` for non-ASCII hostnames:
    # https://github.com/tensorflow/tensorboard/issues/682
    try:
        logging.info("socket.getfqdn(): %r", socket.getfqdn())
    except UnicodeDecodeError as e:
        try:
            binary_hostname = subprocess.check_output(["hostname"]).strip()
        except subprocess.CalledProcessError:
            binary_hostname = b"<unavailable>"
        is_non_ascii = not all(
            0x20
            <= (ord(c) if not isinstance(c, int) else c)
            <= 0x7E  # Python 2
            for c in binary_hostname
        )
        if is_non_ascii:
            message = reflow(
                """
                Your computer's hostname, %r, contains bytes outside of the
                printable ASCII range. Some versions of Python have trouble
                working with such names (https://bugs.python.org/issue26227).
                Consider changing to a hostname that only contains printable
                ASCII bytes.
                """
                % (binary_hostname,)
            )
            yield Suggestion("Use an ASCII hostname", message)
        else:
            message = reflow(
                """
                Python can't read your computer's hostname, %r. This can occur
                if the hostname contains non-ASCII bytes
                (https://bugs.python.org/issue26227). Consider changing your
                hostname, rebooting your machine, and rerunning this diagnosis
                script to see if the problem is resolved.
                """
                % (binary_hostname,)
            )
            yield Suggestion("Use a simpler hostname", message)
        raise e


@check
def stat_tensorboardinfo():
    # We don't use `manager._get_info_dir`, because (a) that requires
    # TensorBoard, and (b) that creates the directory if it doesn't exist.
    path = os.path.join(tempfile.gettempdir(), ".tensorboard-info")
    logging.info("directory: %s", path)
    try:
        stat_result = os.stat(path)
    except OSError as e:
        if e.errno == errno.ENOENT:
            # No problem; this is just fine.
            logging.info(".tensorboard-info directory does not exist")
            return
        else:
            raise
    logging.info("os.stat(...): %r", stat_result)
    logging.info("mode: 0o%o", stat_result.st_mode)
    if stat_result.st_mode & 0o777 != 0o777:
        preamble = reflow(
            """
            The ".tensorboard-info" directory was created by an old version
            of TensorBoard, and its permissions are not set correctly; see
            issue #2010. Change that directory to be world-accessible (may
            require superuser privilege):
            """
        )
        # This error should only appear on Unices, so it's okay to use
        # Unix-specific utilities and shell syntax.
        quote = getattr(shlex, "quote", None) or pipes.quote  # Python <3.3
        command = "chmod 777 %s" % quote(path)
        message = "%s\n\n\t%s" % (preamble, command)
        yield Suggestion('Fix permissions on "%s"' % path, message)


@check
def source_trees_without_genfiles():
    roots = list(sys.path)
    if "" not in roots:
        # Catch problems that would occur in a Python interactive shell
        # (where `""` is prepended to `sys.path`) but not when
        # `diagnose_tensorboard.py` is run as a standalone script.
        roots.insert(0, "")

    def has_tensorboard(root):
        return os.path.isfile(os.path.join(root, "tensorboard", "__init__.py"))

    def has_genfiles(root):
        sample_genfile = os.path.join("compat", "proto", "summary_pb2.py")
        return os.path.isfile(os.path.join(root, "tensorboard", sample_genfile))

    def is_bad(root):
        return has_tensorboard(root) and not has_genfiles(root)

    tensorboard_roots = [root for root in roots if has_tensorboard(root)]
    bad_roots = [root for root in roots if is_bad(root)]

    logging.info(
        "tensorboard_roots (%d): %r; bad_roots (%d): %r",
        len(tensorboard_roots),
        tensorboard_roots,
        len(bad_roots),
        bad_roots,
    )

    if bad_roots:
        if bad_roots == [""]:
            message = reflow(
                """
                Your current directory contains a `tensorboard` Python package
                that does not include generated files. This can happen if your
                current directory includes the TensorBoard source tree (e.g.,
                you are in the TensorBoard Git repository). Consider changing
                to a different directory.
                """
            )
        else:
            preamble = reflow(
                """
                Your Python path contains a `tensorboard` package that does
                not include generated files. This can happen if your current
                directory includes the TensorBoard source tree (e.g., you are
                in the TensorBoard Git repository). The following directories
                from your Python path may be problematic:
                """
            )
            roots = []
            realpaths_seen = set()
            for root in bad_roots:
                label = repr(root) if root else "current directory"
                realpath = os.path.realpath(root)
                if realpath in realpaths_seen:
                    # virtualenvs on Ubuntu install to both `lib` and `local/lib`;
                    # explicitly call out such duplicates to avoid confusion.
                    label += " (duplicate underlying directory)"
                realpaths_seen.add(realpath)
                roots.append(label)
            message = "%s\n\n%s" % (
                preamble,
                "\n".join("  - %s" % s for s in roots),
            )
        yield Suggestion(
            "Avoid `tensorboard` packages without genfiles", message
        )


# Prefer to include this check last, as its output is long.
@check
def full_pip_freeze():
    logging.info(
        "pip freeze --all:\n%s", pip(["freeze", "--all"]).decode("utf-8")
    )


def set_up_logging():
    # Manually install handlers to prevent TensorFlow from stomping the
    # default configuration if it's imported:
    # https://github.com/tensorflow/tensorflow/issues/28147
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(logging.Formatter("%(levelname)s: %(message)s"))
    logger.addHandler(handler)


def main():
    set_up_logging()

    print("### Diagnostics")
    print()

    print("<details>")
    print("<summary>Diagnostics output</summary>")
    print()

    markdown_code_fence = "``````"  # seems likely to be sufficient
    print(markdown_code_fence)
    suggestions = []
    for (i, check) in enumerate(CHECKS):
        if i > 0:
            print()
        print("--- check: %s" % check.__name__)
        try:
            suggestions.extend(check())
        except Exception:
            traceback.print_exc(file=sys.stdout)
            pass
    print(markdown_code_fence)
    print()
    print("</details>")

    for suggestion in suggestions:
        print()
        print("### Suggestion: %s" % suggestion.headline)
        print()
        print(suggestion.description)

    print()
    print("### Next steps")
    print()
    if suggestions:
        print(
            reflow(
                """
                Please try each suggestion enumerated above to determine whether
                it solves your problem. If none of these suggestions works,
                please copy ALL of the above output, including the lines
                containing only backticks, into your GitHub issue or comment. Be
                sure to redact any sensitive information.
                """
            )
        )
    else:
        print(
            reflow(
                """
                No action items identified. Please copy ALL of the above output,
                including the lines containing only backticks, into your GitHub
                issue or comment. Be sure to redact any sensitive information.
                """
            )
        )


if __name__ == "__main__":
    main()