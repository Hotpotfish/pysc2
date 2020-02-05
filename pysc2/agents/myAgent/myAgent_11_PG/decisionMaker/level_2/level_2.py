from pysc2.agents.myAgent.myAgent_11_PG.config import config
import pysc2.agents.myAgent.myAgent_11_PG.smart_actions as sa
from pysc2.agents.myAgent.myAgent_11_PG.tools import handcraft_function


class level_2():
    def __init__(self):
        self.DataShape = (None, config.MAP_SIZE, config.MAP_SIZE, 39)
        self.controllers = handcraft_function.append_controllers()

    # 重训练模式 无需读取外部模型
    def train_action(self, obs, controller_number, save_path):
        action = self.controllers[controller_number].train_action(obs, save_path)
        # print(action)
        return action

    def test_action(self, obs, controller_number):
        action = self.controllers[controller_number].test_action(obs)
        return action

    def train_network(self):
        for i in range(len(sa.controllers)):
            self.controllers[i].controller.network.train_Q_network()

    def load_model(self, modelLoadPath):
        for i in range(len(sa.controllers)):
            self.controllers[i].controller.network.restoreModel(modelLoadPath)
            print('level_2 load complete!')

    def save_model(self, modelSavePath, episode):
        for i in range(len(sa.controllers)):
            self.controllers[i].controller.network.saveModel(modelSavePath, episode)
        print('level_2 episode %d save complete!' % (episode))
