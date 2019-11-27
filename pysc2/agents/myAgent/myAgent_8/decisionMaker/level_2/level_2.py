from pysc2.agents.myAgent.myAgent_8.config import config
import pysc2.agents.myAgent.myAgent_8.smart_actions as sa
from pysc2.agents.myAgent.myAgent_8.tools import handcraft_function



class level_2():
    def __init__(self):
        self.DataShape = (None, config.MAP_SIZE, config.MAP_SIZE, 39)
        self.controllers = handcraft_function.append_controllers()


    # 重训练模式 无需读取外部模型
    def train_action(self, obs, controller_number):
        action = self.controllers[controller_number].train_action(obs)
        return action

    def test_action(self, obs, controller_number):
        self.controllers[controller_number].controller.current_state = handcraft_function.get_all_observation(obs)
        state = self.controllers[controller_number].controller.current_state
        action_and_parameter = self.controllers[controller_number].controller.network.action(state)
        macro_and_parameter = handcraft_function.reflect(obs, action_and_parameter)
        action = handcraft_function.assembly_action(obs, controller_number, macro_and_parameter)
        return action

    def train_network(self,modelSavePath):
        for i in range(len(sa.controllers)):
            self.controllers[i].controller.network.train_Q_network(modelSavePath)

    def load_model(self, modelLoadPath):
        for i in range(len(sa.controllers)):
            self.controllers[i].controller.network.restoreModel(modelLoadPath)
            print('level_2 load complete!')

    def save_model(self, modelSavePath, episode):
        for i in range(len(sa.controllers)):
            self.controllers[i].controller.network.saveModel(modelSavePath, episode)
        print('level_2 episode %d save complete!'%(episode))
