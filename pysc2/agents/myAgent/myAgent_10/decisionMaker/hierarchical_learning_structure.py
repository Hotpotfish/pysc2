import datetime
from pysc2.agents.myAgent.myAgent_10.config import config
from pysc2.agents.myAgent.myAgent_10.decisionMaker.level_1.level_1 import level_1
from pysc2.agents.myAgent.myAgent_10.decisionMaker.level_2.level_2 import level_2
from pysc2.env.environment import StepType
from pysc2.lib import actions


class hierarchical_learning_structure():
    def __init__(self):
        self.episode = 0
        self.win = 0
        self.begin_time = datetime.datetime.now().strftime('%Y%m%d%H%M%S')
        self.load_mark = False

        self.leve1_1 = level_1()
        self.level_2 = level_2()

    def get_model_savePath(self, topPath):
        time = str(self.begin_time)
        modelSavePath = topPath + '/' + time + '/'
        return modelSavePath

    def train_action(self, obs):
        # controller_number = int(self.leve1_1.train_action(obs)[0])
        # controller_number = self.leve1_1.test_action(obs)
        action = self.level_2.train_action(obs, 0)
        return action

    def train_network(self, modelSavePath):
        # self.leve1_1.train_network(modelSavePath)
        self.level_2.train_network(modelSavePath)

    def test_action(self, obs):
        controller_number = self.leve1_1.test_action(obs)
        action = self.level_2.test_action(obs, controller_number)
        print(action)
        return action

    def load_model(self, modelLoadPath):
        # self.leve1_1.load_model(modelLoadPath)
        self.level_2.load_model(modelLoadPath)

    def save_model(self, modelSavePath):
        # self.leve1_1.save_model(modelSavePath, self.episode)
        self.level_2.save_model(modelSavePath, self.episode)

    def execute(self, obs, mark, modelSavePath, modelLoadPath):
        if obs[0] == StepType.FIRST:
            # 更新读取和保存路径
            return actions.RAW_FUNCTIONS.raw_move_camera((config.MAP_SIZE / 2, config.MAP_SIZE / 2))

        if mark == 'TRAIN':
            if modelLoadPath is not None:
                if not self.load_mark:
                    self.load_mark = True
                    self.load_model(modelLoadPath)

            if obs[0] == StepType.LAST:
                save_path = self.get_model_savePath(modelSavePath)
                self.train_network(save_path)

                self.episode += 1
                print('episode:%d   score_cumulative: %f' % (self.episode, obs.observation['score_cumulative'][0]))

                # 模型保存
                if self.episode % config.MODEL_SAVE_EPISODE == 0:
                    self.save_model(save_path)
            return self.train_action(obs)

        elif mark == 'TEST':
            if modelLoadPath is not None:
                if not self.load_mark:
                    self.load_mark = True
                    self.load_model(modelLoadPath)

            return self.test_action(obs)
