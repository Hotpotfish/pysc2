import datetime
from pysc2.agents.myAgent.myAgent_13_BIC_DQN.config import config
from pysc2.agents.myAgent.myAgent_13_BIC_DQN.decisionMaker.level_1.level_1 import level_1
from pysc2.agents.myAgent.myAgent_13_BIC_DQN.decisionMaker.level_2.level_2 import level_2
from pysc2.env.environment import StepType



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
        modelSavePath = topPath + time + '/'
        return modelSavePath

    def train_action(self, obs, save_path):
        # controller_number = int(self.leve1_1.train_action(obs)[0])
        # controller_number = self.leve1_1.test_action(obs)
        action = self.level_2.train_action(obs, 0, save_path)
        return action

    def train_network(self):
        # self.leve1_1.train_network(modelSavePath)
        self.level_2.train_network()

    def test_action(self, obs):
        # controller_number = self.leve1_1.test_action(obs)
        action = self.level_2.test_action(obs, 0)
        print(action)
        return action

    def load_model(self, modelLoadPath):
        # self.leve1_1.load_model(modelLoadPath)
        self.level_2.load_model(modelLoadPath)

    def save_model(self, modelSavePath):
        # self.leve1_1.save_model(modelSavePath, self.episode)
        self.level_2.save_model(modelSavePath, self.episode)

    def execute(self, obs, mark, modelSavePath, modelLoadPath):
        # if obs[0] == StepType.FIRST:
        #     # 更新读取和保存路径
        #     return actions.RAW_FUNCTIONS.raw_move_camera((config.MAP_SIZE / 2, config.MAP_SIZE / 2))

        if mark == 'TRAIN':
            if modelLoadPath is not None:
                if not self.load_mark:
                    self.load_mark = True
                    self.load_model(modelLoadPath)
            save_path = self.get_model_savePath(modelSavePath)

            if obs[0] == StepType.LAST:
                self.episode += 1
                # print('episode:%d   score_cumulative: %f' % (self.episode, obs.observation['score_cumulative'][0]))

                # 模型保存
                if self.episode % config.MODEL_SAVE_EPISODE == 0:
                    self.save_model(save_path)
            self.train_network()
            return self.train_action(obs, save_path)

        elif mark == 'TEST':
            if modelLoadPath is not None:
                if not self.load_mark:
                    self.load_mark = True
                    self.load_model(modelLoadPath)

            return self.test_action(obs)
