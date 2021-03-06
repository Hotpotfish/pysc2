import os

import random

import numpy as np
import tensorflow as tf
import pysc2.agents.myAgent.myAgent_11_PG.config.config as config
from pysc2.agents.myAgent.myAgent_11_PG.tools.SqQueue import SqQueue
from pysc2.agents.myAgent.myAgent_11_PG.net.bicnet_for_level_2.bicnet import bicnet
from pysc2.agents.myAgent.myAgent_11_PG.tools.handcraft_function_for_level_2_attack_controller import discount_and_norm_rewards


class Bicnet():

    def __init__(self, mu, sigma, learning_rate, action_dim, statedim, agents_number, enemy_number, name):  # 初始化
        # 初始化回放缓冲区，用REPLAY_SIZE定义其最大长度
        self.replay_buffer = SqQueue(config.REPLAY_SIZE)

        # 神经网络参数
        self.mu = mu
        self.sigma = sigma
        self.learning_rate = learning_rate

        self.epsilon = config.INITIAL_EPSILON

        # 动作维度数，动作参数维度数（默认为6）,状态维度数
        self.action_dim = action_dim
        # self.parameterdim = parameterdim
        self.state_dim = statedim

        self.agents_number = agents_number
        self.enemy_number = enemy_number

        # 网络结构初始化
        self.name = name

        self.net = bicnet(self.mu, self.sigma, self.learning_rate, self.action_dim, self.state_dim, self.agents_number,
                          self.enemy_number, self.name + '_bicnet')

        # Init session
        self.session = tf.Session()
        self.session.run(tf.initialize_all_variables())

        self.modelSaver = tf.train.Saver()
        self.session.graph.finalize()

        self.epsilon = config.INITIAL_EPSILON

        self.lossSaver = None
        self.loss = 0
        self.epsoide = 0
        self.win = 0

        self.rewardSaver = None
        self.rewardAdd = 0
        self.rewardAvg = 0
        self.rewardStep = 0

        self.timeStep = 0
        self.td_error = 0

        self.restoreModelMark = True

    def restoreModel(self, modelLoadPath):
        self.modelSaver.restore(self.session, modelLoadPath + '/' + self.name + '.ckpt')
        print(self.name + ' ' + 'load!')

    def saveModel(self, modelSavePath, episode):

        thisPath = modelSavePath + 'episode_' + str(episode) + '/'
        try:
            os.makedirs(thisPath)
        except OSError:
            pass
        self.modelSaver.save(self.session, thisPath + self.name + '.ckpt', )

        print(self.name + ' ' + 'saved!')

    def saveLoss(self, modelSavePath):
        # loss save
        if self.lossSaver is None:
            thisPath = modelSavePath
            self.lossSaver = tf.summary.FileWriter(thisPath, self.session.graph)

        data_summary = tf.Summary(
            value=[tf.Summary.Value(tag=self.name + '_' + "LOSS", simple_value=self.loss)])
        self.lossSaver.add_summary(summary=data_summary, global_step=self.epsoide)

    def saveRewardAvg(self, modelSavePath):
        self.rewardSaver = open(modelSavePath + 'PG_reward.txt', 'a+')
        self.rewardSaver.write(str(self.epsoide) + ' ' + str(self.rewardAdd / self.timeStep) + '\n')
        self.rewardAdd = 0
        self.timeStep = 0
        self.rewardSaver.close()

    def saveWinRate(self, modelSavePath):
        self.rewardSaver = open(modelSavePath + 'PG_win_rate.txt', 'a+')
        self.rewardSaver.write(str(self.epsoide) + ' ' + str(self.win / self.epsoide) + '\n')
        self.rewardAdd = 0
        self.timeStep = 0
        self.rewardSaver.close()

    def perceive(self, state, action, reward, done, save_path):  # 感知存储信息
        self.rewardAdd += reward
        self.timeStep += 1

        if done != 0:
            self.epsoide += 1
            if done == 1:
                self.win += 1
            self.saveLoss(save_path)
            self.saveRewardAvg(save_path)
            self.saveWinRate(save_path)

        self.replay_buffer.inQueue([state, action, reward, done])

    def train_Q_network(self):  # 训练网络
        if len(self.replay_buffer.queue) == 0:
            return
        minibatch = self.replay_buffer.queue
        state = np.array([data[0][1] for data in minibatch])
        action_batch = np.array([data[1] for data in minibatch])
        reward_batch = np.array([data[2] for data in minibatch])

        action_batch = np.eye(np.power(self.action_dim, self.agents_number))[action_batch]

        discounted_ep_rs_norm = discount_and_norm_rewards(reward_batch)

        _, self.loss = self.session.run([self.net.trian_op, self.net.loss], {self.net.state: state,
                                                                             self.net.action_input: action_batch,
                                                                             self.net.actions_value: discounted_ep_rs_norm})
        self.replay_buffer.empty()

    def egreedy_action(self, current_state):  # 输出带随机的动作

        action_output = self.session.run(self.net.action_output, {self.net.state: current_state[1][np.newaxis]})
        action_output = np.multiply(current_state[0], action_output)

        action_output = action_output / sum(action_output[0])

        action = np.random.choice(a=len(action_output[0]), p=action_output[0])  # select action w.r.t the actions prob
        return action

    def action(self, current_state):
        action_output = self.session.run(self.net.action_output, {self.net.state: current_state[1][np.newaxis]})
        action_output = np.multiply(current_state[0], action_output)
        return np.argmax(action_output)
