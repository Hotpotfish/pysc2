import os

import random

import numpy as np
import tensorflow as tf
import pysc2.agents.myAgent.myAgent_11.config.config as config
from pysc2.agents.myAgent.myAgent_11.tools.SqQueue import SqQueue
from pysc2.agents.myAgent.myAgent_11.net.bicnet_for_level_2.bicnet import bicnet


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

        # tf.get_default_graph().finalize()

        self.modelSaver = tf.train.Saver()
        self.session.graph.finalize()

        self.lossSaver = None
        self.epsoide = 0

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
            value=[tf.Summary.Value(tag=self.name + '_' + "TD_ERROR", simple_value=self.td_error)])
        self.lossSaver.add_summary(summary=data_summary, global_step=self.epsoide)

    def saveRewardAvg(self, modelSavePath):
        # loss save
        self.rewardSaver = open(modelSavePath + 'reward.txt', 'a+')
        self.rewardSaver.write(str(self.epsoide) + ' ' + str(self.rewardAdd) + '\n')
        self.rewardAdd = 0
        self.timeStep = 0
        # print(self.rewardAdd / self.rewardStep)
        self.rewardSaver.close()

    def perceive(self, state, action, reward, next_state, done, save_path):  # 感知存储信息
        self.rewardAdd += reward
        self.timeStep += 1

        if done:
            self.epsoide += 1
            self.saveLoss(save_path)
            self.saveRewardAvg(save_path)

        self.replay_buffer.inQueue([state, action, reward, next_state, done])

    def train_Q_network(self):  # 训练网络
        if self.replay_buffer.real_size > config.BATCH_SIZE:
            minibatch = random.sample(self.replay_buffer.queue, config.BATCH_SIZE)
            state_input = np.array([data[0][0] for data in minibatch])
            agents_local_observation = np.array([data[0][1] for data in minibatch])
            action_batch = np.array([data[1] for data in minibatch])
            reward_batch = np.array([data[2] for data in minibatch])
            state_input_next = np.array([data[3][0] for data in minibatch])
            agents_local_observation_next = np.array([data[3][1] for data in minibatch])

            action_batch = np.eye(self.action_dim)[action_batch]

            self.session.run(self.net.soft_replace)
            _, q = self.session.run([self.net.atrain, self.net.q], {self.net.state_input: state_input,
                                                                    self.net.agents_local_observation: agents_local_observation})

            __, self.td_error = self.session.run([self.net.ctrain, self.net.td_error],
                                                 {self.net.state_input: state_input,
                                                  self.net.agents_local_observation: agents_local_observation,
                                                  self.net.a: action_batch,
                                                  self.net.reward: reward_batch,
                                                  self.net.state_input_next: state_input_next,
                                                  self.net.agents_local_observation_next: agents_local_observation_next
                                                  })

    def get_execute_action(self, prob_value):
        actions = []

        for i in range(config.MY_UNIT_NUMBER):
            Nt = np.random.randn(self.action_dim)
            actions.append(prob_value[i] + Nt)

        return actions

    def egreedy_action(self, state):  # 输出带随机的动作

        state_input = state[0][np.newaxis, :, :]
        agents_local_observation = state[1][np.newaxis, :, :]
        prob_value = self.session.run(self.net.a, {self.net.state_input: state_input,
                                                   self.net.agents_local_observation: agents_local_observation})[0]
        actions = self.get_execute_action(prob_value)
        return actions

    def action(self, state):
        state_input = state[0][np.newaxis, :, :]
        agents_local_observation = state[1][np.newaxis, :, :]
        prob_value = self.session.run(self.net.a, {self.net.state_input: state_input,
                                                   self.net.agents_local_observation: agents_local_observation})[0]
        return prob_value
