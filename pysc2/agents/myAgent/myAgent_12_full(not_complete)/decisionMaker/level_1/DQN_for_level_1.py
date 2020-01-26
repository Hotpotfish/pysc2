import os
import random

import numpy as np
import tensorflow as tf
import pysc2.agents.myAgent.myAgent_12.config.config as config

from pysc2.agents.myAgent.myAgent_12.net.lenet_for_level_1 import Lenet
from pysc2.agents.myAgent.myAgent_12.tools.SqQueue import SqQueue


class DQN():

    def __init__(self, mu, sigma, learning_rate, actiondim, statedim, name):  # 初始化
        # 初始化回放缓冲区，用REPLAY_SIZE定义其最大长度
        self.replay_buffer = SqQueue(config.REPLAY_SIZE)

        # 神经网络参数
        self.mu = mu
        self.sigma = sigma
        self.learning_rate = learning_rate

        self.time_step = 0
        self.epsilon = config.INITIAL_EPSILON

        # 动作维度数，动作参数维度数（默认为6）,状态维度数
        self.action_dim = actiondim

        self.state_dim = statedim

        # 网络结构初始化
        self.name = name
        self.net = Lenet(self.mu, self.sigma, self.learning_rate, self.action_dim, self.state_dim, self.name + '_dqn')

        # Init session
        self.session = tf.InteractiveSession()
        self.session.run(tf.initialize_all_variables())

        self.modelSaver = tf.train.Saver()
        self.recordSaver = None
        self.recordCount = 0

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

        data_summary = tf.Summary(value=[tf.Summary.Value(tag=self.name + '_' + "TD_ERROR", simple_value=self.td_error)])
        self.lossSaver.add_summary(summary=data_summary, global_step=self.epsoide)

    def saveRewardAvg(self, modelSavePath):
        # loss save
        self.rewardSaver = open(modelSavePath + self.name + '_reward.txt', 'a+')
        self.rewardSaver.write(str(self.epsoide) + ' ' + str(self.rewardAdd) + '\n')
        self.rewardAdd = 0
        self.timeStep = 0
        # print(self.rewardAdd / self.rewardStep)
        self.rewardSaver.close()

    def perceive(self, state, action, reward, next_state, done, save_path):  # 感知存储信息
        self.rewardAdd += np.sum(reward)
        self.timeStep += 1

        if done:
            self.epsoide += 1
            self.saveLoss(save_path)
            self.saveRewardAvg(save_path)

        self.replay_buffer.inQueue([state, action, reward, next_state, done])

    def train_Q_network(self, modelSavePath):  # 训练网络
        if self.replay_buffer.real_size > config.BATCH_SIZE:
            minibatch = random.sample(self.replay_buffer.queue, config.BATCH_SIZE)
            state_batch = np.array([data[0] for data in minibatch])
            action_batch = np.array([data[1] for data in minibatch])
            reward_batch = np.array([data[2] for data in minibatch])
            next_state_batch = np.array([data[3] for data in minibatch])

            # Step 2: calculate y
            y_batch = np.array([])
            Q_value_batch = np.array(self.session.run(self.net.Q_value, {self.net.state_input: next_state_batch}))

            for i in range(0, config.BATCH_SIZE):
                done = minibatch[i][4]
                if done:
                    temp = np.append(np.array(reward_batch[i]), np.array(Q_value_batch[i][self.action_dim:]))
                    temp = temp.reshape((1, 1 + self.parameterdim))
                    y_batch = np.append(y_batch, temp)
                else:
                    temp = np.append(np.array(reward_batch[i] + config.GAMMA * np.max(Q_value_batch[i][0:self.action_dim])),
                                     Q_value_batch[i][self.action_dim:])
                    temp = temp.reshape((1, 1 + self.parameterdim))
                    y_batch = np.append(y_batch, temp)
            y_batch = np.array(y_batch).reshape(config.BATCH_SIZE, 1 + self.parameterdim)

            _, loss = self.session.run([self.net.train_op, self.net.loss],
                                       feed_dict={self.net.y_input: y_batch,
                                                  self.net.action_input: action_batch,
                                                  self.net.state_input: state_batch})

            self.saveRecord(modelSavePath, loss)

        # self.saveModel(modelSavePath, episode)

    def egreedy_action(self, state):  # 输出带随机的动作
        Q_value = self.session.run(self.net.Q_value[0], {self.net.state_input: state})
        # print(logits)
        # self.epsilon -= (config.INITIAL_EPSILON - config.FINAL_EPSILON) / 10000
        if np.random.uniform() <= self.epsilon:
            random_action = np.random.randint(0, self.action_dim)

            random_parameter = np.random.rand(self.parameterdim)
            random_action_and_parameter = np.append(random_action, random_parameter).flatten()

            return random_action_and_parameter

        else:
            action = np.argmax(Q_value[0:self.action_dim])
            parameter = np.array(Q_value[self.action_dim:(self.action_dim + self.parameterdim)])
            action_and_parameter = np.append(action, parameter).flatten()
            # print(action_and_parameter)
            return action_and_parameter

    def action(self, state):
        Q_value = self.session.run(self.net.Q_value, {self.net.state_input: state})[0]
        action = np.argmax(Q_value)
        # parameter = np.array(Q_value[self.action_dim:(self.action_dim + self.parameterdim)])
        # action_and_parameter = np.append(action, parameter)
        return action
