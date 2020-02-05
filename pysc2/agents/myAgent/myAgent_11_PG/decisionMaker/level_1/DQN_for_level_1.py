
import os
import random

import numpy as np
import tensorflow as tf
import pysc2.agents.myAgent.myAgent_11_PG.config.config as config

from pysc2.agents.myAgent.myAgent_11_PG.net.lenet_for_level_1 import Lenet
from pysc2.agents.myAgent.myAgent_11_PG.tools.SqQueue import SqQueue


class DQN():

    def __init__(self, mu, sigma, learning_rate, actiondim, parameterdim, statedim, name):  # 初始化
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
        self.parameterdim = parameterdim
        self.state_dim = statedim

        # 网络结构初始化
        self.name = name
        self.net = Lenet(self.mu, self.sigma, self.learning_rate, self.action_dim, self.parameterdim, self.state_dim, self.name)

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

    def saveRecord(self, modelSavePath, data):
        if self.recordSaver is None:
            thisPath = modelSavePath
            self.recordSaver = tf.summary.FileWriter(thisPath, self.session.graph)

        data_summary = tf.Summary(value=[tf.Summary.Value(tag=self.name + '_' + "loss", simple_value=data)])
        self.recordSaver.add_summary(summary=data_summary, global_step=self.recordCount)
        self.recordCount += 1

    def perceive(self, state, action, reward, next_state, done):  # 感知存储信息
        one_hot_action = np.zeros(self.action_dim + self.parameterdim, dtype=np.float32)
        one_hot_action[int(action[0])] = 1
        if self.parameterdim != 0:
            one_hot_action[self.action_dim:] = action[1:]
        # state = np.squeeze(state)
        # next_state = np.squeeze(next_state)

        self.replay_buffer.inQueue([state[0], one_hot_action, reward, next_state[0], done])

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
