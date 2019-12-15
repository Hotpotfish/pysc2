import os

import random

import numpy as np
import tensorflow as tf
import pysc2.agents.myAgent.myAgent_10.config.config as config

# from pysc2.agents.myAgent.myAgent_10.net.lenet_for_level_2_PG import Lenet

import pysc2.agents.myAgent.myAgent_10.tools.handcraft_function as handcraft_function
from pysc2.agents.myAgent.myAgent_10.net.bicnet_for_level_2.bicnet_for_level_2_actor import bicnet_actor
from pysc2.agents.myAgent.myAgent_10.net.bicnet_for_level_2.bicnet_for_level_2_cirtic import bicnet_critic
from pysc2.agents.myAgent.myAgent_10.tools.SqQueue import SqQueue


class Bicnet():

    def __init__(self, mu, sigma, learning_rate, action_dim, parameterdim, statedim, agents_number, enemy_number, name):  # 初始化
        # 初始化回放缓冲区，用REPLAY_SIZE定义其最大长度
        self.replay_buffer = SqQueue(config.REPLAY_SIZE)

        # 神经网络参数
        self.mu = mu
        self.sigma = sigma
        self.learning_rate = learning_rate

        self.time_step = 0
        self.epsilon = config.INITIAL_EPSILON

        # 动作维度数，动作参数维度数（默认为6）,状态维度数
        self.action_dim = action_dim
        self.parameterdim = parameterdim
        self.state_dim = statedim

        self.agents_number = agents_number
        self.enemy_number = enemy_number

        # 网络结构初始化
        self.name = name
        self.actor_net = bicnet_actor(self.mu, self.sigma, self.learning_rate, self.action_dim, self.parameterdim, self.state_dim, self.agents_number, self.enemy_number, self.name)
        self.critic_net = bicnet_critic(self.mu, self.sigma, self.learning_rate, self.action_dim, self.parameterdim, self.state_dim, self.agents_number, self.enemy_number, self.name)

        # Init session
        self.session = tf.Session()
        self.session.run(tf.initialize_all_variables())

        # tf.get_default_graph().finalize()

        self.modelSaver = tf.train.Saver()
        self.session.graph.finalize()

        self.recordSaver = None
        self.recordCount = 0

        self.restoreModelMark = True

        self.epsiod_record = []
        self.reward_add = 0
        self.reward_avg = 0

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

    # def saveRecord(self, modelSavePath, data):
    #     if self.recordSaver is None:
    #         thisPath = modelSavePath
    #         self.recordSaver = tf.summary.FileWriter(thisPath, self.session.graph)
    #
    #     data_summary = tf.Summary(value=[tf.Summary.Value(tag=self.name + '_' + "loss", simple_value=data)])
    #     self.recordSaver.add_summary(summary=data_summary, global_step=self.recordCount)
    #     self.recordCount += 1

    def perceive(self, state, action, reward, done):  # 感知存储信息

        # print('reward:'+str(reward))

        if done:
            # if (self.reward_add / len(self.epsiod_record)) > self.reward_avg:
            #     self.reward_avg = self.reward_add / len(self.epsiod_record)
            #     self.replay_buffer.inQueue(self.epsiod_record)
            #
            # self.epsiod_record.clear()
            # self.reward_add = 0
            # print('reward_avg:' + str(self.reward_avg))

            self.replay_buffer.inQueue(self.epsiod_record)
            self.reward_add = 0
            self.epsiod_record.clear()


        else:
            action_data = np.array([])
            action_data = np.append(action_data, handcraft_function.one_hot_encoding(int(action[0]), self.action_dim))
            action_data = np.append(action_data, handcraft_function.one_hot_encoding(int(action[1]), config.QUEUED))
            action_data = np.append(action_data, handcraft_function.one_hot_encoding(int(action[2]), config.MY_UNIT_NUMBER))
            action_data = np.append(action_data, handcraft_function.one_hot_encoding(int(action[3]), config.ENEMY_UNIT_NUMBER))
            action_data = np.append(action_data, handcraft_function.one_hot_encoding(int(action[4]), config.POINT_NUMBER))
            self.reward_add = self.reward_add * config.GAMMA + reward
            temp_reward = np.array([self.reward_add for i in range(config.ORDERLENTH)]).flatten()
            self.epsiod_record.append([state[0], action_data, temp_reward, done])

            # self.replay_buffer.inQueue([state[0], action_data, reward, next_state[0], done])

        # 计算单条的y值

    def get_y_value(self, Q_value, reward, mark):
        y_value = []

        if mark == 0:
            start = 0
            end = self.action_dim
            y_value.append(config.GAMMA * np.max(Q_value[start: end]) + reward)

            start = end
            end += config.QUEUED
            y_value.append(config.GAMMA * np.max(Q_value[start: end]) + reward)

            start = end
            end += config.MY_UNIT_NUMBER
            y_value.append(config.GAMMA * np.max(Q_value[start: end]) + reward)

            start += end
            end += config.ENEMY_UNIT_NUMBER
            y_value.append(config.GAMMA * np.max(Q_value[start: end]) + reward)

            start = end
            y_value.append(config.GAMMA * np.max(Q_value[start:]) + reward)

        else:
            y_value = [reward, reward, reward, reward, reward]

        return np.array(y_value)

    def train_Q_network(self, modelSavePath):  # 训练网络
        if self.replay_buffer.real_size >= config.EP_SIZE:
            last_loss = None
            for i in range(config.EP_SIZE):
                episode_data = np.array(random.sample(self.replay_buffer.queue, 1)[0])
                episode_data_len = len(episode_data)
                start = 0
                end = config.BATCH_SIZE
                while 1:
                    if end <= episode_data_len:
                        input_batch = episode_data[start:end]
                        state_batch = np.array([data[0] for data in input_batch])
                        action_batch = np.array([data[1] for data in input_batch])
                        reward_batch = np.array([data[2] for data in input_batch])
                        start += config.BATCH_SIZE
                        end += config.BATCH_SIZE
                        _, loss = self.session.run([self.net.train_op, self.net.loss],
                                                   feed_dict={self.net.reward_input: reward_batch,
                                                              self.net.action_input: action_batch,
                                                              self.net.state_input: state_batch,
                                                              self.net.train: True})
                    else:
                        input_batch = episode_data[start:]
                        state_batch = np.array([data[0] for data in input_batch])
                        action_batch = np.array([data[1] for data in input_batch])
                        reward_batch = np.array([data[2] for data in input_batch])
                        _, loss = self.session.run([self.net.train_op, self.net.loss],
                                                   feed_dict={self.net.reward_input: reward_batch,
                                                              self.net.action_input: action_batch,
                                                              self.net.state_input: state_batch,
                                                              self.net.train: True})
                        last_loss = loss
                        break

                # self.session.close()
                # self.session = tf.Session()
            # self.saveRecord(modelSavePath, last_loss)

    def get_random_action_and_parameter_one_hot(self):
        random_action_and_parameter = np.array([])
        random_action = np.random.randint(0, self.action_dim)
        random_action_one_hot = handcraft_function.one_hot_encoding(random_action, self.action_dim)
        random_action_and_parameter = np.append(random_action_and_parameter, random_action_one_hot)

        random_queued = np.random.randint(0, config.QUEUED)
        random_queued_one_hot = handcraft_function.one_hot_encoding(random_queued, config.QUEUED)
        random_action_and_parameter = np.append(random_action_and_parameter, random_queued_one_hot)

        random_my_unit = np.random.randint(0, config.MY_UNIT_NUMBER)
        random_my_unit_one_hot = handcraft_function.one_hot_encoding(random_my_unit, config.MY_UNIT_NUMBER)
        random_action_and_parameter = np.append(random_action_and_parameter, random_my_unit_one_hot)

        random_enemy_unit = np.random.randint(0, config.ENEMY_UNIT_NUMBER)
        random_enemy_unit_one_hot = handcraft_function.one_hot_encoding(random_enemy_unit, config.ENEMY_UNIT_NUMBER)
        random_action_and_parameter = np.append(random_action_and_parameter, random_enemy_unit_one_hot)

        random_target_point = np.random.randint(0, config.POINT_NUMBER)
        random_target_point_one_hot = handcraft_function.one_hot_encoding(random_target_point, config.POINT_NUMBER)
        random_action_and_parameter = np.append(random_action_and_parameter, random_target_point_one_hot)
        return random_action_and_parameter

    def egreedy_action(self, state):  # 输出带随机的动作

        prob_value = self.session.run(self.net.prob_value, {self.net.state_input: state, self.net.train: True})[0]
        # print(prob_value)
        # self.epsilon -= (config.INITIAL_EPSILON - config.FINAL_EPSILON) / 10000
        if np.random.uniform() <= self.epsilon:
            random_action_and_parameter = self.get_random_action_and_parameter_one_hot()
            return random_action_and_parameter

        else:
            return prob_value

    def action(self, state):
        prob_value = self.session.run(self.net.prob_value, {self.net.state_input: state, self.net.train: False})[0]
        return prob_value
