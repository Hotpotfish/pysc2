import os

import random

import numpy as np
import tensorflow as tf
import pysc2.agents.myAgent.myAgent_15_BIC_DDPG_2.config.config as config
from pysc2.agents.myAgent.myAgent_15_BIC_DDPG_2.tools.SqQueue import SqQueue
from pysc2.agents.myAgent.myAgent_15_BIC_DDPG_2.net.net_for_level_2.net1 import net1
from pysc2.agents.myAgent.myAgent_15_BIC_DDPG_2.tools.handcraft_function_for_level_2_attack_controller import get_k_closest_action, get_action_combination
import pysc2.agents.myAgent.myAgent_15_BIC_DDPG_2.smart_actions as sa


class net():

    def __init__(self, mu, sigma, learning_rate, action_dim, statedim, agents_number, enemy_number, valid_action, name):  # 初始化
        # 初始化回放缓冲区，用REPLAY_SIZE定义其最大长度
        self.replay_buffer = SqQueue(config.REPLAY_SIZE)

        # 神经网络参数
        self.mu = mu
        self.sigma = sigma
        self.var = self.var = np.array([len(sa.attack_controller) - 1,
                                        len(sa.attack_controller) - 1,
                                        config.MAP_SIZE - 1,
                                        config.MAP_SIZE - 1,
                                        len(sa.attack_controller) - 1,
                                        config.MY_UNIT_NUMBER + config.ENEMY_UNIT_NUMBER - 1
                                        ])
        self.learning_rate = learning_rate

        # 动作维度数，动作参数维度数（默认为6）,状态维度数
        self.action_dim = action_dim
        self.state_dim = statedim

        self.agents_number = agents_number
        self.enemy_number = enemy_number

        # 网络结构初始化
        self.name = name
        self.valid_action = valid_action

        self.net = net1(self.mu, self.sigma, self.learning_rate, self.action_dim, self.state_dim, self.agents_number, self.enemy_number, self.name + '_net1')

        # Init session
        self.session = tf.Session()
        self.session.run(tf.initialize_all_variables())

        self.modelSaver = tf.train.Saver()
        self.session.graph.finalize()

        self.epsilon = config.INITIAL_EPSILON

        self.lossSaver = None
        self.loss = 0
        self.win = 0
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
            value=[tf.Summary.Value(tag=self.name + '_' + "td_error", simple_value=self.td_error)])
        self.lossSaver.add_summary(summary=data_summary, global_step=self.epsoide)

    def saveRewardAvg(self, modelSavePath):
        # loss save
        self.rewardSaver = open(modelSavePath + 'BIC_DDPG_reward.txt', 'a+')
        self.rewardSaver.write(str(self.epsoide) + ' ' + str(self.rewardAdd / self.timeStep) + '\n')
        self.rewardAdd = 0
        self.timeStep = 0
        self.rewardSaver.close()

    def saveWinRate(self, modelSavePath):
        self.rewardSaver = open(modelSavePath + 'BIC_DDPG_win_rate.txt', 'a+')
        self.rewardSaver.write(str(self.epsoide) + ' ' + str(self.win / self.epsoide) + '\n')
        self.rewardAdd = 0
        self.timeStep = 0
        self.rewardSaver.close()

    def perceive(self, state, action, reward, next_state, done, save_path):  # 感知存储信息
        self.rewardAdd += reward
        self.timeStep += 1

        if done != 0:
            self.epsoide += 1
            if done == 1:
                self.win += 1
            self.saveLoss(save_path)
            self.saveRewardAvg(save_path)
            self.saveWinRate(save_path)

        self.replay_buffer.inQueue([state, action, reward, next_state, done])

    def train_Q_network(self):  # 训练网络
        if self.replay_buffer.real_size > config.BATCH_SIZE:
            minibatch = random.sample(self.replay_buffer.queue, config.BATCH_SIZE)
            agents_local_observation = np.array([data[0][2] for data in minibatch])
            state = np.array([data[0][1] for data in minibatch])
            bound = np.array([data[0][0] for data in minibatch])

            action_batch = np.array([data[1] for data in minibatch])
            reward_batch = np.array([data[2] for data in minibatch])
            agents_local_observation_next = np.array([data[3][2] for data in minibatch])
            state_next = np.array([data[3][1] for data in minibatch])

            _ = self.session.run(self.net.atrain, {self.net.state_input: state,
                                                   self.net.agents_local_observation: agents_local_observation,
                                                   self.net.bound: bound})

            __, self.td_error = self.session.run([self.net.ctrain, self.net.td_error],
                                                 {self.net.state_input: state,
                                                  self.net.bound: bound,
                                                  self.net.a: action_batch,
                                                  self.net.reward: reward_batch,
                                                  self.net.state_input_next: state_next,
                                                  self.net.agents_local_observation_next: agents_local_observation_next
                                                  })

    def egreedy_action(self, current_state):  # 输出带随机的动作

        actio_proto = self.session.run(self.net.a, {self.net.agents_local_observation: current_state[2][np.newaxis], self.net.bound: current_state[0][np.newaxis]})[0]
        for i in range(config.MY_UNIT_NUMBER):
            actio_proto[i][0] = np.clip(np.random.normal(actio_proto[i][0], self.var[0]), 0, len(sa.attack_controller) - 1)
            actio_proto[i][1] = np.clip(np.random.normal(actio_proto[i][1], self.var[1]), 0, len(sa.attack_controller) - 1)
            actio_proto[i][2] = np.clip(np.random.normal(actio_proto[i][2], self.var[2]), 0, config.MAP_SIZE - 1)
            actio_proto[i][3] = np.clip(np.random.normal(actio_proto[i][3], self.var[3]), 0, config.MAP_SIZE - 1)
            actio_proto[i][4] = np.clip(np.random.normal(actio_proto[i][4], self.var[4]), 0, len(sa.attack_controller) - 1)
            actio_proto[i][5] = np.clip(np.random.normal(actio_proto[i][5], self.var[5]), 0, config.MY_UNIT_NUMBER + config.ENEMY_UNIT_NUMBER - 1)

        self.var = self.var * 0.995
        action_k = get_action_combination(self.valid_action, actio_proto)
        temp_qs = []
        for i in range(np.power(config.K, self.agents_number)):
            temp_q = self.session.run(self.net.q, {self.net.state_input: current_state[1][np.newaxis], self.net.a: action_k[i][np.newaxis]})[0]
            temp_qs.append(temp_q)
        action = action_k[np.argmax(temp_qs)]
        return actio_proto, action

    def action(self, current_state):
        Q_value = self.session.run(self.net.q_value, {self.net.state: current_state[1][np.newaxis], self.net.agents_local_observation: current_state[2][np.newaxis]})[0]
        Q_value = np.multiply(current_state[0], Q_value)
        return np.argmax(Q_value, axis=1)
