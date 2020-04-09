import os

import random

import numpy as np
import tensorflow as tf
from scipy.spatial import KDTree

import pysc2.agents.myAgent.myAgent_15_BIC_DDPG_2.config.config as config
from pysc2.agents.myAgent.myAgent_15_BIC_DDPG_2.tools.SqQueue import SqQueue
from pysc2.agents.myAgent.myAgent_15_BIC_DDPG_2.net.net_for_level_2.net1 import net1
from pysc2.agents.myAgent.myAgent_15_BIC_DDPG_2.tools.handcraft_function_for_level_2_attack_controller import get_action_combination, get_single_agent_closest_action, agent_k_closest_action


class net():

    def __init__(self, mu, sigma, learning_rate, action_dim, statedim, agents_number, enemy_number, valid_action, name):  # 初始化
        # 初始化回放缓冲区，用REPLAY_SIZE定义其最大长度
        self.replay_buffer = SqQueue(config.REPLAY_SIZE)

        # 神经网络参数
        self.mu = mu
        self.sigma = sigma
        self.var = 0.5
        self.learning_rate = learning_rate

        # 动作维度数，动作参数维度数（默认为6）,状态维度数
        self.action_dim = action_dim
        self.state_dim = statedim

        self.agents_number = agents_number
        self.enemy_number = enemy_number

        # 网络结构初始化
        self.name = name
        self.valid_action = valid_action
        self.KDTree = KDTree(np.array(range(len(valid_action)))[:, np.newaxis])
        self.bound = len(valid_action) / 2

        self.net = net1(self.mu, self.sigma, self.learning_rate, self.action_dim, self.state_dim, self.agents_number, self.enemy_number, len(valid_action), self.name + '_net1')

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

        # action = (np.array(action) - self.bound) / self.bound
        self.replay_buffer.inQueue([state, action, np.repeat(reward, config.MY_UNIT_NUMBER), next_state, done])

    def train_Q_network(self):  # 训练网络
        if self.replay_buffer.real_size > config.BATCH_SIZE:
            minibatch = random.sample(self.replay_buffer.queue, config.BATCH_SIZE)
            agents_local_observation = np.array([data[0][1] for data in minibatch])
            # state = np.array([data[0][0] for data in minibatch])

            action_batch = np.array([data[1] for data in minibatch])
            reward_batch = np.array([data[2] for data in minibatch])
            agents_local_observation_next = np.array([data[3][1] for data in minibatch])
            # state_next = np.array([data[3][0] for data in minibatch])

            a_ = self.session.run(self.net.a_, {self.net.agents_local_observation_next: agents_local_observation_next})
            # a_ = np.reshape(a_, (config.BATCH_SIZE, self.agents_number))
            action_next_temp = []
            for i in range(config.BATCH_SIZE):
                action_proto = a_[i] * self.bound
                action_proto += self.bound
                actions = []

                for j in range(self.agents_number):
                    agent_valid_actions = get_single_agent_closest_action(j, agents_local_observation_next[i][j], self.valid_action)

                    # if len(agent_k_closest_action(agent_valid_actions, action_proto[j])) > 1:
                    #     print()
                    actions += agent_k_closest_action(agent_valid_actions, action_proto[j])
                # if i == 31:
                #     print()
                action_next_temp.append(actions)
                # action_k = get_action_combination(self.KDTree, actio_proto)
                # action = None
                # if config.K == 1:
                #     a_[i] = action_k[0][:, np.newaxis]
                # else:
                #     ob_input = np.repeat(agents_local_observation_next, len(action_k), axis=0)
                #     action_k_input = (np.array(action_k) - self.bound) / self.bound
                #     temp_qs = self.session.run(self.net.q, {self.net.agents_local_observation: ob_input, self.net.a: np.array(action_k_input)[:, :, np.newaxis]})
                #     temp_qs = np.sum(temp_qs, axis=1)
                #     a_[i] = action_k[np.argmax(temp_qs)][:, np.newaxis]

            a_input_next = (np.array(action_next_temp) - self.bound) / self.bound
            action_batch = (np.array(action_batch) - self.bound) / self.bound

            _ = self.session.run(self.net.atrain, {self.net.agents_local_observation: agents_local_observation})

            __, self.td_error = self.session.run([self.net.ctrain, self.net.td_error],
                                                 {self.net.agents_local_observation: agents_local_observation,
                                                  self.net.a: action_batch[:, :, np.newaxis],
                                                  self.net.reward: np.reshape(reward_batch, (config.BATCH_SIZE, config.MY_UNIT_NUMBER, 1)),
                                                  self.net.agents_local_observation_next: agents_local_observation_next,
                                                  self.net.a_: a_input_next[:, :, np.newaxis]})
            self.session.run(self.net.soft_replace)

    def egreedy_action(self, current_state):  # 输出带随机的动作

        action_out = self.session.run(self.net.a, {self.net.agents_local_observation: current_state[1][np.newaxis]})[0]

        # action_out = np.clip(np.random.normal(action_out, 0), -1, 1)
        action_proto = action_out * self.bound
        action_proto += self.bound
        print(list(np.squeeze(action_proto)))
        # self.var = self.var * 0.995

        actions = []

        for i in range(self.agents_number):
            agent_valid_actions = get_single_agent_closest_action(i, current_state[1][i], self.valid_action)
            actions += agent_k_closest_action(agent_valid_actions, action_proto[i])
        # print(actions)

        return actions

        # for i in range(self.agents_number):

        # # print(self.var)
        # action_k = get_action_combination(self.KDTree, actio_proto)
        # if config.K == 1:
        #
        #     return action_k[0]
        # else:
        #     ob_input = np.repeat(current_state[1][np.newaxis], len(action_k), axis=0)
        #     action_k_input = (np.array(action_k) - self.bound) / self.bound
        #     temp_qs = self.session.run(self.net.q, {self.net.agents_local_observation: ob_input, self.net.a: np.array(action_k_input)[:, :, np.newaxis]})
        #     temp_qs = np.sum(temp_qs, axis=1)
        #
        #     action = action_k[np.argmax(temp_qs)]
        #     return action

    def action(self, current_state):
        actio_out = self.session.run(self.net.a, {self.net.agents_local_observation: current_state[1][np.newaxis]})[0]
        actio_out = np.clip(np.random.normal(actio_out, 0), 0, 1)
        actio_proto = actio_out * self.bound

        action_k = get_action_combination(self.KDTree, actio_proto)
        if config.K == 1:
            return actio_out, action_k[0]
        else:
            state_input = np.repeat(current_state[0][np.newaxis], np.power(config.K, self.agents_number), axis=0)
            temp_qs = self.session.run(self.net.q, {self.net.state_input: state_input, self.net.a: np.array(action_k)[:, np.newaxis]})
            action = action_k[np.argmax(temp_qs)]
            return actio_out, action
