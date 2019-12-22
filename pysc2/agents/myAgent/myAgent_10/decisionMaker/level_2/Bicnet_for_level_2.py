import os

import random

import numpy as np
import tensorflow as tf
import pysc2.agents.myAgent.myAgent_10.config.config as config
import pysc2.agents.myAgent.myAgent_10.tools.handcraft_function as handcraft_function
from pysc2.agents.myAgent.myAgent_10.net.bicnet_for_level_2.bicnet_for_level_2_actor import bicnet_actor
from pysc2.agents.myAgent.myAgent_10.net.bicnet_for_level_2.bicnet_for_level_2_cirtic import bicnet_critic
from pysc2.agents.myAgent.myAgent_10.tools.SqQueue import SqQueue


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

        self.actor_net = bicnet_actor(self.mu, self.sigma, self.learning_rate, self.action_dim, self.state_dim, self.agents_number, self.enemy_number, self.name + '_actor')
        self.critic_net = bicnet_critic(self.mu, self.sigma, self.learning_rate, self.action_dim, self.state_dim, self.agents_number, self.enemy_number, self.name + '_critic')

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
        self.loss = 0

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

        data_summary = tf.Summary(value=[tf.Summary.Value(tag=self.name + '_' + "loss", simple_value=self.loss)])
        self.lossSaver.add_summary(summary=data_summary, global_step=self.epsoide)

    def saveRewardAvg(self, modelSavePath):
        # loss save
        self.rewardSaver = open(modelSavePath + 'reward.txt', 'a+')
        self.rewardSaver.write(str(self.epsoide) + ' ' + str(self.rewardAdd / self.timeStep) + '\n')
        self.rewardAdd = 0
        self.timeStep = 0
        # print(self.rewardAdd / self.rewardStep)
        self.rewardSaver.close()

    def perceive(self, state, action, reward, next_state, done, save_path):  # 感知存储信息
        self.rewardAdd += np.sum(reward)
        # print(np.sum(reward))
        self.timeStep += 1

        # self.rewardStep += 1
        # print(np.sum(reward))

        if done:
            # print("dsadsada   " + str(self.rewardAdd))
            # print(self.rewardAdd)
            self.epsoide += 1
            # print(self.rewardAdd)
            self.saveLoss(save_path)
            self.saveRewardAvg(save_path)

            # print(self.rewardAvg)

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

            a_ = self.session.run(self.actor_net.a_, {self.actor_net.state_input_next: state_input_next, self.actor_net.agents_local_observation_next: agents_local_observation_next})  # s_
            a_ = np.eye(self.action_dim)[np.argmax(a_, axis=2)].astype(np.float32)
            # q_
            q_ = self.session.run(self.critic_net.q_, {self.critic_net.state_input_next: state_input_next,
                                                       self.critic_net.agents_local_observation_next: agents_local_observation_next,
                                                       self.critic_net.action_input_next: a_})
            # q尖
            q_cusp = reward_batch + config.GAMMA * q_

            action_batch = np.eye(self.action_dim)[action_batch]
            # critic update
            _, self.loss, action_grad = self.session.run([self.critic_net.trian_op, self.critic_net.loss, self.critic_net.action_grad], {self.critic_net.state_input: state_input,
                                                                                                                                         self.critic_net.agents_local_observation: agents_local_observation,
                                                                                                                                         # s_
                                                                                                                                         self.critic_net.action_input: action_batch,
                                                                                                                                         self.critic_net.q_input: q_cusp  # a_
                                                                                                                                         })
            # actor_update
            __ = self.session.run(self.actor_net.train_op, {self.actor_net.state_input: state_input,
                                                            self.actor_net.agents_local_observation: agents_local_observation,  # s_
                                                            self.actor_net.action_gradient: action_grad  # a_
                                                            })

            self.session.run(self.actor_net.soft_replace)
            self.session.run(self.critic_net.soft_replace)

    def get_random_action_and_parameter_one_hot(self):
        actions = []

        for i in range(config.COOP_AGENTS_NUMBER):
            random_action = np.random.randint(0, self.action_dim)
            random_action_one_hot = handcraft_function.one_hot_encoding(random_action, self.action_dim)

            actions.append(random_action_one_hot)

        return actions

    def egreedy_action(self, state):  # 输出带随机的动作
        state_input = state[0][np.newaxis, :, :]
        agents_local_observation = state[1][np.newaxis, :, :]
        prob_value = self.session.run(self.actor_net.a, {self.actor_net.state_input: state_input, self.actor_net.agents_local_observation: agents_local_observation})[0]
        # print(prob_value)
        # self.epsilon -= (config.INITIAL_EPSILON - config.FINAL_EPSILON) / 10000
        if np.random.uniform() <= self.epsilon:
            random_action_and_parameter = self.get_random_action_and_parameter_one_hot()
            return random_action_and_parameter

        else:
            return prob_value

    def action(self, state):
        state_input = state[0][np.newaxis, :, :]
        agents_local_observation = state[1][np.newaxis, :, :]
        prob_value = self.session.run(self.actor_net.a_, {self.actor_net.state_input_next: state_input, self.actor_net.agents_local_observation_next: agents_local_observation})[0]
        return prob_value
