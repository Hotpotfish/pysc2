import tensorflow as tf
from pysc2.agents.myAgent.myAgent_15_BIC_DDPG_2.config import config
import tensorflow.contrib.slim as slim
import numpy as np


class net2(object):

    def __init__(self, mu, sigma, learning_rate, action_dim, statedim, agents_number, enemy_number, name):  # 初始化
        # 神经网络参数
        self.mu = mu
        self.sigma = sigma
        self.learning_rate = learning_rate

        # 动作维度数，动作参数维度数,状态维度数
        self.action_dim = action_dim
        self.state_dim = statedim
        self.agents_number = agents_number
        self.enemy_number = enemy_number

        self.name = name

        self._setup_placeholders_graph()

        self.actions = []
        self.actions_ = []
        with tf.variable_scope('Actor'):
            for i in range(self.agents_number):
                self.actions.append(self._build_graph_a(self.agents_local_observation[:, i], 'eval_' + str(i), train=True))
                self.actions_.append(self._build_graph_a(self.agents_local_observation_next[:, i], 'target_' + str(i), train=False))
        self.actions = tf.reshape(self.actions, (-1, self.agents_number, self.action_dim))
        self.actions_ = tf.reshape(self.actions_, (-1, self.agents_number, self.action_dim))

        self.qs = []
        self.qs_ = []
        with tf.variable_scope('Critic'):
            for i in range(self.agents_number):
                self.qs.append(self._build_graph_c(self.agents_local_observation[:, i], self.actions, 'eval_' + str(i), train=True))
                self.qs_.append(self._build_graph_c(self.agents_local_observation_next[:, i], self.actions_, 'target_' + str(i), train=False))

        self.qs = tf.reshape(self.qs, (-1, self.agents_number))
        self.qs_ = tf.reshape(self.qs_, (-1, self.agents_number))

        self.ae_params = []
        self.at_params = []
        self.ce_params = []
        self.ct_params = []

        for i in range(self.agents_number):
            self.ae_params.append(tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='Actor/eval_' + str(i)))
            self.at_params.append(tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='Actor/target_' + str(i)))
            self.ce_params.append(tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='Critic/eval_' + str(i)))
            self.ct_params.append(tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='Critic/target_' + str(i)))

        self.soft_replace = self._soft_replace()

        self.qs_targets = self.reward + config.GAMMA * self.qs_

        self.atrains = self._build_atrains()
        self.ctrains = self._build_ctrains()

    def _setup_placeholders_graph(self):
        # s

        self.agents_local_observation = tf.placeholder("float", shape=[None, self.agents_number, config.COOP_AGENTS_OBDIM], name='agents_local_observation')

        # s_

        self.agents_local_observation_next = tf.placeholder("float", shape=[None, self.agents_number, config.COOP_AGENTS_OBDIM], name='agents_local_observation_next')

        self.reward = tf.placeholder("float", shape=[None, 1], name='reward')

    def _build_graph_a(self, agents_local_observation, scope_name, train):
        # 环境和智能体本地的共同观察
        with tf.variable_scope(scope_name, reuse=tf.AUTO_REUSE):
            with slim.arg_scope([slim.fully_connected],
                                trainable=train,
                                activation_fn=tf.nn.selu,
                                # weights_initializer=tf.truncated_normal_initializer(stddev=0.1),
                                # weights_regularizer=slim.l2_regularizer(0.05)
                                ):
                fc1 = slim.fully_connected(agents_local_observation, 30, scope='full_connected1')
                fc1 = fc1 * 1/4096
                fc2 = slim.fully_connected(fc1, self.action_dim, activation_fn=tf.sigmoid, scope='full_connected2')
                return fc2

        #################################### critic_net  ####################################

    def _build_graph_c(self, agents_local_observation, union_action, scope_name, train):
        # 环境和智能体本地的共同观察
        with tf.variable_scope(scope_name, reuse=tf.AUTO_REUSE):
            with slim.arg_scope([slim.fully_connected],
                                trainable=train,
                                activation_fn=tf.nn.selu,
                                ):
                union_action = slim.flatten(union_action)
                fc_a_1 = slim.fully_connected(agents_local_observation, 30, scope='full_connected_a_1')
                fc_s_1 = slim.fully_connected(union_action, 30, scope='full_connected_s_1')
                h1 = fc_a_1 + fc_s_1
                q = slim.fully_connected(h1, 1, activation_fn=tf.sigmoid, scope='q')
                return q

    def _build_atrains(self):
        atrains = []
        for i in range(self.agents_number):
            loss = -tf.reduce_mean(self.qs[:, i])
            atrain = tf.train.AdamOptimizer(self.learning_rate).minimize(loss, var_list=self.ae_params[i])
            atrains.append(atrain)
        return atrains

    def _build_ctrains(self):
        ctrains = []
        for i in range(self.agents_number):
            td_error = tf.losses.mean_squared_error(labels=self.qs_targets, predictions=self.qs)
            ctrain = tf.train.AdamOptimizer(self.learning_rate).minimize(td_error, var_list=self.ce_params[i])
            ctrains.append(ctrain)
        return ctrains

    def _soft_replace(self):
        t = None
        for i in range(self.agents_number):
            # temp = zip(self.at_params[i] + self.ct_params[i], self.ae_params[i] + self.ce_params[i])
            t = [tf.assign(t, (1 - config.TAU) * t + config.TAU * e) for t, e in zip(self.at_params[i] + self.ct_params[i], self.ae_params[i] + self.ce_params[i])]
        return t
