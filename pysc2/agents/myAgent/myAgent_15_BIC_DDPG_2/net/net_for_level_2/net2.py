import tensorflow as tf
from pysc2.agents.myAgent.myAgent_15_BIC_DDPG_2.config import config
import tensorflow.contrib.slim as slim
import numpy as np


class maddpg():
    def __init__(self, learning_rate, action_dim, state_dim, agents_number, name):  # 初始化
        # 神经网络参数
        self.learning_rate = learning_rate

        # 动作维度数，动作参数维度数,状态维度数
        self.action_dim = action_dim
        self.state_dim = state_dim
        self.agents_number = agents_number

        self.name = name
        self.ae_params = None
        self.at_params = None
        self.ce_params = None
        self.ct_params = None

        self.a_op = None
        self.c_op = None

    def _get_action(self, agents_local_observation):
        with tf.variable_scope(self.name + '_' + 'Actor', reuse=tf.AUTO_REUSE):
            a = self._build_graph_a(agents_local_observation, 'eval', train=True)
            if self.ae_params is None:
                self.ae_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=self.name + '_' + 'Actor/eval')
        return a

    def _get_action_next(self, agents_local_observation_next):
        with tf.variable_scope(self.name + '_' + 'Actor', reuse=tf.AUTO_REUSE):
            a_ = self._build_graph_a(agents_local_observation_next, 'target', train=False)
            if self.at_params is None:
                self.at_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=self.name + '_' + 'Actor/target')
            return a_

    def _get_q(self, state_input, union_action_input):
        with tf.variable_scope(self.name + '_' + 'Critic', reuse=tf.AUTO_REUSE):
            q = self._build_graph_c(state_input, union_action_input, 'eval', train=True)
            if self.ce_params is None:
                self.ce_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=self.name + '_' + 'Critic/eval')
            return q

    def _get_q_next(self, state_input_next, union_action_input_next):
        with tf.variable_scope(self.name + '_' + 'Critic', reuse=tf.AUTO_REUSE):
            q_ = self._build_graph_c(state_input_next, union_action_input_next, 'target', train=False)
            if self.ct_params is None:
                self.ct_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=self.name + '_' + 'Critic/target')
            return q_

    def _soft_replace(self):
        # self.ae_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=self.name + '_' + 'Actor/eval')
        # self.at_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=self.name + '_' + 'Actor/target')
        # self.ce_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=self.name + '_' + 'Critic/eval')
        # self.ct_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=self.name + '_' + 'Critic/target')
        self.soft_replace = [tf.assign(t, (1 - config.TAU) * t + config.TAU * e)
                             for t, e in zip(self.at_params + self.ct_params, self.ae_params + self.ce_params)]
        return self.soft_replace

    def train_a(self, agents_local_observation, state_input, union_action):
        # a = self._get_action(agents_local_observation)
        # a = tf.reshape(a, [-1, 1, self.action_dim])
        # union_action = tf.concat([a, other_action], 1)
        q = self._get_q(state_input, union_action)
        a_loss = - tf.reduce_mean(q)  # maximize the q
        if self.a_op is None:
            self.a_op = tf.train.AdamOptimizer(self.learning_rate).minimize(a_loss, var_list=self.ae_params)
        return self.a_op

    def train_q(self, reward, state_input, union_action, agents_local_observation_next, state_input_next, other_action_next):
        q = self._get_q(state_input, union_action)

        a_ = self._get_action_next(agents_local_observation_next)
        a_ = tf.reshape(a_, [-1, 1, self.action_dim])
        union_action = tf.concat([a_, other_action_next], 1)
        q_ = self._get_q_next(state_input_next, union_action)
        q_target = reward + config.GAMMA * q_

        td_error = tf.losses.mean_squared_error(labels=q_target, predictions=q)
        if self.c_op is None:
            self.c_op = tf.train.AdamOptimizer(self.learning_rate).minimize(td_error, var_list=self.ce_params)
        return self.c_op

    def _build_graph_a(self, agents_local_observation, scope_name, train):
        # 环境和智能体本地的共同观察
        with tf.variable_scope(scope_name, reuse=tf.AUTO_REUSE):
            with slim.arg_scope([slim.fully_connected],
                                trainable=train,
                                activation_fn=None,
                                weights_initializer=tf.truncated_normal_initializer(stddev=0.1),
                                weights_regularizer=slim.l2_regularizer(0.05)
                                ):
                fc1 = slim.fully_connected(agents_local_observation, 20, scope='full_connected1')
                a = slim.fully_connected(fc1, self.action_dim, activation_fn=tf.nn.sigmoid, scope='a')
                return a

    def _build_graph_c(self, state_input, action_input, scope_name, train):
        # 环境和智能体本地的共同观察
        with tf.variable_scope(scope_name, reuse=tf.AUTO_REUSE):
            with slim.arg_scope([slim.fully_connected],
                                trainable=train,
                                activation_fn=None,
                                weights_initializer=tf.truncated_normal_initializer(stddev=0.1),
                                weights_regularizer=slim.l2_regularizer(0.05)):
                fc1_s = slim.fully_connected(state_input, 20, scope='full_connected_s1')
                fc1_a = slim.fully_connected(action_input, 20, scope='full_connected_a1')

                data = fc1_s + fc1_a

                q = slim.fully_connected(data, 1, scope='q')

                return q


class net2(object):

    def __init__(self, mu, sigma, learning_rate, action_dim, state_dim, agents_number, enemy_number, name):  # 初始化
        # 神经网络参数
        self.mu = mu
        self.sigma = sigma
        self.learning_rate = learning_rate

        self.i = 0

        # 动作维度数，动作参数维度数,状态维度数
        self.action_dim = action_dim
        self.state_dim = state_dim
        self.agents_number = agents_number
        self.enemy_number = enemy_number

        self.name = name

        self._setup_placeholders_graph()

        self.agents = []

        for i in range(self.agents_number):
            self.agents.append(maddpg(learning_rate, action_dim, state_dim, agents_number, name + '_agent_' + str(i)))

        self.union_a = self._get_union_a()
        self.union_a_next = self._get_union_a_next()

        self.union_q = self._get_union_q()

    def _get_union_a(self):
        union_a = []
        for i in range(self.agents_number):
            union_a.append(self.agents[i]._get_action(self.agents_local_observation[:, i]))
        union_a = tf.reshape(union_a, [-1, self.agents_number, self.action_dim])
        return union_a

    def _get_union_q(self):
        union_q = 0
        for i in range(self.agents_number):
            union_q += (self.agents[i]._get_q(self.state_input, self.union_a))
            union_q = tf.reshape(union_q, [-1, 1])
        return union_q

    def _get_union_a_next(self):
        union_a_next = []
        for i in range(self.agents_number):
            union_a_next.append(self.agents[i]._get_action(self.agents_local_observation_next[:, i]))
        union_a_next = tf.reshape(union_a_next, [-1, self.agents_number, self.action_dim])
        return union_a_next

    def agent_i_atrain(self):
        i = 0
        # if i == 0:
        #     other_action = self.union_a[:, 1:]
        #
        # elif i == self.agents_number - 1:
        #     other_action = self.union_a[:, 0:self.agents_number]
        #
        # else:
        #     other_action = tf.concat([self.union_a[:, 0:i], self.union_a[:, i:]], 1)

        self.agents[0].train_a(self.agents_local_observation[:, i], self.state_input, self.union_a)
        return tf.constant(1)

    def agent_i_ctrain(self, i):
        if i == 0:
            other_action_next = self.union_a_next[:, 1:]
        elif i == self.agents_number - 1:
            other_action_next = self.union_a_next[:, 0:self.agents_number - 1]
        else:
            other_action_next = tf.concat([self.union_a_next[:, 0:i], self.union_a_next[:, i:]], 1)

        self.agents[i].train_q(self.reward, self.state_input, self.union_a, self.agents_local_observation_next[:, i], self.state_input_next, other_action_next)
        return other_action_next

    def agent_i_soft_replace(self, i):
        self.agents[i]._soft_replace()
        return tf.constant(1)

    def _setup_placeholders_graph(self):
        # s
        self.state_input = tf.placeholder("float", shape=self.state_dim, name='state_input')  # 全局状态
        self.agents_local_observation = tf.placeholder("float", shape=[None, self.agents_number, config.COOP_AGENTS_OBDIM], name='agents_local_observation')

        # s_
        self.state_input_next = tf.placeholder("float", shape=self.state_dim, name='state_input_next')  # 全局状态
        self.agents_local_observation_next = tf.placeholder("float", shape=[None, self.agents_number, config.COOP_AGENTS_OBDIM], name='agents_local_observation_next')

        self.reward = tf.placeholder("float", shape=[None], name='reward')
        # self.i = tf.placeholder("int32", shape=[None, 1], name='i')
