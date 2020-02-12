import tensorflow as tf
from pysc2.agents.myAgent.myAgent_13.config import config
import tensorflow.contrib.slim as slim
import numpy as np


class net1(object):

    def __init__(self, mu, sigma, learning_rate, action_dim, statedim, agents_number, enemy_number, name):  # 初始化
        # 神经网络参数
        self.mu = mu
        self.sigma = sigma
        self.learning_rate = learning_rate

        # 动作维度数，动作参数维度数,状态维度数
        self.statedim = statedim
        self.action_dim = action_dim
        self.state_dim = statedim
        self.agents_number = agents_number
        self.enemy_number = enemy_number

        self.name = name

        self._setup_placeholders_graph()

        self.q_value = self._build_graph_q(self.state, 'q_value', True)

        self.trian_op, self.loss = self.create_training_method(self.action_input, self.q_value, self.y_input)

    def _setup_placeholders_graph(self):
        # s


        self.state = tf.placeholder("float", shape=[None, config.COOP_AGENTS_OBDIM], name='state')

        self.y_input = tf.placeholder("float", shape=[None], name='y_input')
        self.action_input = tf.placeholder("float", [None, np.power(self.action_dim, self.agents_number)], name='action_input')
        # self.action_bound = tf.placeholder("float", [None, np.power(self.action_dim, self.agents_number)], name='action_bound')

    def _build_graph_q(self, state, scope_name, train):
        # 环境和智能体本地的共同观察
        with tf.variable_scope(scope_name):
            with slim.arg_scope([slim.fully_connected],
                                trainable=train,
                                activation_fn=tf.nn.selu,
                                normalizer_fn=slim.batch_norm):
                fc1 = slim.fully_connected(state, 1000, scope='full_connected_1')
                fc2 = slim.fully_connected(fc1, 400, scope='full_connected_2')
                fc3 = slim.fully_connected(fc2, int(np.power(self.action_dim, self.agents_number)), activation_fn=tf.nn.softmax, scope='full_connected_3')
                return fc3

    def create_training_method(self, action_input, q_value, y_input):
        Q_action = tf.reduce_sum(tf.multiply(q_value, action_input), reduction_indices=1)
        cost = tf.reduce_mean(tf.square(y_input - Q_action))
        train_op = tf.train.AdamOptimizer(self.learning_rate).minimize(cost)
        return train_op, cost
