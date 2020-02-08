import tensorflow as tf
from pysc2.agents.myAgent.myAgent_11_PG.config import config
import tensorflow.contrib.slim as slim
import numpy as np
from keras_radam.training import RAdamOptimizer


class bicnet(object):

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

        self.action_output = self._build_graph_q(self.state, 'action_output', True)

        self.trian_op, self.loss = self.create_training_method(self.action_output, self.action_input, self.actions_value)

    def _setup_placeholders_graph(self):
        # s

        # self.agents_local_observation = tf.placeholder("float", shape=[None, self.agents_number, config.COOP_AGENTS_OBDIM], name='agents_local_observation')
        self.state = tf.placeholder("float", shape=[None, config.COOP_AGENTS_OBDIM], name='state')

        self.actions_value = tf.placeholder("float", shape=[None], name='actions_value')

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
                # output = tf.multiply(fc2, self.action_bound)
                # output = tf.divide(output, tf.expand_dims(tf.reduce_sum(output, axis=1), -1))
                return fc3

    def create_training_method(self, action_output, action_input, actions_value):
        neg_log_prob = tf.reduce_sum(-tf.log(action_output + 1e-7) * action_input, axis=1)
        loss = tf.reduce_mean(neg_log_prob * actions_value)  # reward guided loss
        train_op = tf.train.AdamOptimizer(self.learning_rate).minimize(loss)

        return train_op, loss
