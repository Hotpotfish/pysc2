import tensorflow as tf
from pysc2.agents.myAgent.myAgent_15_BIC_DDPG_2.config import config
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

        self.q_value = self._build_graph_q(self.agents_local_observation, 'q_value', True)

        self.trian_op, self.loss = self.create_training_method(self.action_input, self.q_value, self.y_input)

    def _setup_placeholders_graph(self):
        # s

        # self.state = tf.placeholder("float", shape=[None, config.COOP_AGENTS_OBDIM], name='state')
        self.agents_local_observation = tf.placeholder("float", shape=[None, self.agents_number, config.COOP_AGENTS_OBDIM], name='agents_local_observation')

        self.minimap_input = tf.placeholder("float", shape=[None, config.MAP_SIZE, config.MAP_SIZE, 11], name='minimap_input')

        self.y_input = tf.placeholder("float", shape=[None], name='y_input')
        self.action_input = tf.placeholder("float", [None, self.agents_number, self.action_dim], name='action_input')

    def _build_graph_q(self, agents_local_observation, scope_name, train):
        # 环境和智能体本地的共同观察
        with tf.variable_scope(scope_name, reuse=tf.AUTO_REUSE):
            with slim.arg_scope([slim.conv2d, slim.fully_connected],
                                trainable=train,
                                activation_fn=tf.nn.selu,
                                # normalizer_fn=slim.batch_norm,
                                ):
                encoder_outputs = self._observation_encoder_q(agents_local_observation, self.agents_number, '_observation_encoder')
                bicnet_outputs = self._bicnet_build_q(encoder_outputs, self.agents_number, '_bicnet_build')
                return bicnet_outputs

    def _observation_encoder_q(self, agents_local_observation, agents_number, scope_name):
        with tf.variable_scope(scope_name):
            agents_local_observation = slim.fully_connected(agents_local_observation, 64, scope='full_connected1')
            agents_local_observation = tf.unstack(agents_local_observation, agents_number, 1)
            return agents_local_observation

    # def _observation_encoder_a(self, agents_local_observation, agents_number, scope_name):
    #     with tf.variable_scope(scope_name, reuse=tf.AUTO_REUSE):
    #         agents_local_observation = slim.fully_connected(agents_local_observation, 64, scope='full_connected1')
    #         # agents_local_observation = tf.transpose(agents_local_observation, [1, 0, 2])
    #         agents_local_observation = tf.unstack(agents_local_observation, agents_number, 1)  # (self.agents_number,batch_size,obs_add_dim)
    #         # encoder = []
    #         # for i in range(agents_number):
    #         #     fc1 = slim.fully_connected(agents_local_observation[:, i, :], 500, scope='full_connected1' + '_agent_' + str(i))
    #         #     # fc2 = slim.fully_connected(fc1, 50, scope='full_connected2')
    #         #     encoder.append(fc1)
    #         # encoder = tf.transpose(encoder, [1, 0, 2])
    #         # encoder = tf.unstack(encoder, agents_number, 1)  # (self.agents_number,batch_size,obs_add_dim)
    #         return agents_local_observation
    #
    # def _bicnet_build_a(self, encoder_outputs, agents_number, scope_name):
    #     with tf.variable_scope(scope_name, reuse=tf.AUTO_REUSE):
    #         # outputs = []
    #         lstm_fw_cell = tf.nn.rnn_cell.GRUCell(50, name="lstm_fw_cell")
    #         lstm_bw_cell = tf.nn.rnn_cell.GRUCell(50, name="lstm_bw_cell")
    #         bicnet_outputs, _, _ = tf.nn.static_bidirectional_rnn(lstm_fw_cell, lstm_bw_cell, encoder_outputs, dtype=tf.float32)
    #         bicnet_outputs = tf.reshape(bicnet_outputs, [-1, self.agents_number * 50 * 2, 1])
    #         bicnet_outputs = tf.layers.conv1d(bicnet_outputs, self.action_dim, kernel_size=50 * 2, strides=50 * 2)

    def _bicnet_build_q(self, encoder_outputs, agents_number, scope_name):
        with tf.variable_scope(scope_name):
            # outputs = []
            lstm_fw_cell = tf.nn.rnn_cell.GRUCell(50, name="lstm_fw_cell")
            lstm_bw_cell = tf.nn.rnn_cell.GRUCell(50, name="lstm_bw_cell")
            bicnet_outputs, _, _ = tf.nn.static_bidirectional_rnn(lstm_fw_cell, lstm_bw_cell, encoder_outputs, dtype=tf.float32)
            bicnet_outputs = tf.reshape(bicnet_outputs, [-1, self.agents_number * 50 * 2, 1])
            bicnet_outputs = tf.layers.conv1d(bicnet_outputs, self.action_dim, kernel_size=50 * 2, strides=50 * 2)
            bicnet_outputs = tf.nn.softmax(bicnet_outputs)

            # for i in range(agents_number):
            #     fc1 = slim.fully_connected(bicnet_outputs[i], self.action_dim, activation_fn=tf.nn.softmax, scope='full_connected1')
            #     outputs.append(fc1)

            return bicnet_outputs

    def create_training_method(self, action_input, q_value, y_input):
        Q_action = tf.reduce_sum(tf.reduce_sum(tf.multiply(q_value, action_input), reduction_indices=2), reduction_indices=1)
        cost = tf.reduce_mean(tf.square(y_input - Q_action))
        train_op = tf.train.AdamOptimizer(self.learning_rate).minimize(cost)
        return train_op, cost
