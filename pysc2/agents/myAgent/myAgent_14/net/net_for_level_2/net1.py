import tensorflow as tf
from pysc2.agents.myAgent.myAgent_14.config import config
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

        actions_prob = []
        agent_params = []

        for i in range(self.agents_number):
            actions_prob.append(self._build_graph_a(self.agents_local_observation[i], self.bounds[i], 'actor_' + str(i), True))
            agent_params.append(tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='actor_' + str(i)))

        actions_prob = tf.reshape(actions_prob, [-1, self.agents_number, self.action_dim])

        all_q = self._build_graph_c(self.state, self.agents_local_observation, 'all_q', True)
        all_q_next = self._build_graph_c(self.state_next, self.agents_local_observation_next, 'all_q', True)

        q_s_u = tf.reduce_sum(tf.multiply(all_q, self.action_input), reduction_indices=2)
        q_s_u_next = tf.reduce_sum(tf.multiply(all_q_next, self.action_input_next), reduction_indices=2)

        actor_train_ops = self.actor_learn(q_s_u, all_q, self.action_input, actions_prob, agent_params)

    def _setup_placeholders_graph(self):
        # s

        self.state = tf.placeholder("float", shape=[None, config.COOP_AGENTS_OBDIM], name='state')
        self.agents_local_observation = tf.placeholder("float", shape=[None, self.agents_number, config.COOP_AGENTS_OBDIM], name='agents_local_observation')
        self.bounds = tf.placeholder("float", shape=[None, self.agents_number, self.action_dim], name='bounds')

        self.state_next = tf.placeholder("float", shape=[None, config.COOP_AGENTS_OBDIM], name='state_next')
        self.agents_local_observation_next = tf.placeholder("float", shape=[None, self.agents_number, config.COOP_AGENTS_OBDIM], name='agents_local_observation_next')
        self.bounds_next = tf.placeholder("float", shape=[None, self.agents_number, self.action_dim], name='bounds_next')

        self.reward = tf.placeholder("float", [None, 1], name='reward')

        # self.y_input = tf.placeholder("float", shape=[None], name='y_input')
        self.action_input = tf.placeholder("float", [None, self.agents_number, self.action_dim], name='action_input')
        self.action_input_next = tf.placeholder("float", [None, self.agents_number, self.action_dim], name='action_input_next')

    def _build_graph_a(self, agent_local_observation, bound, scope_name, train):
        with tf.variable_scope(scope_name):
            with slim.arg_scope([slim.fully_connected],
                                trainable=train,
                                activation_fn=tf.nn.selu,
                                normalizer_fn=slim.batch_norm,
                                weights_initializer=tf.truncated_normal_initializer(stddev=0.1),
                                weights_regularizer=slim.l2_regularizer(0.05)):
                fc1 = slim.fully_connected(agent_local_observation, 30, scope='full_connected_1')
                action_prob = slim.fully_connected(fc1, self.action_dim, activation_fn=tf.nn.softmax, scope='full_connected_1')
                action_prob = tf.multiply(action_prob, bound)
                action_prob = action_prob / tf.reduce_sum(action_prob)

                # action = tf.argmax(tf.multiply(action_prob, bound))
                # real_prob = tf.reduce_max(tf.multiply(action_prob, bound))
                return action_prob

    def _build_graph_c(self, state_input, agents_local_observation, scope_name, train):
        # 环境和智能体本地的共同观察
        with tf.variable_scope(scope_name, reuse=tf.AUTO_REUSE):
            with slim.arg_scope([slim.fully_connected],
                                trainable=train,
                                activation_fn=tf.nn.selu,
                                normalizer_fn=slim.batch_norm,
                                weights_initializer=tf.truncated_normal_initializer(stddev=0.1),
                                weights_regularizer=slim.l2_regularizer(0.05)):
                encoder_outputs = self._observation_encoder_c(state_input, agents_local_observation, self.agents_number, '_observation_encoder')
                bicnet_outputs = self._bicnet_build_c(encoder_outputs, self.agents_number, '_bicnet_build')
                return bicnet_outputs

    def _observation_encoder_c(self, state_input, agents_local_observation, agents_number, scope_name):
        with tf.variable_scope(scope_name):
            encoder = []
            fc1_s = slim.fully_connected(state_input, 30, scope='full_connected_s1')
            for i in range(agents_number):
                fc1_o = slim.fully_connected(agents_local_observation[:, i], 30, scope='full_connected_o1')
                data = tf.concat([fc1_s, fc1_o], 1)
                fc1 = slim.fully_connected(data, 30, scope='full_connected1')
                encoder.append(fc1)
            encoder = tf.transpose(encoder, [1, 0, 2])
            encoder = tf.unstack(encoder, agents_number, 1)  # (self.agents_number,batch_size,obs_add_dim)
            return encoder

    def _bicnet_build_c(self, encoder_outputs, agents_number, scope_name):
        with tf.variable_scope(scope_name):
            outputs = []
            lstm_fw_cell = tf.nn.rnn_cell.GRUCell(self.action_dim, name="lstm_fw_cell")
            lstm_bw_cell = tf.nn.rnn_cell.GRUCell(self.action_dim, name="lstm_bw_cell")
            bicnet_outputs, _, _ = tf.nn.static_bidirectional_rnn(lstm_fw_cell, lstm_bw_cell, encoder_outputs, dtype=tf.float32)
            for i in range(agents_number):
                fc1 = slim.fully_connected(bicnet_outputs[i], self.action_dim, activation_fn=tf.nn.softmax, scope='full_connected1')
                outputs.append(fc1)
            outputs = tf.unstack(outputs, self.agents_number)
            outputs = tf.transpose(outputs, [1, 0, 2])
            return outputs  # (1,agent_number,action_dim)

    def actor_learn(self, q_s_u, all_q, action_input, actions_prob, agent_params):
        train_ops = []
        expectation = tf.reduce_sum(tf.multiply(all_q, actions_prob), reduction_indices=2)
        log_actions_prob = tf.log(tf.reduce_sum(tf.multiply(action_input, actions_prob), reduction_indices=2))
        for i in range(self.agents_number):
            A = tf.square(q_s_u[0][i] - expectation[0][i])
            exp_v = tf.reduce_mean(log_actions_prob[0][i] * A)
            train_op = tf.train.AdamOptimizer(self.learning_rate).minimize(-exp_v, var_list=agent_params[i])
            train_ops.append(train_op)
        return train_ops

    def critic_learn(self, q_s_u, q_s_u_next):
        td_error = self.reward + config.GAMMA * q_s_u_next - q_s_u
        loss = tf.square(td_error)
        train_op = tf.train.AdamOptimizer(self.learning_rate).minimize(loss)

        return train_op


