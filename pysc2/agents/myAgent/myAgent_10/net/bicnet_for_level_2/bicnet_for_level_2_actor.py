import tensorflow as tf
import tensorflow.contrib.slim as slim

from pysc2.agents.myAgent.myAgent_10.config import config
import numpy as np


class bicnet_actor():

    def __init__(self, mu, sigma, learning_rate, action_dim, statedim, agents_number, enemy_number, name):

        self.mu = mu
        self.sigma = sigma
        self.learning_rate = learning_rate

        self.action_dim = action_dim
        # self.parameterdim = parameterdim
        self.statedim = statedim

        self.agents_number = agents_number
        self.enemy_number = enemy_number

        self.name = name

        # 建立输入管道

        # 两个A网络
        with tf.variable_scope(name, reuse=tf.AUTO_REUSE):
            self._setup_placeholders_graph()
            # a
            self.a = self._build_graph(self.state_input, self.agents_local_observation, 'eval_net', True)
            # a_
            self.a_ = self._build_graph(self.state_input_next, self.agents_local_observation_next, 'target_net', False)

        self.e_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=name + '/eval_net')
        self.t_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=name + '/target_net')

        self.soft_replace = [tf.assign(t, (1 - config.GAMMA_FOR_UPDATE) * t + config.GAMMA_FOR_UPDATE * e) for t, e in zip(self.t_params, self.e_params)]

        self.train_op = self._optimizer(self.a, self.action_gradient)

    def _setup_placeholders_graph(self):
        # s
        self.state_input = tf.placeholder("float", shape=self.statedim, name='state_input')  # 全局状态
        self.agents_local_observation = tf.placeholder("float", shape=[None, self.agents_number, config.COOP_AGENTS_OBDIM], name='agents_local_observation')

        self.action_gradient = tf.placeholder(tf.float32, [self.agents_number, None, self.agents_number, self.action_dim], name="action_gradient")

        self.state_input_next = tf.placeholder("float", shape=self.statedim, name='state_input_next')  # 全局状态
        self.agents_local_observation_next = tf.placeholder("float", shape=[None, self.agents_number, config.COOP_AGENTS_OBDIM], name='agents_local_observation_next')

    def _build_graph(self, state_input, agents_local_observation, scope_name, train):
        # 环境和智能体本地的共同观察
        with tf.variable_scope(scope_name, reuse=tf.AUTO_REUSE):
            encoder_outputs = self._observation_encoder(state_input, agents_local_observation, config.COOP_AGENTS_NUMBER, '_observation_encoder', train)
            bicnet_outputs = self._bicnet_build(encoder_outputs, '_bicnet_build', train)
            return bicnet_outputs

    def _observation_encoder(self, state_input, agents_local_observation, agents_number, scope_name, train):
        with tf.variable_scope(scope_name, reuse=tf.AUTO_REUSE):
            encoder = []
            conv1 = slim.conv2d(state_input, 1, [5, 5], stride=4, padding="VALID", scope='layer_1_conv')
            pool1 = slim.max_pool2d(conv1, [2, 2], stride=2, padding="VALID", scope='layer_1_pooling')
            # bn1 = tf.layers.batch_normalization(pool1, training=train)

            # 传给下一阶段
            state_input_flatten = slim.flatten(pool1, scope="flatten")

            for i in range(agents_number):
                encoder.append(tf.concat([agents_local_observation[:, i, :], state_input_flatten], axis=1))
            encoder = tf.transpose(encoder, [1, 0, 2])
            fc1 = slim.fully_connected(encoder, 30, scope='full_connected1')
            # bn3 = tf.layers.batch_normalization(fc1, training=train)
            encoder = tf.unstack(fc1, agents_number, 1)  # (self.agents_number,batch_size,obs_add_dim)
            return encoder

    def _bicnet_build(self, encoder_outputs, scope_name, train):
        with tf.variable_scope(scope_name, reuse=tf.AUTO_REUSE):
            lstm_fw_cell = tf.nn.rnn_cell.BasicLSTMCell(self.action_dim, forget_bias=1.0, name="lstm_fw_cell")
            lstm_bw_cell = tf.nn.rnn_cell.BasicLSTMCell(self.action_dim, forget_bias=1.0, name="lstm_bw_cell")
            bicnet_outputs, _, _ = tf.nn.static_bidirectional_rnn(lstm_fw_cell, lstm_bw_cell, encoder_outputs, dtype=tf.float32)
            fc1 = slim.fully_connected(bicnet_outputs, self.action_dim, scope='full_connected1')
            bicnet_outputs = tf.nn.softmax(fc1, axis=2)

            bicnet_outputs = tf.unstack(bicnet_outputs, self.agents_number)  # (agents_number, batch_size, action_dim)
            bicnet_outputs = tf.transpose(bicnet_outputs, [1, 0, 2])
            return bicnet_outputs  # (batch_size,agents_number,action_dim)

    def _soft_replace(self):
        self.soft_replace = [tf.assign(t, (1 - config.GAMMA_FOR_UPDATE) * t + config.GAMMA_FOR_UPDATE * e) for t, e in zip(self.t_params, self.e_params)]

    def _optimizer(self, aout, action_gradient):
        grads = []
        batch_size = tf.to_float(tf.shape(aout)[0])
        for i in range(self.agents_number):
            for j in range(self.agents_number):
                grads.append(tf.gradients(aout[:, j], self.e_params, action_gradient[j][:, i]))
            # grads.append(tf.gradients(aout, self.e_params, -action_gradient[:, i]))
        grads = np.array(grads)
        unnormalized_actor_gradients = [tf.reduce_sum(list(grads[:, i]), axis=0) for i in range(len(self.e_params))]
        actor_gradients = list(map(lambda x: tf.div(x, batch_size), unnormalized_actor_gradients))

        optimizer = tf.train.AdamOptimizer(self.learning_rate)
        train_op = optimizer.apply_gradients(list(zip(actor_gradients, self.e_params)))

        return train_op
