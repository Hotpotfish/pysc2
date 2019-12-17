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
        self._setup_placeholders_graph()

        # 两个A网络

        # a
        self.a = self._build_graph(self.state_input, self.agents_local_observation, 'eval_net', True)
        # a_
        self.a_ = self._build_graph(self.state_input_next, self.agents_local_observation_next, 'target_net', False)

        self.e_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='eval_net')
        self.t_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='target_net')

        self.soft_replace = [tf.assign(t, (1 - config.GAMMA_FOR_UPDATE) * t + config.GAMMA_FOR_UPDATE * e) for t, e in zip(self.t_params, self.e_params)]

        self.train_op = self._optimizer(self.a, self.action_gradient)

    def _setup_placeholders_graph(self):
        # s
        self.state_input = tf.placeholder("float", shape=self.statedim, name=self.name + '_' + 'state_input')  # 全局状态
        self.agents_local_observation = tf.placeholder("float", shape=[None, self.agents_number, config.COOP_AGENTS_OBDIM], name='agents_local_observation')

        self.action_gradient = tf.placeholder(tf.float32, [self.agents_number, None, self.agents_number, self.action_dim], name="action_gradient")

        # # a
        # self.action_input = tf.placeholder("float", shape=[None, self.agents_number, self.action_dim + self.parameterdim], name='action_input')
        #
        # # r
        # self.reward_input = tf.placeholder("float", shape=[None, self.agents_number, 1], name='action_input')
        #
        # # s_
        self.state_input_next = tf.placeholder("float", shape=self.statedim, name=self.name + '_' + 'state_input_next')  # 全局状态
        self.agents_local_observation_next = tf.placeholder("float", shape=[None, self.agents_number, config.COOP_AGENTS_OBDIM], name='agents_local_observation_next')

    def _build_graph(self, state_input, agents_local_observation, scope_name, train):
        # 环境和智能体本地的共同观察
        with tf.variable_scope(scope_name, reuse=tf.AUTO_REUSE):
            encoder_outputs = self._observation_encoder(state_input, agents_local_observation, config.COOP_AGENTS_NUMBER, '_observation_encoder', train)
            bicnet_outputs = self._bicnet_build(encoder_outputs, '_bicnet_build', train)
            # action_outputs = self._action_network_graph(bicnet_outputs, '_action_network_graph', train)
            # queued_outputs = self._queued_network_graph(encoder_outputs, action_outputs, '_queued_network_graph', train)
            # enemy_unit_outputs = self._enemy_unit_network_graph(encoder_outputs, action_outputs, queued_outputs, '_enemy_unit_network_graph', train)
            # target_point_x_outputs = self._target_point_network_x_graph(encoder_outputs, action_outputs, queued_outputs, enemy_unit_outputs, '_target_point_x_network_graph', train)
            # target_point_y_outputs = self._target_point_network_y_graph(encoder_outputs, action_outputs, queued_outputs, enemy_unit_outputs, target_point_x_outputs, '_target_point_y_network_graph',
            #                                                             train)
            # outputs = self._get_outputs(action_outputs, queued_outputs, enemy_unit_outputs, target_point_x_outputs, target_point_y_outputs)
            return bicnet_outputs

    def _observation_encoder(self, state_input, agents_local_observation, agents_number, scope_name, train):
        with tf.variable_scope(scope_name, reuse=tf.AUTO_REUSE):
            encoder = []
            conv1 = slim.conv2d(state_input, 1, [5, 5], stride=1, padding="VALID", scope='layer_1_conv')
            pool1 = slim.max_pool2d(conv1, [2, 2], stride=2, padding="VALID", scope='layer_1_pooling')
            bn1 = tf.layers.batch_normalization(pool1, training=train)

            # 传给下一阶段
            state_input_flatten = slim.flatten(bn1, scope="flatten")

            for i in range(agents_number):
                encoder.append(tf.concat([agents_local_observation[:, i, :], state_input_flatten], axis=1))
            encoder = tf.transpose(encoder, [1, 0, 2])
            fc1 = slim.fully_connected(encoder, 60, scope='full_connected1')
            bn3 = tf.layers.batch_normalization(fc1, training=train)
            encoder = tf.unstack(bn3, agents_number, 1)  # (self.agents_number,batch_size,obs_add_dim)
            return encoder

    def _bicnet_build(self, encoder_outputs, scope_name, train):
        with tf.variable_scope(scope_name, reuse=tf.AUTO_REUSE):
            lstm_fw_cell = tf.nn.rnn_cell.BasicLSTMCell(self.action_dim/2, forget_bias=1.0, name="lstm_fw_cell")
            lstm_bw_cell = tf.nn.rnn_cell.BasicLSTMCell(self.action_dim/2, forget_bias=1.0, name="lstm_bw_cell")
            bicnet_outputs, _, _ = tf.nn.static_bidirectional_rnn(lstm_fw_cell, lstm_bw_cell, encoder_outputs, dtype=tf.float32)
            fc1 = slim.fully_connected(bicnet_outputs, self.action_dim, scope='full_connected1')
            bn1 = tf.layers.batch_normalization(fc1, training=train)
            bicnet_outputs = tf.nn.softmax(bn1)

            bicnet_outputs = tf.unstack(bicnet_outputs, self.agents_number)  # (agents_number, batch_size, action_dim)
            bicnet_outputs = tf.transpose(bicnet_outputs, [1, 0, 2])
            return bicnet_outputs  # (batch_size,agents_number,action_dim)

    # def _action_network_graph(self, bicnet_outputs, scope_name, train):
    #     action_outputs = []
    #     with tf.variable_scope(scope_name, reuse=tf.AUTO_REUSE):
    #         with slim.arg_scope([slim.conv2d, slim.fully_connected],
    #                             activation_fn=None,
    #                             weights_initializer=tf.truncated_normal_initializer(self.mu, self.sigma),  # mu，sigma
    #                             weights_regularizer=slim.l2_regularizer(0.1)):
    #             for i in range(self.agents_number):
    #                 encoder_output = bicnet_outputs[i]
    #                 with tf.variable_scope('_action_network_graph_fc', reuse=tf.AUTO_REUSE):
    #                     # fc1 = slim.fully_connected(encoder_output, 100, scope='full_connected1')
    #                     #
    #                     # fc2 = slim.fully_connected(fc1, 80, scope='full_connected2')
    #
    #                     action_logits = slim.fully_connected(encoder_output, self.action_dim, scope='action_logits')
    #                     action_logits_bn = tf.contrib.layers.batch_norm(action_logits, is_training=train)
    #
    #                     action_output = tf.nn.softmax(action_logits_bn)  # (batch_size,obs_dim)
    #                     action_outputs.append(action_output)  # (agents_number,batch_size,action_dim)
    #
    #             return action_outputs

    # def _queued_network_graph(self, encoder_outputs, action_outputs, scope_name, train):
    #     queued_outputs = []
    #     with tf.variable_scope(scope_name, reuse=tf.AUTO_REUSE):
    #         with slim.arg_scope([slim.conv2d, slim.fully_connected],
    #                             activation_fn=None,
    #                             weights_initializer=tf.truncated_normal_initializer(self.mu, self.sigma),  # mu，sigma
    #                             weights_regularizer=slim.l2_regularizer(0.1)):
    #             for i in range(self.agents_number):
    #
    #                 encoder_output = tf.concat([encoder_outputs[i], action_outputs[i]], axis=1)
    #                 with tf.variable_scope('_queued_network_graph_fc', reuse=tf.AUTO_REUSE):
    #                     # fc1 = slim.fully_connected(encoder_output, 100, scope='full_connected1')
    #                     #
    #                     # fc2 = slim.fully_connected(fc1, 80, scope='full_connected2')
    #
    #                     queued_logits = slim.fully_connected(encoder_output, config.QUEUED, scope='queued_logits')
    #                     queued_logits_bn = tf.contrib.layers.batch_norm(queued_logits, is_training=train)
    #
    #                     queued_output = tf.nn.softmax(queued_logits_bn)  # (batch_size,obs_dim)
    #                     queued_outputs.append(queued_output)  # (agents_number,batch_size,queued_dim)
    #
    #             return queued_outputs

    # def _my_unit_network_graph(self, encoder_outputs, action_outputs, queued_outputs, scope_name, train):
    #     my_unit_outputs = []
    #     with tf.variable_scope(scope_name, reuse=tf.AUTO_REUSE):
    #         with slim.arg_scope([slim.conv2d, slim.fully_connected],
    #                             activation_fn=None,
    #                             weights_initializer=tf.truncated_normal_initializer(self.mu, self.sigma),  # mu，sigma
    #                             weights_regularizer=slim.l2_regularizer(0.1)):
    #             for i in range(self.agents_number):
    #                 encoder_output = tf.concat([encoder_outputs[i, :, :], action_outputs[i, :, :], queued_outputs[i, :, :]], axis=1)
    #                 # encoder_output = tf.concat(encoder_output, self.queued_outputs[i, :, :], axis=1)
    #                 fc1 = slim.fully_connected(encoder_output, 120, scope='full_connected1')
    #
    #                 fc2 = slim.fully_connected(fc1, 84, scope='full_connected2')
    #
    #                 my_unit_logits = slim.fully_connected(fc2, self.agents_number, scope='my_unit_logits')
    #                 my_unit_logits_bn = tf.contrib.layers.batch_norm(my_unit_logits, is_training=train)
    #
    #                 my_unit_output = tf.nn.softmax(my_unit_logits_bn)  # (batch_size,obs_dim)
    #                 my_unit_outputs.append(my_unit_output)  # (agents_number,batch_size,my_unit_dim)
    #
    #             return my_unit_outputs

    # def _enemy_unit_network_graph(self, encoder_outputs, action_outputs, queued_outputs, scope_name, train):
    #     enemy_unit_outputs = []
    #     with tf.variable_scope(scope_name, reuse=tf.AUTO_REUSE):
    #         with slim.arg_scope([slim.conv2d, slim.fully_connected],
    #                             activation_fn=None,
    #                             weights_initializer=tf.truncated_normal_initializer(self.mu, self.sigma),  # mu，sigma
    #                             weights_regularizer=slim.l2_regularizer(0.1)):
    #             for i in range(self.agents_number):
    #                 encoder_output = tf.concat([encoder_outputs[i], action_outputs[i], queued_outputs[i]], axis=1)
    #                 # encoder_output = tf.concat(encoder_output, self.queued_outputs[i, :, :], axis=1)
    #                 # encoder_output = tf.concat(encoder_output, self.my_unit_outputs[i, :, :], axis=1)
    #                 with tf.variable_scope('_enemy_unit_network_graph_fc', reuse=tf.AUTO_REUSE):
    #                     # fc1 = slim.fully_connected(encoder_output, 100, scope='full_connected1')
    #                     #
    #                     # fc2 = slim.fully_connected(fc1, 80, scope='full_connected2')
    #
    #                     enemy_unit_logits = slim.fully_connected(encoder_output, self.enemy_number, scope='enemy_unit_logits')
    #                     enemy_unit_logits_bn = tf.contrib.layers.batch_norm(enemy_unit_logits, is_training=train)
    #
    #                     enemy_unit_output = tf.nn.softmax(enemy_unit_logits_bn)  # (batch_size,obs_dim)
    #                     enemy_unit_outputs.append(enemy_unit_output)  # (agents_number,batch_size,enemy_unit_dim)
    #             return enemy_unit_outputs
    #
    # def _target_point_network_x_graph(self, encoder_outputs, action_outputs, queued_outputs, enemy_unit_outputs, scope_name, train):
    #     target_point_x_outputs = []
    #     with tf.variable_scope(scope_name, reuse=tf.AUTO_REUSE):
    #         with slim.arg_scope([slim.conv2d, slim.fully_connected],
    #                             activation_fn=None,
    #                             weights_initializer=tf.truncated_normal_initializer(self.mu, self.sigma),  # mu，sigma
    #                             weights_regularizer=slim.l2_regularizer(0.1)):
    #             for i in range(self.agents_number):
    #                 encoder_output = tf.concat([encoder_outputs[i], action_outputs[i], queued_outputs[i], enemy_unit_outputs[i]],
    #                                            axis=1)
    #                 with tf.variable_scope('_target_point_network_x_graph_fc', reuse=tf.AUTO_REUSE):
    #                     # fc1 = slim.fully_connected(encoder_output, 100, scope='full_connected1')
    #                     #
    #                     # fc2 = slim.fully_connected(fc1, 80, scope='full_connected2')
    #
    #                     target_point_x_logits = slim.fully_connected(encoder_output, config.MAP_SIZE, scope='target_point_logits')
    #                     target_point_x_logits_bn = tf.contrib.layers.batch_norm(target_point_x_logits, is_training=train)
    #
    #                     target_point_x_output = tf.nn.softmax(target_point_x_logits_bn)  # (batch_size,obs_dim)
    #                     target_point_x_outputs.append(target_point_x_output)  # (agents_number,batch_size,target_point_dim)
    #
    #             return target_point_x_outputs
    #
    # def _target_point_network_y_graph(self, encoder_outputs, action_outputs, queued_outputs, enemy_unit_outputs, target_point_x_outputs, scope_name, train):
    #     target_point_y_outputs = []
    #     with tf.variable_scope(scope_name, reuse=tf.AUTO_REUSE):
    #         with slim.arg_scope([slim.conv2d, slim.fully_connected],
    #                             activation_fn=None,
    #                             weights_initializer=tf.truncated_normal_initializer(self.mu, self.sigma),  # mu，sigma
    #                             weights_regularizer=slim.l2_regularizer(0.1)):
    #             for i in range(self.agents_number):
    #                 encoder_output = tf.concat([encoder_outputs[i], action_outputs[i], queued_outputs[i], enemy_unit_outputs[i], target_point_x_outputs[i]],
    #                                            axis=1)
    #                 with tf.variable_scope('_target_point_network_y_graph_fc', reuse=tf.AUTO_REUSE):
    #                     # fc1 = slim.fully_connected(encoder_output, 100, scope='full_connected1')
    #                     #
    #                     # fc2 = slim.fully_connected(fc1, 80, scope='full_connected2')
    #
    #                     target_point_y_logits = slim.fully_connected(encoder_output, config.MAP_SIZE, scope='target_point_logits')
    #                     target_point_y_logits_bn = tf.contrib.layers.batch_norm(target_point_y_logits, is_training=train)
    #
    #                     target_point_x_output = tf.nn.softmax(target_point_y_logits_bn)  # (batch_size,obs_dim)
    #                     target_point_y_outputs.append(target_point_x_output)  # (agents_number,batch_size,target_point_dim)
    #
    #             return target_point_y_outputs
    #
    # def _get_outputs(self, action_outputs, queued_outputs, enemy_unit_outputs, target_point_x_outputs, target_point_y_outputs):
    #     outputs = tf.concat([action_outputs, queued_outputs, enemy_unit_outputs, target_point_x_outputs, target_point_y_outputs], axis=2)
    #     outputs = tf.transpose(outputs, [1, 0, 2])  # (batch_size, agents_number,outputs_prob)
    #     return outputs

    def _soft_replace(self):
        self.soft_replace = [tf.assign(t, (1 - config.GAMMA_FOR_UPDATE) * t + config.GAMMA_FOR_UPDATE * e) for t, e in zip(self.t_params, self.e_params)]

    def _optimizer(self, aout, action_gradient):
        grads = []
        batch_size = tf.to_float(tf.shape(aout)[0])
        for i in range(self.agents_number):
            # for j in range(self.agents_number):
                # grads.append(tf.gradients(aout[:, j], self.e_params, -action_gradient[j][:, i]))
            grads.append(tf.gradients(aout, self.e_params, -action_gradient[:, i]))
        grads = np.array(grads)
        unnormalized_actor_gradients = [tf.reduce_sum(list(grads[:, i]), axis=0) for i in range(len(self.e_params))]
        actor_gradients = list(map(lambda x: tf.div(x, batch_size), unnormalized_actor_gradients))

        optimizer = tf.train.AdamOptimizer(self.learning_rate)
        optimizer = optimizer.apply_gradients(zip(actor_gradients, self.e_params))

        return optimizer
