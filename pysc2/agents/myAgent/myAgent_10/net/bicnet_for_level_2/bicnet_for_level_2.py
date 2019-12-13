import tensorflow as tf
import tensorflow.contrib.slim as slim

from pysc2.agents.myAgent.myAgent_10.config import config


class bicnet_actor():

    def __init__(self, mu, sigma, learning_rate, action_dim, parameterdim, statedim, agents_number, enemy_number, name):
        self.mu = mu
        self.sigma = sigma
        self.learning_rate = learning_rate

        self.action_dim = action_dim
        self.parameterdim = parameterdim
        self.statedim = statedim

        self.agents_number = agents_number
        self.enemy_number = enemy_number

        self.name = name

        # 建立输入管道
        self._setup_placeholders_graph()

        # 两个A网络
        with tf.variable_scope('Actor'):
            # a
            self.a = self._build_graph(self.state_input, self.agents_local_observation, 'eval_net', True)
            # a_
            self.a_ = self._build_graph(self.state_input_next, self.agents_local_observation_next, 'target_net', False)

        self.e_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='Actor/eval_net')
        self.t_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='Actor/target_net')

        self.soft_replace = [tf.assign(t, (1 - config.GAMMA_FOR_UPDATE) * t + config.GAMMA_FOR_UPDATE * e) for t, e in zip(self.t_params, self.e_params)]

    def _setup_placeholders_graph(self):
        # s
        self.state_input = tf.placeholder("float", shape=self.statedim, name=self.name + '_' + 'state_input')  # 全局状态
        self.agents_local_observation = tf.placeholder("float", shape=[None, self.agents_number, config.COOP_AGENTS_OBDIM], name='agents_local_observation')

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
            bicnet_outputs = self._bicnet_build(encoder_outputs, '_bicnet_build')
            action_outputs = self._action_network_graph(bicnet_outputs, '_action_network_graph', train)
            queued_outputs = self._queued_network_graph(encoder_outputs, action_outputs, '_queued_network_graph', train)
            my_unit_outputs = self._my_unit_network_graph(encoder_outputs, action_outputs, queued_outputs, '_my_unit_network_graph', train)
            enemy_unit_outputs = self._enemy_unit_network_graph(encoder_outputs, action_outputs, queued_outputs, my_unit_outputs, '_enemy_unit_network_graph', train)
            target_point_outputs = self._target_point_network_graph(encoder_outputs, action_outputs, queued_outputs, my_unit_outputs, enemy_unit_outputs, '_target_point_network_graph', train)
            outputs = self._get_outputs(action_outputs, queued_outputs, my_unit_outputs, enemy_unit_outputs, target_point_outputs)
            return outputs

    def _observation_encoder(self, state_input, agents_local_observation, agents_number, scope_name, train):
        with tf.variable_scope(scope_name, reuse=tf.AUTO_REUSE):
            encoder = []
            for i in range(agents_number):
                encoder.append(tf.concat([agents_local_observation[:, i, :], state_input], axis=1))
            encoder = tf.transpose(encoder, [1, 0, 2])
            fc1 = slim.fully_connected(encoder, 4096, scope='full_connected1')
            bn1 = tf.layers.batch_normalization(fc1, training=train)
            fc2 = slim.fully_connected(bn1, 512, scope='full_connected2')
            bn2 = tf.layers.batch_normalization(fc2, training=train)
            fc3 = slim.fully_connected(bn2, 64, scope='full_connected3')
            bn3 = tf.layers.batch_normalization(fc3, training=train)
            encoder = tf.unstack(bn3, agents_number, 1)  # (self.agents_number,batch_size,obs_add_dim)
            return encoder

    def _bicnet_build(self, encoder_outputs, scope_name):
        with tf.variable_scope(scope_name, reuse=tf.AUTO_REUSE):
            lstm_fw_cell = tf.nn.rnn_cell.BasicLSTMCell(64, forget_bias=1.0, name="lstm_fw_cell")
            lstm_bw_cell = tf.nn.rnn_cell.BasicLSTMCell(64, forget_bias=1.0, name="lstm_bw_cell")
            bicnet_outputs, _, _ = tf.nn.static_bidirectional_rnn(lstm_fw_cell, lstm_bw_cell, encoder_outputs, dtype=tf.float32)
            bicnet_outputs = tf.unstack(bicnet_outputs, self.agents_number, axis=1)
            return bicnet_outputs  # (agents_number,batch_size,64*2)

    def _action_network_graph(self, bicnet_outputs, scope_name, train):
        action_outputs = []
        with tf.variable_scope(scope_name, reuse=tf.AUTO_REUSE):
            with slim.arg_scope([slim.conv2d, slim.fully_connected],
                                activation_fn=None,
                                weights_initializer=tf.truncated_normal_initializer(self.mu, self.sigma),  # mu，sigma
                                weights_regularizer=slim.l2_regularizer(0.1)):
                for i in range(self.agents_number):
                    encoder_output = bicnet_outputs[i, :, :]
                    fc1 = slim.fully_connected(encoder_output, 120, scope='full_connected1')

                    fc2 = slim.fully_connected(fc1, 84, scope='full_connected2')

                    action_logits = slim.fully_connected(fc2, self.action_dim, scope='action_logits')
                    action_logits_bn = tf.contrib.layers.batch_norm(action_logits, is_training=train)

                    action_output = tf.nn.softmax(action_logits_bn)  # (batch_size,obs_dim)
                    action_outputs.append(action_output)  # (agents_number,batch_size,action_dim)

                return action_outputs

    def _queued_network_graph(self, encoder_outputs, action_outputs, scope_name, train):
        queued_outputs = []
        with tf.variable_scope(scope_name, reuse=tf.AUTO_REUSE):
            with slim.arg_scope([slim.conv2d, slim.fully_connected],
                                activation_fn=None,
                                weights_initializer=tf.truncated_normal_initializer(self.mu, self.sigma),  # mu，sigma
                                weights_regularizer=slim.l2_regularizer(0.1)):
                for i in range(self.agents_number):
                    encoder_output = tf.concat([encoder_outputs[i, :, :], action_outputs[i, :, :]], axis=1)
                    fc1 = slim.fully_connected(encoder_output, 120, scope='full_connected1')

                    fc2 = slim.fully_connected(fc1, 84, scope='full_connected2')

                    queued_logits = slim.fully_connected(fc2, config.QUEUED, scope='queued_logits')
                    queued_logits_bn = tf.contrib.layers.batch_norm(queued_logits, is_training=train)

                    queued_output = tf.nn.softmax(queued_logits_bn)  # (batch_size,obs_dim)
                    queued_outputs.append(queued_output)  # (agents_number,batch_size,queued_dim)

                return queued_outputs

    def _my_unit_network_graph(self, encoder_outputs, action_outputs, queued_outputs, scope_name, train):
        my_unit_outputs = []
        with tf.variable_scope(scope_name, reuse=tf.AUTO_REUSE):
            with slim.arg_scope([slim.conv2d, slim.fully_connected],
                                activation_fn=None,
                                weights_initializer=tf.truncated_normal_initializer(self.mu, self.sigma),  # mu，sigma
                                weights_regularizer=slim.l2_regularizer(0.1)):
                for i in range(self.agents_number):
                    encoder_output = tf.concat([encoder_outputs[i, :, :], action_outputs[i, :, :], queued_outputs[i, :, :]], axis=1)
                    # encoder_output = tf.concat(encoder_output, self.queued_outputs[i, :, :], axis=1)
                    fc1 = slim.fully_connected(encoder_output, 120, scope='full_connected1')

                    fc2 = slim.fully_connected(fc1, 84, scope='full_connected2')

                    my_unit_logits = slim.fully_connected(fc2, self.agents_number, scope='my_unit_logits')
                    my_unit_logits_bn = tf.contrib.layers.batch_norm(my_unit_logits, is_training=train)

                    my_unit_output = tf.nn.softmax(my_unit_logits_bn)  # (batch_size,obs_dim)
                    my_unit_outputs.append(my_unit_output)  # (agents_number,batch_size,my_unit_dim)

                return my_unit_outputs

    def _enemy_unit_network_graph(self, encoder_outputs, action_outputs, queued_outputs, my_unit_outputs, scope_name, train):
        enemy_unit_outputs = []
        with tf.variable_scope(scope_name, reuse=tf.AUTO_REUSE):
            with slim.arg_scope([slim.conv2d, slim.fully_connected],
                                activation_fn=None,
                                weights_initializer=tf.truncated_normal_initializer(self.mu, self.sigma),  # mu，sigma
                                weights_regularizer=slim.l2_regularizer(0.1)):
                for i in range(self.agents_number):
                    encoder_output = tf.concat([encoder_outputs[i, :, :], action_outputs[i, :, :], queued_outputs[i, :, :], my_unit_outputs[i, :, :]], axis=1)
                    # encoder_output = tf.concat(encoder_output, self.queued_outputs[i, :, :], axis=1)
                    # encoder_output = tf.concat(encoder_output, self.my_unit_outputs[i, :, :], axis=1)
                    fc1 = slim.fully_connected(encoder_output, 120, scope='full_connected1')

                    fc2 = slim.fully_connected(fc1, 84, scope='full_connected2')

                    enemy_unit_logits = slim.fully_connected(fc2, self.enemy_number, scope='enemy_unit_logits')
                    enemy_unit_logits_bn = tf.contrib.layers.batch_norm(enemy_unit_logits, is_training=train)

                    enemy_unit_output = tf.nn.softmax(enemy_unit_logits_bn)  # (batch_size,obs_dim)
                    enemy_unit_outputs.append(enemy_unit_output)  # (agents_number,batch_size,enemy_unit_dim)
                return enemy_unit_outputs

    def _target_point_network_graph(self, encoder_outputs, action_outputs, queued_outputs, my_unit_outputs, enemy_unit_outputs, scope_name, train):
        target_point_outputs = []
        with tf.variable_scope(scope_name, reuse=tf.AUTO_REUSE):
            with slim.arg_scope([slim.conv2d, slim.fully_connected],
                                activation_fn=None,
                                weights_initializer=tf.truncated_normal_initializer(self.mu, self.sigma),  # mu，sigma
                                weights_regularizer=slim.l2_regularizer(0.1)):
                for i in range(self.agents_number):
                    encoder_output = tf.concat([encoder_outputs[i, :, :], action_outputs[i, :, :], queued_outputs[i, :, :], my_unit_outputs[i, :, :], enemy_unit_outputs[i, :, :]],
                                               axis=1)
                    fc1 = slim.fully_connected(encoder_output, 120, scope='full_connected1')

                    fc2 = slim.fully_connected(fc1, 84, scope='full_connected2')

                    target_point_logits = slim.fully_connected(fc2, config.POINT_NUMBER, scope='target_point_logits')
                    target_point_logits_bn = tf.contrib.layers.batch_norm(target_point_logits, is_training=train)

                    target_point_output = tf.nn.softmax(target_point_logits_bn)  # (batch_size,obs_dim)
                    target_point_outputs.append(target_point_output)  # (agents_number,batch_size,target_point_dim)

                return target_point_outputs

    def _get_outputs(self, action_outputs, queued_outputs, my_unit_outputs, enemy_unit_outputs, target_point_outputs):
        outputs = tf.concat([action_outputs, queued_outputs, my_unit_outputs, enemy_unit_outputs, target_point_outputs], axis=2)
        outputs = tf.transpose(outputs, [1, 0, 2])  # (batch_size, agents_number,outputs_prob)
        return outputs

    def _soft_replace(self):
        self.soft_replace = [tf.assign(t, (1 - config.GAMMA_FOR_UPDATE) * t + config.GAMMA_FOR_UPDATE * e) for t, e in zip(self.t_params, self.e_params)]

#
# def _compute_loss_graph(self):
#     with tf.name_scope(self.name + "_loss_function"):
#             self.action_prob = tf.multiply(self.prob_value, self.action_input)
#             start = 0
#             end = self.action_dim
#             self.action_Q = tf.reduce_sum(self.action_prob[:, start:end], axis=1)
#
#             start = end
#             end += config.QUEUED
#             self.queued_Q = tf.reduce_sum(self.action_prob[:, start:end], axis=1)
#
#             start = end
#             end += config.MY_UNIT_NUMBER
#             self.my_unit_Q = tf.reduce_sum(self.action_prob[:, start:end], axis=1)
#
#             start += end
#             end += config.ENEMY_UNIT_NUMBER
#             self.enemy_unit_Q = tf.reduce_sum(self.action_prob[:, start:end], axis=1)
#             start = end
#             self.target_point_Q = tf.reduce_sum(self.action_prob[:, start:], axis=1)
#             self.Q_action_1 = tf.stack([self.action_Q, self.queued_Q, self.my_unit_Q, self.enemy_unit_Q, self.target_point_Q], 1)
#
#             self.loss = tf.reduce_mean(tf.multiply(self.Q_action_1, self.reward_input))
#     # tf.summary.scalar(self.name + "_loss_function", self.loss)
#
# #
# # def _compute_acc_graph(self):
# #     with tf.name_scope(self.name + "_acc_function"):
# #         self.accuracy = \
# #             tf.metrics.accuracy(labels=tf.argmax(self.y, axis=1), predictions=tf.argmax(self.y_predicted, axis=1))[
# #                 1]
# #         tf.summary.scalar("accuracy", self.accuracy)
#
# def _create_train_op_graph(self):
#     optimizer = tf.train.AdamOptimizer(self.learning_rate)
#     extra_update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
#     with tf.control_dependencies(extra_update_ops):
#         self.train_op = optimizer.minimize(self.loss)
