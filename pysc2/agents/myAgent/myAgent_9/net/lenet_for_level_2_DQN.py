import tensorflow as tf
import tensorflow.contrib.slim as slim

from pysc2.agents.myAgent.myAgent_9.config import config


class Lenet():

    def __init__(self, mu, sigma, learning_rate, action_dim, parameterdim, statedim, name):
        self.mu = mu
        self.sigma = sigma
        self.learning_rate = learning_rate

        self.action_dim = action_dim
        self.parameterdim = parameterdim
        self.statedim = statedim

        self.name = name

        self._build_graph()


    def _build_graph(self):

        self._setup_placeholders_graph()
        self._build_network_graph(self.name)
        self._compute_loss_graph()
        self._create_train_op_graph()
        # self.merged_summary = tf.summary.merge_all()

    def _build_network_graph(self, name):
        self._action_network_graph(name + '_' + 'action')
        self._queued_network_graph(name + '_' + 'queued')
        self._my_unit_network_graph(name + '_' + 'my_unit')
        self._enemy_unit_network_graph(name + '_' + 'enemy_unit')
        self._target_point_network_graph(name + '_' + 'target_point')

    def _setup_placeholders_graph(self):
        self.action_input = tf.placeholder("float", shape=[None, self.action_dim + self.parameterdim], name=self.name + '_' + 'action_input')
        self.y_input = tf.placeholder("float", shape=[None, config.ORDERLENTH], name=self.name + '_' + 'y_input')
        self.state_input = tf.placeholder("float", shape=self.statedim, name=self.name + '_' + 'state_input')

    def _action_network_graph(self, scope_name):
        with tf.variable_scope(scope_name, reuse=tf.AUTO_REUSE):
            with slim.arg_scope([slim.conv2d, slim.fully_connected],
                                activation_fn=None,
                                weights_initializer=tf.truncated_normal_initializer(self.mu, self.sigma),  # mu，sigma
                                weights_regularizer=slim.l2_regularizer(0.1)):
                conv1 = slim.conv2d(self.state_input, 6, [5, 5], stride=1, padding="VALID", scope='layer_1_conv')
                pool1 = slim.max_pool2d(conv1, [2, 2], stride=2, padding="VALID", scope='layer_1_pooling')

                conv2 = slim.conv2d(pool1, 16, [5, 5], stride=1, padding="VALID", scope='layer_2_conv')
                pool2 = slim.max_pool2d(conv2, [2, 2], stride=2, padding="VALID", scope='layer_2_pooling')
                # 传给下一阶段
                self.action_flatten = slim.flatten(pool2, scope="flatten")

                fc1 = slim.fully_connected(self.action_flatten, 120, scope='full_connected1')
                fc2 = slim.fully_connected(fc1, 84, scope='full_connected2')

                self.action = slim.fully_connected(fc2, self.action_dim, activation_fn=tf.nn.softmax, scope='action')

    def _queued_network_graph(self, scope_name):
        with tf.variable_scope(scope_name, reuse=tf.AUTO_REUSE):
            with slim.arg_scope([slim.conv2d, slim.fully_connected],
                                activation_fn=None,
                                weights_initializer=tf.truncated_normal_initializer(self.mu, self.sigma),  # mu，sigma
                                weights_regularizer=slim.l2_regularizer(0.1)):
                self.queued_flatten = tf.concat([self.action_flatten, self.action], axis=1)

                fc1 = slim.fully_connected(self.queued_flatten, 120, scope='full_connected1')
                fc2 = slim.fully_connected(fc1, 84, scope='full_connected2')
                self.queued = slim.fully_connected(fc2, config.QUEUED, activation_fn=tf.nn.softmax, scope='queued')

    def _my_unit_network_graph(self, scope_name):
        with tf.variable_scope(scope_name, reuse=tf.AUTO_REUSE):
            with slim.arg_scope([slim.conv2d, slim.fully_connected],
                                activation_fn=None,
                                weights_initializer=tf.truncated_normal_initializer(self.mu, self.sigma),  # mu，sigma
                                weights_regularizer=slim.l2_regularizer(0.1)):
                self.my_unit_flatten = tf.concat([self.queued_flatten, self.queued], axis=1)

                fc1 = slim.fully_connected(self.my_unit_flatten, 120, scope='full_connected1')
                fc2 = slim.fully_connected(fc1, 84, scope='full_connected2')

                self.my_unit = slim.fully_connected(fc2, config.MY_UNIT_NUMBER, activation_fn=tf.nn.softmax, scope='my_unit')

    def _enemy_unit_network_graph(self, scope_name):
        with tf.variable_scope(scope_name, reuse=tf.AUTO_REUSE):
            with slim.arg_scope([slim.conv2d, slim.fully_connected],
                                activation_fn=None,
                                weights_initializer=tf.truncated_normal_initializer(self.mu, self.sigma),  # mu，sigma
                                weights_regularizer=slim.l2_regularizer(0.1)):
                self.enemy_unit_flatten = tf.concat([self.my_unit_flatten, self.my_unit], axis=1)

                fc1 = slim.fully_connected(self.enemy_unit_flatten, 120, scope='full_connected1')
                fc2 = slim.fully_connected(fc1, 84, scope='full_connected2')

                self.enemy_unit = slim.fully_connected(fc2, config.ENEMY_UNIT_NUMBER, activation_fn=tf.nn.softmax, scope='enemy_unit')

    def _target_point_network_graph(self, scope_name):
        with tf.variable_scope(scope_name, reuse=tf.AUTO_REUSE):
            with slim.arg_scope([slim.conv2d, slim.fully_connected],
                                activation_fn=None,
                                weights_initializer=tf.truncated_normal_initializer(self.mu, self.sigma),  # mu，sigma
                                weights_regularizer=slim.l2_regularizer(0.1)):
                self.target_point_flatten = tf.concat([self.enemy_unit_flatten, self.enemy_unit], axis=1)

                fc1 = slim.fully_connected(self.target_point_flatten, 120, scope='full_connected1')
                fc2 = slim.fully_connected(fc1, 84, scope='full_connected2')

                self.target_point = slim.fully_connected(fc2, config.POINT_NUMBER, activation_fn=tf.nn.softmax, scope='target_point')
                self.Q_value = tf.concat([self.action, self.queued, self.my_unit, self.enemy_unit, self.target_point], axis=1)

    def _compute_loss_graph(self):
        with tf.name_scope(self.name + "_loss_function"):
            self.Q_action = tf.multiply(self.Q_value, self.action_input)

            start = 0
            end = self.action_dim
            self.action_Q = tf.reduce_sum(self.Q_action[:, start:end], axis=1)

            start = end
            end += config.QUEUED
            self.queued_Q = tf.reduce_sum(self.Q_action[:, start:end], axis=1)

            start = end
            end += config.MY_UNIT_NUMBER
            self.my_unit_Q = tf.reduce_sum(self.Q_action[:, start:end], axis=1)

            start += end
            end += config.ENEMY_UNIT_NUMBER
            self.enemy_unit_Q = tf.reduce_sum(self.Q_action[:, start:end], axis=1)
            start = end
            self.target_point_Q = tf.reduce_sum(self.Q_action[:, start:], axis=1)
            self.Q_action_1 = tf.stack([self.action_Q, self.queued_Q, self.my_unit_Q, self.enemy_unit_Q, self.target_point_Q], 1)

            self.loss = tf.reduce_mean(tf.square(self.y_input - self.Q_action_1))
        # tf.summary.scalar(self.name + "_loss_function", self.loss)

    def _compute_acc_graph(self):
        with tf.name_scope(self.name + "_acc_function"):
            self.accuracy = \
                tf.metrics.accuracy(labels=tf.argmax(self.y, axis=1), predictions=tf.argmax(self.y_predicted, axis=1))[
                    1]
            tf.summary.scalar("accuracy", self.accuracy)

    def _create_train_op_graph(self):
        self.train_op = tf.train.AdamOptimizer(self.learning_rate).minimize(self.loss)
