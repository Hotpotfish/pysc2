import datetime
import os
from collections import deque
import random

import numpy as np
import tensorflow as tf

# Hyper Parameters for DQN
from tensorflow_core.contrib import slim

GAMMA = 0.9  # discount factor for target Q
INITIAL_EPSILON = 0.5  # starting value of epsilon
FINAL_EPSILON = 0.01  # final value of epsilon
REPLAY_SIZE = 20000  # experience replay buffer size
LOOP = 4
BATCH_SIZE = 32  # size of minibatch


class DQN():

    def _cnn_layer(self, scope_name, W_name, b_name, x, filter_shape, conv_strides, padding_tag='VALID'):
        with tf.variable_scope(scope_name):
            conv_W = tf.get_variable(W_name,
                                     dtype=tf.float32,
                                     initializer=tf.truncated_normal(shape=filter_shape, mean=self.mu,
                                                                     stddev=self.sigma))
            conv_b = tf.get_variable(b_name,
                                     dtype=tf.float32,
                                     initializer=tf.zeros(filter_shape[3]))
            conv = tf.nn.conv2d(x, conv_W,
                                strides=conv_strides,
                                padding=padding_tag) + conv_b

            return conv

    def _pooling_layer(self, scope_name, x, pool_ksize, pool_strides, padding_tag='VALID'):
        with tf.variable_scope(scope_name):
            pool = tf.nn.avg_pool(x, pool_ksize, pool_strides, padding=padding_tag)
            return pool

    def _fully_connected_layer(self, scope_name, W_name, b_name, x, W_shape):
        with tf.variable_scope(scope_name):
            x = tf.reshape(x, [-1, W_shape[0]])
            w = tf.get_variable(W_name,
                                dtype=tf.float32,
                                initializer=tf.truncated_normal(shape=W_shape, mean=self.mu,
                                                                stddev=self.sigma))
            b = tf.get_variable(b_name,
                                dtype=tf.float32,
                                initializer=tf.zeros(W_shape[1]))

            r = tf.add(tf.matmul(x, w), b)

        return r

    # DQN Agent
    def __init__(self, mu, sigma, learning_rate, actiondim, parameterdim, datadim, name):  # 初始化
        # init experience replay
        self.replay_buffer = deque(maxlen=REPLAY_SIZE)
        # init some parameters
        # 神经网络参数
        self.mu = mu
        self.sigma = sigma
        self.learning_rate = learning_rate

        self.time_step = 0
        self.epsilon = INITIAL_EPSILON

        self.state_dim = datadim
        self.action_dim = actiondim
        self.parameterdim = parameterdim

        self.create_Q_network(name)
        self.create_training_method()
        self.name = name

        self.data = []

        self.saver = tf.train.Saver()

        # Init session
        self.session = tf.InteractiveSession()
        self.session.run(tf.initialize_all_variables())

        self.restoreModelMark = True

    def restoreModel(self, modelLoadPath):
        self.saver.restore(self.session, modelLoadPath + '/' + self.name + '.ckpt')

    def create_Q_network(self, name):  # 创建Q网络(vgg16结构)
        self.state_input = tf.placeholder("float", shape=self.state_dim, name='state_input')
        with tf.variable_scope(name, reuse=tf.AUTO_REUSE):
            with slim.arg_scope([slim.conv2d],
                                activation_fn=tf.nn.relu,
                                padding='SAME',
                                weights_initializer=tf.truncated_normal_initializer(self.mu, self.sigma),
                                weights_regularizer=slim.l2_regularizer(0.0005),
                                ):
                # 112 * 112 * 64
                net = slim.conv2d(self.state_input, 64, [7, 7], stride=2, scope='conv1')

                # 56 * 56 * 64
                net = slim.max_pool2d(net, [3, 3], stride=2, scope='pool1')

                temp = net

                # 第一残差块
                net = slim.conv2d(net, 64, [3, 3], scope='conv2_1_1')
                net = slim.conv2d(net, 64, [3, 3], scope='conv2_1_2')
                # 残差相加
                net = tf.nn.relu(tf.add(temp, net))

                temp = net
                # 残差块
                net = slim.conv2d(net, 64, [3, 3], scope='conv2_2_1')
                net = slim.conv2d(net, 64, [3, 3], scope='conv2_2_2')
                # 残差相加
                net = tf.nn.relu(tf.add(temp, net))

                temp = net
                # 28 * 28 * 128
                temp = slim.conv2d(temp, 128, [1, 1], stride=2, scope='r1')

                # 第二残差块
                net = slim.conv2d(net, 128, [3, 3], stride=2, scope='conv3_1_1')
                net = slim.conv2d(net, 128, [3, 3], scope='conv3_1_2')
                # 残差相加
                net = tf.nn.relu(tf.add(temp, net))

                temp = net
                # 残差块
                net = slim.conv2d(net, 128, [3, 3], scope='conv3_2_1')
                net = slim.conv2d(net, 128, [3, 3], scope='conv3_2_2')
                # 残差相加
                net = tf.nn.relu(tf.add(temp, net))

                temp = net
                # 14 * 14 * 256
                temp = slim.conv2d(temp, 256, [1, 1], stride=2, scope='r2')

                # 第三残差块
                net = slim.conv2d(net, 256, [3, 3], stride=2, scope='conv4_1_1')
                net = slim.conv2d(net, 256, [3, 3], scope='conv4_1_2')
                # 残差相加
                net = tf.nn.relu(tf.add(temp, net))

                temp = net
                # 残差块
                net = slim.conv2d(net, 256, [3, 3], scope='conv4_2_1')
                net = slim.conv2d(net, 256, [3, 3], scope='conv4_2_2')
                # 残差相加
                net = tf.nn.relu(tf.add(temp, net))

                temp = net
                # 7 * 7 * 512
                temp = slim.conv2d(temp, 512, [1, 1], stride=2, scope='r3')

                # 第四残差块
                net = slim.conv2d(net, 512, [3, 3], stride=2, scope='conv5_1_1')
                net = slim.conv2d(net, 512, [3, 3], scope='conv5_1_2')
                # 残差相加
                net = tf.nn.relu(tf.add(temp, net))

                temp = net
                # 残差块
                net = slim.conv2d(net, 512, [3, 3], scope='conv5_2_1')
                net = slim.conv2d(net, 512, [3, 3], scope='conv5_2_2')
                # 残差相加
                net = tf.nn.relu(tf.add(temp, net))

                net = slim.avg_pool2d(net, [4, 4], stride=1, scope='pool2')

                net = slim.flatten(net, scope='flatten')
                fc1 = slim.fully_connected(net, 1000, scope='fc1')

                self.logits = slim.fully_connected(fc1, self.action_dim + self.parameterdim, activation_fn=None, scope='fc2')
                self.Q_value = tf.nn.softmax(self.logits)
        # with tf.variable_scope(name, reuse=tf.AUTO_REUSE):
        #     self.conv1_1 = tf.nn.relu(
        #         self._cnn_layer('layer_1_1_conv', 'conv_w', 'conv_b', self.state_input, (3, 3, self.state_dim[3], 8),
        #                         [1, 1, 1, 1],
        #                         padding_tag='SAME'))
        #
        #     self.pool1 = self._pooling_layer('layer_1_pooling', self.conv1_1, [1, 2, 2, 1], [1, 2, 2, 1])
        #
        #     self.fc1 = tf.nn.relu(
        #         self._fully_connected_layer('full_connected_1', 'full_connected_w', 'full_connected_b',
        #                                     self.pool1, (self.pool1._shape[1] * self.pool1._shape[2] *
        #                                                  self.pool1._shape[3], 1024)))
        #     self.dropOut1 = tf.nn.dropout(self.fc1, 0.5)
        #
        #     self.logits = self._fully_connected_layer('full_connected_2', 'full_connected_w', 'full_connected_b',
        #                                               self.dropOut1, (1024, self.action_dim + self.parameterdim))
        #
        #     self.Q_value = tf.nn.softmax(self.logits)

    def create_training_method(self):  # 创建训练方法
        self.action_input = tf.placeholder("float", [None, self.action_dim + self.parameterdim])  # one hot presentation
        self.y_input = tf.placeholder("float", [None, 1 + self.parameterdim])
        Q_action = tf.reduce_sum(tf.multiply(self.Q_value, self.action_input))
        self.cost = tf.reduce_mean(tf.square(self.y_input - Q_action))
        self.optimizer = tf.train.AdamOptimizer(self.learning_rate).minimize(self.cost)

    def perceive(self, state, action, reward, next_state, done):  # 感知存储信息
        one_hot_action = np.zeros(self.action_dim + self.parameterdim)
        one_hot_action[int(action[0])] = 1
        if self.parameterdim != 0:
            one_hot_action[self.action_dim:] = action[1:]
        state = np.squeeze(state)
        next_state = np.squeeze(next_state)

        self.replay_buffer.append([state, one_hot_action, reward, next_state, done])

    def train_Q_network(self, modelSavePath, episode):  # 训练网络
        # Step 1: obtain random minibatch from replay memory

        if len(self.replay_buffer) > BATCH_SIZE:
            for mark in range(LOOP):
                for i in range(BATCH_SIZE):
                    minibatch = random.sample(self.replay_buffer, BATCH_SIZE)
                    state_batch = np.array([data[0] for data in minibatch])
                    action_batch = np.array([data[1] for data in minibatch])
                    reward_batch = np.array([data[2] for data in minibatch])
                    next_state_batch = np.array([data[3] for data in minibatch])

                # Step 2: calculate y
                y_batch = np.array([])
                Q_value_batch = np.array(self.session.run(self.Q_value, {self.state_input: next_state_batch}))

                for i in range(0, BATCH_SIZE):
                    done = minibatch[i][4]
                    if done:
                        temp = np.append(np.array(reward_batch[i]), np.array(Q_value_batch[i][self.action_dim:]))
                        temp = temp.reshape((1, 1 + self.parameterdim))
                        y_batch = np.append(y_batch, temp)
                    else:
                        temp = np.append(np.array(reward_batch[i] + GAMMA * np.max(Q_value_batch[i][0:self.action_dim])),
                                         Q_value_batch[i][self.action_dim:])
                        temp = temp.reshape((1, 1 + self.parameterdim))
                        y_batch = np.append(y_batch, temp)
                y_batch = np.array(y_batch).reshape(BATCH_SIZE, 1 + self.parameterdim)
                self.optimizer.run(feed_dict={self.y_input: y_batch, self.action_input: action_batch, self.state_input: state_batch})
        if episode % 2 == 0 :
            self.replay_buffer.clear()

        if episode % 20 == 0:
            thisPath = modelSavePath + 'episode_' + str(episode) + '/'
            try:
                os.makedirs(thisPath)
            except OSError:
                pass

            self.saver.save(self.session, thisPath + self.name + '.ckpt', )

    def egreedy_action(self, state):  # 输出带随机的动作
        Q_value = self.session.run(self.Q_value, {self.state_input: state})[0]
        self.epsilon -= (INITIAL_EPSILON - FINAL_EPSILON) / 10000
        if np.random.uniform() <= self.epsilon:
            # return random.randint(0, self.action_dim - 1)
            random_action = np.random.randint(0, self.action_dim)
            random_parameter = np.random.rand(self.parameterdim)
            random_action_and_parameter = np.append(random_action, random_parameter).flatten()
            return random_action_and_parameter

        else:
            action = np.argmax(Q_value[0:self.action_dim])
            parameter = np.array(Q_value[self.action_dim:(self.action_dim + self.parameterdim)])
            action_and_parameter = np.append(action, parameter).flatten()
            return action_and_parameter

    def action(self, state, modelLoadPath):

        if self.restoreModelMark == True and modelLoadPath is not None:
            self.restoreModelMark = False
            self.restoreModel(modelLoadPath)
            print(self.name + 'read!')

        Q_value = self.session.run(self.Q_value, {self.state_input: state})[0]
        action = np.argmax(Q_value[0:self.action_dim])
        parameter = np.array(Q_value[self.action_dim:(self.action_dim + self.parameterdim)])
        action_and_parameter = np.append(action, parameter)
        return action_and_parameter
