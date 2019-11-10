from collections import deque
import random

import numpy as np
import tensorflow as tf

# Hyper Parameters for DQN
GAMMA = 0.9  # discount factor for target Q
INITIAL_EPSILON = 0.5  # starting value of epsilon
FINAL_EPSILON = 0.01  # final value of epsilon
REPLAY_SIZE = 10000  # experience replay buffer size
BATCH_SIZE = 300  # size of minibatch


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
    def __init__(self, mu, sigma, learning_rate, actiondim, datadim, name):  # 初始化
        # init experience replay
        self.replay_buffer = deque()
        # init some parameters
        # 神经网络参数
        self.mu = mu
        self.sigma = sigma
        self.learning_rate = learning_rate

        self.time_step = 0
        self.epsilon = INITIAL_EPSILON

        self.state_dim = datadim
        self.action_dim = actiondim

        self.create_Q_network(name)
        self.create_training_method()

        # Init session
        self.session = tf.InteractiveSession()
        self.session.run(tf.initialize_all_variables())

    def create_Q_network(self, name):  # 创建Q网络(vgg16结构)
        self.state_input = tf.placeholder("float", shape=self.state_dim, name='state_input')
        with tf.variable_scope(name, reuse=tf.AUTO_REUSE):
            # self.conv1_1 = tf.nn.relu(
            #     self._cnn_layer('layer_1_1_conv', 'conv_w', 'conv_b', self.state_input, (3, 3, self.state_dim[3], 64),
            #                     [1, 1, 1, 1],
            #                     padding_tag='SAME'))
            #
            # self.conv1_2 = tf.nn.relu(
            #     self._cnn_layer('layer_1_2_conv', 'conv_w', 'conv_b', self.conv1_1, (3, 3, 64, 64), [1, 1, 1, 1],
            #                     padding_tag='SAME'))
            #
            # self.pool1 = self._pooling_layer('layer_1_pooling', self.conv1_2, [1, 2, 2, 1], [1, 2, 2, 1])
            #
            # self.conv2_1 = tf.nn.relu(
            #     self._cnn_layer('layer_2_1_conv', 'conv_w', 'conv_b', self.pool1, (3, 3, 64, 128), [1, 1, 1, 1],
            #                     padding_tag='SAME'))
            #
            # self.conv2_2 = tf.nn.relu(
            #     self._cnn_layer('layer_2_2_conv', 'conv_w', 'conv_b', self.conv2_1, (3, 3, 128, 128), [1, 1, 1, 1],
            #                     padding_tag='SAME'))
            #
            # self.pool2 = self._pooling_layer('layer_2_pooling', self.conv2_2, [1, 2, 2, 1], [1, 2, 2, 1])
            #
            # self.conv3_1 = tf.nn.relu(
            #     self._cnn_layer('layer_3_1_conv', 'conv_w', 'conv_b', self.pool2, (3, 3, 128, 256), [1, 1, 1, 1],
            #                     padding_tag='SAME'))
            # self.conv3_2 = tf.nn.relu(
            #     self._cnn_layer('layer_3_2_conv', 'conv_w', 'conv_b', self.conv3_1, (3, 3, 256, 256), [1, 1, 1, 1],
            #                     padding_tag='SAME'))
            # self.conv3_3 = tf.nn.relu(
            #     self._cnn_layer('layer_3_3_conv', 'conv_w', 'conv_b', self.conv3_2, (3, 3, 256, 256), [1, 1, 1, 1],
            #                     padding_tag='SAME'))
            #
            # self.pool3 = self._pooling_layer('layer_3_pooling', self.conv3_3, [1, 2, 2, 1], [1, 2, 2, 1])
            #
            # self.conv4_1 = tf.nn.relu(
            #     self._cnn_layer('layer_4_1_conv', 'conv_w', 'conv_b', self.pool3, (3, 3, 256, 512), [1, 1, 1, 1],
            #                     padding_tag='SAME'))
            # self.conv4_2 = tf.nn.relu(
            #     self._cnn_layer('layer_4_2_conv', 'conv_w', 'conv_b', self.conv4_1, (3, 3, 512, 512), [1, 1, 1, 1],
            #                     padding_tag='SAME'))
            # self.conv4_3 = tf.nn.relu(
            #     self._cnn_layer('layer_4_3_conv', 'conv_w', 'conv_b', self.conv4_2, (3, 3, 512, 512), [1, 1, 1, 1],
            #                     padding_tag='SAME'))
            #
            # self.pool4 = self._pooling_layer('layer_4_pooling', self.conv4_3, [1, 2, 2, 1], [1, 2, 2, 1])
            #
            # self.conv5_1 = tf.nn.relu(
            #     self._cnn_layer('layer_5_1_conv', 'conv_w', 'conv_b', self.pool4, (3, 3, 512, 512), [1, 1, 1, 1],
            #                     padding_tag='SAME'))
            # self.conv5_2 = tf.nn.relu(
            #     self._cnn_layer('layer_5_2_conv', 'conv_w', 'conv_b', self.conv5_1, (3, 3, 512, 512), [1, 1, 1, 1],
            #                     padding_tag='SAME'))
            #
            # self.conv5_3 = tf.nn.relu(
            #     self._cnn_layer('layer_5_3_conv', 'conv_w', 'conv_b', self.conv5_2, (3, 3, 512, 512), [1, 1, 1, 1],
            #                     padding_tag='SAME'))
            #
            # self.pool5 = self._pooling_layer('layer_5_pooling', self.conv5_3, [1, 2, 2, 1], [1, 2, 2, 1])
            #
            # self.fc6 = tf.nn.relu(self._fully_connected_layer('full_connected6', 'full_connected_w', 'full_connected_b',
            #                                                   self.pool5, (512 * 7 * 7, 4096)))
            # self.dropOut1 = tf.nn.dropout(self.fc6, 0.5)
            #
            # self.fc7 = tf.nn.relu(self._fully_connected_layer('full_connected7', 'full_connected_w', 'full_connected_b',
            #                                                   self.dropOut1, (4096, 4096)))
            # self.dropOut2 = tf.nn.dropout(self.fc7, 0.5)
            #
            # self.logits = self._fully_connected_layer('full_connected8', 'full_connected_w', 'full_connected_b',
            #                                           self.dropOut2, (4096, self.action_dim))

            self.conv1_1 = tf.nn.relu(
                self._cnn_layer('layer_1_1_conv', 'conv_w', 'conv_b', self.state_input, (3, 3, self.state_dim[3], 64),
                                [1, 1, 1, 1],
                                padding_tag='SAME'))

            self.pool1 = self._pooling_layer('layer_1_pooling', self.conv1_1, [1, 4, 4, 1], [1, 4, 4, 1])

            self.fc6 = tf.nn.relu(self._fully_connected_layer('full_connected6', 'full_connected_w', 'full_connected_b',
                                                              self.pool1, (16 * 16 * 64, 4096)))
            self.dropOut1 = tf.nn.dropout(self.fc6, 0.5)

            self.logits = self._fully_connected_layer('full_connected8', 'full_connected_w', 'full_connected_b',
                                                      self.dropOut1, (4096, self.action_dim))

            self.Q_value = tf.nn.softmax(self.logits)

    def create_training_method(self):  # 创建训练方法
        self.action_input = tf.placeholder("float", [None, self.action_dim])  # one hot presentation
        self.y_input = tf.placeholder("float", [None])
        Q_action = tf.reduce_sum(tf.multiply(self.Q_value, self.action_input), reduction_indices=1)
        self.cost = tf.reduce_mean(tf.square(self.y_input - Q_action))
        self.optimizer = tf.train.AdamOptimizer(self.learning_rate).minimize(self.cost)

    def perceive(self, state, action, reward, next_state, done):  # 感知存储信息
        one_hot_action = np.zeros(self.action_dim)
        one_hot_action[action] = 1
        state = np.squeeze(state)
        next_state = np.squeeze(next_state)

        self.replay_buffer.append([state, one_hot_action, reward, next_state, done])

        if len(self.replay_buffer) > REPLAY_SIZE:
            self.replay_buffer.popleft()

        if len(self.replay_buffer) > BATCH_SIZE:
            self.train_Q_network()

    def train_Q_network(self):  # 训练网络
        self.time_step += 1
        # Step 1: obtain random minibatch from replay memory
        minibatch = random.sample(self.replay_buffer, BATCH_SIZE)
        state_batch = np.array([data[0] for data in minibatch])
        action_batch = np.array([data[1] for data in minibatch])
        reward_batch = np.array([data[2] for data in minibatch])
        next_state_batch = np.array([data[3] for data in minibatch])

        # Step 2: calculate y
        y_batch = []
        Q_value_batch = np.array(self.session.run([self.Q_value], {self.state_input: next_state_batch}))
        # Q_value_batch = self.Q_value.eval(feed_dict={self.state_input: next_state_batch})
        for i in range(0, BATCH_SIZE):
            done = minibatch[i][4]
            if done:
                y_batch.append(reward_batch[i])
            else:
                y_batch.append(reward_batch[i] + GAMMA * np.max(Q_value_batch[i]))

        self.optimizer.run(feed_dict={
            self.y_input: y_batch,
            self.action_input: action_batch,
            self.state_input: state_batch
        })

    def egreedy_action(self, state):  # 输出带随机的动作
        Q_value = self.session.run([self.Q_value], {self.state_input: state})

        self.epsilon -= (INITIAL_EPSILON - FINAL_EPSILON) / 10000
        if np.random.uniform() <= self.epsilon:
            return random.randint(0, self.action_dim - 1)
        else:
            return np.argmax(Q_value)

    def action(self, state):
        return np.argmax(self.session.run([self.Q_value], {self.state_input: state}))
