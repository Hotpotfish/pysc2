from collections import deque
import random

import numpy as np
import tensorflow as tf

# Hyper Parameters for DQN
GAMMA = 0.9  # discount factor for target Q
INITIAL_EPSILON = 0.5  # starting value of epsilon
FINAL_EPSILON = 0.01  # final value of epsilon
REPLAY_SIZE = 10000  # experience replay buffer size
EPISODES = 32
BATCH_SIZE = 1  # size of minibatch


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
    def __init__(self, mu, sigma, learning_rate, actiondim, parameterdim, datadim, name, replay_path):  # 初始化
        # init experience replay
        self.replay_path = replay_path
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

        # self.states = []
        # self.one_hot_actions = []
        # self.rewards = []
        # self.next_states = []
        # self.dones = []
        self.data = []

        # Init session
        self.session = tf.InteractiveSession()
        self.session.run(tf.initialize_all_variables())

    def create_Q_network(self, name):  # 创建Q网络(vgg16结构)
        self.state_input = tf.placeholder("float", shape=self.state_dim, name='state_input')
        with tf.variable_scope(name, reuse=tf.AUTO_REUSE):
            self.conv1_1 = tf.nn.relu(
                self._cnn_layer('layer_1_1_conv', 'conv_w', 'conv_b', self.state_input, (3, 3, self.state_dim[3], 8),
                                [1, 1, 1, 1],
                                padding_tag='SAME'))

            self.pool1 = self._pooling_layer('layer_1_pooling', self.conv1_1, [1, 4, 4, 1], [1, 4, 4, 1])

            self.fc1 = tf.nn.relu(
                self._fully_connected_layer('full_connected_1', 'full_connected_w', 'full_connected_b',
                                            self.pool1, (self.pool1._shape[1] * self.pool1._shape[2] *
                                                         self.pool1._shape[3], 1024)))
            self.dropOut1 = tf.nn.dropout(self.fc1, 0.5)

            self.logits = self._fully_connected_layer('full_connected_2', 'full_connected_w', 'full_connected_b',
                                                      self.dropOut1, (1024, self.action_dim + self.parameterdim))

            self.Q_value = tf.nn.softmax(self.logits)

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

        # np.savetxt(
        #     self.replay_path + self.name + '.txt',
        #     np.array([state, one_hot_action, reward, next_state, done]),
        # )
        # self.states.append(state)
        # self.one_hot_actions.append(one_hot_action)
        # self.rewards.append(reward)
        # self.next_states.append(next_state)
        # self.dones.append(done)
        self.data.append([state, one_hot_action, reward, next_state, done])

        if done == True:
            np.savez(self.replay_path + self.name,
                     data=self.data)
            # state=self.states,
            # one_hot_action=self.one_hot_actions,
            # reward=self.rewards,
            # next_state=self.next_states,
            # done=self.dones)
            self.data.clear()

    def train_Q_network(self):  # 训练网络
        # self.time_step += 1
        file = np.load(self.replay_path + self.name + '.npz')
        data = list(file['data'])
        # Step 1: obtain random minibatch from replay memory

        for mark in range(EPISODES):
            for i in range(BATCH_SIZE):
                minibatch = random.sample(data, BATCH_SIZE)
                state_batch = np.array([data[0] for data in minibatch])
                action_batch = np.array([data[1] for data in minibatch])
                reward_batch = np.array([data[2] for data in minibatch])
                next_state_batch = np.array([data[3] for data in minibatch])

            # Step 2: calculate y
            y_batch = np.array([])
            Q_value_batch = np.array(self.session.run(self.Q_value, {self.state_input: next_state_batch}))
            # Q_value_batch = np.squeeze(Q_value_batch)

            # Q_value_batch = self.Q_value.eval(feed_dict={self.state_input: next_state_batch})
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
            self.optimizer.run(
                feed_dict={self.y_input: y_batch, self.action_input: action_batch, self.state_input: state_batch})

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

    def action(self, state):
        Q_value = self.session.run(self.Q_value, {self.state_input: state})[0]
        action = np.argmax(Q_value[0:self.action_dim])
        parameter = np.array(Q_value[self.action_dim:(self.action_dim + self.parameterdim)])
        action_and_parameter = np.append(action, parameter)
        return action_and_parameter
