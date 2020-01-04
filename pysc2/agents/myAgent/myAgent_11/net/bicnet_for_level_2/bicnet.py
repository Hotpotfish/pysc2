import tensorflow as tf
from pysc2.agents.myAgent.myAgent_11.config import config
from tensorflow_core.contrib import slim


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
        with tf.variable_scope('Actor'):
            self.a = self._build_graph_a(self.state_input, self.agents_local_observation, 'eval', train=True)
            a_ = self._build_graph_a(self.state_input_next, self.agents_local_observation_next, 'target', train=False)

        with tf.variable_scope('Critic'):
            q = self._build_graph_c(self.state_input, self.agents_local_observation, self.a, 'eval', train=True)
            q_ = self._build_graph_c(self.state_input_next, self.agents_local_observation_next, a_, 'target', train=False)

        # networks parameters
        self.ae_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='Actor/eval')
        self.at_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='Actor/target')
        self.ce_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='Critic/eval')
        self.ct_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='Critic/target')

        # target net replacement
        self.soft_replace = [tf.assign(t, (1 - config.GAMMA_FOR_UPDATE) * t + config.GAMMA_FOR_UPDATE * e)
                             for t, e in zip(self.at_params + self.ct_params, self.ae_params + self.ce_params)]

        q_target = self.reward + config.GAMMA * q_

        self.td_error = tf.losses.mean_squared_error(labels=q_target, predictions=q)
        self.ctrain = tf.train.AdamOptimizer(self.learning_rate).minimize(self.td_error, var_list=self.ce_params)
        self.a_loss = - tf.reduce_mean(q)  # maximize the q
        self.atrain = tf.train.AdamOptimizer(self.learning_rate).minimize(self.a_loss, var_list=self.ae_params)

    def _setup_placeholders_graph(self):
        # s
        self.state_input = tf.placeholder("float", shape=self.statedim, name='state_input')  # 全局状态
        self.agents_local_observation = tf.placeholder("float", shape=[None, self.agents_number, config.COOP_AGENTS_OBDIM], name='agents_local_observation')

        # s_
        self.state_input_next = tf.placeholder("float", shape=self.statedim, name='state_input_next')  # 全局状态
        self.agents_local_observation_next = tf.placeholder("float", shape=[None, self.agents_number, config.COOP_AGENTS_OBDIM], name='agents_local_observation_next')

        self.reward = tf.placeholder("float", shape=[None, self.agents_number], name='reward')

    #################################### actor_net  ####################################

    def _build_graph_a(self, state_input, agents_local_observation, scope_name, train):
        # 环境和智能体本地的共同观察
        with tf.variable_scope(scope_name, reuse=tf.AUTO_REUSE):
            with slim.arg_scope([slim.conv2d, slim.fully_connected],
                                trainable=train,
                                weights_initializer=tf.truncated_normal_initializer(stddev=0.01),
                                weights_regularizer=slim.l2_regularizer(0.0001)):
                encoder_outputs = self._observation_encoder_a(state_input, agents_local_observation, config.MY_UNIT_NUMBER, '_observation_encoder', train)
                bicnet_outputs = self._bicnet_build_a(encoder_outputs, '_bicnet_build', train)
                return bicnet_outputs

    def _observation_encoder_a(self, state_input, agents_local_observation, agents_number, scope_name, train):
        with tf.variable_scope(scope_name, reuse=tf.AUTO_REUSE):
            encoder = []
            conv1 = slim.conv2d(state_input, 1, [5, 5], stride=4, padding="VALID", scope='layer_1_conv')
            conv1 = tf.nn.relu(conv1)
            pool1 = slim.max_pool2d(conv1, [2, 2], stride=2, padding="VALID", scope='layer_1_pooling')

            conv2 = slim.conv2d(pool1, 1, [5, 5], stride=1, padding="VALID", scope='layer_2_conv')
            conv2 = tf.nn.relu(conv2)
            pool2 = slim.max_pool2d(conv2, [2, 2], stride=2, padding="VALID", scope='layer_2_pooling')
            # 传给下一阶段
            state_input_flatten = slim.flatten(pool2, scope="flatten")

            for i in range(agents_number):
                encoder.append(tf.concat([agents_local_observation[:, i, :], state_input_flatten], axis=1))
            encoder = tf.transpose(encoder, [1, 0, 2])
            fc1 = slim.fully_connected(encoder, 30, scope='full_connected1')
            encoder = tf.unstack(fc1, agents_number, 1)  # (self.agents_number,batch_size,obs_add_dim)
            return encoder

    def _bicnet_build_a(self, encoder_outputs, scope_name, train):
        with tf.variable_scope(scope_name, reuse=tf.AUTO_REUSE):
            lstm_fw_cell = tf.nn.rnn_cell.GRUCell(self.action_dim, name="lstm_fw_cell")
            lstm_bw_cell = tf.nn.rnn_cell.GRUCell(self.action_dim, name="lstm_bw_cell")
            bicnet_outputs, _, _ = tf.nn.static_bidirectional_rnn(lstm_fw_cell, lstm_bw_cell, encoder_outputs, dtype=tf.float32)
            fc1 = slim.fully_connected(bicnet_outputs, self.action_dim, scope='full_connected1')
            bicnet_outputs = tf.nn.softmax(fc1, axis=2)

            bicnet_outputs = tf.unstack(bicnet_outputs, self.agents_number)  # (agents_number, batch_size, action_dim)
            bicnet_outputs = tf.transpose(bicnet_outputs, [1, 0, 2])
            return bicnet_outputs  # (batch_size,agents_number,action_dim)

    #################################### critic_net  ####################################

    def _build_graph_c(self, state_input, agents_local_observation, action_input, scope_name, train):
        # 环境和智能体本地的共同观察
        with tf.variable_scope(scope_name, reuse=tf.AUTO_REUSE):
            with slim.arg_scope([slim.conv2d, slim.fully_connected],
                                trainable=train,
                                weights_initializer=tf.truncated_normal_initializer(stddev=0.01),
                                weights_regularizer=slim.l2_regularizer(0.0001)):
                encoder_outputs = self._observation_encoder_c(state_input, agents_local_observation, action_input, config.MY_UNIT_NUMBER, '_observation_encoder', train)
                bicnet_outputs = self._bicnet_build_c(encoder_outputs, '_bicnet_build', train)
                q_out = self._get_Q_c(bicnet_outputs, action_input, '_get_Q')
                return q_out

    def _observation_encoder_c(self, state_input, agents_local_observation, action_input, agents_number, scope_name, train):
        with tf.variable_scope(scope_name, reuse=tf.AUTO_REUSE):
            encoder = []
            conv1 = slim.conv2d(state_input, 1, [5, 5], stride=1, padding="VALID", scope='layer_1_conv')
            conv1 = tf.nn.relu(conv1)
            pool1 = slim.max_pool2d(conv1, [2, 2], stride=2, padding="VALID", scope='layer_1_pooling')

            conv2 = slim.conv2d(pool1, 1, [5, 5], stride=1, padding="VALID", scope='layer_2_conv')
            conv2 = tf.nn.relu(conv2)
            pool2 = slim.max_pool2d(conv2, [2, 2], stride=2, padding="VALID", scope='layer_2_pooling')

            # 传给下一阶段
            state_input_flatten = slim.flatten(pool2, scope="flatten")
            for i in range(agents_number):
                encoder.append(tf.concat([agents_local_observation[:, i, :], state_input_flatten, action_input[:, i]], axis=1))
            encoder = tf.transpose(encoder, [1, 0, 2])
            fc1 = slim.fully_connected(encoder, 60, scope='full_connected3')
            fc1 = tf.nn.relu(fc1)
            encoder = tf.unstack(fc1, agents_number, 1)  # (self.agents_number,batch_size,obs_add_dim)
            return encoder

    def _bicnet_build_c(self, encoder_outputs, scope_name, train):
        with tf.variable_scope(scope_name, reuse=tf.AUTO_REUSE):
            lstm_fw_cell = tf.nn.rnn_cell.GRUCell(self.action_dim, name="lstm_fw_cell")
            lstm_bw_cell = tf.nn.rnn_cell.GRUCell(self.action_dim, name="lstm_bw_cell")
            bicnet_outputs, _, _ = tf.nn.static_bidirectional_rnn(lstm_fw_cell, lstm_bw_cell, encoder_outputs, dtype=tf.float32)
            fc1 = slim.fully_connected(bicnet_outputs, self.action_dim, scope='full_connected1')
            bicnet_outputs = tf.nn.softmax(fc1, axis=2)

            bicnet_outputs = tf.unstack(bicnet_outputs, self.agents_number)  # (agents_number, batch_size, action_dim)
            bicnet_outputs = tf.transpose(bicnet_outputs, [1, 0, 2])
            return bicnet_outputs  # (batch_size,agents_number,action_dim)

    def _get_Q_c(self, bicnet_outputs, action_input, scope_name):
        with tf.variable_scope(scope_name, reuse=tf.AUTO_REUSE):
            q_out = tf.multiply(bicnet_outputs, action_input)  # (batch_size, agents_number,outputs_prob)
            q_out = tf.reduce_sum(q_out, axis=2)  # (batch_size, agents_number)

            return q_out
