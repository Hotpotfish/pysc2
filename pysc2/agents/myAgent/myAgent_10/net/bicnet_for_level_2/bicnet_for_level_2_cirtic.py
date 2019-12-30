import tensorflow as tf
import tensorflow.contrib.slim as slim

from pysc2.agents.myAgent.myAgent_10.config import config


class bicnet_critic():

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
        # self._setup_placeholders_graph()

        # 两个q网络

        with tf.variable_scope(name, reuse=tf.AUTO_REUSE):
            self._setup_placeholders_graph()
            # q
            self.q = self._build_graph(self.state_input, self.agents_local_observation, self.action_input, 'eval_net', True)
            # q_
            self.q_ = self._build_graph(self.state_input_next, self.agents_local_observation_next, self.action_input_next, 'target_net', False)

        self.e_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=name + '/eval_net')
        self.t_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=name + '/target_net')

        self.soft_replace = [tf.assign(t, (1 - config.GAMMA_FOR_UPDATE) * t + config.GAMMA_FOR_UPDATE * e) for t, e in zip(self.t_params, self.e_params)]

        self.td_error, self.loss = self._compute_loss_graph(self.q_input, self.q, 'Critic')
        self.trian_op = self._create_train_op_graph()
        # self.action_grad = self._compute_action_grad(self.q, self.action_input)

    def _setup_placeholders_graph(self):
        # s
        self.state_input = tf.placeholder("float", shape=self.statedim, name='state_input')  # 全局状态
        self.agents_local_observation = tf.placeholder("float", shape=[None, self.agents_number, config.COOP_AGENTS_OBDIM], name='agents_local_observation')

        # a
        self.action_input = tf.placeholder("float", shape=[None, self.agents_number, self.action_dim], name='action_input')

        # s_
        self.state_input_next = tf.placeholder("float", shape=self.statedim, name='state_input_next')  # 全局状态
        self.agents_local_observation_next = tf.placeholder("float", shape=[None, self.agents_number, config.COOP_AGENTS_OBDIM], name='agents_local_observation_next')

        # a_
        self.action_input_next = tf.placeholder("float", shape=[None, self.agents_number, self.action_dim], name='action_input_next')

        # q_input
        self.q_input = tf.placeholder("float", shape=[None, self.agents_number], name='q_input')

    def _build_graph(self, state_input, agents_local_observation, action_input, scope_name, train):
        # 环境和智能体本地的共同观察
        with tf.variable_scope(scope_name, reuse=tf.AUTO_REUSE):
            encoder_outputs = self._observation_encoder(state_input, agents_local_observation, action_input, config.MY_UNIT_NUMBER, '_observation_encoder', train)
            bicnet_outputs = self._bicnet_build(encoder_outputs, '_bicnet_build', train)
            q_out = self._get_Q(bicnet_outputs, action_input, '_get_Q')
            return q_out

    def _observation_encoder(self, state_input, agents_local_observation, action_input, agents_number, scope_name, train):
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

    def _bicnet_build(self, encoder_outputs, scope_name, train):
        with tf.variable_scope(scope_name, reuse=tf.AUTO_REUSE):
            lstm_fw_cell = tf.nn.rnn_cell.GRUCell(self.action_dim, name="lstm_fw_cell")
            lstm_bw_cell = tf.nn.rnn_cell.GRUCell(self.action_dim, name="lstm_bw_cell")
            bicnet_outputs, _, _ = tf.nn.static_bidirectional_rnn(lstm_fw_cell, lstm_bw_cell, encoder_outputs, dtype=tf.float32)
            fc1 = slim.fully_connected(bicnet_outputs, self.action_dim, scope='full_connected1')
            bicnet_outputs = tf.nn.softmax(fc1, axis=2)

            bicnet_outputs = tf.unstack(bicnet_outputs, self.agents_number)  # (agents_number, batch_size, action_dim)
            bicnet_outputs = tf.transpose(bicnet_outputs, [1, 0, 2])
            return bicnet_outputs  # (batch_size,agents_number,action_dim)

    def _get_Q(self, bicnet_outputs, action_input, scope_name):
        with tf.variable_scope(scope_name, reuse=tf.AUTO_REUSE):
            q_out = tf.multiply(bicnet_outputs, action_input)  # (batch_size, agents_number,outputs_prob)
            q_out = tf.reduce_sum(q_out, axis=2)  # (batch_size, agents_number)

            return q_out

    def _soft_replace(self):
        self.soft_replace = [tf.assign(t, (1 - config.GAMMA_FOR_UPDATE) * t + config.GAMMA_FOR_UPDATE * e) for t, e in zip(self.t_params, self.e_params)]

    def _compute_loss_graph(self, qin, qout, scope_name):
        with tf.name_scope(scope_name + "_compute_loss_graph"):
            td_error = qin - qout
            loss = tf.reduce_mean(tf.square(td_error))


            # loss = tf.squared_difference(qin, qout)
            # loss = tf.reduce_mean(loss)
            return td_error, loss

    def _create_train_op_graph(self):
        train_op = tf.train.AdamOptimizer(self.learning_rate).minimize(self.loss, var_list=self.e_params)
        return train_op

    # def _compute_action_grad(self, qout, action_input):
    #     action_grads = [tf.gradients(qout[:, i], action_input) for i in range(self.agents_number)]  # (batch_size,agent_number,agent_number,action_dim)
    #     action_grads = tf.reshape(action_grads, [self.agents_number   , -1, self.agents_number, self.action_dim])
    #     return action_grads
