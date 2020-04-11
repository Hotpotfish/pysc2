import tensorflow as tf
from pysc2.agents.myAgent.myAgent_15_BIC_DDPG_2.config import config
import tensorflow.contrib.slim as slim


class net1(object):

    def __init__(self, mu, sigma, learning_rate, action_dim, statedim, agents_number, enemy_number, name):  # 初始化
        # 神经网络参数
        self.mu = mu
        self.sigma = sigma
        self.learning_rate = learning_rate

        # 动作维度数，动作参数维度数,状态维度数
        self.action_dim = action_dim
        self.state_dim = statedim
        self.agents_number = agents_number
        self.enemy_number = enemy_number

        self.name = name

        self._setup_placeholders_graph()
        with tf.variable_scope('Actor'):
            self.a = self._build_graph_a(self.agents_local_observation, 'eval', train=True)
            self.a_ = self._build_graph_a(self.agents_local_observation_next, 'target', train=False)

        with tf.variable_scope('Critic'):
            self.q = self._build_graph_c(self.agents_local_observation, self.a, 'eval', train=True)
            self.q_ = self._build_graph_c(self.agents_local_observation_next, self.a_, 'target', train=False)

        # networks parameters
        self.ae_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='Actor/eval')
        self.at_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='Actor/target')
        self.ce_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='Critic/eval')
        self.ct_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='Critic/target')

        # target net replacement
        self.soft_replace = [tf.assign(t, (1 - config.TAU) * t + config.TAU * e)
                             for t, e in zip(self.at_params + self.ct_params, self.ae_params + self.ce_params)]

        q_target = self.reward + config.GAMMA * self.q_

        self.td_error = tf.losses.mean_squared_error(labels=q_target, predictions=self.q)
        self.ctrain = tf.train.AdamOptimizer(self.learning_rate).minimize(self.td_error, var_list=self.ce_params)

        self.a_loss = - tf.reduce_mean(self.q)  # maximize the q
        self.atrain = tf.train.AdamOptimizer(self.learning_rate).minimize(self.a_loss, var_list=self.ae_params)

    def _setup_placeholders_graph(self):
        # s
        # self.state_input = tf.placeholder("float", shape=self.state_dim, name='state_input')  # 全局状态
        self.agents_local_observation = tf.placeholder("float", shape=[None, self.agents_number, config.COOP_AGENTS_OBDIM], name='agents_local_observation')

        # s_
        # self.state_input_next = tf.placeholder("float", shape=self.state_dim, name='state_input_next')  # 全局状态
        self.agents_local_observation_next = tf.placeholder("float", shape=[None, self.agents_number, config.COOP_AGENTS_OBDIM], name='agents_local_observation_next')

        self.reward = tf.placeholder("float", shape=[None, 1], name='reward')

    def _build_graph_a(self, agents_local_observation, scope_name, train):
        # 环境和智能体本地的共同观察
        with tf.variable_scope(scope_name, reuse=tf.AUTO_REUSE):
            with slim.arg_scope([slim.fully_connected],
                                trainable=train,
                                # normalizer_fn=slim.batch_norm,
                                activation_fn=None):
                encoder_outputs = self._observation_encoder_a(agents_local_observation, self.agents_number, '_observation_encoder')
                bicnet_outputs = self._bicnet_build_a(encoder_outputs, self.agents_number, '_bicnet_build')
                return bicnet_outputs

    def _observation_encoder_a(self, agents_local_observation, agents_number, scope_name):
        with tf.variable_scope(scope_name, reuse=tf.AUTO_REUSE):
            encoder = []
            for i in range(agents_number):
                fc1 = slim.fully_connected(agents_local_observation[:, i, :], 100, scope='full_connected1' + '_agent_' + str(i))
                # fc2 = slim.fully_connected(fc1, 50, scope='full_connected2')
                encoder.append(fc1)
            encoder = tf.transpose(encoder, [1, 0, 2])
            encoder = tf.unstack(encoder, agents_number, 1)  # (self.agents_number,batch_size,obs_add_dim)
            return encoder

    def _bicnet_build_a(self, encoder_outputs, agents_number, scope_name):
        with tf.variable_scope(scope_name, reuse=tf.AUTO_REUSE):
            outputs = []
            lstm_fw_cell = tf.nn.rnn_cell.GRUCell(50, name="lstm_fw_cell")
            lstm_bw_cell = tf.nn.rnn_cell.GRUCell(50, name="lstm_bw_cell")
            bicnet_outputs, _, _ = tf.nn.static_bidirectional_rnn(lstm_fw_cell, lstm_bw_cell, encoder_outputs, dtype=tf.float32)
            for i in range(agents_number):
                fc1 = slim.fully_connected(bicnet_outputs[i], 25, scope='full_connected1' + '_agent_' + str(i))
                # fc1 = fc1 * 0.1
                # fc1 = tf.Print(fc1, [fc1])
                # bicnet_outputs[i] = bicnet_outputs[i] * 0.1
                # bicnet_outputs[i] = tf.Print(bicnet_outputs[i], [bicnet_outputs[i]])
                # fc1 = tf.Print(fc1, [fc1])
                # fc1 = fc1 * 1 / 4096
                fc1 = fc1 * 0.1
                fc2 = slim.fully_connected(fc1, self.action_dim, activation_fn=tf.nn.tanh, scope='full_connected2' + '_agent_' + str(i))
                # fc2 = fc2 / 2 + 1
                # fc2 = tf.Print(fc2, [fc2])

                outputs.append(fc2)

            outputs = tf.unstack(outputs, self.agents_number)  # (agents_number, batch_size, action_dim)
            outputs = tf.transpose(outputs, [1, 0, 2])
            # outputs = tf.clip_by_value(outputs, 0, 1)
            # outputs = tf.Print(outputs, [outputs])
            return outputs  # (batch_size,agents_number,action_dim)

        #################################### critic_net  ####################################

    def _build_graph_c(self, state_input, action_input, scope_name, train):
        # 环境和智能体本地的共同观察
        with tf.variable_scope(scope_name, reuse=tf.AUTO_REUSE):
            with slim.arg_scope([slim.fully_connected],
                                trainable=train,
                                # normalizer_fn=slim.batch_norm,

                                activation_fn=None):
                encoder_outputs = self._observation_encoder_c(state_input, action_input, self.agents_number, '_observation_encoder')
                bicnet_outputs = self._bicnet_build_c(encoder_outputs, self.agents_number, '_bicnet_build')
                return bicnet_outputs

    def _observation_encoder_c(self, state_input, action_input, agents_number, scope_name):
        with tf.variable_scope(scope_name, reuse=tf.AUTO_REUSE):
            encoder = []
            for i in range(agents_number):
                # state_input = tf.Print( state_input, [ state_input])
                input_data = tf.concat([state_input[:, i], action_input[:, i]], 1)
                # input_data = tf.Print(input_data, [input_data])
                fc1 = slim.fully_connected(input_data, 100, scope='full_connected1' + '_agent_' + str(i))
                # fc2 = slim.fully_connected(fc1, 50, scope='full_connected2')
                # fc1_s = slim.fully_connected(state_input, 100, scope='full_connected_s1')
                # fc1_a = slim.fully_connected(action_input[:, i], 100, scope='full_connected_a1')
                # fc2_a = slim.fully_connected(fc1_a, 200, scope='full_connected_a2')
                # data = fc1_s + fc1_a
                encoder.append(fc1)
            encoder = tf.transpose(encoder, [1, 0, 2])
            encoder = tf.unstack(encoder, agents_number, 1)  # (self.agents_number,batch_size,obs_add_dim)
            return encoder

    def _bicnet_build_c(self, encoder_outputs, agents_number, scope_name):
        with tf.variable_scope(scope_name, reuse=tf.AUTO_REUSE):
            outputs = []
            lstm_fw_cell = tf.nn.rnn_cell.GRUCell(50, name="lstm_fw_cell")
            lstm_bw_cell = tf.nn.rnn_cell.GRUCell(50, name="lstm_bw_cell")
            bicnet_outputs, _, _ = tf.nn.static_bidirectional_rnn(lstm_fw_cell, lstm_bw_cell, encoder_outputs, dtype=tf.float32)
            for i in range(agents_number):
                fc1 = slim.fully_connected(bicnet_outputs[i], 1, scope='full_connected1' + '_agent_' + str(i))
                # fc2 = slim.fully_connected(fc1, 1, scope='full_connected2')
                outputs.append(fc1)
            outputs = tf.unstack(outputs, self.agents_number)  # (agents_number, batch_size,1)
            outputs = tf.transpose(outputs, [1, 0, 2])  # (batch_size,agents_number,1)
            # outputs= tf.Print(outputs, [outputs])
            outputs = slim.flatten(outputs)

            fc3 = slim.fully_connected(outputs, 1, activation_fn=None, scope='full_connected3')
            # fc3 = tf.clip_by_value(fc3, 0, 1)
            # fc3 = tf.Print(fc3, [fc3])

            return  fc3
