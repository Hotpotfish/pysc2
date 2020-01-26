import tensorflow as tf
from pysc2.agents.myAgent.myAgent_12.config import config
from tensorflow_core.contrib import slim


class ddpg(object):

    def __init__(self, mu, sigma, learning_rate, action_dim, statedim, name):  # 初始化
        # 神经网络参数
        self.mu = mu
        self.sigma = sigma
        self.learning_rate = learning_rate

        # 动作维度数，动作参数维度数,状态维度数
        self.action_dim = action_dim
        self.state_dim = statedim

        self.name = name

        self._setup_placeholders_graph()
        with tf.variable_scope(self.name+'_Actor'):
            self.a = self._build_graph_a(self.state_input, 'eval', train=True)
            a_ = self._build_graph_a(self.state_input_next, 'target', train=False)

        with tf.variable_scope(self.name+'_Critic'):
            q = self._build_graph_c(self.state_input, self.action_input, 'eval', train=True)
            q_ = self._build_graph_c(self.state_input_next, a_, 'target', train=False)

        # networks parameters
        self.ae_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=self.name+'_Actor/eval')
        self.at_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=self.name+'_Actor/target')
        self.ce_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=self.name+'_Critic/eval')
        self.ct_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=self.name+'_Critic/target')

        # target net replacement
        self.soft_replace = [tf.assign(t, (1 - config.GAMMA_FOR_UPDATE) * t + config.GAMMA_FOR_UPDATE * e)
                             for t, e in zip(self.at_params + self.ct_params, self.ae_params + self.ce_params)]

        q_target = self.reward + config.GAMMA * q_

        self.td_error = tf.losses.mean_squared_error(labels=q_target, predictions=q)
        self.ctrain = tf.train.AdamOptimizer(self.learning_rate).minimize(self.td_error, var_list=self.ce_params)
        q_temp = self._build_graph_c(self.state_input, self.a, self.name+'_Critic/eval', train=False)
        self.a_loss = - tf.reduce_mean(q_temp)  # maximize the q
        self.atrain = tf.train.AdamOptimizer(self.learning_rate).minimize(self.a_loss, var_list=self.ae_params)

    def _setup_placeholders_graph(self):
        self.state_input = tf.placeholder("float", shape=self.state_dim, name='state_input')  # 全局状态
        self.action_input = tf.placeholder("float", shape=[None, self.action_dim], name='action_input')  # 全局状态
        self.reward = tf.placeholder("float", shape=[None], name='reward')
        self.state_input_next = tf.placeholder("float", shape=self.state_dim, name='state_input_next')  # 全局状态

    def _build_graph_a(self, state_input, scope_name, train):
        # 环境和智能体本地的共同观察
        with tf.variable_scope(scope_name, reuse=tf.AUTO_REUSE):
            with slim.arg_scope([slim.conv2d, slim.fully_connected],
                                trainable=train,
                                weights_initializer=tf.truncated_normal_initializer(stddev=0.1),
                                weights_regularizer=slim.l2_regularizer(0.05)):
                fc1 = slim.fully_connected(state_input, 100, scope='full_connected1')
                relu1 = tf.nn.relu(fc1)
                fc2 = slim.fully_connected(relu1, 100, scope='full_connected2')
                relu2 = tf.nn.relu(fc2)
                fc3 = slim.fully_connected(relu2, self.action_dim, scope='full_connected3')
                output = tf.nn.softmax(fc3)
                return output

        #################################### critic_net  ####################################

    def _build_graph_c(self, state_input, action, scope_name, train):
        # 环境和智能体本地的共同观察
        with tf.variable_scope(scope_name, reuse=tf.AUTO_REUSE):
            with slim.arg_scope([slim.conv2d, slim.fully_connected],
                                trainable=train,
                                weights_initializer=tf.truncated_normal_initializer(stddev=0.1),
                                weights_regularizer=slim.l2_regularizer(0.05)):
                fc1_s = slim.fully_connected(state_input, 100, scope='full_connected1_s')
                fc1_a = slim.fully_connected(action, 100, scope='full_connected1_a')
                concat = tf.concat([fc1_s, fc1_a], axis=1)

                fc1 = slim.fully_connected(concat, 100, scope='full_connected1')
                output = slim.fully_connected(fc1, 1, scope='full_connected2')

                return output
