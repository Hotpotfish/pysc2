import tensorflow as tf
import tensorflow.contrib.slim as slim




class Lenet():

    def __init__(self, mu, sigma, learning_rate, action_dim, statedim, name):
        self.mu = mu
        self.sigma = sigma
        self.learning_rate = learning_rate

        self.action_dim = action_dim
        self.statedim = statedim

        self.name = name

        self._build_graph()

    def _build_graph(self):
        self._setup_placeholders_graph()
        # self._build_action_network_graph(self.name + '_action')
        self._build_network_graph(self.name)
        self._compute_loss_graph()
        # self._compute_acc_graph()
        self._create_train_op_graph()
        self.merged_summary = tf.summary.merge_all()

    def _build_network_graph(self, name):
        self._action_network_graph(name + '_' + 'action')


    def _setup_placeholders_graph(self):
        self.action_input = tf.placeholder("float", shape=[None, self.action_dim ], name=self.name + '_' + 'action_input')
        self.y_input = tf.placeholder("float", shape=[None, 1 ], name=self.name + '_' + 'y_input')
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

                self.Q_value = slim.fully_connected(fc2, self.action_dim, activation_fn=tf.nn.softmax, scope='Q_value')

    def _compute_loss_graph(self):
        with tf.name_scope(self.name + "_loss_function"):
            self.Q_action = tf.reduce_sum(tf.multiply(self.Q_value, self.action_input))
            self.loss = tf.reduce_mean(tf.square(self.y_input - self.Q_action))


    def _create_train_op_graph(self):
        self.train_op = tf.train.AdamOptimizer(self.learning_rate).minimize(self.loss)
