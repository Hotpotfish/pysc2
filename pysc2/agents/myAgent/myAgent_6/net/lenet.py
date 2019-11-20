import tensorflow as tf


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
        # self._compute_acc_graph()
        self._create_train_op_graph()
        # self.merged_summary = tf.summary.merge_all()

    def _setup_placeholders_graph(self):
        self.action_input = tf.placeholder("float", shape=[None, self.action_dim + self.parameterdim], name='action_input')
        self.y_input = tf.placeholder("float", shape=[None, 1 + self.parameterdim], name='y_input')
        self.state_input = tf.placeholder("float", shape=self.statedim, name='state_input')

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

    def _build_network_graph(self, scope_name):
        with tf.variable_scope(scope_name,reuse=tf.AUTO_REUSE):
            # 28 * 28 * 6
            self.conv1 = self._cnn_layer('layer_1_conv', 'conv_w', 'conv_b', self.state_input, (5, 5, self.statedim[3], 6), [1, 1, 1, 1])
            # 14 * 14 * 6
            self.pool1 = self._pooling_layer('layer_1_pooling', self.conv1, [1, 2, 2, 1], [1, 2, 2, 1])

            # 10 * 10 * 16
            self.conv2 = self._cnn_layer('layer_2_conv', 'conv_w', 'conv_b', self.pool1, (5, 5, 6, 16), [1, 1, 1, 1])

            # 5 * 5 * 16
            self.pool2 = self._pooling_layer('layer_2_pooling', self.conv2, [1, 2, 2, 1], [1, 2, 2, 1])

            # w.shape=[5 * 5 * 16, 120]
            self.fc1 = self._fully_connected_layer('full_connected1', 'full_connected_w', 'full_connected_b',
                                                   self.pool2, (self.pool2._shape[1] * self.pool2._shape[2] * self.pool2._shape[3], 120))

            # w.shape=[120, 84]
            self.fc2 = self._fully_connected_layer('full_connected2', 'full_connected_w',
                                                   'full_connected_b',
                                                   self.fc1, (120, 84))
            # w.shape=[84, 10]
            self.logits = self._fully_connected_layer('full_connected3', 'full_connected_w', 'full_connected_b',
                                                      self.fc2, (84, self.action_dim + self.parameterdim))

            self.Q_value = tf.nn.softmax(self.logits)
            # tf.summary.histogram("y_predicted", self.y_predicted)

    def _compute_loss_graph(self):
        with tf.name_scope(self.name + "_loss_function"):
            self.Q_action = tf.reduce_sum(tf.multiply(self.Q_value, self.action_input))
            self.loss = tf.reduce_mean(tf.square(self.y_input - self.Q_action))
            tf.summary.scalar("loss", self.loss)

    def _compute_acc_graph(self):
        with tf.name_scope(self.name + "_acc_function"):
            self.accuracy = \
                tf.metrics.accuracy(labels=tf.argmax(self.y, axis=1), predictions=tf.argmax(self.y_predicted, axis=1))[
                    1]
            tf.summary.scalar("accuracy", self.accuracy)

    def _create_train_op_graph(self):
        self.train_op = tf.train.AdamOptimizer(self.learning_rate).minimize(self.loss)
