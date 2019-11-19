import tensorflow as tf


class alexNet():

    def __init__(self, mu, sigma):
        self.mu = mu
        self.sigma = sigma
        self._build_graph()

    def _build_graph(self, network_name='alexNet'):
        self._setup_placeholders_graph()
        self._build_network_graph(network_name)
        self._compute_loss_graph()
        self._compute_acc_graph()
        self._create_train_op_graph()
        self.merged_summary = tf.summary.merge_all()

    def _setup_placeholders_graph(self):
        self.x = tf.placeholder("float", shape=[None, 227, 227, 3], name='x')
        self.y = tf.placeholder("float", shape=[None, 101], name='y')

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
        with tf.variable_scope(scope_name):
            # 55 * 55 * 96
            self.conv1 = tf.nn.relu(
                self._cnn_layer('layer_1_conv', 'conv_w', 'conv_b', self.x, (11, 11, 3, 96), [1, 4, 4, 1]))
            self.lrn1 = tf.nn.lrn(self.conv1, 2, 2e-05, 0.75, name='layer_1_lrn')
            self.pool1 = self._pooling_layer('layer_1_pooling', self.lrn1, [1, 3, 3, 1], [1, 2, 2, 1])

            self.conv2 = tf.nn.relu(self._cnn_layer('layer_2_conv', 'conv_w', 'conv_b', self.pool1, (5, 5, 96, 256), [1, 1, 1, 1]))
            self.lrn2 = tf.nn.lrn(self.conv2, 2, 2e-05, 0.75, name='layer_2_lrn')
            self.pool2 = self._pooling_layer('layer_2_pooling', self.lrn2, [1, 3, 3, 1], [1, 2, 2, 1])

            self.conv3 = tf.nn.relu(self._cnn_layer('layer_3_conv', 'conv_w', 'conv_b', self.pool2, (3, 3, 256, 384), [1, 1, 1, 1]))

            self.conv4 = tf.nn.relu(self._cnn_layer('layer_4_conv', 'conv_w', 'conv_b', self.conv3, (3, 3, 384, 384), [1, 1, 1, 1]))

            self.conv5 = tf.nn.relu(self._cnn_layer('layer_5_conv', 'conv_w', 'conv_b', self.conv4, (3, 3, 384, 256), [1, 1, 1, 1]))

            self.pool3 = self._pooling_layer('layer_2_pooling', self.conv5, [1, 3, 3, 1], [1, 2, 2, 1])

            # w.shape=[5 * 5 * 16, 120]
            self.fc1 = tf.nn.relu(self._fully_connected_layer('full_connected1', 'full_connected_w', 'full_connected_b',
                                                              self.pool3, (2 * 2 * 256, 4096)))

            # w.shape=[120, 84]
            self.fc2 = tf.nn.relu(self._fully_connected_layer('full_connected2', 'full_connected_w',
                                                              'full_connected_b',
                                                              self.fc1, (4096, 4096)))

            self.fc3 = tf.nn.relu(self._fully_connected_layer('full_connected3', 'full_connected_w',
                                                              'full_connected_b',
                                                              self.fc2, (4096, 1000)))
            # w.shape=[84, 10]
            self.logits = self._fully_connected_layer('full_connected4', 'full_connected_w', 'full_connected_b',
                                                      self.fc3, (1000, 101))

            self.y_predicted = tf.nn.softmax(self.logits)
            # tf.summary.histogram("y_predicted", self.y_predicted)

    def _compute_loss_graph(self):
        with tf.name_scope("loss_function"):
            loss = tf.nn.softmax_cross_entropy_with_logits_v2(labels=self.y, logits=self.logits)
            self.loss = tf.reduce_mean(loss)
            tf.summary.scalar("loss", self.loss)

    def _compute_acc_graph(self):
        with tf.name_scope("acc_function"):
            self.accuracy = \
                tf.metrics.accuracy(labels=tf.argmax(self.y, axis=1), predictions=tf.argmax(self.y_predicted, axis=1))[
                    1]
            tf.summary.scalar("accuracy", self.accuracy)

    def _create_train_op_graph(self):
        self.train_op = tf.train.AdamOptimizer(learning_rate=0.001).minimize(self.loss)
