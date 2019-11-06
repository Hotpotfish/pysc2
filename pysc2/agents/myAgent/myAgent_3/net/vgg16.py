import pysc2.agents.myAgent.myAgent_3.macro_operation as mo
import tensorflow as tf


class VGG16():

    def __init__(self, mu, sigma, layerSize, outSize):
        self.mu = mu
        self.sigma = sigma
        self.layerSize = layerSize
        self.outSize = outSize

        self._build_graph()

    def _build_graph(self, network_name='VGG16'):
        self._setup_placeholders_graph()
        self._build_network_graph(network_name)
        self._compute_loss_graph()
        self._compute_acc_graph()
        self._create_train_op_graph()
        self.merged_summary = tf.summary.merge_all()

    def _setup_placeholders_graph(self):
        self.x = tf.placeholder("float", shape=[None, self.layerSize, self.layerSize, 1], name='x')
        self.y = tf.placeholder("float", shape=[None, self.outSize], name='y')

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
        with tf.variable_scope(scope_name, reuse=tf.AUTO_REUSE):
            self.conv1_1 = tf.nn.relu(
                self._cnn_layer('layer_1_1_conv', 'conv_w', 'conv_b', self.x, (3, 3, 3, 64), [1, 1, 1, 1],
                                padding_tag='SAME'))

            self.conv1_2 = tf.nn.relu(
                self._cnn_layer('layer_1_2_conv', 'conv_w', 'conv_b', self.conv1_1, (3, 3, 64, 64), [1, 1, 1, 1],
                                padding_tag='SAME'))

            self.pool1 = self._pooling_layer('layer_1_pooling', self.conv1_2, [1, 2, 2, 1], [1, 2, 2, 1])

            self.conv2_1 = tf.nn.relu(
                self._cnn_layer('layer_2_1_conv', 'conv_w', 'conv_b', self.pool1, (3, 3, 64, 128), [1, 1, 1, 1],
                                padding_tag='SAME'))

            self.conv2_2 = tf.nn.relu(
                self._cnn_layer('layer_2_2_conv', 'conv_w', 'conv_b', self.conv2_1, (3, 3, 128, 128), [1, 1, 1, 1],
                                padding_tag='SAME'))

            self.pool2 = self._pooling_layer('layer_2_pooling', self.conv2_2, [1, 2, 2, 1], [1, 2, 2, 1])

            self.conv3_1 = tf.nn.relu(
                self._cnn_layer('layer_3_1_conv', 'conv_w', 'conv_b', self.pool2, (3, 3, 128, 256), [1, 1, 1, 1],
                                padding_tag='SAME'))
            self.conv3_2 = tf.nn.relu(
                self._cnn_layer('layer_3_2_conv', 'conv_w', 'conv_b', self.conv3_1, (3, 3, 256, 256), [1, 1, 1, 1],
                                padding_tag='SAME'))
            self.conv3_3 = tf.nn.relu(
                self._cnn_layer('layer_3_3_conv', 'conv_w', 'conv_b', self.conv3_2, (3, 3, 256, 256), [1, 1, 1, 1],
                                padding_tag='SAME'))

            self.pool3 = self._pooling_layer('layer_3_pooling', self.conv3_3, [1, 2, 2, 1], [1, 2, 2, 1])

            self.conv4_1 = tf.nn.relu(
                self._cnn_layer('layer_4_1_conv', 'conv_w', 'conv_b', self.pool3, (3, 3, 256, 512), [1, 1, 1, 1],
                                padding_tag='SAME'))
            self.conv4_2 = tf.nn.relu(
                self._cnn_layer('layer_4_2_conv', 'conv_w', 'conv_b', self.conv4_1, (3, 3, 512, 512), [1, 1, 1, 1],
                                padding_tag='SAME'))
            self.conv4_3 = tf.nn.relu(
                self._cnn_layer('layer_4_3_conv', 'conv_w', 'conv_b', self.conv4_2, (3, 3, 512, 512), [1, 1, 1, 1],
                                padding_tag='SAME'))

            self.pool4 = self._pooling_layer('layer_4_pooling', self.conv4_3, [1, 2, 2, 1], [1, 2, 2, 1])

            self.conv5_1 = tf.nn.relu(
                self._cnn_layer('layer_5_1_conv', 'conv_w', 'conv_b', self.pool4, (3, 3, 512, 512), [1, 1, 1, 1],
                                padding_tag='SAME'))
            self.conv5_2 = tf.nn.relu(
                self._cnn_layer('layer_5_2_conv', 'conv_w', 'conv_b', self.conv5_1, (3, 3, 512, 512), [1, 1, 1, 1],
                                padding_tag='SAME'))

            self.conv5_3 = tf.nn.relu(
                self._cnn_layer('layer_5_3_conv', 'conv_w', 'conv_b', self.conv5_2, (3, 3, 512, 512), [1, 1, 1, 1],
                                padding_tag='SAME'))

            self.pool5 = self._pooling_layer('layer_5_pooling', self.conv5_3, [1, 2, 2, 1], [1, 2, 2, 1])

            self.fc6 = tf.nn.relu(self._fully_connected_layer('full_connected6', 'full_connected_w', 'full_connected_b',
                                                              self.pool5, (512 * 7 * 7, 4096)))
            self.dropOut1 = tf.nn.dropout(self.fc6, 0.5)

            self.fc7 = tf.nn.relu(self._fully_connected_layer('full_connected7', 'full_connected_w', 'full_connected_b',
                                                              self.dropOut1, (4096, 4096)))
            self.dropOut2 = tf.nn.dropout(self.fc7, 0.5)

            self.logits = self._fully_connected_layer('full_connected8', 'full_connected_w', 'full_connected_b',
                                                      self.dropOut2, (4096, self.outSize))

            self.y_predicted = tf.nn.softmax(self.logits)

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
        self.train_op = tf.train.AdamOptimizer(learning_rate=0.1).minimize(self.loss)
