import tensorflow as tf
import tensorflow.contrib.slim as slim


class resNet():

    def __init__(self, mu, sigma):
        self.mu = mu
        self.sigma = sigma
        self._build_graph()

    def _build_graph(self):
        self._setup_placeholders_graph()
        self._build_network_graph()
        self._compute_loss_graph()
        self._compute_acc_graph()
        self._create_train_op_graph()
        self.merged_summary = tf.summary.merge_all()

    def _setup_placeholders_graph(self):
        self.x = tf.placeholder("float", shape=[None, 224, 224, 3], name='x')
        self.y = tf.placeholder("float", shape=[None, 101], name='y')

    def _build_network_graph(self):
        with slim.arg_scope([slim.conv2d],
                            activation_fn=tf.nn.relu,
                            padding='SAME',
                            weights_initializer=tf.truncated_normal_initializer(self.mu, self.sigma),
                            weights_regularizer=slim.l2_regularizer(0.0005)):
            # 112 * 112 * 64
            net = slim.conv2d(self.x, 64, [7, 7], stride=2, scope='conv1')

            # 56 * 56 * 64
            net = slim.max_pool2d(net, [3, 3], stride=2, scope='pool1')

            temp = net

            # 第一残差块
            net = slim.conv2d(net, 64, [3, 3], scope='conv2_1_1')
            net = slim.conv2d(net, 64, [3, 3], scope='conv2_1_2')
            # 残差相加
            net = tf.nn.relu(tf.add(temp, net))

            temp = net
            # 残差块
            net = slim.conv2d(net, 64, [3, 3], scope='conv2_2_1')
            net = slim.conv2d(net, 64, [3, 3], scope='conv2_2_2')
            # 残差相加
            net = tf.nn.relu(tf.add(temp, net))

            temp = net
            # 28 * 28 * 128
            temp = slim.conv2d(temp, 128, [1, 1], stride=2, scope='r1')

            # 第二残差块
            net = slim.conv2d(net, 128, [3, 3], stride=2, scope='conv3_1_1')
            net = slim.conv2d(net, 128, [3, 3], scope='conv3_1_2')
            # 残差相加
            net = tf.nn.relu(tf.add(temp, net))

            temp = net
            # 残差块
            net = slim.conv2d(net, 128, [3, 3], scope='conv3_2_1')
            net = slim.conv2d(net, 128, [3, 3], scope='conv3_2_2')
            # 残差相加
            net = tf.nn.relu(tf.add(temp, net))

            temp = net
            # 14 * 14 * 256
            temp = slim.conv2d(temp, 256, [1, 1], stride=2, scope='r2')

            # 第三残差块
            net = slim.conv2d(net, 256, [3, 3], stride=2, scope='conv4_1_1')
            net = slim.conv2d(net, 256, [3, 3], scope='conv4_1_2')
            # 残差相加
            net = tf.nn.relu(tf.add(temp, net))

            temp = net
            # 残差块
            net = slim.conv2d(net, 256, [3, 3], scope='conv4_2_1')
            net = slim.conv2d(net, 256, [3, 3], scope='conv4_2_2')
            # 残差相加
            net = tf.nn.relu(tf.add(temp, net))

            temp = net
            # 7 * 7 * 512
            temp = slim.conv2d(temp, 512, [1, 1], stride=2, scope='r3')

            # 第四残差块
            net = slim.conv2d(net, 512, [3, 3], stride=2, scope='conv5_1_1')
            net = slim.conv2d(net, 512, [3, 3], scope='conv5_1_2')
            # 残差相加
            net = tf.nn.relu(tf.add(temp, net))

            temp = net
            # 残差块
            net = slim.conv2d(net, 512, [3, 3], scope='conv5_2_1')
            net = slim.conv2d(net, 512, [3, 3], scope='conv5_2_2')
            # 残差相加
            net = tf.nn.relu(tf.add(temp, net))

            net = slim.avg_pool2d(net, [7, 7], stride=1, scope='pool2')

            net = slim.flatten(net, scope='flatten')
            fc1 = slim.fully_connected(net, 1000, scope='fc1')

            self.logits = slim.fully_connected(fc1, 101, activation_fn=None, scope='fc2')
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
        self.train_op = tf.train.AdamOptimizer(learning_rate=1e-4).minimize(self.loss)
