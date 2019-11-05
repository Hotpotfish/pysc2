import pysc2.agents.myAgent.myAgent_3.net.vgg16 as vgg16
import pysc2.agents.myAgent.myAgent_3.smart_actions as sa
import tensorflow as tf


class First():

    def __init__(self, mu, sigma, data):
        self.net = vgg16(mu, sigma, len(data[0]), len(sa.controllers))
        self.data = data

    def train(self):
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer(),
                     sess.run(tf.local_variables_initializer())
                     )
