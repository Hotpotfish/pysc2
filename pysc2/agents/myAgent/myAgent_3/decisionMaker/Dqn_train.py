import pysc2.agents.myAgent.myAgent_3.net.vgg16 as vgg16
import pysc2.agents.myAgent.myAgent_3.smart_actions as sa
import tensorflow as tf
import pysc2.agents.myAgent.myAgent_3.macro_operation as mo


class Dqn_train():

    def __init__(self, mu, sigma, mapSize, channels):
        self.eval_net = vgg16(mu, sigma, mapSize, channels)
        self.target_net = vgg16(mu, sigma, mapSize, channels)
        self.data = data

    def train(self, gameLoop):
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer(),
                     sess.run(tf.local_variables_initializer())
                     )
            self.target_y_predicted = sess.run([self.target_net.y_predicted, ],
                                               {self.target_net.x: self.data,
                                                self.target.y: self.data, })

            if gameLoop % 100 == 0:
                self.target_net = self.eval_net
