from Brains.Brain import Brain
import tensorflow as tf
import numpy as np
import datetime
from NN_Translate import NNTranslate


class GeneBrain(Brain):
    def __init__(
        self,
        features=NNTranslate.features,
        actions=NNTranslate.actions
    ):
        self.features, self.actions = features, actions
        self.session = tf.Session()
        self._build_net()

    def _build_net(self):
        self.state = tf.placeholder(tf.float64, shape=(None, self.features))
        w_init, b_init = tf.random_normal_initializer(0.0, 0.3), tf.constant_initializer(0.1)
        hidden = tf.layers.dense(self.state, 30, tf.nn.leaky_relu, True,
                                 kernel_initializer=w_init, bias_initializer=b_init)
        self.output = tf.layers.dense(hidden, self.actions, kernel_initializer=w_init)
        self.session.run(tf.global_variables_initializer())

    def get(self, status):
        status = status[np.newaxis, :]
        if np.random.uniform() < self.eplison:
            action_values = self.session.run([self.eval_out], feed_dict={self.s: status})
            action = np.argmax(action_values)
        else:
            action = np.random.randint(0, self.actions)
        return action

    def add_observation(self, s, a, r, s_):
        return

    def learn(self):
        return

