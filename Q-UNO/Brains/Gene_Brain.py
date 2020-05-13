from Brains.Brain import Brain
import tensorflow as tf
import numpy as np
import datetime
from NN_Translate import NNTranslate


class GeneBrain(Brain):
    def __init__(
        self,
        session, id,
        features=NNTranslate.features,
        actions=NNTranslate.actions
    ):
        self.features, self.actions = features, actions
        self.session, self.id = session, id
        self.layers = []
        self._build_net()

    def _build_net(self):
        self.state = tf.placeholder(tf.float64, shape=(1, self.features))
        w_init, b_init = tf.random_normal_initializer(0.0, 0.3), tf.constant_initializer(0.1)
        with tf.variable_scope('network_' + str(self.id)):
            self.layers.append(tf.layers.dense(self.state, 30, tf.nn.leaky_relu, True,
                                               kernel_initializer=w_init, bias_initializer=b_init))
            # self.layers.append(tf.layers.dense(self.state, 40, tf.nn.leaky_relu, True,
            #                                    kernel_initializer=w_init, bias_initializer=b_init))
            # self.layers.append(tf.layers.dense(self.state, 30, tf.nn.leaky_relu, True,
            #                                    kernel_initializer=w_init, bias_initializer=b_init))
            self.layers.append(tf.layers.dense(self.layers[-1], self.actions, tf.nn.sigmoid, True,
                                               kernel_initializer=w_init, bias_initializer=b_init))
        network_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='network_' + str(self.id))

        self.suicide = [tf.assign(var, tf.random.normal(var.shape, dtype=tf.float64)) for var in network_vars]
        self.mutation = [tf.assign_add(var, tf.random.normal(var.shape, dtype=tf.float64)) for var in network_vars]
        self.session.run(tf.global_variables_initializer())

    def get(self, status):
        available = Brain.get_available(status)
        status_nn = NNTranslate.state_to_nn(status)[np.newaxis, :]
        action_values = self.session.run([self.layers[-1]], feed_dict={self.state: status_nn})
        action_values = np.array(action_values) + 1
        action_values = action_values * NNTranslate.get_available_mask(available)
        return NNTranslate.nn_to_state(np.argmax(action_values))

    def add_observation(self, s, a, r, s_, batch_size):
        return

    def learn(self):
        return

    def mutate(self):
        self.session.run(self.mutation)

    def reset(self):
        self.session.run(self.suicide)

    def fertilization(self, mate_id):
        self_param = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='network_' + str(self.id))
        mate_param = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='network_' + str(mate_id))
        fertilize = [tf.assign(s, tf.div(tf.add(s, m), 2)) for s, m in zip(self_param, mate_param)]
        self.session.run(fertilize)
