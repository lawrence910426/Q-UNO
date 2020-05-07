from Brains.Brain import Brain
import tensorflow as tf
import numpy as np
import datetime
from NN_Translate import NNTranslate
import threading


class RLBrain:
    def __init__(
        self,
        session=None,
        features=NNTranslate.features,
        actions=NNTranslate.actions,
        gamma=0.9, memory_size=3000,
        eplison=0.9, batch_size=1024
    ):
        self.features, self.actions, self.gamma, self.eplison = features, actions, gamma, eplison
        self.learn_step, self.memory_size, self.memory_count, self.batch_size = 0, memory_size, 0, batch_size
        self.memory = {
            "s": np.zeros([self.memory_size, self.features]), "s_": np.zeros([self.memory_size, self.features]),
            "a": np.zeros([self.memory_size, 1]), "r": np.zeros([self.memory_size, 1])
        }
        self.session = tf.Session() if session is None else session
        self._build_net()
        self.writer = tf.summary.FileWriter("logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S"), self.session.graph)
        self.summary = tf.summary.scalar(name='loss', tensor=self.loss)
        # threading.Thread(target=self.learn, daemon=True).start()

    def _build_net(self):
        self.s = tf.placeholder(tf.float64, shape=(None, self.features))
        self.s_ = tf.placeholder(tf.float64, shape=(None, self.features))
        self.r = tf.placeholder(tf.float64, shape=(None, ))
        self.a = tf.placeholder(tf.int32, shape=(None, ))
        w_init, b_init = tf.random_normal_initializer(0.0, 0.3), tf.constant_initializer(0.1)

        with tf.variable_scope('eval_net'):
            layer = tf.layers.dense(self.s, 30, tf.nn.leaky_relu, True,
                                    kernel_initializer=w_init, bias_initializer=b_init)
            self.eval_out = tf.layers.dense(layer, self.actions, tf.nn.sigmoid, True,
                                            kernel_initializer=w_init, bias_initializer=b_init, name='eval_net')
        with tf.variable_scope('target_net'):
            layer = tf.layers.dense(self.s_, 30, tf.nn.leaky_relu, True,
                                    kernel_initializer=w_init, bias_initializer=b_init)
            self.target_out = tf.layers.dense(layer, self.actions, tf.nn.sigmoid, True,
                                              kernel_initializer=w_init, bias_initializer=b_init, name='target_net')
        with tf.variable_scope('q_eval'):
            indices = tf.stack([tf.range(tf.shape(self.a)[0], dtype=tf.int32), self.a], axis=1)
            self.q_eval = tf.gather_nd(self.eval_out, indices)
        with tf.variable_scope('q_target'):
            self.q_target = self.r + self.gamma * tf.reduce_max(self.target_out)
            self.q_target = tf.stop_gradient(self.q_target, name='q_target')
        with tf.variable_scope('train'):
            self.loss = tf.reduce_min(tf.squared_difference(self.q_target, self.q_eval), name='loss')
            self.trainer = tf.train.RMSPropOptimizer(0.01).minimize(self.loss)
        t_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='target_net')
        e_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='eval_net')
        with tf.variable_scope('swap'):
            self.swap = [tf.assign(t, e) for t, e in zip(t_params, e_params)]
        self.session.run(tf.global_variables_initializer())

    def get(self, status):
        available = Brain.get_available(status)
        status_nn = NNTranslate.state_to_nn(status)[np.newaxis, :]
        if np.random.uniform() < self.eplison:
            action_values = self.session.run([self.eval_out], feed_dict={self.s: status_nn})
            action_values = np.array(action_values) + 1
        else:
            action_values = 1 / (1 + np.exp(-np.random.normal(size=self.actions)))
            action_values = action_values + 1
        action_values = action_values * NNTranslate.get_available_mask(available)
        action = np.argmax(action_values)
        return NNTranslate.nn_to_state(action)

    def add_observation(self, s, a, r, s_):
        self.memory["s"][self.memory_count % self.memory_size, :] = NNTranslate.state_to_nn(s)[np.newaxis, :]
        self.memory["a"][self.memory_count % self.memory_size, :] = NNTranslate.card_to_nn(a)
        self.memory["r"][self.memory_count % self.memory_size, :] = r
        self.memory["s_"][self.memory_count % self.memory_size, :] = NNTranslate.state_to_nn(s_)[np.newaxis, :]
        self.memory_count += 1
        self.learn()

    def learn(self):
        self.session.run([self.swap])
        batch = np.random.choice(min(self.memory_size, self.memory_count), size=self.batch_size)
        _, loss, summary = self.session.run(
            [self.trainer, self.loss, self.summary],
            feed_dict={
                self.s: self.memory["s"][batch, :],
                self.a: self.memory["a"][batch, 0],
                self.r: self.memory["r"][batch, 0],
                self.s_: self.memory["s_"][batch, :]
            }
        )
        self.writer.add_summary(summary, self.learn_step)
        self.learn_step += 1
