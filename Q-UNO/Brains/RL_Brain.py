from Brains.Brain import Brain
import tensorflow as tf
import numpy as np
import datetime
from NN_Translate import NNTranslate
import threading


class RLBrain:
    def __init__(
        self,
        session, id,
        features=NNTranslate.features,
        actions=NNTranslate.actions,
        gamma=0.99, memory_size=5000,
        eplison=0.9, batch_size=256, **kwargs
    ):
        self.features, self.actions, self.gamma, self.eplison = features, actions, gamma, eplison
        self.learn_step, self.memory_size, self.memory_count, self.batch_size = 0, memory_size, 0, batch_size
        self.session, self.id = session, str(id)
        if "restore" in kwargs:
            self.memory = kwargs["memory"]
            self.memory_count = kwargs["memory_count"]
            self.suicide_vars = kwargs["suicide_vars"]
            self.swap_vars = kwargs["swap_vars"]
        else:
            self.memory = {
                "s": np.zeros([self.memory_size, self.features]), "s_": np.zeros([self.memory_size, self.features]),
                "a": np.zeros([self.memory_size, 1]), "r": np.zeros([self.memory_size, 1]),
                "m": np.zeros([self.memory_size, self.actions])
            }
            self._build_net()

    def _build_net(self):
        s = tf.placeholder(tf.float64, shape=(None, self.features), name='s' + self.id)
        s_ = tf.placeholder(tf.float64, shape=(None, self.features), name='s_' + self.id)
        r = tf.placeholder(tf.float64, shape=(None, ), name='r' + self.id)
        a = tf.placeholder(tf.int32, shape=(None, ), name='a' + self.id)
        mask = tf.placeholder(tf.float64, shape=(None, self.actions), name='m' + self.id)
        w_init, b_init = tf.random_normal_initializer(0.0, 0.01), tf.constant_initializer(0.01)

        def hidden(previous):
            previous = tf.layers.dense(previous, 30, tf.nn.leaky_relu, True,
                                       kernel_initializer=w_init, bias_initializer=b_init)
            previous = tf.layers.dense(previous, 40, tf.nn.leaky_relu, True,
                                       kernel_initializer=w_init, bias_initializer=b_init)
            previous = tf.layers.dense(previous, 50, tf.nn.leaky_relu, True,
                                       kernel_initializer=w_init, bias_initializer=b_init)
            previous = tf.layers.dense(previous, 30, tf.nn.leaky_relu, True,
                                       kernel_initializer=w_init, bias_initializer=b_init)
            return previous

        with tf.variable_scope('eval_net_' + self.id):
            eval_out = tf.layers.dense(hidden(s), self.actions, tf.nn.sigmoid, True,
                                       kernel_initializer=w_init, bias_initializer=b_init)
        with tf.variable_scope('target_net_' + self.id):
            target_out = tf.layers.dense(hidden(s_), self.actions, tf.nn.sigmoid, True,
                                         kernel_initializer=w_init, bias_initializer=b_init)
        tf.identity(eval_out, name='eval_' + self.id)
        # Q eval net #
        indices = tf.stack([tf.range(tf.shape(a)[0], dtype=tf.int32), a], axis=1)
        q_eval = tf.gather_nd(eval_out, indices)
        # Q target net #
        action_values = tf.add(target_out, 1)
        action_values = tf.multiply(action_values, mask)
        action_values = tf.add(action_values, -1)
        q_target = r + self.gamma * tf.reduce_max(action_values)
        q_target = tf.stop_gradient(q_target)
        # Training #
        loss = tf.reduce_min(tf.squared_difference(q_target, q_eval))
        tf.train.RMSPropOptimizer(0.01, name='trainer_' + self.id).minimize(loss)
        # Swap #
        self.swap_vars = 0
        t_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='target_net_' + self.id)
        e_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='eval_net_' + self.id)
        for t, e in zip(t_params, e_params):
            tf.assign(t, e, name='swap_' + self.id + "_" + str(self.swap_vars))
            self.swap_vars += 1
        # Suicide #
        self.suicide_vars = 0
        network_vars = []
        network_vars.extend([item for item in t_params])
        network_vars.extend([item for item in e_params])
        for var in network_vars:
            tf.assign(var, tf.zeros(var.shape, dtype=tf.float64),
                      name='suicide_' + self.id + "_" + str(self.suicide_vars))
            self.suicide_vars += 1

        self.session.run(tf.global_variables_initializer())

    def get(self, status):
        available = Brain.get_available(status)
        status_nn = NNTranslate.state_to_nn(status)[np.newaxis, :]
        if np.random.uniform() < self.eplison + self.learn_step * 1e-6:
            action_values = self.session.run(
                ['eval_' + self.id + ":0"],
                feed_dict={"s" + self.id + ":0": status_nn}
            )
            action_values = np.array(action_values) + 1
        else:
            action_values = 1 / (1 + np.exp(-np.random.normal(size=self.actions)))
            action_values = action_values + 1
        action_values = action_values * NNTranslate.get_available_mask(available)
        action = np.argmax(action_values)
        return NNTranslate.nn_to_state(action)

    def add_observation(self, s, a, r, s_, batch_size=None, learn=True):
        if batch_size is not None:
            self.batch_size = batch_size
        pos = self.memory_count % self.memory_size
        self.memory["s"][pos, :] = NNTranslate.state_to_nn(s)[np.newaxis, :]
        self.memory["a"][pos, :] = NNTranslate.card_to_nn(a)
        self.memory["r"][pos, :] = r
        self.memory["s_"][pos, :] = NNTranslate.state_to_nn(s_)[np.newaxis, :]
        self.memory["m"][pos, :] = NNTranslate.get_available_mask(Brain.get_available(s_))
        self.memory_count += 1
        if learn:
            threading.Thread(target=self.learn).start()

    def learn(self):
        self.session.run(['swap_' + self.id + "_" + str(i) for i in range(self.swap_vars)])
        batch = np.random.choice(min(self.memory_size, self.memory_count), size=self.batch_size)
        self.session.run(
            ["trainer_" + self.id],
            feed_dict={
                "s" + self.id + ":0": self.memory["s"][batch, :],
                "a" + self.id + ":0": self.memory["a"][batch, 0],
                "r" + self.id + ":0": self.memory["r"][batch, 0],
                "s_" + self.id + ":0": self.memory["s_"][batch, :],
                "m" + self.id + ":0": self.memory["m"][batch, :]
            }
        )
        self.learn_step += 1

    def reset(self):
        self.session.run(['suicide_' + self.id + "_" + str(i) for i in range(self.suicide_vars)])
        self.learn_step, self.memory_count = 0, 0
        self.memory = {
            "s": np.zeros([self.memory_size, self.features]), "s_": np.zeros([self.memory_size, self.features]),
            "a": np.zeros([self.memory_size, 1]), "r": np.zeros([self.memory_size, 1]),
            "m": np.zeros([self.memory_size, self.actions])
        }
