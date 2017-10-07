import numpy as np
import tensorflow as tf
import tensorflow.contrib.rnn as rnn
from ops import flatten, conv2d, linear


def normalized_columns_initializer(std=1.0):
    def _initializer(shape, dtype=None, partition_info=None):
        out = np.random.randn(*shape).astype(np.float32)
        out *= std / np.sqrt(np.square(out).sum(axis=0, keepdims=True))
        return tf.constant(out)
    return _initializer


def categorical_sample(logits, d):
    value = tf.squeeze(tf.multinomial(logits - tf.reduce_max(logits, [1], keep_dims=True), 1), [1])
    return tf.one_hot(value, d)


class SubPolicy(object):
    def __init__(self, ob_space, ac_space, subgoal_space, intrinsic_type):
        self.x = x = tf.placeholder(tf.float32, [None] + list(ob_space), name='x')
        self.action_prev = action_prev = tf.placeholder(tf.float32, [None, ac_space], name='action_prev')
        self.reward_prev = reward_prev = tf.placeholder(tf.float32, [None, 1], name='reward_prev')
        self.subgoal = subgoal = tf.placeholder(tf.float32, [None, subgoal_space], name='subgoal')
        self.intrinsic_type = intrinsic_type

        with tf.variable_scope('encoder'):
            x = tf.image.resize_images(x, [84, 84])
            x = x / 255.0
            self.p = x
            x = tf.nn.relu(conv2d(x, 16, "l1", [8, 8], [4, 4]))
            x = tf.nn.relu(conv2d(x, 32, "l2", [4, 4], [2, 2]))
            self.f = tf.reduce_mean(x, axis=[1, 2])
            x = flatten(x)

        with tf.variable_scope('sub_policy'):
            x = tf.nn.relu(linear(x, 256, "fc",
                                  normalized_columns_initializer(0.01)))
            x = tf.concat([x, action_prev], axis=1)
            x = tf.concat([x, reward_prev], axis=1)
            x = tf.concat([x, subgoal], axis=1)

            # introduce a "fake" batch dimension of 1 after flatten
            # so that we can do LSTM over time dim
            x = tf.expand_dims(x, [0])

            size = 256
            lstm = rnn.BasicLSTMCell(size, state_is_tuple=True)
            self.state_size = lstm.state_size
            step_size = tf.shape(self.x)[:1]

            c_init = np.zeros((1, lstm.state_size.c), np.float32)
            h_init = np.zeros((1, lstm.state_size.h), np.float32)
            self.state_init = [c_init, h_init]
            c_in = tf.placeholder(tf.float32, [1, lstm.state_size.c])
            h_in = tf.placeholder(tf.float32, [1, lstm.state_size.h])
            self.state_in = [c_in, h_in]

            state_in = rnn.LSTMStateTuple(c_in, h_in)

            lstm_outputs, lstm_state = tf.nn.dynamic_rnn(
                lstm, x, initial_state=state_in, sequence_length=step_size,
                time_major=False
            )
            lstm_c, lstm_h = lstm_state
            lstm_outputs = tf.reshape(lstm_outputs, [-1, size])
            self.logits = linear(lstm_outputs, ac_space, "action",
                                 normalized_columns_initializer(0.01))
            self.vf = tf.reshape(linear(lstm_outputs, 1, "value",
                                        normalized_columns_initializer(1.0)), [-1])
            self.state_out = [lstm_c[:1, :], lstm_h[:1, :]]
            self.sample = categorical_sample(self.logits, ac_space)[0, :]
        self.var_list = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES,
                                          tf.get_variable_scope().name)

    def get_initial_features(self):
        return self.state_init

    def act(self, ob, action_prev, reward_prev, subgoal, c, h):
        sess = tf.get_default_session()
        return sess.run([self.sample, self.vf] + self.state_out,
                        {self.x: [ob], self.state_in[0]: c, self.state_in[1]: h,
                         self.action_prev: [action_prev],
                         self.reward_prev: [reward_prev],
                         self.subgoal: [subgoal]})

    def value(self, ob, action_prev, reward_prev, subgoal, c, h):
        sess = tf.get_default_session()
        return sess.run(self.vf, {self.x: [ob], self.state_in[0]: c,
                                  self.state_in[1]: h,
                                  self.action_prev: [action_prev],
                                  self.reward_prev: [reward_prev],
                                  self.subgoal: [subgoal]})[0]

    def feature(self, state):
        sess = tf.get_default_session()
        if self.intrinsic_type == 'feature':
            return sess.run(self.f, {self.x: [state]})[0, :]
        else:
            return sess.run(self.p, {self.x: [state]})[0, :]


class MetaPolicy(object):
    def __init__(self, ob_space, subgoal_space, intrinsic_type):
        self.x = x = \
            tf.placeholder(tf.float32, [None] + list(ob_space), name='x_meta')
        self.subgoal_prev = subgoal_prev = \
            tf.placeholder(tf.float32, [None, subgoal_space], name='subgoal_prev')
        self.reward_prev = reward_prev = \
            tf.placeholder(tf.float32, [None, 1], name='reward_prev_meta')
        self.intrinsic_type = intrinsic_type

        with tf.variable_scope('encoder', reuse=True):
            x = tf.image.resize_images(x, [84, 84])
            x = x / 255.0
            x = tf.nn.relu(conv2d(x, 16, "l1", [8, 8], [4, 4]))
            x = tf.nn.relu(conv2d(x, 32, "l2", [4, 4], [2, 2]))
            x = flatten(x)

        with tf.variable_scope('meta_policy'):
            x = tf.nn.relu(linear(x, 256, "fc",
                                  normalized_columns_initializer(0.01)))
            x = tf.concat([x, subgoal_prev], axis=1)
            x = tf.concat([x, reward_prev], axis=1)

            # introduce a "fake" batch dimension of 1 after flatten
            # so that we can do LSTM over time dim
            x = tf.expand_dims(x, [0])

            size = 256
            lstm = rnn.BasicLSTMCell(size, state_is_tuple=True)
            self.state_size = lstm.state_size
            step_size = tf.shape(self.x)[:1]

            c_init = np.zeros((1, lstm.state_size.c), np.float32)
            h_init = np.zeros((1, lstm.state_size.h), np.float32)
            self.state_init = [c_init, h_init]
            c_in = tf.placeholder(tf.float32, [1, lstm.state_size.c])
            h_in = tf.placeholder(tf.float32, [1, lstm.state_size.h])
            self.state_in = [c_in, h_in]

            state_in = rnn.LSTMStateTuple(c_in, h_in)
            lstm_outputs, lstm_state = tf.nn.dynamic_rnn(
                lstm, x, initial_state=state_in, sequence_length=step_size,
                time_major=False)
            lstm_c, lstm_h = lstm_state
            lstm_outputs = tf.reshape(lstm_outputs, [-1, size])
            self.logits = linear(lstm_outputs, subgoal_space, "action",
                                 normalized_columns_initializer(0.01))
            self.vf = tf.reshape(linear(lstm_outputs, 1, "value",
                                        normalized_columns_initializer(1.0)), [-1])
            self.state_out = [lstm_c[:1, :], lstm_h[:1, :]]
            self.sample = categorical_sample(self.logits, subgoal_space)[0, :]
        self.var_list = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, tf.get_variable_scope().name)

    def get_initial_features(self):
        return self.state_init

    def act(self, ob, subgoal_prev, reward_prev, c, h):
        sess = tf.get_default_session()
        return sess.run([self.sample, self.vf] + self.state_out,
                        {self.x: [ob], self.state_in[0]: c, self.state_in[1]: h,
                         self.subgoal_prev: [subgoal_prev],
                         self.reward_prev: [reward_prev]})

    def value(self, ob, subgoal_prev, reward_prev, c, h):
        sess = tf.get_default_session()
        return sess.run(self.vf, {self.x: [ob], self.state_in[0]: c,
                                  self.state_in[1]: h,
                                  self.subgoal_prev: [subgoal_prev],
                                  self.reward_prev: [reward_prev]})[0]
