import numpy as np
import tensorflow as tf
import tensorflow.contrib.rnn as rnn
import distutils.version
use_tf100_api = distutils.version.LooseVersion(tf.VERSION) >= distutils.version.LooseVersion('1.0.0')

def normalized_columns_initializer(std=1.0):
    def _initializer(shape, dtype=None, partition_info=None):
        out = np.random.randn(*shape).astype(np.float32)
        out *= std / np.sqrt(np.square(out).sum(axis=0, keepdims=True))
        return tf.constant(out)
    return _initializer

def flatten(x):
    return tf.reshape(x, [-1, np.prod(x.get_shape().as_list()[1:])])

def conv2d(x, num_filters, name, filter_size=(3, 3), stride=(1, 1), pad="SAME", dtype=tf.float32, collections=None):
    with tf.variable_scope(name):
        stride_shape = [1, stride[0], stride[1], 1]
        filter_shape = [filter_size[0], filter_size[1], int(x.get_shape()[3]), num_filters]

        # there are "num input feature maps * filter height * filter width"
        # inputs to each hidden unit
        fan_in = np.prod(filter_shape[:3])
        # each unit in the lower layer receives a gradient from:
        # "num output feature maps * filter height * filter width" /
        #   pooling size
        fan_out = np.prod(filter_shape[:2]) * num_filters
        # initialize weights with random weights
        w_bound = np.sqrt(6. / (fan_in + fan_out))

        w = tf.get_variable("W", filter_shape, dtype, tf.random_uniform_initializer(-w_bound, w_bound),
                            collections=collections)
        b = tf.get_variable("b", [1, 1, 1, num_filters], initializer=tf.constant_initializer(0.0),
                            collections=collections)
        return tf.nn.conv2d(x, w, stride_shape, pad) + b

def linear(x, size, name, initializer=None, bias_init=0):
    w = tf.get_variable(name + "/w", [x.get_shape()[1], size], initializer=initializer)
    b = tf.get_variable(name + "/b", [size], initializer=tf.constant_initializer(bias_init))
    return tf.matmul(x, w) + b

def categorical_sample(logits, d):
    value = tf.squeeze(tf.multinomial(logits - tf.reduce_max(logits, [1], keep_dims=True), 1), [1])
    return tf.one_hot(value, d)


class SubPolicy(object):
    def __init__(self, ob_space, ac_space, subgoal_space):
        self.x = x = tf.placeholder(tf.float32, [None] + list(ob_space), name='x')
        self.action_prev = action_prev = tf.placeholder(tf.float32, [None, ac_space], name='action_prev')
        self.reward_prev = reward_prev = tf.placeholder(tf.float32, [None, 1], name='reward_prev')
        self.subgoal = subgoal = tf.placeholder(tf.float32, [None, subgoal_space], name='subgoal')

        with tf.variable_scope('encoder'):
            x = tf.nn.relu(conv2d(x, 16, "l1", [8, 8], [4, 4]))
            x = tf.nn.relu(conv2d(x, 32, "l2", [4, 4], [2, 2]))
            dim = int(x.get_shape()[1]) * int(x.get_shape()[2]) * int(x.get_shape()[3])
            x = tf.reshape(x, [-1, dim])

            self.f = x

            x = tf.nn.relu(linear(x, 256, "fc", normalized_columns_initializer(0.01)))
            x = tf.concat([x, action_prev], axis=1)
            x = tf.concat([x, reward_prev], axis=1)
            x = tf.concat([x, subgoal], axis=1)

            # introduce a "fake" batch dimension of 1 after flatten so that we can do LSTM over time dim
            x = tf.expand_dims(flatten(x), [0])

        with tf.variable_scope('sub_policy'):
            size = 256
            if use_tf100_api:
                lstm = rnn.BasicLSTMCell(size, state_is_tuple=True)
            else:
                lstm = rnn.rnn_cell.BasicLSTMCell(size, state_is_tuple=True)
            self.state_size = lstm.state_size
            step_size = tf.shape(self.x)[:1]

            c_init = np.zeros((1, lstm.state_size.c), np.float32)
            h_init = np.zeros((1, lstm.state_size.h), np.float32)
            self.state_init = [c_init, h_init]
            c_in = tf.placeholder(tf.float32, [1, lstm.state_size.c])
            h_in = tf.placeholder(tf.float32, [1, lstm.state_size.h])
            self.state_in = [c_in, h_in]

            if use_tf100_api:
                state_in = rnn.LSTMStateTuple(c_in, h_in)
            else:
                state_in = rnn.rnn_cell.LSTMStateTuple(c_in, h_in)
            lstm_outputs, lstm_state = tf.nn.dynamic_rnn(
                lstm, x, initial_state=state_in, sequence_length=step_size,
                time_major=False)
            lstm_c, lstm_h = lstm_state
            x = tf.reshape(lstm_outputs, [-1, size])
            self.logits = linear(x, ac_space, "action", normalized_columns_initializer(0.01))
            self.vf = tf.reshape(linear(x, 1, "value", normalized_columns_initializer(1.0)), [-1])
            self.state_out = [lstm_c[:1, :], lstm_h[:1, :]]
            self.sample = categorical_sample(self.logits, ac_space)
        self.var_list = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, tf.get_variable_scope().name)

    def get_initial_features(self):
        return self.state_init

    def act(self, ob, action_prev, reward_prev, subgoal, c, h):
        sess = tf.get_default_session()
        return sess.run([self.sample, self.vf] + self.state_out,
                        {self.x: [ob], self.state_in[0]: c, self.state_in[1]: h,
                         self.action_prev: action_prev,
                         self.reward_prev: reward_prev,
                         self.subgoal: subgoal})

    def value(self, ob, action_prev, reward_prev, subgoal, c, h):
        sess = tf.get_default_session()
        return sess.run(self.vf, {self.x: [ob], self.state_in[0]: c,
                                  self.state_in[1]: h,
                                  self.action_prev: action_prev,
                                  self.reward_prev: reward_prev,
                                  self.subgoal: subgoal})[0]

    def feature(self, state):
        sess = tf.get_default_session()
        return sess.run(self.f, {self.x: [state]})[0, :]


class MetaPolicy(object):
    def __init__(self, ob_space, subgoal_space):
        self.x = x = tf.placeholder(tf.float32, [None] + list(ob_space), name='x_meta')
        self.subgoal_prev = subgoal_prev = tf.placeholder(tf.float32, [None, subgoal_space], name='subgoal_prev')
        self.reward_prev = reward_prev = tf.placeholder(tf.float32, [None, 1], name='reward_prev_meta')

        with tf.variable_scope('encoder', reuse=True):
            x = tf.nn.relu(conv2d(x, 16, "l1", [8, 8], [4, 4]))
            x = tf.nn.relu(conv2d(x, 32, "l2", [4, 4], [2, 2]))
            dim = int(x.get_shape()[1]) * int(x.get_shape()[2]) * int(x.get_shape()[3])
            x = tf.reshape(x, [-1, dim])

            self.f = x

            x = tf.nn.relu(linear(x, 256, "fc", normalized_columns_initializer(0.01)))
            x = tf.concat([x, subgoal_prev], axis=1)
            x = tf.concat([x, reward_prev], axis=1)

            # introduce a "fake" batch dimension of 1 after flatten so that we can do LSTM over time dim
            x = tf.expand_dims(flatten(x), [0])

        with tf.variable_scope('meta_policy'):
            size = 256
            if use_tf100_api:
                lstm = rnn.BasicLSTMCell(size, state_is_tuple=True)
            else:
                lstm = rnn.rnn_cell.BasicLSTMCell(size, state_is_tuple=True)
            self.state_size = lstm.state_size
            step_size = tf.shape(self.x)[:1]

            c_init = np.zeros((1, lstm.state_size.c), np.float32)
            h_init = np.zeros((1, lstm.state_size.h), np.float32)
            self.state_init = [c_init, h_init]
            c_in = tf.placeholder(tf.float32, [1, lstm.state_size.c])
            h_in = tf.placeholder(tf.float32, [1, lstm.state_size.h])
            self.state_in = [c_in, h_in]

            if use_tf100_api:
                state_in = rnn.LSTMStateTuple(c_in, h_in)
            else:
                state_in = rnn.rnn_cell.LSTMStateTuple(c_in, h_in)
            lstm_outputs, lstm_state = tf.nn.dynamic_rnn(
                lstm, x, initial_state=state_in, sequence_length=step_size,
                time_major=False)
            lstm_c, lstm_h = lstm_state
            x = tf.reshape(lstm_outputs, [-1, size])
            self.logits = linear(x, subgoal_space, "action", normalized_columns_initializer(0.01))
            self.vf = tf.reshape(linear(x, 1, "value", normalized_columns_initializer(1.0)), [-1])
            self.state_out = [lstm_c[:1, :], lstm_h[:1, :]]
            self.sample = categorical_sample(self.logits, subgoal_space)
        self.var_list = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, tf.get_variable_scope().name)

    def get_initial_features(self):
        return self.state_init

    def act(self, ob, subgoal_prev, reward_prev, c, h):
        sess = tf.get_default_session()
        return sess.run([self.sample, self.vf] + self.state_out,
                        {self.x: [ob], self.state_in[0]: c, self.state_in[1]: h,
                         self.subgoal_prev: subgoal_prev,
                         self.reward_prev: reward_prev})

    def value(self, ob, subgoal_prev, reward_prev, c, h):
        sess = tf.get_default_session()
        return sess.run(self.vf, {self.x: [ob], self.state_in[0]: c,
                                  self.state_in[1]: h,
                                  self.subgoal_prev: subgoal_prev,
                                  self.reward_prev: reward_prev})[0]
