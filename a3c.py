from __future__ import print_function
from collections import namedtuple
import numpy as np
import tensorflow as tf
from model import SubPolicy, MetaPolicy
import six.moves.queue as queue
import scipy.signal
import threading
import distutils.version
use_tf12_api = distutils.version.LooseVersion(tf.VERSION) >= distutils.version.LooseVersion('0.12.0')


def discount(x, gamma):
    return scipy.signal.lfilter([1], [1, -gamma], x[::-1], axis=0)[::-1]

class Batch:
    def __init__(self, si, a, adv, r, terminal, features):
        self.si = si
        self.a = a
        self.adv = adv
        self.r = r
        self.terminal = terminal
        self.features = features

def process_rollout(rollout, gamma, lambda_=1.0):
    """
given a rollout, compute its returns and the advantage
"""
    batch_si = {}
    for key in rollout.states[0].keys():
        batch_si[key] = np.stack([s[key] for s in rollout.states])
    #batch_si = np.asarray(rollout.states)
    batch_a = np.asarray(rollout.actions)
    rewards = np.asarray(rollout.rewards)
    vpred_t = np.asarray(rollout.values + [rollout.r])

    rewards_plus_v = np.asarray(rollout.rewards + [rollout.r])
    batch_r = discount(rewards_plus_v, gamma)[:-1]
    delta_t = rewards + gamma * vpred_t[1:] - vpred_t[:-1]
    # this formula for the advantage comes "Generalized Advantage Estimation":
    # https://arxiv.org/abs/1506.02438
    batch_adv = discount(delta_t, gamma * lambda_)

    features = rollout.features[0]
    return Batch(batch_si, batch_a, batch_adv, batch_r, rollout.terminal, features)


class PartialRollout(object):
    """
a piece of a complete rollout.  We run our agent, and process its experience
once it has processed enough steps.
"""
    def __init__(self):
        self.states = []
        self.actions = []
        self.rewards = []
        self.values = []
        self.r = 0.0
        self.terminal = False
        self.features = []

    def add(self, state, action, reward, value, terminal, features):
        self.states += [state]
        self.actions += [action]
        self.rewards += [reward]
        self.values += [value]
        self.terminal = terminal
        self.features += [features]

    def extend(self, other):
        assert not self.terminal
        self.states.extend(other.states)
        self.actions.extend(other.actions)
        self.rewards.extend(other.rewards)
        self.values.extend(other.values)
        self.r = other.r
        self.terminal = other.terminal
        self.features.extend(other.features)


class RunnerThread(threading.Thread):
    """
One of the key distinctions between a normal environment and a universe environment
is that a universe environment is _real time_.  This means that there should be a thread
that would constantly interact with the environment and tell it what to do.  This thread is here.
"""
    def __init__(self, env, sub_policy, meta_policy, num_local_steps, visualise):
        threading.Thread.__init__(self)
        self.sub_queue = queue.Queue(5)
        self.meta_queue = queue.Queue(5)
        self.num_local_steps = num_local_steps
        self.env = env
        self.last_features = None
        self.sub_policy = sub_policy
        self.meta_policy = meta_policy
        self.daemon = True
        self.sess = None
        self.summary_writer = None
        self.visualise = visualise

    def start_runner(self, sess, summary_writer):
        self.sess = sess
        self.summary_writer = summary_writer
        self.start()

    def run(self):
        with self.sess.as_default():
            self._run()

    def _run(self):
        rollout_provider = env_runner(self.env, self.sub_policy, self.meta_policy, self.num_local_steps, self.summary_writer, self.visualise)
        while True:
            # the timeout variable exists because apparently, if one worker dies, the other workers
            # won't die with it, unless the timeout is set to some large number.  This is an empirical
            # observation.

            t, rollout = next(rollout_provider)
            if t == 0:
                self.sub_queue.put(rollout, timeout=600.0)
            else:
                self.meta_queue.put(rollout, timeout=600.0)



def env_runner(env, sub_policy, meta_policy, num_local_steps, summary_writer, render):
    """
The logic of the thread runner.  In brief, it constantly keeps on running
the policy, and as long as the rollout exceeds a certain length, the thread
runner appends the policy to the queue.
"""
    num_local_meta_steps = 20

    subgoal_space = 32
    action_space = env.action_space.n

    last_meta_state = last_state = env.reset()
    last_features = sub_policy.get_initial_features()
    last_meta_features = meta_policy.get_initial_features()

    last_subgoal = np.zeros((1, subgoal_space))
    last_action = np.zeros((1, action_space))
    last_meta_reward = np.zeros((1, 1))
    last_reward = np.zeros((1, 1))

    length = 0
    rewards = 0
    extrinsic_reward = 0
    intrinsic_reward = 0
    beta = 0.75
    steps = 0

    while True:
        terminal_end = False
        meta_rollout = PartialRollout()

        for _ in range(num_local_meta_steps):
            fetched = meta_policy.act(last_meta_state, last_subgoal, last_meta_reward, *last_meta_features)
            subgoal, value_, meta_features = fetched[0], fetched[1], fetched[2:]

            for _ in range(5):
                sub_rollout = PartialRollout()
                meta_reward = 0
                for _ in range(num_local_steps):
                    fetched = sub_policy.act(last_state, last_action, last_reward, subgoal, *last_features)
                    action, value_, features = fetched[0], fetched[1], fetched[2:]

                    # argmax to convert from one-hot
                    state, extrinsic_reward, terminal, info = env.step(action[0, :].argmax())
                    def compute_intrinsic(state, last_state, subgoal):
                        f = sub_policy.feature(state)
                        last_f = sub_policy.feature(last_state)
                        diff = np.abs(f - last_f)
                        eta = 0.05
                        return eta * np.sum(diff[subgoal]) / (np.sum(diff) + 1e-10)
                    intrinsic_reward = compute_intrinsic(state, last_state, subgoal[0, :].argmax())
                    reward = beta * extrinsic_reward + (1 - beta) * intrinsic_reward
                    meta_reward += extrinsic_reward

                    # collect the experience
                    si = {
                        'x': last_state,
                        'action_prev': last_action[0],
                        'reward_prev': last_reward[0],
                        'subgoal': subgoal[0]
                    }

                    sub_rollout.add(si, action[0, :], reward, value_, terminal, last_features)
                    length += 1
                    rewards += reward

                    last_state = state
                    last_action = action
                    last_features = features
                    last_reward = [[reward]]

                    if info:
                        summary = tf.Summary()
                        for k, v in info.items():
                            summary.value.add(tag=k, simple_value=float(v))
                        summary_writer.add_summary(summary, sub_policy.global_step.eval())
                        summary_writer.flush()

                    timestep_limit = env.spec.tags.get('wrapper_config.TimeLimit.max_episode_steps')
                    if terminal or length >= timestep_limit:
                        terminal_end = True
                        if length >= timestep_limit or not env.metadata.get('semantics.autoreset'):
                            last_meta_state = last_state = env.reset()
                            steps = 0
                        last_features = sub_policy.get_initial_features()
                        last_meta_features = meta_policy.get_initial_features()
                        print("Episode finished. Sum of rewards: %d. Length: %d" % (rewards, length))
                        length = 0
                        rewards = 0
                        break

                if not terminal_end:
                    sub_rollout.r = sub_policy.value(last_state, last_action, last_reward, subgoal, *last_features)

                yield (0, sub_rollout)

                if terminal_end:
                    break

            si = {
                'x': last_meta_state,
                'subgoal_prev': last_subgoal[0],
                'reward_prev': last_meta_reward[0]
            }

            meta_rollout.add(si, subgoal[0], meta_reward, value_, terminal_end, last_meta_features)

            last_meta_state = state
            last_meta_features = meta_features
            last_meta_reward = [[meta_reward]]
            last_subgoal = subgoal

            last_state = state
            last_action = action
            last_reward = [[reward]]
            last_features = features

            if terminal_end:
                last_meta_features = meta_policy.get_initial_features()
                break

        if terminal_end:
            meta_rollout.r = meta_policy.value(last_state, last_subgoal, last_meta_reward, *last_features)

        # once we have enough experience, yield it, and have the ThreadRunner place it on a queue
        yield (1, meta_rollout)


class A3C(object):
    def __init__(self, env, task, visualise):
        """
An implementation of the A3C algorithm that is reasonably well-tuned for the VNC environments.
Below, we will have a modest amount of complexity due to the way TensorFlow handles data parallelism.
But overall, we'll define the model, specify its inputs, and describe how the policy gradients step
should be computed.
"""

        self.env = env
        self.task = task
        worker_device = "/job:worker/task:{}/cpu:0".format(task)
        with tf.device(tf.train.replica_device_setter(1, worker_device=worker_device)):
            with tf.variable_scope("global"):
                self.sub_network = SubPolicy(env.observation_space.shape, env.action_space.n, 32)
                self.meta_network = MetaPolicy(env.observation_space.shape, 32)
                self.global_step = tf.get_variable("global_step", [], tf.int32, initializer=tf.constant_initializer(0, dtype=tf.int32),
                                                   trainable=False)

        with tf.device(worker_device):
            with tf.variable_scope("local"):
                self.local_sub_network = pi = SubPolicy(env.observation_space.shape, env.action_space.n, 32)
                self.local_meta_network = meta_pi = MetaPolicy(env.observation_space.shape, 32)
                pi.global_step = self.global_step

            self.ac = tf.placeholder(tf.float32, [None, env.action_space.n], name="ac")
            self.adv = tf.placeholder(tf.float32, [None], name="adv")
            self.r = tf.placeholder(tf.float32, [None], name="r")

            log_prob_tf = tf.nn.log_softmax(pi.logits)
            prob_tf = tf.nn.softmax(pi.logits)

            # the "policy gradients" loss:  its derivative is precisely the policy gradient
            # notice that self.ac is a placeholder that is provided externally.
            # adv will contain the advantages, as calculated in process_rollout
            pi_loss = - tf.reduce_sum(tf.reduce_sum(log_prob_tf * self.ac, [1]) * self.adv)

            # loss of value function
            vf_loss = 0.5 * tf.reduce_sum(tf.square(pi.vf - self.r))
            entropy = - tf.reduce_sum(prob_tf * log_prob_tf)

            bs = tf.to_float(tf.shape(pi.x)[0])
            self.loss = pi_loss + 0.5 * vf_loss - entropy * 0.01


            self.meta_ac = tf.placeholder(tf.float32, [None, 32], name="meta_ac")
            self.meta_adv = tf.placeholder(tf.float32, [None], name="meta_adv")
            self.meta_r = tf.placeholder(tf.float32, [None], name="meta_r")

            meta_log_prob_tf = tf.nn.log_softmax(meta_pi.logits)
            meta_prob_tf = tf.nn.softmax(meta_pi.logits)

            # the "policy gradients" loss:  its derivative is precisely the policy gradient
            # notice that self.ac is a placeholder that is provided externally.
            # adv will contain the advantages, as calculated in process_rollout
            meta_pi_loss = - tf.reduce_sum(tf.reduce_sum(meta_log_prob_tf * self.meta_ac, [1]) * self.meta_adv)

            # loss of value function
            meta_vf_loss = 0.5 * tf.reduce_sum(tf.square(meta_pi.vf - self.meta_r))
            meta_entropy = - tf.reduce_sum(meta_prob_tf * meta_log_prob_tf)

            meta_bs = tf.to_float(tf.shape(meta_pi.x)[0])
            self.meta_loss = meta_pi_loss + 0.5 * meta_vf_loss - meta_entropy * 0.01

            # 20 represents the number of "local steps":  the number of timesteps
            # we run the policy before we update the parameters.
            # The larger local steps is, the lower is the variance in our policy gradients estimate
            # on the one hand;  but on the other hand, we get less frequent parameter updates, which
            # slows down learning.  In this code, we found that making local steps be much
            # smaller than 20 makes the algorithm more difficult to tune and to get to work.
            self.runner = RunnerThread(env, pi, meta_pi, 20, visualise)


            grads = tf.gradients(self.loss, pi.var_list)
            meta_grads = tf.gradients(self.meta_loss, meta_pi.var_list)
#
#
#          if use_tf12_api:
#              tf.summary.scalar("model/policy_loss", pi_loss / bs)
#              tf.summary.scalar("model/value_loss", vf_loss / bs)
#              tf.summary.scalar("model/entropy", entropy / bs)
#              tf.summary.image("model/state", pi.x)
#              tf.summary.scalar("model/grad_global_norm", tf.global_norm(grads))
#              tf.summary.scalar("model/var_global_norm", tf.global_norm(pi.var_list))
#              tf.summary.scalar("meta_model/policy_loss", meta_pi_loss / meta_bs)
#              tf.summary.scalar("meta_model/value_loss", meta_vf_loss / meta_bs)
#              tf.summary.scalar("meta_model/entropy", meta_entropy / meta_bs)
#              tf.summary.scalar("meta_model/grad_global_norm", tf.global_norm(meta_grads))
#              tf.summary.scalar("meta_model/var_global_norm", tf.global_norm(meta_pi.var_list))
#              self.summary_op = tf.summary.merge_all()
#
#          else:
#              tf.scalar_summary("model/policy_loss", pi_loss / bs)
#              tf.scalar_summary("model/value_loss", vf_loss / bs)
#              tf.scalar_summary("model/entropy", entropy / bs)
#              tf.image_summary("model/state", pi.x)
#              tf.scalar_summary("model/grad_global_norm", tf.global_norm(grads))
#              tf.scalar_summary("model/var_global_norm", tf.global_norm(pi.var_list))
#              self.summary_op = tf.merge_all_summaries()
#
            grads, _ = tf.clip_by_global_norm(grads, 40.0)
            meta_grads, _ = tf.clip_by_global_norm(meta_grads, 40.0)

            # copy weights from the parameter server to the local model
            self.sync = tf.group(*[v1.assign(v2) for v1, v2 in zip(pi.var_list, self.sub_network.var_list)])
            self.meta_sync = tf.group(*[v1.assign(v2) for v1, v2 in zip(meta_pi.var_list, self.meta_network.var_list)])

            grads_and_vars = list(zip(grads, self.sub_network.var_list))
            inc_step = self.global_step.assign_add(tf.shape(pi.x)[0])
            meta_grads_and_vars = list(zip(meta_grads, self.meta_network.var_list))
            meta_inc_step = self.global_step.assign_add(tf.shape(meta_pi.x)[0])

            # each worker has a different set of adam optimizer parameters
            opt = tf.train.AdamOptimizer(1e-4)
            self.train_op = tf.group(opt.apply_gradients(grads_and_vars), inc_step)
            meta_opt = tf.train.AdamOptimizer(1e-4)
            self.meta_train_op = tf.group(meta_opt.apply_gradients(meta_grads_and_vars), meta_inc_step)

            self.summary_writer = None
            self.local_steps = 0

    def start(self, sess, summary_writer):
        self.runner.start_runner(sess, summary_writer)
        self.summary_writer = summary_writer

    def pull_batch_from_sub_queue(self):
        """
self explanatory:  take a rollout from the queue of the thread runner.
"""
        rollout = self.runner.sub_queue.get(timeout=600.0)
        while not rollout.terminal:
            try:
                rollout.extend(self.runner.sub_queue.get_nowait())
            except queue.Empty:
                break
        return rollout

    def pull_batch_from_meta_queue(self):
        """
self explanatory:  take a rollout from the queue of the thread runner.
"""
        rollout = self.runner.meta_queue.get(timeout=600.0)
        while not rollout.terminal:
            try:
                rollout.extend(self.runner.meta_queue.get_nowait())
            except queue.Empty:
                break
        return rollout

    def process(self, sess):

        sess.run(self.meta_sync)  # copy weights from shared to local
        sess.run(self.sync)  # copy weights from shared to local

        env = self.env
        num_local_meta_steps = 20
        num_local_steps = 20

        subgoal_space = 32
        action_space = self.env.action_space.n

        sub_policy = self.local_sub_network
        meta_policy = self.local_meta_network

        last_meta_state = last_state = self.env.reset()
        last_features = sub_policy.get_initial_features()
        last_meta_features = meta_policy.get_initial_features()

        last_subgoal = np.zeros((1, subgoal_space))
        last_action = np.zeros((1, action_space))
        last_meta_reward = np.zeros((1, 1))
        last_reward = np.zeros((1, 1))

        length = 0
        rewards = 0
        extrinsic_reward = 0
        intrinsic_reward = 0
        beta = 0.75
        steps = 0

        while True:
            terminal_end = False
            meta_rollout = PartialRollout()

            for _ in range(num_local_meta_steps):
                fetched = meta_policy.act(last_meta_state, last_subgoal, last_meta_reward, *last_meta_features)
                subgoal, value_, meta_features = fetched[0], fetched[1], fetched[2:]

                for _ in range(5):
                    sub_rollout = PartialRollout()
                    meta_reward = 0
                    for _ in range(num_local_steps):
                        fetched = sub_policy.act(last_state, last_action, last_reward, subgoal, *last_features)
                        action, value_, features = fetched[0], fetched[1], fetched[2:]

                        # argmax to convert from one-hot
                        state, extrinsic_reward, terminal, info = self.env.step(action[0, :].argmax())
                        def compute_intrinsic(state, last_state, subgoal):
                            f = sub_policy.feature(state)
                            last_f = sub_policy.feature(last_state)
                            diff = np.abs(f - last_f)
                            eta = 0.05
                            return eta * np.sum(diff[subgoal]) / (np.sum(diff) + 1e-10)
                        intrinsic_reward = compute_intrinsic(state, last_state, subgoal[0, :].argmax())
                        reward = beta * extrinsic_reward + (1 - beta) * intrinsic_reward
                        meta_reward += extrinsic_reward

                        # collect the experience
                        si = {
                            'x': last_state,
                            'action_prev': last_action[0],
                            'reward_prev': last_reward[0],
                            'subgoal': subgoal[0]
                        }

                        sub_rollout.add(si, action[0, :], reward, value_, terminal, last_features)
                        length += 1
                        rewards += reward

                        last_state = state
                        last_action = action
                        last_features = features
                        last_reward = [[reward]]

                        timestep_limit = self.env.spec.tags.get('wrapper_config.TimeLimit.max_episode_steps')
                        if terminal or length >= timestep_limit:
                            terminal_end = True
                            if length >= timestep_limit or not self.env.metadata.get('semantics.autoreset'):
                                last_meta_state = last_state = self.env.reset()
                                steps = 0
                            last_features = sub_policy.get_initial_features()
                            last_meta_features = meta_policy.get_initial_features()
                            print("Episode finished. Sum of rewards: %d. Length: %d" % (rewards, length))
                            length = 0
                            rewards = 0
                            break

                    if not terminal_end:
                        sub_rollout.r = sub_policy.value(last_state, last_action, last_reward, subgoal, *last_features)

                    # Sub rollout
                    batch = process_rollout(sub_rollout, gamma=0.99, lambda_=1.0)
                    fetches = [self.train_op, self.global_step]
                    feed_dict = {
                        self.local_sub_network.x: batch.si['x'],
                        self.local_sub_network.action_prev: batch.si['action_prev'],
                        self.local_sub_network.reward_prev: batch.si['reward_prev'],
                        self.local_sub_network.subgoal: batch.si['subgoal'],
                        self.ac: batch.a,
                        self.adv: batch.adv,
                        self.r: batch.r,
                        self.local_sub_network.state_in[0]: batch.features[0],
                        self.local_sub_network.state_in[1]: batch.features[1],
                    }
                    fetched = sess.run(fetches, feed_dict=feed_dict)

                    if terminal_end:
                        break

                si = {
                    'x': last_meta_state,
                    'subgoal_prev': last_subgoal[0],
                    'reward_prev': last_meta_reward[0]
                }

                meta_rollout.add(si, subgoal[0], meta_reward, value_, terminal_end, last_meta_features)

                last_meta_state = state
                last_meta_features = meta_features
                last_meta_reward = [[meta_reward]]
                last_subgoal = subgoal

                last_state = state
                last_action = action
                last_reward = [[reward]]
                last_features = features

                if terminal_end:
                    last_meta_features = meta_policy.get_initial_features()
                    break

            if terminal_end:
                meta_rollout.r = meta_policy.value(last_state, last_subgoal, last_meta_reward, *last_features)

            # meta rollout
            batch = process_rollout(meta_rollout, gamma=0.99, lambda_=1.0)
            fetches = [self.meta_train_op, self.global_step]

            feed_dict = {
                self.local_meta_network.x: batch.si['x'],
                self.local_meta_network.subgoal_prev: batch.si['subgoal_prev'],
                self.local_meta_network.reward_prev: batch.si['reward_prev'],
                self.meta_ac: batch.a,
                self.meta_adv: batch.adv,
                self.meta_r: batch.r,
                self.local_meta_network.state_in[0]: batch.features[0],
                self.local_meta_network.state_in[1]: batch.features[1],
            }
            fetched = sess.run(fetches, feed_dict=feed_dict)

            self.local_steps += 1
            break

    def meta_process(self, sess):
        """
process grabs a rollout that's been produced by the thread runner,
and updates the parameters.  The update is then sent to the parameter
server.
"""

        sess.run(self.meta_sync)  # copy weights from shared to local
        rollout = self.pull_batch_from_meta_queue()
        batch = process_rollout(rollout, gamma=0.99, lambda_=1.0)

        should_compute_summary = self.task == 0 and self.local_steps % 11 == 0

        if should_compute_summary:
            #fetches = [self.summary_op, self.train_op, self.global_step]
            fetches = [self.meta_train_op, self.global_step]
        else:
            fetches = [self.meta_train_op, self.global_step]

        feed_dict = {
            self.local_meta_network.x: batch.si['x'],
            self.local_meta_network.subgoal_prev: batch.si['subgoal_prev'],
            self.local_meta_network.reward_prev: batch.si['reward_prev'],
            self.meta_ac: batch.a,
            self.meta_adv: batch.adv,
            self.meta_r: batch.r,
            self.local_meta_network.state_in[0]: batch.features[0],
            self.local_meta_network.state_in[1]: batch.features[1],
        }

        fetched = sess.run(fetches, feed_dict=feed_dict)

        if should_compute_summary:
            #self.summary_writer.add_summary(tf.Summary.FromString(fetched[0]), fetched[-1])
            self.summary_writer.flush()
        self.local_steps += 1

    def sub_process(self, sess):
        """
process grabs a rollout that's been produced by the thread runner,
and updates the parameters.  The update is then sent to the parameter
server.
"""

        sess.run(self.sync)  # copy weights from shared to local
        rollout = self.pull_batch_from_sub_queue()
        batch = process_rollout(rollout, gamma=0.99, lambda_=1.0)

        should_compute_summary = self.task == 0 and self.local_steps % 11 == 0

        if should_compute_summary:
            #fetches = [self.summary_op, self.train_op, self.global_step]
            fetches = [self.train_op, self.global_step]
        else:
            fetches = [self.train_op, self.global_step]

        feed_dict = {
            self.local_sub_network.x: batch.si['x'],
            self.local_sub_network.action_prev: batch.si['action_prev'],
            self.local_sub_network.reward_prev: batch.si['reward_prev'],
            self.local_sub_network.subgoal: batch.si['subgoal'],
            self.ac: batch.a,
            self.adv: batch.adv,
            self.r: batch.r,
            self.local_sub_network.state_in[0]: batch.features[0],
            self.local_sub_network.state_in[1]: batch.features[1],
        }

        fetched = sess.run(fetches, feed_dict=feed_dict)

        if should_compute_summary:
            #self.summary_writer.add_summary(tf.Summary.FromString(fetched[0]), fetched[-1])
            self.summary_writer.flush()
        self.local_steps += 1
