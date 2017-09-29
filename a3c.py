import numpy as np
import tensorflow as tf
import scipy.signal

from model import SubPolicy, MetaPolicy


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


class A3C(object):
    def __init__(self, env, task, visualise, intrinsic_type, bptt):
        """
An implementation of the A3C algorithm that is reasonably well-tuned for the VNC environments.
Below, we will have a modest amount of complexity due to the way TensorFlow handles data parallelism.
But overall, we'll define the model, specify its inputs, and describe how the policy gradients step
should be computed.
"""

        self.env = env
        self.task = task
        self.visualise = visualise
        self.intrinsic_type = intrinsic_type
        self.bptt = bptt
        self.subgoal_space = 32 if intrinsic_type == 'feature' else 37
        self.action_space = env.action_space.n

        self.summary_writer = None
        self.local_steps = 0

        self.beta = 0.75
        self.eta = 0.05
        self.num_local_steps = self.bptt
        self.num_local_meta_steps = 20

        # Testing
        if task is None:
            with tf.variable_scope("global"):
                self.local_sub_network = SubPolicy(env.observation_space.shape,
                                                   env.action_space.n,
                                                   self.subgoal_space,
                                                   self.intrinsic_type)
                self.local_meta_network = MetaPolicy(env.observation_space.shape,
                                                     self.subgoal_space,
                                                     self.intrinsic_type)
                return

        # Training
        worker_device = "/job:worker/task:{}/cpu:0".format(task)
        with tf.device(tf.train.replica_device_setter(1, worker_device=worker_device)):
            with tf.variable_scope("global"):
                print(env.observation_space.shape)
                self.sub_network = SubPolicy(env.observation_space.shape,
                                             env.action_space.n,
                                             self.subgoal_space,
                                             self.intrinsic_type)
                self.meta_network = MetaPolicy(env.observation_space.shape,
                                               self.subgoal_space,
                                               self.intrinsic_type)
                self.global_step = tf.get_variable("global_step", [], tf.int32, initializer=tf.constant_initializer(0, dtype=tf.int32),
                                                   trainable=False)

        with tf.device(worker_device):
            with tf.variable_scope("local"):
                self.local_sub_network = pi = SubPolicy(env.observation_space.shape,
                                                        env.action_space.n,
                                                        self.subgoal_space,
                                                        self.intrinsic_type)
                self.local_meta_network = meta_pi = MetaPolicy(env.observation_space.shape,
                                                               self.subgoal_space,
                                                               self.intrinsic_type)
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


            self.meta_ac = tf.placeholder(tf.float32, [None, self.subgoal_space], name="meta_ac")
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
            # self.runner = RunnerThread(env, pi, meta_pi, 20, visualise)

            grads = tf.gradients(self.loss, pi.var_list)
            meta_grads = tf.gradients(self.meta_loss, meta_pi.var_list)

            summary = [
                tf.summary.scalar("model/policy_loss", pi_loss / bs),
                tf.summary.scalar("model/value_loss", vf_loss / bs),
                tf.summary.scalar("model/entropy", entropy / bs),
                tf.summary.image("model/state", pi.x),
                tf.summary.scalar("model/grad_global_norm", tf.global_norm(grads)),
                tf.summary.scalar("model/var_global_norm", tf.global_norm(pi.var_list))
            ]

            meta_summary = [
                tf.summary.scalar("meta_model/policy_loss", meta_pi_loss / meta_bs),
                tf.summary.scalar("meta_model/value_loss", meta_vf_loss / meta_bs),
                tf.summary.scalar("meta_model/entropy", meta_entropy / meta_bs),
                tf.summary.scalar("meta_model/grad_global_norm", tf.global_norm(meta_grads)),
                tf.summary.scalar("meta_model/var_global_norm", tf.global_norm(meta_pi.var_list))
            ]
            self.summary_op = tf.summary.merge(summary)
            self.meta_summary_op = tf.summary.merge(meta_summary)

            grads, _ = tf.clip_by_global_norm(grads, 40.0)
            meta_grads, _ = tf.clip_by_global_norm(meta_grads, 40.0)
            self.grads = grads

            # copy weights from the parameter server to the local model
            self.sync = tf.group(*[v1.assign(v2) for v1, v2 in zip(pi.var_list, self.sub_network.var_list)])
            self.meta_sync = tf.group(*[v1.assign(v2) for v1, v2 in zip(meta_pi.var_list, self.meta_network.var_list)])

            grads_and_vars = list(zip(grads, self.sub_network.var_list))
            meta_grads_and_vars = list(zip(meta_grads, self.meta_network.var_list))

            inc_step = self.global_step.assign_add(tf.shape(pi.x)[0])

            # each worker has a different set of adam optimizer parameters
            opt = tf.train.AdamOptimizer(1e-4)
            self.train_op = tf.group(opt.apply_gradients(grads_and_vars), inc_step)
            meta_opt = tf.train.AdamOptimizer(1e-4)
            self.meta_train_op = meta_opt.apply_gradients(meta_grads_and_vars)

    def start(self, sess, summary_writer):
        self.summary_writer = summary_writer

    def process(self, sess):
        """
        run one episode and process experience to train both meta and sub networks
        """
        env = self.env

        sub_policy = self.local_sub_network
        meta_policy = self.local_meta_network

        self.last_state = env.reset()
        self.last_meta_state = env.reset()
        self.last_features = sub_policy.get_initial_features()
        self.last_meta_features = meta_policy.get_initial_features()

        self.last_action = np.zeros(self.action_space)
        self.last_subgoal = np.zeros(self.subgoal_space)
        self.last_reward = np.zeros(1)
        self.last_meta_reward = np.zeros(1)

        self.length = 0
        self.rewards = 0
        self.extrinsic_rewards = 0
        self.intrinsic_rewards = 0

        terminal_end = False
        while not terminal_end:
            terminal_end = self._meta_process(sess)

    def _meta_process(self, sess):
        sess.run(self.meta_sync)  # copy weights from shared to local
        meta_rollout = PartialRollout()
        meta_policy = self.local_meta_network

        terminal_end = False
        for _ in range(self.num_local_meta_steps):
            fetched = meta_policy.act(self.last_meta_state, self.last_subgoal,
                                      self.last_meta_reward, *self.last_meta_features)
            subgoal, meta_value, meta_features = fetched[0], fetched[1], fetched[2:]

            assert self.bptt in [20, 100], 'bptt (%d) should be 20 or 100' % self.bptt

            if self.bptt == 20:
                meta_reward = 0
                for _ in range(5):
                    state, reward, terminal_end = self._sub_process(sess, subgoal)
                    meta_reward += reward
                    if terminal_end:
                        break
            elif self.bptt == 100:
                state, meta_reward, terminal_end = self._sub_process(sess, subgoal)


            si = {
                'x': self.last_meta_state,
                'subgoal_prev': self.last_subgoal,
                'reward_prev': self.last_meta_reward
            }
            meta_rollout.add(si, subgoal, meta_reward, meta_value,
                             terminal_end, self.last_meta_features)

            self.last_meta_state = state
            self.last_meta_features = meta_features
            self.last_meta_reward = [meta_reward]
            self.last_subgoal = subgoal

            if terminal_end:
                break

        if not terminal_end:
            meta_rollout.r = meta_policy.value(self.last_state,
                                               self.last_subgoal,
                                               self.last_meta_reward,
                                               *self.last_meta_features)

        # meta rollout
        batch = process_rollout(meta_rollout, gamma=0.99, lambda_=1.0)
        fetches = [self.meta_summary_op, self.meta_train_op, self.global_step]

        feed_dict = {
            meta_policy.x: batch.si['x'],
            meta_policy.subgoal_prev: batch.si['subgoal_prev'],
            meta_policy.reward_prev: batch.si['reward_prev'],
            self.meta_ac: batch.a,
            self.meta_adv: batch.adv,
            self.meta_r: batch.r,
            meta_policy.state_in[0]: batch.features[0],
            meta_policy.state_in[1]: batch.features[1],
        }
        fetched = sess.run(fetches, feed_dict=feed_dict)
        self.summary_writer.add_summary(fetched[0], fetched[-1])

        return terminal_end

    def _sub_process(self, sess, subgoal):
        sess.run(self.sync)  # copy weights from shared to local
        sub_rollout = PartialRollout()
        sub_policy = self.local_sub_network
        meta_reward = 0

        terminal_end = False
        for _ in range(self.num_local_steps):
            fetched = sub_policy.act(self.last_state, self.last_action,
                                     self.last_reward, subgoal, *self.last_features)
            action, value, features = fetched[0], fetched[1], fetched[2:]

            # argmax to convert from one-hot
            state, episode_reward, terminal, info = self.env.step(action.argmax())
            # reward clipping to the range of [-1, 1]
            extrinsic_reward = max(min(episode_reward, 1), -1)

            def get_mask(shape, subgoal):
                mask = np.zeros(shape)
                if subgoal < 36:
                    y = subgoal // 6
                    x = subgoal % 6
                    mask[y*14:(y+1)*14, x*14:(x+1)*14] = 1
                mask = np.stack([mask] * 3)
                return mask

            def compute_intrinsic(state, last_state, subgoal):
                f = sub_policy.feature(state)
                last_f = sub_policy.feature(last_state)
                if self.intrinsic_type == 'feature':
                    diff = np.abs(f - last_f)
                    return self.eta * diff[subgoal] / (np.sum(diff) + 1e-10)
                else:
                    diff = f - last_f
                    diff = diff * diff
                    mask = get_mask(diff.shape, subgoal)
                    return self.eta * np.sum(mask * diff) / (np.sum(diff) + 1e-10)

            intrinsic_reward = compute_intrinsic(state, self.last_state, subgoal.argmax())
            reward = self.beta * extrinsic_reward + (1 - self.beta) * intrinsic_reward

            meta_reward += extrinsic_reward
            # meta_reward += reward
            self.intrinsic_rewards += intrinsic_reward
            self.extrinsic_rewards += extrinsic_reward

            # collect the experience
            si = {
                'x': self.last_state,
                'action_prev': self.last_action,
                'reward_prev': self.last_reward,
                'subgoal': subgoal
            }
            sub_rollout.add(si, action, reward, value, terminal, self.last_features)

            self.length += 1
            self.rewards += episode_reward

            self.last_state = state
            self.last_action = action
            self.last_features = features
            self.last_reward = [reward]

            timestep_limit = self.env.spec.tags.get('wrapper_config.TimeLimit.max_episode_steps')
            if terminal or self.length >= timestep_limit:
                terminal_end = True

                summary = tf.Summary()
                summary.value.add(tag='global/episode_reward',
                                  simple_value=self.rewards)
                summary.value.add(tag='global/extrinsic_reward',
                                  simple_value=self.extrinsic_rewards)
                summary.value.add(tag='global/intrinsic_reward',
                                  simple_value=self.intrinsic_rewards)
                summary.value.add(tag='global/episode_length',
                                  simple_value=self.length)
                self.summary_writer.add_summary(summary, sub_policy.global_step.eval())
                self.summary_writer.flush()

                print("Episode finished. Ep rewards: %.5f (In: %.5f, Ex: %.5f). Length: %d" %
                      (self.rewards, self.intrinsic_rewards, self.extrinsic_rewards, self.length))
                break

        if not terminal_end:
            sub_rollout.r = sub_policy.value(self.last_state,
                                             self.last_action,
                                             self.last_reward,
                                             subgoal, *self.last_features)

        self.local_steps += 1
        should_compute_summary = self.task == 0 and self.local_steps % 11 == 0

        # sub rollout
        batch = process_rollout(sub_rollout, gamma=0.99, lambda_=1.0)
        fetches = [self.train_op, self.global_step]
        if should_compute_summary:
            fetches = [self.summary_op] + fetches
        feed_dict = {
            sub_policy.x: batch.si['x'],
            sub_policy.action_prev: batch.si['action_prev'],
            sub_policy.reward_prev: batch.si['reward_prev'],
            sub_policy.subgoal: batch.si['subgoal'],
            self.ac: batch.a,
            self.adv: batch.adv,
            self.r: batch.r,
            sub_policy.state_in[0]: batch.features[0],
            sub_policy.state_in[1]: batch.features[1],
        }

        fetched = sess.run(fetches, feed_dict=feed_dict)
        if should_compute_summary:
            self.summary_writer.add_summary(fetched[0], fetched[-1])

        return self.last_state, meta_reward, terminal_end


    def evaluate(self, sess):
        """
        run one episode and process experience to train both meta and sub networks
        """
        env = self.env

        sub_policy = self.local_sub_network
        meta_policy = self.local_meta_network

        self.last_state = env.reset()
        self.last_meta_state = env.reset()
        self.last_features = sub_policy.get_initial_features()
        self.last_meta_features = meta_policy.get_initial_features()

        self.last_action = np.zeros(self.action_space)
        self.last_subgoal = np.zeros(self.subgoal_space)
        self.last_reward = np.zeros(1)
        self.last_meta_reward = np.zeros(1)

        self.length = 0
        self.rewards = 0
        self.extrinsic_rewards = 0
        self.intrinsic_rewards = 0

        terminal_end = False
        frames = [self.last_state]
        while not terminal_end:
            frames_, terminal_end = self._meta_evaluate(sess)
            frames.extend(frames_)
        frames = np.stack(frames)
        return frames, self.rewards, self.length

    def _meta_evaluate(self, sess):
        meta_policy = self.local_meta_network

        terminal_end = False
        frames = []
        for _ in range(self.num_local_meta_steps):
            fetched = meta_policy.act(self.last_meta_state, self.last_subgoal,
                                      self.last_meta_reward, *self.last_meta_features)
            subgoal, meta_features = fetched[0], fetched[2:]

            if self.bptt == 20:
                meta_reward = 0
                for _ in range(5):
                    frames_, state, reward, terminal_end = self._sub_evaluate(sess, subgoal)
                    frames.extend(frames_)
                    meta_reward += reward
                    if terminal_end:
                        break
            elif self.bptt == 100:
                frames_, state, meta_reward, terminal_end = self._sub_evaluate(sess, subgoal)
                frames.extend(frames_)

            self.last_meta_state = state
            self.last_subgoal = subgoal
            self.last_meta_reward = [meta_reward]
            self.last_meta_features = meta_features
            if terminal_end:
                break
        return frames, terminal_end

    def _sub_evaluate(self, sess, subgoal):
        sub_policy = self.local_sub_network
        meta_reward = 0
        frames = []

        for _ in range(self.num_local_steps):
            fetched = sub_policy.act(self.last_state, self.last_action,
                                     self.last_reward, subgoal, *self.last_features)
            action, features = fetched[0], fetched[2:]

            # argmax to convert from one-hot
            state, episode_reward, terminal, info = self.env.step(action.argmax())
            # reward clipping to the range of [-1, 1]
            extrinsic_reward = max(min(episode_reward, 1), -1)
            frames.append(state)

            if self.visualise:
                self.env.render()

            def get_mask(shape, subgoal):
                mask = np.zeros(shape)
                if subgoal < 36:
                    y = subgoal // 6
                    x = subgoal % 6
                    mask[y*14:(y+1)*14, x*14:(x+1)*14] = 1
                mask = np.stack([mask] * 3)
                return mask

            def compute_intrinsic(state, last_state, subgoal):
                if self.intrinsic_type == 'feature':
                    f = sub_policy.feature(state)
                    last_f = sub_policy.feature(last_state)
                    diff = np.abs(f - last_f)
                    return self.eta * diff[subgoal] / (np.sum(diff) + 1e-10)
                else:
                    diff = state - last_state
                    diff = diff * diff
                    mask = get_mask(diff.shape, subgoal)
                    return self.eta * np.sum(mask * diff) / (np.sum(diff) + 1e-10)

            intrinsic_reward = compute_intrinsic(state, self.last_state, subgoal.argmax())
            reward = self.beta * extrinsic_reward + (1 - self.beta) * intrinsic_reward

            meta_reward += extrinsic_reward
            # meta_reward += reward
            self.intrinsic_rewards += intrinsic_reward
            self.extrinsic_rewards += extrinsic_reward

            self.length += 1
            self.rewards += episode_reward

            self.last_state = state
            self.last_action = action
            self.last_features = features
            self.last_reward = [reward]

            timestep_limit = self.env.spec.tags.get('wrapper_config.TimeLimit.max_episode_steps')
            if terminal or self.length >= timestep_limit:
                print("Episode finished. Ep rewards: %.5f (In: %.5f, Ex: %.5f). Length: %d" %
                      (self.rewards, self.intrinsic_rewards, self.extrinsic_rewards, self.length))
                return frames, state, meta_reward, True
        return frames, state, meta_reward, False
