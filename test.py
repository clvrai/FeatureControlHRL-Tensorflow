#!/usr/bin/env python
import argparse
import logging
import sys
import signal
import os

import tensorflow as tf
import imageio

from a3c import A3C
from envs import create_env


logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# parsing cmd arguments
parser = argparse.ArgumentParser(description="Test commands")
parser.add_argument('-e', '--env-id', type=str, default="MontezumaRevenge-v0",
                    help="Environment id")
parser.add_argument('-l', '--log-dir', type=str, default="/tmp/montezuma",
                    help="Log directory path")

# Add visualisation argument
parser.add_argument('--visualise', action='store_true',
                    help="Visualise the gym environment by running env.render() between each timestep")

# Add model parameters
parser.add_argument('--intrinsic-type', type=str, default='feature',
                    choices=['feature', 'pixel'], help="Feature-control or Pixel-control")
parser.add_argument('--bptt', type=int, default=100,
                    help="BPTT")


# Disables write_meta_graph argument, which freezes entire process and is mostly useless.
class FastSaver(tf.train.Saver):
    def save(self, sess, save_path, global_step=None, latest_filename=None,
             meta_graph_suffix="meta", write_meta_graph=True):
        super(FastSaver, self).save(sess, save_path, global_step, latest_filename,
                                    meta_graph_suffix, False)


def run(args):
    env = create_env(args.env_id)
    trainer = A3C(env, None, args.visualise, args.intrinsic_type, args.bptt)

    # Variable names that start with "local" are not saved in checkpoints.
    variables_to_save = [v for v in tf.global_variables() if not v.name.startswith("local")]
    init_op = tf.variables_initializer(variables_to_save)
    init_all_op = tf.global_variables_initializer()
    saver = FastSaver(variables_to_save)

    var_list = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, tf.get_variable_scope().name)
    logger.info('Trainable vars:')
    for v in var_list:
        logger.info('  %s %s', v.name, v.get_shape())

    def init_fn(ses):
        logger.info("Initializing all parameters.")
        ses.run(init_all_op)

    logdir = os.path.join(args.log_dir, 'train')
    summary_writer = tf.summary.FileWriter(logdir)
    logger.info("Events directory: %s", logdir)

    sv = tf.train.Supervisor(is_chief=True,
                             logdir=logdir,
                             saver=saver,
                             summary_op=None,
                             init_op=init_op,
                             init_fn=init_fn,
                             summary_writer=summary_writer,
                             ready_op=tf.report_uninitialized_variables(variables_to_save),
                             global_step=None,
                             save_model_secs=0,
                             save_summaries_secs=0)

    video_dir = os.path.join(args.log_dir, 'test_videos_' + args.intrinsic_type)
    if not os.path.exists(video_dir):
        os.makedirs(video_dir)
    video_filename = video_dir + "/%s_%02d_%d.gif"
    print("Video saved at %s" % video_dir)

    with sv.managed_session() as sess, sess.as_default():
        trainer.start(sess, summary_writer)
        rewards = []
        lengths = []
        for i in range(10):
            frames, reward, length = trainer.evaluate(sess)
            rewards.append(reward)
            lengths.append(length)
            imageio.mimsave(video_filename % (args.env_id, i, reward), frames, fps=30)

        print('Evaluation: avg. reward %.2f    avg.length %.2f' %
              (sum(rewards) / 10.0, sum(lengths) / 10.0))

    # Ask for all the services to stop.
    sv.stop()


def main(_):
    args, unparsed = parser.parse_known_args()

    def shutdown(signal, frame):
        logger.warn('Received signal %s: exiting', signal)
        sys.exit(128+signal)
    signal.signal(signal.SIGHUP, shutdown)
    signal.signal(signal.SIGINT, shutdown)
    signal.signal(signal.SIGTERM, shutdown)

    run(args)

if __name__ == "__main__":
    tf.app.run()
