#!/usr/bin/env python
import argparse
import logging
import sys
import signal
import time
import os

import imageio
from tqdm import tqdm
import tensorflow as tf

from a3c import A3C
from envs import create_env

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


# Disables write_meta_graph argument, which freezes entire process and is mostly useless.
class FastSaver(tf.train.Saver):
    def save(self, sess, save_path, global_step=None, latest_filename=None,
             meta_graph_suffix="meta", write_meta_graph=True):
        super(FastSaver, self).save(sess, save_path, global_step, latest_filename,
                                    meta_graph_suffix, False)


def run(args, server):
    env = create_env(args.env_id)
    trainer = A3C(env, args.task, args.visualise, args.intrinsic_type, args.bptt)

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

    config = tf.ConfigProto(device_filters=["/job:ps", "/job:worker/task:{}/cpu:0".format(args.task)])
    logdir = os.path.join(args.log_dir, 'train')

    summary_writer = tf.summary.FileWriter(logdir + "_%d" % args.task)

    logger.info("Events directory: %s_%s", logdir, args.task)
    sv = tf.train.Supervisor(is_chief=(args.task == 0),
                             logdir=logdir,
                             saver=saver,
                             summary_op=None,
                             init_op=init_op,
                             init_fn=init_fn,
                             summary_writer=summary_writer,
                             ready_op=tf.report_uninitialized_variables(variables_to_save),
                             global_step=trainer.global_step,
                             save_model_secs=30,
                             save_summaries_secs=30)

    video_dir = os.path.join(args.log_dir, 'train_videos_' + args.intrinsic_type)
    if not os.path.exists(video_dir):
        os.makedirs(video_dir)
    video_filename = video_dir + "/%s_%010d_%d.gif"
    print("Video saved at %s" % video_dir)

    num_global_steps = 300000000
    num_record_steps = 1000000
    last_record_step = 0

    logger.info(
        "Starting session. If this hangs, we're mostly likely waiting to connect to the parameter server. " +
        "One common cause is that the parameter server DNS name isn't resolving yet, or is misspecified.")
    with sv.managed_session(server.target, config=config) as sess, sess.as_default():
        sess.run(trainer.meta_sync)
        sess.run(trainer.sync)
        trainer.start(sess, summary_writer)
        global_step = sess.run(trainer.global_step)
        logger.info("Starting training at step=%d", global_step)

        pbar = tqdm(total=num_global_steps)
        pbar.update(global_step)

        while not sv.should_stop() and (not num_global_steps or global_step < num_global_steps):
            trainer.process(sess)

            new_global_step = sess.run(trainer.global_step)
            pbar.set_description('')
            pbar.update(max(1, new_global_step - global_step))
            global_step = new_global_step

            if args.task == 0 and global_step - last_record_step > num_record_steps:
                sess.run(trainer.meta_sync)
                sess.run(trainer.sync)
                last_record_step = global_step
                frames, reward, length = trainer.evaluate(sess)
                imageio.mimsave(video_filename % (args.env_id, global_step, reward), frames, fps=30)

    # Ask for all the services to stop.
    sv.stop()
    logger.info('reached %s steps. worker stopped.', global_step)


def cluster_spec(num_workers, num_ps):
    """
More tensorflow setup for data parallelism
"""
    cluster = {}
    port = 12222

    all_ps = []
    host = '127.0.0.1'
    for _ in range(num_ps):
        all_ps.append('{}:{}'.format(host, port))
        port += 1
    cluster['ps'] = all_ps

    all_workers = []
    for _ in range(num_workers):
        all_workers.append('{}:{}'.format(host, port))
        port += 1
    cluster['worker'] = all_workers
    return cluster


def main(_):
    """
Setting up Tensorflow for data parallel work
"""

    parser = argparse.ArgumentParser(description=None)
    parser.add_argument('-v', '--verbose', action='count', dest='verbosity', default=0, help='Set verbosity.')
    parser.add_argument('--task', default=0, type=int, help='Task index')
    parser.add_argument('--job-name', default="worker", help='worker or ps')
    parser.add_argument('--num-workers', default=1, type=int, help='Number of workers')
    parser.add_argument('--log-dir', default="/tmp/pong", help='Log directory path')
    parser.add_argument('--env-id', default="PongDeterministic-v3", help='Environment id')

    # Add visualisation argument
    parser.add_argument('--visualise', action='store_true',
                        help="Visualise the gym environment by running env.render() between each timestep")

    # Add model parameters
    parser.add_argument('--intrinsic-type', type=str, default='feature',
                        choices=['feature', 'pixel'], help="feature or pixel")
    parser.add_argument('--bptt', type=int, default=100,
                        help="BPTT")

    args = parser.parse_args()
    spec = cluster_spec(args.num_workers, 1)
    cluster = tf.train.ClusterSpec(spec).as_cluster_def()

    def shutdown(signal, frame):
        logger.warn('Received signal %s: exiting', signal)
        sys.exit(128+signal)
    signal.signal(signal.SIGHUP, shutdown)
    signal.signal(signal.SIGINT, shutdown)
    signal.signal(signal.SIGTERM, shutdown)

    if args.job_name == "worker":
        server = tf.train.Server(cluster, job_name="worker", task_index=args.task,
                                 config=tf.ConfigProto(intra_op_parallelism_threads=1, inter_op_parallelism_threads=2))
        run(args, server)
    else:
        server = tf.train.Server(cluster, job_name="ps", task_index=args.task,
                                 config=tf.ConfigProto(device_filters=["/job:ps"]))
        while True:
            time.sleep(1000)

if __name__ == "__main__":
    tf.app.run()
