"""
Evaluation of a given checkpoint in the standard and adversarial sense.  Can be
called as an infinite loop going through the checkpoints in the model directory
as they appear and evaluating them. Accuracy and average loss are printed and
added as tensorboard summaries.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
from datetime import datetime
import json
import math
import os
import sys
import time

import numpy as np
import tensorflow as tf
from tqdm import trange

import cifar10_input
import resnet
from spatial_attack import SpatialAttack
import utilities

# A function for evaluating a single checkpoint
def evaluate(model, attack, sess, config, summary_writer=None):
    num_eval_examples = config.eval.num_eval_examples
    eval_batch_size = config.eval.batch_size
    data_path = config.data.data_path

    model_dir = config.model.output_dir
    # Setting up the Tensorboard and checkpoint outputs
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)

    cifar = cifar10_input.CIFAR10Data(data_path)
    global_step = tf.contrib.framework.get_or_create_global_step()
    # Iterate over the samples batch-by-batch
    num_batches = int(math.ceil(num_eval_examples / eval_batch_size))
    total_xent_nat = 0.
    total_xent_adv = 0.
    total_corr_nat = 0
    total_corr_adv = 0

    for ibatch in trange(num_batches):
      bstart = ibatch * eval_batch_size
      bend = min(bstart + eval_batch_size, num_eval_examples)

      x_batch = cifar.eval_data.xs[bstart:bend, :]
      y_batch = cifar.eval_data.ys[bstart:bend]

      noop_trans = np.zeros([len(x_batch), 3])
      if config.eval.adversarial_eval:
          x_batch_adv, adv_trans = attack.perturb(x_batch, y_batch, sess)
      else:
          x_batch_adv, adv_trans = x_batch, noop_trans

      dict_nat = {model.x_input: x_batch,
                  model.y_input: y_batch,
                  model.transform: noop_trans,
                  model.is_training: False}

      dict_adv = {model.x_input: x_batch_adv,
                  model.y_input: y_batch,
                  model.transform: adv_trans,
                  model.is_training: False}

      cur_corr_nat, cur_xent_nat = sess.run([model.num_correct, model.xent],
                                            feed_dict = dict_nat)
      cur_corr_adv, cur_xent_adv = sess.run([model.num_correct, model.xent],
                                            feed_dict = dict_adv)

      total_xent_nat += cur_xent_nat
      total_xent_adv += cur_xent_adv
      total_corr_nat += cur_corr_nat
      total_corr_adv += cur_corr_adv

    avg_xent_nat = total_xent_nat / num_eval_examples
    avg_xent_adv = total_xent_adv / num_eval_examples
    acc_nat = total_corr_nat / num_eval_examples
    acc_adv = total_corr_adv / num_eval_examples

    if summary_writer:
        summary = tf.Summary(value=[
              tf.Summary.Value(tag='xent_adv_eval', simple_value= avg_xent_adv),
              tf.Summary.Value(tag='xent_nat_eval', simple_value= avg_xent_nat),
              tf.Summary.Value(tag='xent_adv', simple_value= avg_xent_adv),
              tf.Summary.Value(tag='xent_nat', simple_value= avg_xent_nat),
              tf.Summary.Value(tag='accuracy_adv_eval', simple_value= acc_adv),
              tf.Summary.Value(tag='accuracy_nat_eval', simple_value= acc_nat),
              tf.Summary.Value(tag='accuracy_adv', simple_value= acc_adv),
              tf.Summary.Value(tag='accuracy_nat', simple_value= acc_nat)])
        summary_writer.add_summary(summary, global_step.eval(sess))

    step = global_step.eval(sess)
    print('Eval at step: {}'.format(step))
    print('  natural: {:.2f}%'.format(100 * acc_nat))
    print('  adversarial: {:.2f}%'.format(100 * acc_adv))
    print('  avg nat xent: {:.4f}'.format(avg_xent_nat))
    print('  avg adv xent: {:.4f}'.format(avg_xent_adv))

    result = {'nat': '{:.2f}%'.format(100 * acc_nat),
              'adv': '{:.2f}%'.format(100 * acc_adv)}
    with open('job_result.json', 'w') as result_file:
        json.dump(result, result_file, sort_keys=True, indent=4)

def loop(model, attack, config, summary_writer=None):

    last_checkpoint_filename = ''
    already_seen_state = False
    model_dir = config.model.output_dir
    saver = tf.train.Saver()

    while True:
      cur_checkpoint = tf.train.latest_checkpoint(model_dir)

      # Case 1: No checkpoint yet
      if cur_checkpoint is None:
        if not already_seen_state:
          print('No checkpoint yet, waiting ...', end='')
          already_seen_state = True
        else:
          print('.', end='')
        sys.stdout.flush()
        time.sleep(10)
      # Case 2: Previously unseen checkpoint
      elif cur_checkpoint != last_checkpoint_filename:
        print('\nCheckpoint {}, evaluating ...   ({})'.format(cur_checkpoint,
                                                              datetime.now()))
        sys.stdout.flush()
        last_checkpoint_filename = cur_checkpoint
        already_seen_state = False
        with tf.Session() as sess:
            # Restore the checkpoint
            saver.restore(sess, cur_checkpoint)
            evaluate(model, attack, sess, config, summary_writer)
      # Case 3: Previously evaluated checkpoint
      else:
        if not already_seen_state:
          print('Waiting for the next checkpoint ...   ({})   '.format(
                datetime.now()),
                end='')
          already_seen_state = True
        else:
          print('.', end='')
        sys.stdout.flush()
        time.sleep(10)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
                        description='Eval script options',
                        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-c', '--config', type=str,
                        help='path to config file',
                        default="config.json", required=False)
    parser.add_argument('--loop', help='continuously monitor model_dir'
                                       'evaluating new ckpt', 
                        action="store_true")
    args = parser.parse_args()

    config_dict = utilities.get_config(args.config)
    config = utilities.config_to_namedtuple(config_dict)

    model = resnet.Model(config.model)
    model_dir = config.model.output_dir

    global_step = tf.contrib.framework.get_or_create_global_step()
    attack = SpatialAttack(model, config.attack)

    if args.loop:
        eval_dir = os.path.join(model_dir, 'eval')
        if not os.path.exists(eval_dir):
          os.makedirs(eval_dir)
        summary_writer = tf.summary.FileWriter(eval_dir)

        loop(model, attack, config, summary_writer)
    else:
        saver = tf.train.Saver()

        cur_checkpoint = tf.train.latest_checkpoint(model_dir)
        if cur_checkpoint is None:
            print('No checkpoint found.')
        else:
            with tf.Session() as sess:
                # Restore the checkpoint
                print('Evaluating checkpoint {}'.format(cur_checkpoint))
                saver.restore(sess, cur_checkpoint)
                evaluate(model, attack, sess, config)
