"""
Implementation of a PGD attack bounded in L_infty.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import numpy as np

class LinfPGDAttack:
  def __init__(self, model, config):
    """Attack parameter initialization. The attack performs k steps of
       size a, while always staying within epsilon from the initial
       point."""
    self.model = model
    self.epsilon = config.epsilon
    self.num_steps = config.num_steps
    self.step_size = config.step_size
    self.rand = config.random_start

    if config.loss_function == 'xent':
      loss = model.xent
    elif config.loss_function == 'cw':
      label_mask = tf.one_hot(model.y_input,
                              10,
                              on_value=1.0,
                              off_value=0.0,
                              dtype=tf.float32)
      correct_logit = tf.reduce_sum(label_mask * model.pre_softmax, axis=1)
      wrong_logit = tf.reduce_max((1-label_mask) * model.pre_softmax
                                                    - 1e4 * label_mask, axis=1)
      loss = wrong_logit - correct_logit 
    else:
      print('Unknown loss function. Defaulting to cross-entropy')
      loss = model.xent

    self.grad = tf.gradients(loss, model.x_input)[0]

  def perturb(self, x_nat, y, sess, trans=None):
    """
    Given a set of examples (x_nat, y), returns a set of adversarial
    examples within epsilon of x_nat in l_infinity norm. An optional
    spatial perturbations can be given as (trans_x, trans_y, rot).
    """

    if self.rand:
        x = x_nat + np.random.uniform(-self.epsilon, self.epsilon, x_nat.shape)
    else:
        x = np.copy(x_nat)

    if trans is None:
        trans = np.zeros([len(x_nat),3])

    for i in range(self.num_steps):
        curr_dict = {self.model.x_input: x,
                     self.model.y_input: y,
                     self.model.transform: trans,
                     self.model.is_training: False}
        grad = sess.run(self.grad, feed_dict=curr_dict)

        x = np.add(x, self.step_size * np.sign(grad), out=x, casting='unsafe')

        x = np.clip(x, x_nat - self.epsilon, x_nat + self.epsilon)
        x = np.clip(x, 0, 255) # ensure valid pixel range

    return x
