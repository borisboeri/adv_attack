"""
Dfns for loss, optimizer, gradients computation, summaries ... i.e. anything
that has to do with training
"""
from six.moves import urllib
import tensorflow as tf
from math import sqrt
import numpy as np
from tensorflow.python.framework import ops
from tensorflow.python.ops import gen_nn_ops
from tensorflow.python.ops import array_ops

def loss_classif(logits, labels):
    """Add Loss to all the trainable variables.

    Add summary for for "Loss" and "Loss/avg".
    Args:
      logits: Logits from inference().
      labels: Labels from distorted_inputs or inputs(). 2-D tensor
              of shape [batch_size, 1]

    Returns:
      Loss tensor of type float.
    """
    labels = tf.cast(labels, tf.int64)
    loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=labels))
    correct_prediction = tf.equal(tf.argmax(logits, 1), labels)
    acc = tf.reduce_mean(tf.cast(correct_prediction,tf.float32))
    
    tf.summary.scalar('loss', loss)
    tf.summary.scalar('acc', acc)
    tf.add_to_collection('losses', loss)
    return tf.add_n(tf.get_collection('losses'), name='total_loss'), acc

def _add_loss_summaries(total_loss):
  """Add summaries for losses in CIFAR-10 model.

  Generates moving average for all losses and associated summaries for
  visualizing the performance of the network.

  Args:
    total_loss: Total loss from loss().
  Returns:
    loss_averages_op: op for generating moving averages of losses.
  """
  # Compute the moving average of all individual losses and the total loss.
  loss_averages = tf.train.ExponentialMovingAverage(0.9, name='avg')
  losses = tf.get_collection('losses')
  loss_averages_op = loss_averages.apply(losses + [total_loss])

  # Attach a scalar summary to all individual losses and the total loss; do the
  # same for the averaged version of the losses.
  for l in losses + [total_loss]:
    # Name each loss as '(raw)' and name the moving average version of the loss
    # as the original loss name.
    tf.summary.scalar(l.op.name +' (raw)', l)
    tf.summary.scalar(l.op.name, loss_averages.average(l))

  return loss_averages_op


def train(total_loss, global_step, args):
    """
    Create an optimizer and apply to all trainable variables. Add moving
    average for all trainable variables.
    Args:
      total_loss: Total loss from loss().
      global_step: Integer Variable counting the number of training steps
        processed.
    Returns:
      train_op: op for training.
    """
    # Generate moving averages of all losses and associated summaries.
    loss_averages_op = _add_loss_summaries(total_loss)
    
    # debug: display trainable variables
    #var_to_train = tf.trainable_variables()
    #print('\nvar to train')
    #for var in var_to_train:
    #    print(var.op.name)

    # Compute gradients.
    with tf.control_dependencies([loss_averages_op]):
        opt = tf.train.AdamOptimizer(args.lr, args.adam_b1, args.adam_b2, args.adam_eps)
        
        # Warning: if you net has bn use this block 
        update_ops =  tf.get_collection(tf.GraphKeys.UPDATE_OPS) #line for BN
        with tf.control_dependencies(update_ops):
            grads = opt.compute_gradients(total_loss, var_list=tf.trainable_variables())
            apply_gradient_op = opt.apply_gradients(grads, global_step=global_step)
    
    # Warning: if your net does not use bn, use this block and comment the one above
    #   grads = opt.compute_gradients(total_loss, var_list=tf.trainable_variables())
    #apply_gradient_op = opt.apply_gradients(grads, global_step=global_step)

    # Add histograms for trainable variables.
    for var in tf.trainable_variables():
        tf.summary.histogram(var.op.name, var)

    # Add histograms for gradients.
    for grad, var in grads:
        if grad is not None:
            tf.summary.histogram(var.op.name + '/gradients', grad)

    # Track the moving averages of all trainable variables.
    variable_averages = tf.train.ExponentialMovingAverage(args.moving_average_decay, global_step)

    with tf.control_dependencies([apply_gradient_op]):
        variables_averages_op = variable_averages.apply(tf.trainable_variables())

    return variables_averages_op


