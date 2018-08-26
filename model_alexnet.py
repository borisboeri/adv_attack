
"""
- netmork model
- loss
- optimizer
- summary
"""
# pylint: disable=missing-docstring
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import gzip
import os
import re
import sys
import tarfile


from six.moves import urllib
import tensorflow as tf
from math import sqrt
import numpy as np
from tensorflow.python.framework import ops
from tensorflow.python.ops import gen_nn_ops
from tensorflow.python.ops import array_ops

FLAGS = tf.app.flags.FLAGS

@ops.RegisterGradient("MaxPoolGradWithArgmax")
def _MaxPoolGradGradWithArgmax(op, grad):
  print(len(op.outputs))
  print(len(op.inputs))
  print(op.name)
  return (array_ops.zeros(
      shape=array_ops.shape(op.inputs[0]),
      dtype=op.inputs[0].dtype), array_ops.zeros(
          shape=array_ops.shape(op.inputs[1]), dtype=op.inputs[1].dtype),
          gen_nn_ops._max_pool_grad_grad_with_argmax(
              op.inputs[0],
              grad,
              op.inputs[2],
              op.get_attr("ksize"),
              op.get_attr("strides"),
              padding=op.get_attr("padding")))
              
def model(images, is_training, reuse=False):
    """
    Args:
      images: 
    Returns:
      Logits.
    """
    
    print('inference::input', images.get_shape())
    bn = True
    lrn = False
 
    with tf.variable_scope('conv1', reuse=reuse) as scope: #1
        conv1 = tf.layers.conv2d(inputs=images, filters=96, kernel_size=(11,11), strides=(4,4),
                padding="same", kernel_initializer=tf.contrib.layers.xavier_initializer())
        if bn:
            conv1 = tf.contrib.layers.batch_norm(conv1, fused=True, decay=0.9, is_training=is_training)
        if lrn:
            conv1 = tf.nn.local_response_normalization(conv1,depth_radius=2,alpha=1.99999994948e-05,beta=0.75,bias=1.0,name='norm1')
        conv1 = tf.nn.relu(conv1)
        print('conv1', conv1.get_shape())
        pool1 = tf.nn.max_pool(conv1, ksize=[1,3,3,1], strides=[1,2,2,1], padding='SAME', name='pool') #pool1
        print('pool1', pool1.get_shape())
 
    pool1_1, pool1_2 = tf.split(pool1, 2, 3)
    print('pool1_1.shape: ', pool1_1.get_shape())
    print('pool1_2.shape: ', pool1_2.get_shape())
    
    with tf.variable_scope('conv2_1', reuse=reuse) as scope:#3
        conv2_1 = tf.layers.conv2d(inputs=pool1_1, filters=128, kernel_size=(5, 5),
                padding="same", kernel_initializer=tf.contrib.layers.xavier_initializer())
        conv2_1 = tf.nn.relu(conv2_1)
        print('conv2_1', conv2_1.get_shape())
        
    with tf.variable_scope('conv2_2', reuse=reuse) as scope:#3
        conv2_2 = tf.layers.conv2d(inputs=pool1_2, filters=128, kernel_size=(5, 5),
                padding="same", kernel_initializer=tf.contrib.layers.xavier_initializer())
        conv2_2 = tf.nn.relu(conv2_2)
        print('conv2_2', conv2_2.get_shape())
        
    conv2 = tf.concat([conv2_1, conv2_2], 3)
    if bn:
        conv2 = tf.contrib.layers.batch_norm(conv2,fused=True, decay=0.9, is_training=is_training)
    if lrn:
        conv2 = tf.nn.local_response_normalization(conv2,depth_radius=2,alpha=1.99999994948e-05,beta=0.75,bias=1.0,name='norm2')
    pool2 = tf.nn.max_pool(conv2, ksize=[1,3,3,1], strides=[1,2,2,1], padding='VALID', name='pool') #pool2
    print('pool2', pool2.get_shape())
  
    with tf.variable_scope('conv3', reuse=reuse) as scope:#5
        conv3 = tf.layers.conv2d(inputs=pool2, filters=384, kernel_size=(3, 3),
                padding="same", kernel_initializer=tf.contrib.layers.xavier_initializer())
        if bn:
            conv3 = tf.contrib.layers.batch_norm(conv3, fused=True, decay=0.9, is_training=is_training)
        conv3 = tf.nn.relu(conv3)
        print('conv3', conv3.get_shape())

    conv3_1, conv3_2 = tf.split(conv3, 2, 3)
    print('conv3_1.shape: ', conv3_1.get_shape())
    print('conv3_2.shape: ', conv3_2.get_shape())

    with tf.variable_scope('conv4_1', reuse=reuse) as scope:
        conv4_1 = tf.layers.conv2d(inputs=conv3_1, filters=192, kernel_size=(3, 3),
                padding="same", kernel_initializer=tf.contrib.layers.xavier_initializer())
        if bn:
            conv4_1 = tf.contrib.layers.batch_norm(conv4_1,fused=True, decay=0.9, is_training=is_training)
        conv4_1 = tf.nn.relu(conv4_1)
        print('conv4_1', conv4_1.get_shape())

    with tf.variable_scope('conv4_2', reuse=reuse) as scope:
        conv4_2 = tf.layers.conv2d(inputs=conv3_2, filters=192, kernel_size=(3, 3),
                padding="same", kernel_initializer=tf.contrib.layers.xavier_initializer())
        if bn:
            conv4_2 = tf.contrib.layers.batch_norm(conv4_2,fused=True, decay=0.9, is_training=is_training)
        conv4_2 = tf.nn.relu(conv4_2)
        print('conv4_2', conv4_2.get_shape())

    with tf.variable_scope('conv5_1', reuse=reuse) as scope:
        conv5_1 = tf.layers.conv2d(inputs=conv4_1, filters=128, kernel_size=(3, 3),
                padding="same", kernel_initializer=tf.contrib.layers.xavier_initializer())
        conv5_1 = tf.nn.relu(conv5_1)
        print('conv5_1', conv5_1.get_shape())

    with tf.variable_scope('conv5_2', reuse=reuse) as scope:
        conv5_2 = tf.layers.conv2d(inputs=conv4_2, filters=128, kernel_size=(3, 3),
                padding="same", kernel_initializer=tf.contrib.layers.xavier_initializer())
        conv5_2 = tf.nn.relu(conv5_2)
        print('conv5_2', conv5_2.get_shape())
    
    conv5 = tf.concat([conv5_1, conv5_2], 3)
    if bn:
        conv5 = tf.contrib.layers.batch_norm(conv5,fused=True,decay=0.9,is_training=is_training)

    pool5 = tf.nn.max_pool(conv5, ksize=[1,3,3,1], strides=[1,2,2,1], padding='VALID', name='pool')
    print('pool5', pool5.get_shape())

    with tf.variable_scope('fc6') as scope:
        shape = int(np.prod(pool5.get_shape()[1:]))
        in_ = tf.reshape(pool5, [-1, shape])
        fc6 = tf.nn.relu( tf.layers.dense(in_, 4096) )
        print('fc6', fc6.get_shape())

    with tf.variable_scope('fc7') as scope:
        fc7 = tf.nn.relu(tf.layers.dense(fc6, 4096))
        print('fc7', fc7.get_shape())

    with tf.variable_scope('fc8') as scope:
        fc8 = tf.nn.relu(tf.layers.dense(fc7, 1000))
        print('fc8', fc8.get_shape())
    
    return fc8

def loss(feat1, feat2, labels):
    """Add Loss to all the trainable variables.

    Add summary for for "Loss" and "Loss/avg".
    Args:
      logits: Logits from inference().
      labels: Labels from distorted_inputs or inputs(). 2-D tensor
              of shape [batch_size, 1]

    Returns:
      Loss tensor of type float.
    """
    margin = 0
    #d_op = tf.reduce_sum(tf.square(feat1 - feat2), (1,2,3))
    #d_op = tf.reduce_sum(tf.square(feat1 - feat2), (1))
    #d = tf.nn.l2_loss(feat1 - feat2)# / (header.BATCH_SIZE ) #* header.IMAGE_SIZE)
    d_op = tf.reduce_sum(tf.abs(feat1 - feat2), (1,2,3)) # paper recommends L1 to avoid local minima

    print('d.shape: ', d_op.get_shape())
    print('labels.shape: ', labels.get_shape())
    #d_sqrt = tf.sqrt(d)
    #loss = labels * tf.square(tf.maximum(0., margin - d_sqrt)) + (1 - labels) * d
    #loss = (1-labels) * tf.maximum(0., margin - d) + labels * d
    loss_b = labels * d_op + (1 - labels) * (margin - d_op)
    print('loss_b.shape: ', loss_b.get_shape())
    #loss = tf.reduce_sum(loss_b)
    loss = tf.reduce_mean(loss_b)
    tf.summary.scalar('loss', loss)
    
    tf.add_to_collection('losses', loss)
    # The total loss is defined as the l2 loss plus all of the weight
    # decay terms (L2 loss).
    return tf.add_n(tf.get_collection('losses'), name='total_loss'), loss_b#, d_op
    #return tf.add_n(tf.get_collection('losses'), name='total_loss')


def triplet_loss(feat1, feat2, feat3, args):
    """
        Triplet loss
    """
    margin = args.margin
    #d = tf.nn.l2_loss(feat1 - feat2)# / (header.BATCH_SIZE ) #* header.IMAGE_SIZE)
    dp = tf.reduce_sum(tf.abs(feat1 - feat2), (1,2,3)) # P example
    dn = tf.reduce_sum(tf.abs(feat1 - feat3), (1,2,3)) # N example
    dp_mean = tf.reduce_mean(dp)
    dn_mean = tf.reduce_mean(dn)
    tf.summary.scalar('dn', dn_mean)
    tf.summary.scalar('dp', dp_mean)
    loss_b = tf.maximum(0.0, margin + dp - dn)
    loss = tf.reduce_mean(loss_b) 
    #print('dn.shape: ', dn.get_shape())
    #print('loss_b.shape: ', loss_b.get_shape())
    tf.summary.scalar('loss', loss)
    
    tf.add_to_collection('losses', loss)
    # The total loss is defined as the l2 loss plus all of the weight
    # decay terms (L2 loss).
    return tf.add_n(tf.get_collection('losses'), name='total_loss'), dp, dn
    #return tf.add_n(tf.get_collection('losses'), name='total_loss')

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
    """Train CIFAR-10 model.
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

    #var_to_train = tf.trainable_variables()
    #print('\nvar to train')
    #for var in var_to_train:
    #    print(var.op.name)

    # Compute gradients.
    with tf.control_dependencies([loss_averages_op]):
        opt = tf.train.AdamOptimizer(args.lr, args.adam_b1, args.adam_b2, args.adam_eps)
        
        # TODO BN
        update_ops =  tf.get_collection(tf.GraphKeys.UPDATE_OPS) #line for BN
        with tf.control_dependencies(update_ops):
            grads = opt.compute_gradients(total_loss, var_list=tf.trainable_variables())
            apply_gradient_op = opt.apply_gradients(grads, global_step=global_step)
    
    # no BN
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
        #train_op = tf.no_op(name='train')

    return variables_averages_op #train_op


