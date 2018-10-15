#  Copyright 2016 The TensorFlow Authors. All Rights Reserved.
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.
"""Convolutional Neural Network Estimator for MNIST, built with tf.layers."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf

tf.logging.set_verbosity(tf.logging.INFO)

def model(images, is_training, scope_name, reuse=False):
    """Model function for CNN."""
    # Input Layer
    # Reshape X to 4-D tensor: [batch_size, width, height, channels]
    # MNIST images are 28x28 pixels, and have one color channel
    feat_dict = {} # init dict
    with tf.variable_scope(scope_name, reuse) as scope:
        # Input Tensor Shape: [batch_size, 28, 28, 1]
        # Output Tensor Shape: [batch_size, 28, 28, 32]
        conv1 = tf.layers.conv2d(inputs=images, filters=32, kernel_size=[5, 5],
              padding="same", activation=tf.nn.relu)
        feat_dict['conv1'] = conv1

        # Pooling Layer #1
        # First max pooling layer with a 2x2 filter and stride of 2
        # Input Tensor Shape: [batch_size, 28, 28, 32]
        # Output Tensor Shape: [batch_size, 14, 14, 32]
        pool1 = tf.layers.max_pooling2d(inputs=conv1, pool_size=[2, 2], strides=2)
        feat_dict['pool1'] = conv1
        
        # Convolutional Layer #2
        # Computes 64 features using a 5x5 filter.
        # Padding is added to preserve width and height.
        # Input Tensor Shape: [batch_size, 14, 14, 32]
        # Output Tensor Shape: [batch_size, 14, 14, 64]
        conv2 = tf.layers.conv2d( inputs=pool1, filters=64, kernel_size=[5, 5],
              padding="same", activation=tf.nn.relu)
        feat_dict['conv2'] = conv1
        
        # Pooling Layer #2
        # Second max pooling layer with a 2x2 filter and stride of 2
        # Input Tensor Shape: [batch_size, 14, 14, 64]
        # Output Tensor Shape: [batch_size, 7, 7, 64]
        pool2 = tf.layers.max_pooling2d(inputs=conv2, pool_size=[2, 2], strides=2)
        feat_dict['pool2'] = conv1
        
        # Flatten tensor into a batch of vectors
        # Input Tensor Shape: [batch_size, 7, 7, 64]
        # Output Tensor Shape: [batch_size, 7 * 7 * 64]
        pool2_flat = tf.reshape(pool2, [-1, 7 * 7 * 64])
        feat_dict['pool2_flat'] = conv1
        
        # Dense Layer
        # Densely connected layer with 1024 neurons
        # Input Tensor Shape: [batch_size, 7 * 7 * 64]
        # Output Tensor Shape: [batch_size, 1024]
        dense = tf.layers.dense(inputs=pool2_flat, units=1024, activation=tf.nn.relu)
        feat_dict['dense'] = conv1
       
       # Add dropout operation; 0.6 probability that element will be kept
        dropout = tf.layers.dropout( inputs=dense, rate=0.4, training=is_training)
        
        # Logits layer
        # Input Tensor Shape: [batch_size, 1024]
        # Output Tensor Shape: [batch_size, 10]
        logits = tf.layers.dense(inputs=dropout, units=10)

        return logits #,feat_dict


def grad_perturbed(loss_op, feat_pertubed_dict, feat_dict, loss_value):
    """ 
    WARNING: 
    """
    grad_dict ={}
    with tf.variable_scope("perturbed", reuse) as scope:
        grad_dict['loss'] = tf.gradients(loss_op, logits, loss_value)
        grad_dict['logits'] = tf.gradients(loss_op, dense, grad_dict['loss'])
        grad_dict['dense'] = tf.gradients(feat_dict['dense'], feat_dict['pool2'], grad_dict['logits']) # maybe pool2_flat
        grad_dict['pool2'] = tf.gradients(feat_dict['pool2'], feat_dict['conv2'], grad_dict['dense'])
        grad_dict['conv2'] = tf.gradients(feat_dict['conv2'], feat_dict['pool1'], grad_dict['pool2'])
        grad_dict['pool1'] = tf.gradients(feat_dict['pool1'], feat_dict['conv1'], grad_dict['conv2'])
        #grad_dict['conv1'] = tf.gradients(feat_dict['conv1'], feat_dict['dsad


