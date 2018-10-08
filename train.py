"""A binary to train using a single GPU.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys
import argparse
import os.path
from datetime import datetime
import time
import math
import numpy as np
from six.moves import xrange
import tensorflow as tf

import loss
import init_weights

def train_network(args): 
    if args.model=='vgg':
        import model_vgg as model
    elif args.model=='alexnet':
        import model_alexnet as model
    elif args.model=='tf':
        import model_tf as model
    else:
        print('Error: %s is not a valid model' %args.model)
        exit(1)

    if args.data=='cifar':
        import data_cifar as data
    elif args.data=='mnist':
        import data_mnist as data
    else:
        print('Error: %s is not a valid dataset' %args.data)
        exit(1)

    train_log_dir = os.path.join(args.train_log_dir, args.xp_name, 'train')
    data_dir = os.path.join(args.data_dir, args.data)
    
    with tf.Graph().as_default():
        global_step = tf.Variable(0, trainable=False) 
        img_op, labels_op = data.inputs(False, data_dir, args.batch_size) 
        logits_op = model.model(img_op, is_training=True, scope_name='original') # generate the graph model
        perturbed_logits_op = model.model(img_op, is_training=True, scope_name='perturbed') # generate the graph model
        loss_op, acc_op = loss.loss_classif(logits_op, labels_op)
        train_op = loss.train(loss_op, global_step, args) # trains

        var_list =  tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES)
        for var in var_list:
            print(var.op.name)
     
        # Set saver to restore network before eval
        variable_averages = tf.train.ExponentialMovingAverage(args.moving_average_decay)
        variables_to_restore = variable_averages.variables_to_restore()
        saver = tf.train.Saver(variables_to_restore)
   
        # Set summary op, restore vars
        summary_op = tf.summary.merge_all()
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            
            ### begin var  assignment OK !!! Do this one !
            if args.start == 0:
                if args.model=='vgg':
                    init_weights.restore_vgg(sess)
                elif args.model=='alexnet':
                    restore_alexnet(sess)
                global_step = 0
            else:
                ckpt = tf.train.get_checkpoint_state(train_log_dir)
                print("checkpoint path: ", ckpt.model_checkpoint_path)
                if ckpt and ckpt.model_checkpoint_path:
                    saver.restore(sess, ckpt.model_checkpoint_path)
                    global_step = ckpt.model_checkpoint_path.split('/')[-1].split('-')[-1]
                else:
                    print('No checkpoint file found')
                    return
            #### end var assignment

            # Start the queue runners.
            coord = tf.train.Coordinator()
            try:
                threads = []
                for qr in tf.get_collection(tf.GraphKeys.QUEUE_RUNNERS):
                  threads.extend(qr.create_threads(sess, coord=coord, daemon=True,start=True))
                summary_writer = tf.summary.FileWriter(train_log_dir, graph_def=sess.graph_def)

                max_step_train = args.epochs * int(math.ceil(args.train_set_size/args.batch_size))
                min_step = int(global_step)
                max_step = min_step + max_step_train
                for step in range(min_step, max_step):
                    start_time = time.time()
                    _, loss_np, acc_np = sess.run([train_op, loss_op, acc_op])
                    assert not np.isnan(loss_np), 'Model diverged with loss = NaN'
                    duration = time.time() - start_time
                    step+=1
                    
                    if step % args.display_interval == 0:
                        num_examples_per_step = args.batch_size
                        examples_per_sec = num_examples_per_step / duration
                        sec_per_batch = float(duration)
                        format_str = ('%s: step %d, loss=%.3f, acc=%.3f, (%.1f examples/sec; %.3f ' 'sec/batch)')
                        print (format_str % (datetime.now(), step, loss_np, acc_np, examples_per_sec, sec_per_batch))
                                          
                    if (step % args.summary_interval==0):
                        summary_str = sess.run(summary_op)
                        summary_writer.add_summary(summary_str, step)

                    if step == 200:
                        break
 
                # Save model
                summary_str = sess.run(summary_op)
                summary_writer.add_summary(summary_str, step)
                checkpoint_path = os.path.join(train_log_dir, 'model.ckpt')
                saver.save(sess, checkpoint_path, global_step=step)

            except Exception as e:  # pylint: disable=broad-except
                coord.request_stop(e)
            
            coord.request_stop()
            coord.join(threads, stop_grace_period_secs=10)


if __name__ == '__main__':  
   
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_log_dir', type=str, help='path to log dir')
    parser.add_argument('--xp_name', type=str, help='xp name')
    parser.add_argument('--display_interval', type=int, default=10, help='')
    parser.add_argument('--summary_interval', type=int, default=10, help='')
    parser.add_argument('--data_dir', type=str, help='full path to datasets')
    parser.add_argument('--data', type=str, help='dataset name {cifar, mnist}')
    parser.add_argument('--h', type=int, default=480, help='new height')
    parser.add_argument('--w', type=int, default=704, help='new width')
    parser.add_argument('--train_set_size', type=int, default=50000, help='training set size')
    parser.add_argument('--test_set_size', type=int, default=50000, help='test set size')
    parser.add_argument('--model', type=str, help='net model')
    parser.add_argument('--batch_size', type=int, help='batch size')
    parser.add_argument('--epochs', type=int, help='num of epochs to train')
    parser.add_argument('--lr', type=float, default=1e-03, help='learning rate')
    parser.add_argument('--adam_b1', type=float, default=0.9, help='adam beta1')
    parser.add_argument('--adam_b2', type=float, default=0.999, help='adam beta2')
    parser.add_argument('--adam_eps', type=float, default=1e-08, help='adam epsilon')
    parser.add_argument('--moving_average_decay', type=float, default=0.9999, help='')
    parser.add_argument('--start', type=int, default=0, help='Set to 0 to load vgg weights')
    args = parser.parse_args()
    
    #train_log_dir = os.path.join(args.train_log_dir, args.xp_name, 'train')
    #if tf.gfile.Exists(train_log_dir):
    #    tf.gfile.DeleteRecursively(train_log_dir)
    #    tf.gfile.MakeDirs(train_log_dir)

    train_network(args) #epochs_to_train, train_log_dir, data_dir, dataset_id)

   
