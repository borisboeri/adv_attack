"""Evaluation.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys
import argparse
import os
from datetime import datetime
import math
import time

import numpy as np
import tensorflow as tf
import loss

def main(eval_log_dir, train_log_dir, args):
    """
    Eval the model for a number of steps.
    Args:
        eval_log_dir: directory where to write eval data
        train_log_dir: directory where the model to evaluate is
        data_dir: dataset directory
        dataset_id: id of the dataset to use
        save2file: Set to True to save outputs
    """

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

    data_dir = os.path.join(args.data_dir, args.data)

    with tf.Graph().as_default():
        # Build model
        img_op, labels_op = data.inputs(True, data_dir, args.batch_size)
        logits_op = model.model(img_op, is_training=False, scope_name='original')
        loss_op, acc_op = loss.loss_classif(logits_op, labels_op)
        #acc_op = tf.nn.in_top_k(logits_op, labels_op, 1)
        cm_op=tf.confusion_matrix(labels_op, tf.argmax(logits_op, axis=1),10)
        
        # Restore the moving average version of the learned variables for eval.
        variable_averages = tf.train.ExponentialMovingAverage(args.moving_average_decay)
        variables_to_restore = variable_averages.variables_to_restore()
        saver = tf.train.Saver(variables_to_restore)

        # Build the summary operation based on the TF collection of Summaries.
        summary_op = tf.summary.merge_all()
        graph_def = tf.get_default_graph().as_graph_def()
        summary_writer = tf.summary.FileWriter(eval_log_dir, graph_def=graph_def)
        
        with tf.Session() as sess:
            ckpt = tf.train.get_checkpoint_state(train_log_dir)
            if ckpt and ckpt.model_checkpoint_path:
                print("checkpoint path: ", ckpt.model_checkpoint_path)
                saver.restore(sess, ckpt.model_checkpoint_path)
                global_step = ckpt.model_checkpoint_path.split('/')[-1].split('-')[-1]
            else:
                print('No checkpoint file found')
                return

            # Start the queue runners.
            coord = tf.train.Coordinator()
            try:
                threads = []
                for qr in tf.get_collection(tf.GraphKeys.QUEUE_RUNNERS):
                    threads.extend(qr.create_threads(sess, coord=coord, daemon=True, start=True))

                num_iter = int(math.ceil(args.test_set_size / args.batch_size))
                mean_loss=0
                mean_acc=0
                loss_count = 0
                cm = np.zeros((10,10))
                while (loss_count < (num_iter+1)) and not coord.should_stop():
                    (loss_np, acc_np, cm_np)= sess.run([loss_op, acc_op, cm_op])
                    mean_acc += np.sum(acc_np)
                    mean_loss += np.sum(loss_np)
                    cm += cm_np
                    loss_count += 1 # net already return the mean loss over batch

                mean_loss /= loss_count
                mean_acc /= loss_count                    

                # Compute precision @ 1.
                print('%s: mean_loss = %.3f, mean_acc: %.3f' % (datetime.now(), mean_loss, mean_acc))
                summary = tf.Summary()
                summary.ParseFromString(sess.run(summary_op))
                summary.value.add(tag='mean_loss', simple_value=mean_loss)
                summary.value.add(tag='mean_acc', simple_value=mean_acc)
                summary_writer.add_summary(summary, global_step)

                np.savetxt(os.path.join(eval_log_dir, 'cm.txt'), cm)

            
            except Exception as e:  # pylint: disable=broad-except
                coord.request_stop(e)

            coord.request_stop()
            coord.join(threads, stop_grace_period_secs=10)


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--log_dir', type=str, help='path to log dir')
    parser.add_argument('--xp_name', type=str, help='xp name')
    parser.add_argument('--train_log_dir', type=str, help='path to log dir')
    parser.add_argument('--data_dir', type=str, help='full path to datasets')
    parser.add_argument('--data', type=str, help='dataset name {cifar, mnist}')
    parser.add_argument('--h', type=int, default=480, help='new height')
    parser.add_argument('--w', type=int, default=704, help='new width')
    parser.add_argument('--train_set_size', type=int, default=50000, help='training set size')
    parser.add_argument('--test_set_size', type=int, default=50000, help='test set size')
    parser.add_argument('--model', type=str, help='net model')
    parser.add_argument('--batch_size', type=int, help='batch size')
    parser.add_argument('--margin', type=float, default=0.0, help='margin of the triplet loss')
    parser.add_argument('--moving_average_decay', type=float, default=0.9999, help='')
    args = parser.parse_args()
    
    train_log_dir = os.path.join(args.train_log_dir, args.xp_name, 'train')
    if not os.path.exists(train_log_dir):
        print('Error: train log dir does not exists: %s' %train_log_dir)
        exit(1)

    eval_log_dir = os.path.join(args.train_log_dir, args.xp_name, 'val')
    #if tf.gfile.Exists(eval_log_dir):
    #    tf.gfile.DeleteRecursively(eval_log_dir)
    #tf.gfile.MakeDirs(eval_log_dir)

    main(eval_log_dir, train_log_dir, args)

