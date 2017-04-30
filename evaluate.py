#############################################################################################
# Author: Safa Messaoud                                                                     #
# E-MAIL: messaou2@illinois.edu                                                             #
# Instituation: University of Illinois at Urbana-Champaign                                  #
# Date: February 2017                                                                       #
# Description: Evaluate the model.This script should be run concurrently with training      #
# so that summaries show up in TensorBoard.                                                 #
#############################################################################################

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math
import os.path
import time


import numpy as np
import tensorflow as tf
import configuration


import configuration
import DeepEEG_model
import argparse


tf.logging.set_verbosity(tf.logging.INFO)


def do_eval(model, saver, summary_writer, summary_op,checkpoint_dir,num_eval_examples,min_global_step):
  #Evaluates the latest model checkpoint.
  #Args:
  # model: Instance of DeepEcog; the model to evaluate.
  # saver: Instance of tf.train.Saver for restoring model Variables.
  # summary_writer: Instance of SummaryWriter.
  # summary_op: Op for generating model summaries.
  # checkpoint_dir: path to the directory with the checkpoint
  # num_eval_examples: number of examples in the evaluation set
  # min_global_step: minimum global step to run evaluation
  
  model_path = tf.train.latest_checkpoint(checkpoint_dir)
  if not model_path:
    tf.logging.info("Skipping evaluation. No checkpoint found in: %s",checkpoint_dir)
    return

  with tf.Session() as sess:
    # Load model from checkpoint.
    tf.logging.info("Loading model from checkpoint: %s", model_path)
    saver.restore(sess, model_path)
    global_step = tf.train.global_step(sess, model.global_step.name)
    tf.logging.info("Successfully loaded %s at global step = %d.",os.path.basename(model_path), global_step)
    
    if global_step < min_global_step:
      tf.logging.info("Skipping evaluation. Global step = %d < %d", global_step,min_global_step)
      return

    # Start the queue runners.
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(coord=coord)

    # Run evaluation on the latest checkpoint.
    evaluate_model(
    sess=sess,
    model=model,
    global_step=global_step,
    num_eval_examples=num_eval_examples,
    summary_writer=summary_writer,
    summary_op=summary_op)
  
    coord.request_stop()
    coord.join(threads, stop_grace_period_secs=10)



def evaluate_model(sess, model, global_step, summary_writer, summary_op,num_eval_examples):
  #Computes accuracy over the evaluation dataset.
  #	Args:
  #		sess: Session object.
  #		model: Instance of ShowAndTellModel; the model to evaluate.
  #		global_step: Integer; global step of the model checkpoint.
  #		summary_writer: Instance of SummaryWriter.
  #		summary_op: Op for generating model summaries.
  #   num_eval_examples: Number of examples for evaluation
  
  # Log model summaries on a single batch.
  summary_str = sess.run(summary_op)
  summary_writer.add_summary(summary_str, global_step)
  
  # Compute accuracy over the entire eval dataset.
  num_eval_batches = int(math.ceil(num_eval_examples / model.config.batch_size))
  sum_losses=0
  sum_accuracies=0
  
  start_time = time.time()
  
  for i in range(num_eval_batches):
    cross_entropy_loss = sess.run([model.batch_loss])
    sum_losses = sum_losses+np.sum(cross_entropy_loss)

        #compute accuracy
        accuracy=sess.run([model.batch_accuracy])
        sum_accuracies=sum_accuracies+np.sum(accuracy)

        #compute recall
        recall=sess.run([model.batch_recall])
        sum_recalls=sum_recalls+np.sum(recall)

        #compute precision
        precision=sess.run([model.batch_precision])
        sum_precisions=sum_precisions+np.sum(precision)


        #compute F1 score
        f1_score=sess.run([model.batch_f1_score])
        sum_f1_score=sum_f1_score+np.sum(f1_score)


        if not i % 100:
            tf.logging.info("Computed evaluation metrics for %d of %d batches.", i + 1,num_eval_batches)


    eval_time = time.time() - start_time

    sum_losses=sum_losses/num_eval_batches
    sum_accuracies=sum_accuracies/num_eval_batches
    sum_recalls=sum_recalls/num_eval_batches
    sum_precisions=sum_precisions/num_eval_batches
    sum_f1_score=sum_f1_score/num_eval_batches

    tf.logging.info("sum_losses = %f (%.2g sec)", sum_losses, eval_time)
    tf.logging.info("sum_accuracies = %f (%.2g sec)", sum_accuracies, eval_time)
    tf.logging.info("sum_recalls = %f (%.2g sec)", sum_recalls, eval_time)
    tf.logging.info("sum_precisions = %f (%.2g sec)", sum_precisions, eval_time)
    tf.logging.info("sum_f1_score = %f (%.2g sec)", sum_f1_score, eval_time)


    # Log loss to the SummaryWriter.
    summary = tf.Summary()
    value = summary.value.add()
    value.simple_value = sum_losses
    value.tag = "loss"
    summary_writer.add_summary(summary, global_step)

    # Log accuracy to the SummaryWriter.
    summary = tf.Summary()
    value = summary.value.add()
    value.simple_value = sum_accuracies
    value.tag = "accuracy"
    summary_writer.add_summary(summary, global_step)

    # Log recall to the SummaryWriter.
    summary = tf.Summary()
    value = summary.value.add()
    value.simple_value = sum_recalls
    value.tag = "recall"
    summary_writer.add_summary(summary, global_step)


    # Log precision to the SummaryWriter.
    summary = tf.Summary()
    value = summary.value.add()
    value.simple_value = sum_precisions
    value.tag = "precision"
    summary_writer.add_summary(summary, global_step)


    # Log f1 to the SummaryWriter.
    summary = tf.Summary()
    value = summary.value.add()
    value.simple_value = sum_precisions
    value.tag = "f1_score"
    summary_writer.add_summary(summary, global_step)


    # Write the Events file to the eval directory.
    summary_writer.flush()
    tf.logging.info("Finished processing evaluation at global step %d.",global_step)



