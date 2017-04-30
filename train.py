###########################################################################
# Author: Safa Messaoud                                                   #
# E-MAIL: messaou2@illinois.edu                                           #
# Instituation: University of Illinois at Urbana-Champaign                #
# Date: February 2017                                                     #
# Description: Train the model                                            #
###########################################################################

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys
sys.path.insert(0, 'path to DeepEEG')


import tensorflow as tf
import argparse
import configuration
import DeepEEG_model

tf.logging.set_verbosity(tf.logging.INFO)

def parse_arguments(parser):
  parser.add_argument('checkpoint_dir', type=str, default= 'DeepEEG/checkpoint_dir', metavar='<checkpoint_dir>', help='Directory for saving and loading model checkpoints')	
  parser.add_argument('input_file_pattern', type=str,default='DeepEEG/train_dir' , metavar='<input_file_pattern>', help='File pattern of sharded TFRecord input files.')
  parser.add_argument('--number_of_steps', type=int, default=1000000 ,metavar='<number_of_steps>', help='Number of training steps')
  parser.add_argument('--log_every_n_steps', type=int,default=1 , metavar='<log_every_n_steps>', help='Frequency at which loss and global step are logged')	
  parser.add_argument('--model_choice', type=int,default=1 , metavar='<model_choice>', help='choose the model to be trained:\n (1)CNN+maxpool')
  
  args = parser.parse_args()
  return args


def main():
  parser = argparse.ArgumentParser()
  args = parse_arguments(parser)
  
  model_config = configuration.ModelConfig()
  model_config.input_file_pattern = args.input_file_pattern
  model_config.model_choice = args.model_choice


  #check if the model number is valid
  if (args.model_choice>model_config.nb_models):
    tf.logging.info("the number of the model should be between 1 and %d", model_config.nb_models)
    return

  # Create checking directory
  checkpoint_dir = args.checkpoint_dir
  
  if (not tf.gfile.IsDirectory(checkpoint_dir)):
    tf.logging.info("Creating logging directory: %s", checkpoint_dir)
    tf.gfile.MakeDirs(checkpoint_dir)
    
  # Build the TensorFlow graph.
  g = tf.Graph()

  with g.as_default():
    # Build the model.
    model = DeepEEG_model.DeepEEG_model(model_config, mode="train")
    model.build()

    # Set up the learning rate.
    learning_rate_decay_fn = None

    learning_rate = tf.constant(model_config.initial_learning_rate)

    if model_config.learning_rate_decay_factor > 0:
      num_batches_per_epoch = (model_config.num_examples_per_epoch / model_config.batch_size)
      
      #the number of batches after which the learning rate is decayed
      decay_steps = int(num_batches_per_epoch * model_config.num_epochs_per_decay)

      def _learning_rate_decay_fn(learning_rate, global_step):
        return tf.train.exponential_decay(learning_rate,global_step,decay_steps=decay_steps,decay_rate=model_config.learning_rate_decay_factor,staircase=True)

      learning_rate_decay_fn = _learning_rate_decay_fn
			

    # Set up the training ops.
    train_op = tf.contrib.layers.optimize_loss(
    loss=model.total_loss,
    global_step=model.global_step,
    learning_rate=learning_rate,
    optimizer=model_config.optimizer,
    clip_gradients=model_config.clip_gradients,
    learning_rate_decay_fn=learning_rate_decay_fn)
    
    # Set up the Saver for saving and restoring model checkpoints.
    saver = tf.train.Saver(max_to_keep=model_config.max_checkpoints_to_keep)

    # Run training.
    tf.contrib.slim.learning.train(
    train_op,
    checkpoint_dir,
    log_every_n_steps=args.log_every_n_steps,
    graph=g,  
    global_step=model.global_step,
    number_of_steps=args.number_of_steps,
    init_fn=model.init_fn,
    save_summaries_secs=model_config.save_summaries_secs,
    save_interval_secs=model_config.save_interval_secs,
    saver=saver)

if __name__ == "__main__":
  main()
