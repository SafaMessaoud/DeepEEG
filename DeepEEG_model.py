###########################################################################
# Author: Safa Messaoud                                                   #
# E-MAIL: messaou2@illinois.edu                                           #
# Instituation: University of Illinois at Urbana-Champaign                #
# Date: February 2017                                                     #
# Description: Implementation of DeepEEG models                           #
#                                                                         #
###########################################################################

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


import tensorflow as tf
import numpy as np
import sys
import input_op 
import configuration as config
import model0
import model1
import model2
import model3
import model4
import model5
import model6
import model7


class DeepEcog_model(object):
#DeepEcog_model implementation 
  
  def __init__(self,config, mode):
    #Basic setup
    #Args:
    #  config: Object containing configuration parameters.
    #  mode: "train" or "eval" 
	 
    assert mode in ["train", "eval"]
    self.config = config
    self.mode = mode

    # Reader for the input data.
    self.reader = tf.TFRecordReader()

    # Initialize all variables (apart from the weights) with a random uniform initializer.
    self.initializer = tf.global_variables_initializer()

    # Initialize the weights with xavier initializer
    self.weight_initializer = tf.contrib.layers.xavier_initializer(dtype=tf.float32)

    # Weight regularizer
    self.weights_regularizer=tf.contrib.layers.l2_regularizer(self.config.weight_decay)

    # Bach normalization
    self.normalizer_fn = tf.contrib.layers.batch_norm

    # A float Tensor with shape [batch_size].
    self.label = None

    # An float Tensor with shape [batch_size,nb_channels,nb_freq,nb_time_samples].
    self.trial_data = None

    # Evaluation metrics
    self.total_loss = None
    self.batch_loss = None
    self.batch_accuracy = None
    
    # Global step Tensor (number of batches seen so far by the graph).
    self.global_step = None

    def is_training(self):
    #Returns true if the model is built for training mode
      return self.mode == "train"

    def setup_global_step(self):
      global_step = tf.Variable(initial_value=0,
      name="global_step",
      trainable=False,
      collections=[tf.GraphKeys.GLOBAL_STEP, tf.GraphKeys.GLOBAL_VARIABLES])
      
    self.global_step=global_step
        

    def build_inputs(self):
      # Input prefetching and batching.
      #Outputs:
      # self.label
      # self.trial_data
       
      # Prefetch serialized SequenceExample protos.
      input_queue = input_op.prefetch_input_data(
      self.reader,
      input_file_pattern,
      is_training=self.is_training(),
      batch_size=self.config.batch_size,
      values_per_shard=self.config.values_per_input_shard,
      input_queue_capacity_factor=self.config.input_queue_capacity_factor,
      num_reader_threads=self.config.num_input_reader_threads)

      assert self.config.num_fetching_threads % 2 == 0

      trials_and_label=[]
      
      for thread_id in range(num_fetching_threads):
        serialized_sequence_example = input_queue.dequeue()
        enc_label,enc_trial_data =parse_sequence_example(
        serialized_sequence_example,               
        nb_freq=self.config.nb_freq,
        nb_time_points=self.config.nb_time_windows,
        sample=self.config.sample_name,
        label=self.config.label_name)
        
        trials_and_label.append([enc_trial_data, enc_label  ])

        # Batch inputs.
        queue_capacity = (2 * self.config.num_fetching_threads *self.config.batch_size)

        b_label, b_trial_data= input_op.batch_data(trials_and_label,batch_size=self.config.batch_size,queue_capacity=queue_capacity)
        
      self.label = b_label
      self.trial_data = b_trial_data
      
    def build_model(self):
      if (self.config.model_choice==0):
        logits=model0.model(self)
      elif (self.config.model_choice==1):
        logits=model1(self)      
      elif (self.config.model_choice==2):
        logits=model2.model(self)      
      elif (self.config.model_choice==3):
        logits=model3.model(self)      
      elif (self.config.model_choice==4):
        logits=model4.model(self)      
      elif (self.config.model_choice==5):
        logits=model5.model(self)      
      elif (self.config.model_choice==6):
        logits=model6.model(self)      
      elif (self.config.model_choice==7):
        logits=model7.model(self)            
      
      # cast Logits and Label
      logits = tf.cast(logits, tf.float64)
      label=tf.cast(self.label, tf.int64)  #(batch)
      
      #Compute batch loss
      losses_t = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=label,logits=logits)
      batch_loss = tf.reduce_mean(losses_t, name='batch_loss') 
      self.batch_loss= batch_loss
      
      #add batch loss to the collection of losses 
      batch_loss = tf.cast(batch_loss, tf.float32)
      self.batch_loss=batch_loss
      tf.losses.add_loss(batch_loss)
      total_loss = tf.losses.get_total_loss(add_regularization_losses=False)
      
      #determine the total number of parameters
      total_parameters=0
      
      for var in tf.trainable_variables():
        tf.summary.histogram("parameters/" + var.op.name, var)
          shape = var.get_shape()
          variable_parametes = 1
          for dim in shape:
            variable_parametes *= dim.value
            total_parameters += variable_parametes
            
      print('variable_parametes: ',total_parameters)            
      vars=tf.trainable_variables()

      # Compure regularized loss
      lossL2 = tf.add_n([ tf.nn.l2_loss(v) for v in vars if (('bias' not in v.name) and ('BatchNorm'  not in v.name)) ]) * 0.001        
      self.total_loss = tf.add(total_loss,lossL2)
      
         
      #compute batch accuracy
      softmax_out=tf.nn.softmax(logits)
      correct_prediction = tf.equal(label, tf.argmax(softmax_out,1))
      accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
      tf.logging.info('batch_accuracy: ',accuracy)
      self.batch_accuracy = accuracy
        
  def build(self):
  #Creates all ops for training and evaluation
    self.build_inputs()
    self.setup_global_step()
    self.build_model()

