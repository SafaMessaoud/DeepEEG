####################################################################
# Author: Safa Messaoud                                            #
# E-MAIL: messaou2@illinois.edu                                    #
# Instituation: University of Illinois at Urbana-Champaign         #
# Date: February 2017                                              #
# Description: DeepEEG model and training configuration            #
####################################################################

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


class ModelConfig(object):
  # Wrapper class for model hyperparameters
  def __init__(self):
  ################ Model Parameters ################
    # Number of electrodes 
    self.nb_channels=64,

    # Number of frequencies
    self.nb_freq=3

    # Number of time windows
    self.nb_time_windows=7

    # Number of samples
    self.nb_samples=2670

    # Total number of models
    self.nb_models=8

    # Number of Classes
    self.num_classes=4
    
    # Trial data name
    self.sample_name="trial/sample"
    
    # Label name
    self.label_name="trial/label"

    # Number of Models
    nb_models=7

    # Model Choice
    model_choice=2

    ################# Model0 ###############
    # RNN state size
    self.model0_rnn_state_size=40
    
    ################# Model1 ###############
    # Output size of the conv layer
    self.model1_output1_dim=5
    
    # RNN state size	
    self.model1_rnn_state_size=40

    # Size of the CNN output
    self.model1_cnn_representation_size=10
    
    ################# Model2 ###############
    # CNN Kernel 
    self.model2_kernel_width=3
    self.model2_kernel_hight=3
    
    # Cov stride	
    self.model2_stride_width=1

    # Output size of the conv layer
    self.model2_output1_dim=5

    # RNN state size	
    self.model2_rnn_state_size=40
    
    ################# Model3 ###############
    # RNN state size	 
    self.model3_rnn_state_size=40

    # Electrode representation dimension	 
    self.model3_att_elec_dim=self.model3_rnn_state_size

  ################ Training Parameters ################
    # Batch size
    self.batch_size=128
	
    # number of values per input shard
    self.values_per_input_shard = 100

    # input queue capacity factor
    self.input_queue_capacity_factor =  2

    # number of input reader threads
    self.num_input_reader_threads = 1

    # number of prefetching threads
    self.num_fetching_threads = 4

    # weight decay factor
    self.weight_decay=0.00004

    # frequency for saving summaries
    self.save_summaries_secs=3
    self.save_interval_secs=3

    # initial learning rate
    self.initial_learning_rate = 2.0

    # learning rate decay factor
    self.learning_rate_decay_factor = 0.5

    # number of epochs for the learning rate decay
    self.num_epochs_per_decay = 4.0

    # maximum size to clip gradients
    self.clip_gradients = 5.0

    # optimizer
    self.optimizer = "SGD"

    # maximum number of checkpoints to keep
    self.max_checkpoints_to_keep = 5

    # number of examples per epoch
    self.num_examples_per_epoch = 1000

    # total number of global steps
    self.number_of_steps=2000000

    #frequency of logging
    self.log_every_n_steps=1
