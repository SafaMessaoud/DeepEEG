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
    nb_models=8

    # Model Choice
    model_choice=1
    
    
    
    ################# Model0 ###############
    # RNN state size
    self.model0_rnn_state_size=40
    
    ################# Model1 ###############
    #output size of the conv layer
    self.model1_output1_dim=5
    self.model1_rnn_state_size=40
    self.model1_cnn_representation_size=10
    
    ################# Model2 ###############
    self.model2_kernel_width=3
    self.model2_kernel_hight=3
    self.model2_stride_width=2
    self.model2_output1_dim=5
    
    ################# Model3 ###############
    self.model3_rnn_state_size=40
    self.model3_att_elec_dim=self.model3_rnn_state_size

  ################ Training Parameters ################
    self.batch_size=128
	

		learning_rate_decay_factor
		optimizer

		clip_gradients

   


sample_name
label_name







