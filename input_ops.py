#######################################################################################
# Author: Safa Messaoud                                                               # 
# E-mail: messaou2@illinois.edu                                                       # 
# Instituation: University of Illinois at Urbana-Champaign                            #
# Date: February 2017                                                                 # 
# Description: input ops for data prefetching, parsing sequence example and batching  #                                                                              #
#######################################################################################

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import configuration as config
import numpy as np



def parse_sequence_example(serialized,nb_freq,nb_time_points,sample, label ):
  #Parses a tensorflow.SequenceExample into a label and an EEG recording
  #Args:
  # serialized: A scalar string Tensor; a single serialized SequenceExample.
  # nb_freq: number of frequencies
  # number of time points
  # sample: Name of SequenceExample context feature containing the EEG recording
  # label: Name of SequenceExample feature list containing the label.
  #Returns:
  #  encoded_label: an integer corresponding to the label of the trial.
  #  encoded_trial: a 3D tensor with the EEG-recording.
  
  print('parsing start....')
  
  context, sequence = tf.parse_single_sequence_example(serialized,
  context_features={label: tf.FixedLenFeature([], dtype=tf.int64)},
  sequence_features={sample: tf.FixedLenSequenceFeature([], dtype=tf.float32)})
  
  encoded_label = context[label]-1
  encoded_trial = sequence[sample]
  
  #reshaping
  encoded_trial=tf.reshape(encoded_trial,(config.nb_channels,config.nb_freq,config.nb_time_windows)
  
  print('parsing end....')
  return encoded_label, encoded_trial

def prefetch_input_data(reader,
  file_pattern,
  is_training,
  batch_size,
  values_per_shard,
  input_queue_capacity_factor=16,
  num_reader_threads=1,
  shard_queue_name="filename_queue",
  value_queue_name="input_queue"):
  
  #Prefetches string values from disk into an input queue.
  #Args:
  # reader: Instance of tf.ReaderBase.
  # file_pattern: Comma-separated list of file patterns 
  # is_training: Boolean; whether prefetching for training or eval.
  # batch_size: Model batch size used to determine queue capacity.
  # values_per_shard: Approximate number of values per shard.
  # input_queue_capacity_factor: Minimum number of values to keep in the queue in multiples of values_per_shard. See comments above.
  # num_reader_threads: Number of reader threads to fill the queue.
  # shard_queue_name: Name for the shards filename queue.
  # value_queue_name: Name for the values input queue.
  #Returns:
  # A Queue containing prefetched string values.
  

  data_files = []
  for pattern in file_pattern.split(","):
    data_files.extend(tf.gfile.Glob(pattern))
    
  if not data_files:
    tf.logging.fatal("Found no input files matching %s", file_pattern)
  else:
    tf.logging.info("Prefetching values from %d files matching %s", len(data_files), file_pattern)

  if is_training:
    filename_queue = tf.train.string_input_producer(data_files, shuffle=True, capacity=16, name=shard_queue_name)
    min_queue_examples = values_per_shard * input_queue_capacity_factor
    capacity = min_queue_examples + 100 * batch_size
    values_queue = tf.RandomShuffleQueue(capacity=capacity,min_after_dequeue=min_queue_examples,dtypes=[tf.string],name="random_" + value_queue_name)
  else:
    filename_queue = tf.train.string_input_producer(data_files, shuffle=False, capacity=1, name=shard_queue_name)
    capacity = values_per_shard + 3 * batch_size
    values_queue = tf.FIFOQueue(capacity=capacity, dtypes=[tf.string], name="fifo_" + value_queue_name)

  enqueue_ops = []

  for _ in range(num_reader_threads):
    _, value = reader.read(filename_queue)
    print('prefetch value: ',value)
    enqueue_ops.append(values_queue.enqueue([value]))

  tf.train.queue_runner.add_queue_runner(tf.train.queue_runner.QueueRunner(values_queue, enqueue_ops))
  print("queue/%s/fraction_of_%d_full" % (values_queue.name, capacity), tf.cast(values_queue.size(), tf.float32) * (1. / capacity))
  tf.summary.scalar("queue/%s/fraction_of_%d_full" % (values_queue.name, capacity), tf.cast(values_queue.size(), tf.float32) * (1. / capacity))

  return values_queue



def batch_data(trial_data,batch_size,queue_capacity,add_summaries=True):
  #Batches labels and EEG_recording data
  #Args:
  #   trial_data: A list of pairs [EEG_recording, label], 
  #   Each pair will be processed and added to the queue in a separate thread.
  #   batch_size: Batch size.
  #   queue_capacity: Queue capacity.
  #   add_summaries: If true, add caption length summaries.
  #Returns:
  #   batch_sample: A Tensor of shape [batch_size, nb_channel, nb_freq, nb_time_windows].
  #   batch_label: An int32 Tensor of shape [batch_size].
  
  enqueue_list = []
  
  for  sample,label in trial_data:
    enqueue_list.append([ label, sample])
    batch_label, batch_sample  = tf.train.batch_join(enqueue_list,batch_size=batch_size,capacity=queue_capacity,dynamic_pad=True,name="batch_and_pad")    
  
  return batch_label, batch_sample




