#########################################################################################################################
# Author: Safa Messaoud                                                                                                 #
# E-Mail: messaou2@illinois.edu                                                                                         #
# Instituation: University of Illinois at Urbana-Champaign                                                              #
# Date: February 2017                                                                                                   #
#                                                                                                                       #
# Description:                                                                                                          #
#   * Convert the EEG data from https://github.com/pbashivan/EEGLearn/tree/master/Sample%20data into tf_records         #
#   * The files FeatureMat_timeWin.mat, Neuroscan_locs_orig.mat and trials_subNums.mat are expected to reside in        #
#   the folder 'DeepEEG/data'                                                                                           #
#   * The data for every patient is converted into a tf_record.                                                         #
#   * Each record within the TFRecord file is a serialized SequenceExample proto consisting of precisely one            #
#   label-EEG_recording pair corresponding to one trial                                                                 #
#########################################################################################################################

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import sys
from collections import Counter
from collections import namedtuple
from datetime import datetime
import os.path
import random
from random import shuffle
import threading
import pickle
import argparse
import os
import pandas as pd
from six.moves import xrange  
import numpy as np
import scipy.io as sio
import itertools
import glob
import scipy
import numpy as np
import tensorflow as tf
import shutil


def _int64_feature(value):
  #Wrapper for inserting an int64 Feature into a SequenceExample proto.	
  return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

def _float_feature(value):
  #Wrapper for inserting an float Feature into a SequenceExample proto.	
  return tf.train.Feature(float_list=tf.train.FloatList(value=[value]))

def _float_feature_list(values):
  #Wrapper for inserting a list of float Features into a SequenceExample proto.	
  return tf.train.FeatureList(feature=[_float_feature(v) for v in values])


def _to_sequence_example(trial,path):
  #Builds a SequenceExample proto for one trial (label-EEG_recorfing) pair
  #Args:
  #	trial: name of the pickled file containig the trial data
  #	path: path to the directory containing the trial data
  #Returns:
  #	A SequenceExample proto.
  
  #load the pickled file with thw trial data
  trial_file_name=  path+'/'+trial
  trial_file=open(trial_file_name, 'rb')
  sample = pickle.load(trial_file)
  label = pickle.load(trial_file)
  
  #reshape
  sample = np.reshape(sample, -1)
  label =np.reshape(label, -1)
  
  #transform into a SequenceExample proto
  context = tf.train.Features(feature={"trial/label": _int64_feature(int(label))})
  feature_lists = tf.train.FeatureLists(feature_list={"trial/sample": _float_feature_list(sample )})
  
  sequence_example = tf.train.SequenceExample(context=context, feature_lists=feature_lists)
  return sequence_example
  
def _process_trial_files(thread_index, ranges, name, trials, num_shards,output_dir,path):
  #Processes and saves a subset of trials as TFRecord files in one thread.
  #Args:
  #	thread_index: Integer thread identifier within [0, len(ranges)].
  #	ranges: A list of pairs of integers specifying the ranges of the dataset to process in parallel.
  #	name: Unique identifier specifying the dataset.
  #	trials: List of the pickled files containing the trials data
  #	num_shards: Integer number of shards for the output files.
  #	output_dir: directory were the tf_records are stored
  #	path: path to the directory where the trials data is stored
  
  #total number of threads to process all the batches
  num_threads = len(ranges)
  assert not num_shards % num_threads
  num_shards_per_batch = int(num_shards / num_threads)
  
  #determine the range of shards for each TF_record
  shard_ranges = np.linspace(ranges[thread_index][0], ranges[thread_index][1],num_shards_per_batch + 1).astype(int)
  
  #number of trials processed by the thread
  num_trials_in_thread = ranges[thread_index][1] - ranges[thread_index][0]
  
  #counter for the number of processes trials in this thread
  counter = 0
  
  for s in xrange(num_shards_per_batch):
    # Generate a sharded version of the file name
    shard = thread_index * num_shards_per_batch + s
    output_filename = "%s" % (name)
    output_file = os.path.join(output_dir, output_filename)
    
    #writer to the shard file
    writer = tf.python_io.TFRecordWriter(output_file)
    
    #counter for the number of trials in one shard
    shard_counter = 0
    
    trials_in_shard = np.arange(shard_ranges[s], shard_ranges[s + 1], dtype=int)
    
    for i in trials_in_shard:
      trial = trials[i]
      sequence_example = _to_sequence_example(trial,path)
      
      if sequence_example is not None:
        writer.write(sequence_example.SerializeToString())
        shard_counter += 1
        counter += 1
        print ("thread_index %d:%d ",thread_index ,counter)
        
      if not counter % 100:
        print("%s [thread %d]: Processed %d of %d items in thread batch." %(datetime.now(), thread_index, counter, num_trials_in_thread))
        sys.stdout.flush()
      
    #finish writing
    writer.close() 
      
    print("%s [thread %d]: Wrote %d trials to %s" %(datetime.now(), thread_index, shard_counter, output_file))
    sys.stdout.flush()
    shard_counter = 0 
      
  print("%s [thread %d]: Wrote %d trials to %d shards." %(datetime.now(), thread_index, counter, num_shards_per_batch))
  sys.stdout.flush() 
    
def _process_dataset(name, trials, num_shards, num_threads,output_dir,path):
  #Processes a complete data set and saves it as a TFRecord.
  #Args:
  #	name: Unique identifier specifying the dataset.
  #	trials: List of pickled files containg the trials data.
  #	num_shards: Integer number of shards for the output files.
  #	num_threads: Number of threads to convert the data in TFRecords
  #	output_dir: directory were the tf_records are stored
  #	path: path to the directory where the trials data is stored
  
  num_threads = min(num_shards, num_threads)
  spacing = np.linspace(0, len(trials), num_threads + 1).astype(np.int)
  ranges = []
  threads = []
  
  for i in xrange(len(spacing) - 1):
    ranges.append([spacing[i], spacing[i + 1]])
    
  # Create a mechanism for monitoring when all threads are finished.
  coord = tf.train.Coordinator()	
  
  # Launch a thread for each batch.
  print("Launching %d threads for spacings: %s" % (num_threads, ranges))
  
  for thread_index in xrange(len(ranges)):
    args = (thread_index, ranges, name, trials, num_shards,output_dir,path)
    t = threading.Thread(target=_process_trial_files, args=args)
    t.start()
    threads.append(t)
  
  # Wait for all the threads to terminate.
  coord.join(threads)
  print("%s: Finished processing all %d trials pairs in data set '%s'." %(datetime.now(), len(trials), name))
 
def generate_pickled_files(subj_num_file,feature_file,location_file,nb_time_windows,nb_freq):
  #create a directory for every patients where every trial ia saved as a pickled file 
  #Args:
  #	subj_num_file: path to the matfile with the subject numbers
  #	feature_file: path to the matfile with the EEG features
  #	location_file: path to the electrode location file
  #	nb_time_windows: number of time windows
  #	nb_freq: number of frequencies for each trial

  feats = scipy.io.loadmat(feature_file)['features']
  subj_nums = np.squeeze(scipy.io.loadmat(subj_num_file)['subjectNum'])
  locs = scipy.io.loadmat(location_file)['A']

  nb_samples=len(feats);
  nb_electrodes=len(locs);

  #extract the label
  Y=feats[:,-1]
  Y=Y.tolist()

  #remove the column corresponding to the level
  feats=feats[:,0:np.shape(feats)[1]-1]

  #reshape features
  X=np.transpose(np.reshape(feats, (nb_samples,nb_time_windows,nb_freq,nb_electrodes)), (0,3,2,1))
  X=X.tolist()

  #generate pickled files with the data
  for i in range(len(subj_nums)):
    print(i)
    sample=X[i]
    label=Y[i]
		
    dir_name='patient_data/s_'+str(subj_nums[i])
    if not tf.gfile.IsDirectory(dir_name): tf.gfile.MakeDirs(dir_name)

    file_name=dir_name+'/s_'+str(subj_nums[i])+'_'+str(i)+'.pckl'
    filehandler = open(file_name,"wb")
    pickle.dump(sample,filehandler)
    pickle.dump(label,filehandler)
    filehandler.close()
  
  #create directory to store the evaluation data
  if not tf.gfile.IsDirectory('patient_data/eval'): tf.gfile.MakeDirs('patient_data/eval')
  patient_dir_list= glob.glob('patient_data/s*')

  #store all the trials for one patient in tf_records
  for i in range(len(patient_dir_list)):
    patient_dir=patient_dir_list[i]
    dataset=os.listdir(patient_dir)
    for j in range(0,20):
      shutil.move(patient_dir+'/'+dataset[j],'patient_data/eval/')
	
	
	
def parse_arguments(parser):
  parser.add_argument('tf_records_dir', type=str,default= '/DeepEEG/tf_records_dir', metavar='<tf_records_dir>', help='The path to TF_Records')	
  parser.add_argument('subj_num_file', type=str,default= '/DeepEEG/data/trials_subNums.mat', metavar='<subj_num_file>', help='The path to mat file with the subject numbers')
  parser.add_argument('feature_file', type=str,default= '/DeepEEG/data/FeatureMat_timeWin.mat', metavar='<feature_file>', help='The path to the mat file with the EEG data')
  parser.add_argument('location_file', type=str,default= '/DeepEEG/data/Neuroscan_locs_orig.mat', metavar='<location_file>', help='The path to the mat file with the electrode location')
  parser.add_argument('--nb_shards', type=int,default= 1, metavar='<nb_shards>', help='Number of shards in TFRecord files')
  parser.add_argument('--nb_threads', type=int,default= 8,  metavar='<num_threads>', help='Number of threads to convert the data in TFRecords')
  parser.add_argument('--nb_time_windows', type=int,default= 7,  metavar='<nb_time_windows>', help='Number of time windows')
  parser.add_argument('--nb_freq', type=int,default= 3,  metavar='<nb_freq>', help='Number of frequencies')
  args = parser.parse_args()
  return args

def main():
  parser = argparse.ArgumentParser()
  args = parse_arguments(parser)

  #sanity checks
  def _is_valid_num_shards(num_shards):
    #Returns True if num_shards is compatible with num_threads
    return num_shards < args.nb_threads or not num_shards % args.num_threads

  assert _is_valid_num_shards(args.nb_shards), ("Please make the args.num_threads commensurate with args.nb_shards")

  #read the mat files and process them into pickled files
  generate_pickled_files(args.subj_num_file,args.feature_file,args.location_file,args.nb_time_windows,args.nb_freq)

  #if the directory to store tf_records does not exist, make one
  if not tf.gfile.IsDirectory(args.tf_records_dir): tf.gfile.MakeDirs(args.tf_records_dir)

  #extract the list of patient directories
  patient_dir_list=os.listdir('patient_data/')

  #store all the trials for one patient in tf_records
  for i in range(len(patient_dir_list)):
    patient_dir='patient_data/'+patient_dir_list[i]
    if not tf.gfile.IsDirectory(patient_dir): continue
    train_dataset=os.listdir(patient_dir)
    _process_dataset(patient_dir_list[i], train_dataset, args.nb_shards,args.nb_threads,args.tf_records_dir,patient_dir)


if __name__ == "__main__":
  main()



