#########################################################################################################################
# Author: Safa Messaoud                                                                                                 #
#  E-Mail: messaou2@illinois.edu                                                                                        #
#  Instituation: University of Illinois at Urbana-Champaign                                                             #
#  Date: February 2017                                                                                                  #
#                                                                                                                       #
#  Description:                                                                                                         #
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
