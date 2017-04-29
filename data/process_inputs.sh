#####################################################################################
# Author: Safa Messaoud                                                             #
# E-MAIL: messaou2@illinois.edu                                                     #
# Instituation: University of Illinois at Urbana-Champaign                          #
# Date: February 2017                                                               #
#                                                                                   #
# Description:                                                                      #
#	The outputs of this script are sharded TFRecord files containing serialized #
# 	SequenceExample protocol buffers. See input_data.py for details of how      #
# 	the SequenceExample protocol buffers are constructed.                       #
#                                                                                   #
# usage:                                                                            #
#  ./process_inputs.sh                                                              #
#####################################################################################

#current directory
WORK_DIR='/Users/safamessaoud/Desktop/class_project/data'

#path to the directory with the TFRecords
TF_RECORD_DIR='/Users/safamessaoud/Desktop/class_project/data/tf_record_dir'

#path to the matfile with the subject numbers
SUBJ_NUM_FILE='/Users/safamessaoud/Desktop/class_project/data/trials_subNums.mat'

#path to the matfile with the EEG features
FEATURE_FILE='/Users/safamessaoud/Desktop/class_project/data/FeatureMat_timeWin.mat'

#path to the electrode location file
LOCATION_FILE='/Users/safamessaoud/Desktop/class_project/data/Neuroscan_locs_orig.mat'

#number of time windows
NB_TIME_WINDOWS=7

#number of frequencies
NB_FREQ=3

#Number of shards for each patient
NB_SHARDS=1

#Number of threads to convert the data in TFRecords
NB_THREADS=8


cd "${CURRENT_DIR}"

BUILD_SCRIPT="${WORK_DIR}/input_data.py"


python "${BUILD_SCRIPT}" \
  "${TF_RECORD_DIR}" \
  "${SUBJ_NUM_FILE}" \
  "${FEATURE_FILE}" \
  "${LOCATION_FILE}" \
  --nb_shards "${NB_SHARDS}" \
  --nb_time_windows "${NB_TIME_WINDOWS} "\
  --nb_freq "${NB_FREQ}" \
  --nb_threads "${NB_THREADS}" \

	
	
