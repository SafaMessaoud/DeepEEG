##########################################################################
# Author: Safa Messaoud                                                  #
# E-MAIL: messaou2@illinois.edu                                          #
# Instituation: University of Illinois at Urbana-Champaign               #
# Date: February 2017                                                    #
# Description: evaluation script                                         #
# usage:  ./eval_script.sh                                               #
##########################################################################

export CUDA_VISIBLE_DEVICES=""

#current directory
CURRENT_DIR='/Users/safamessaoud/Documents/RAM_Project/DeepLearningPy'

#Directory for saving and loading model checkpoints
CHECKPOINT_DIR='/Users/safamessaoud/Documents/RAM_Project/deep_models/model/train'

#Directory for saving the evets logs
EVAL_DIR_LOG='/Users/safamessaoud/Documents/RAM_Project/deep_models/model/eval'


#File pattern of sharded TFRecord input files
INPUT_FILE_PATTERN='/Users/safamessaoud/Documents/RAM_Project/deep_models/val_dir'

#number of examples for the evaluation process
NB_EVAL_EXAMPLES=48

#Interval between evaluation runs
EVAL_INTERVAL_SECS=60


#Frequency at which loss and global step are logged
LOG_EVERY_N_STEPS=1

#Minimum global step to run evaluation
MIN_GLOBAL_STEP=10

#Determine the type of CNN Architecture (time signal processing)
CNN_TIME_ARCHITECTURE=1

#Determine the type of CNN Architecture (frequency signal processing)
CNN_FREQ_ARCHITECTURE=2


#specify if we want to insert the electrode position info
INSERT_ELECTRODE_POS_INFO=false

#specify if we want to insert the patient info
INSERT_PATIENT_INFO=false

#process the time data
PROCESS_TIME_DATA=true


#process the frequency data
PROCESS_FREQ_DATA=true

#Specify the model to be trained
#MODEL1: CNN+MAXpool (need to input CNN window)
#MODEL2: 
MODEL_CHOICE=1



cd "${CURRENT_DIR}"

BUILD_SCRIPT="${CURRENT_DIR}/evaluate.py"

echo $CURRENT_DIR

# Run the training script.
python "${BUILD_SCRIPT}" \
	"${CHECKPOINT_DIR}" \
	"${INPUT_FILE_PATTERN}/val-?????-of-00008"  \
	"${EVAL_DIR_LOG}" \
	--number_eval_examples "${NB_EVAL_EXAMPLES}" \
	--log_every_n_steps "${LOG_EVERY_N_STEPS}" \
	--eval_interval_secs "${EVAL_INTERVAL_SECS}" \
	--min_global_step "${MIN_GLOBAL_STEP}" \
	--model_choice "${MODEL_CHOICE}" \
	--insert_elec_pos_info "${INSERT_ELECTRODE_POS_INFO}" \
	--insert_patient_info "${INSERT_PATIENT_INFO}" \
	--cnn_time_architecture "${CNN_TIME_ARCHITECTURE}" \
	--cnn_freq_architecture "${CNN_FREQ_ARCHITECTURE}" \
	--process_time_data "${PROCESS_TIME_DATA}" \
	--process_freq_data "${PROCESS_FREQ_DATA}" \
	


