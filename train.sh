#################################################################################
# Author: Safa Messaoud                                                         #
# E-MAIL: messaou2@illinois.edu                                                 #
# Instituation: University of Illinois at Urbana-Champaign                      #
# Date: February 2017                                                           #
# Description: use this script for training one of the DeepEEG models           #
# usage:  ./train.sh                                                            #
#################################################################################

#current directory
CURRENT_DIR='${HOME}/DeepEEG'

#Directory for saving and loading model checkpoints
CHECKPOINT_DIR='${HOME}/DeepEEG/model/train'


#Folder Containing the tf_records
TF_RECORD_FOLDER='${HOME}/DeepEEG/data/tf_record_dir'

#path to the evaluation directory
EVAL_DIR='${HOME}/DeepEEG/data/eval_dir'

#path to the testing directory
TEST_DIR='${HOME}/DeepEEG/data/test_dir'

#path to the training directory.
INPUT_FILE_PATTERN='${HOME}/DeepEEG/data/train_dir'

#create an evaluation directory
mkdir -p ${EVAL_DIR}

#create a test directory
mkdir -p ${TEST_DIR}

#create a train directory
mkdir -p ${INPUT_FILE_PATTERN}

#copy all the data in tf_record into train_dir
cp -r ${TF_RECORD_FOLDER} ${INPUT_FILE_PATTERN}

#path to the tf_record of the subject considered for testing
TEST_SUBJ='${HOME}/DeepEEG/data/train_dir/s1'

#path to the tf_record for evaluation
EVAL_RECORD='${HOME}/DeepEEG/data/train_dir/eval'

#copy PATIENT tf_record into test_dir and copy eval into eval_dir
cp ${TEST_SUBJ} ${TEST_DIR}

#copy the evaluation tf_record into eval_dir
cp ${EVAL_RECORD} ${EVAL_DIR}

#remove the eval tf_record and the subject considered for testing from the training directory
rm ${TEST_SUBJ}
rm ${EVAL_RECORD}


#Number of training steps
NUMBER_OF_STEPS=1000000

#Frequency at which loss and global step are logged
LOG_EVERY_N_STEPS=1

#Specify the model to be trained
#MODEL1: CNN+MAXpool (need to input CNN window)
#MODEL2: 
MODEL_CHOICE=1

cd "${CURRENT_DIR}"

BUILD_SCRIPT="${CURRENT_DIR}/train.py"

echo $CURRENT_DIR

# Run the training script.
python "${BUILD_SCRIPT}" \
	"${CHECKPOINT_DIR}" \
	"${INPUT_FILE_PATTERN}/s?"  \
	--number_of_steps "${NUMBER_OF_STEPS}" \
	--log_every_n_steps "${LOG_EVERY_N_STEPS}" \
	--model_choice "${MODEL_CHOICE}" \
		
