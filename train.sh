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


#File pattern of sharded TFRecord input files.
INPUT_FILE_PATTERN='${HOME}/train_dir'

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
	"${INPUT_FILE_PATTERN}/train-?????-of-00016"  \
	--number_of_steps "${NUMBER_OF_STEPS}" \
	--log_every_n_steps "${LOG_EVERY_N_STEPS}" \
	--log_every_n_steps "${LOG_EVERY_N_STEPS}" \
	--model_choice "${MODEL_CHOICE}" \
		
