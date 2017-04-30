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
CURRENT_DIR='${HOME}/DeepEEG'

#Directory for saving and loading model checkpoints
CHECKPOINT_DIR='${HOME}/DeepEEG/model/train'

#Directory for saving the evets logs
EVAL_DIR_LOG='${HOME}/DeepEEG/model/eval'

#File pattern of sharded TFRecord input files
INPUT_FILE_PATTERN='${HOME}/DeepEEG/data/eval_dir'

#number of examples for the evaluation process
NB_EVAL_EXAMPLES=260

#Interval between evaluation runs
EVAL_INTERVAL_SECS=60

#Frequency at which loss and global step are logged
LOG_EVERY_N_STEPS=1

#Minimum global step to run evaluation
MIN_GLOBAL_STEP=10

#Specify the model to be trained
MODEL_CHOICE=1

cd "${CURRENT_DIR}"

BUILD_SCRIPT="${CURRENT_DIR}/evaluate.py"

echo $CURRENT_DIR

# Run the training script.
python "${BUILD_SCRIPT}" \
	"${CHECKPOINT_DIR}" \
	"${INPUT_FILE_PATTERN}/eval"  \
	"${EVAL_DIR_LOG}" \
	--number_eval_examples "${NB_EVAL_EXAMPLES}" \
	--log_every_n_steps "${LOG_EVERY_N_STEPS}" \
	--eval_interval_secs "${EVAL_INTERVAL_SECS}" \
	--min_global_step "${MIN_GLOBAL_STEP}" \
	--model_choice "${MODEL_CHOICE}" \
