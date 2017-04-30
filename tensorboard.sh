#################################################################
# Author: Safa Messaoud                                         #
# E-MAIL: messaou2@illinois.edu                                 #
# Instituation: University of Illinois at Urbana-Champaign      #
# Date: February 2017                                           #
# usage:  ./tensorboard.sh                                      #
#################################################################


#model dirctory
MODEL_DIR="${HOME}/DeepEEG/model"

# Run a TensorBoard server.
tensorboard --logdir="${MODEL_DIR}" --debug
