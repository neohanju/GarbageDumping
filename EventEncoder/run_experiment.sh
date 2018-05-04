#!/usr/bin/env bash

# paths
data_path="/home/neohanju/Workspace/dataset/etri_action_data/30_10"

BASEDIR=$(pwd)
TIMESTAMP=$(date '+%Y-%m-%d_%H-%M-%S')
RESULT_PATH="training_results/$TIMESTAMP"


echo " ===< TRAINING >==========================================================="

# other options
num_filters="512 256 128"
kernel_sizes="8 8 8"
#  output:   23,16,9
num_z="256"

# num_filters="2048 1024 512 256 128 64"
# size_kernels="5 5 5 5 5 5"
# #  output:  26,22,18,14,10,6
# num_z = 128

epochs="2000"

TRAIN_OPTS="--data_path $data_path --save_path $RESULT_PATH --tb_path $RESULT_PATH \
            --epochs $epochs --nfs $num_filters --sks $kernel_sizes --nz $num_z"

python train.py $TRAIN_OPTS


echo " ===< TESTING >============================================================"

TEST_OPTS="--data_path $data_path --model_path $RESULT_PATH/best_loss.hdf5 --save_path $RESULT_PATH --save_latent"

python test.py $TEST_OPTS


echo " ===< VISUALIZATION >======================================================"

random_sampling="30"

VIS_OPTS="--input_path $data_path --recon_path $RESULT_PATH/recons --save_path $RESULT_PATH/videos \
          --random_sample $random_sampling"

python visualize_input_recon.py $VIS_OPTS

# ()()
# ('') HAANJU.YOO
