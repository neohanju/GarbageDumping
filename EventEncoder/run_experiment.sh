#!/usr/bin/env bash

# paths
data_path="/home/mlpa/Workspace/dataset/etri_action_data/30_10/posetrack"


BASEDIR=$(pwd)
TIMESTAMP=$(date '+%Y-%m-%d_%H-%M-%S')
RESULT_PATH="training_results/$TIMESTAMP"

mkdir $RESULT_PATH

echo ""
echo " ===< TRAINING >==========================================================="
echo ""

# other options
model="AE"

num_filters="512 256 128"
kernel_sizes="8 8 8"
#  output:   23,16,9
num_z="256"

# num_filters="2048 1024 512 256 128 64"
# size_kernels="5 5 5 5 5 5"
# #  output:  26,22,18,14,10,6
# num_z = 128

batch_size="512"
epochs="1000"
denoising=true
dropout=true
latent_reg=true

if [ $denoising = true ]; then
	denoising="--denoising"
else
	denoising=""
fi
if [ $dropout = true ]; then
	dropout="--dropout"
else
	dropout=""
fi
if [ $activation_reg = true ]; then
	latent_reg="--latent_reg"
else
	latent_reg=""
fi

TRAIN_OPTS="--model $model --data_path $data_path --save_path $RESULT_PATH --tb_path $RESULT_PATH \
            --epochs $epochs --nfs $num_filters --sks $kernel_sizes --nz $num_z --batch_size $batch_size \
            $denoising $dropout $latent_reg"

python train.py $TRAIN_OPTS |& tee $RESULT_PATH/training.log


echo ""
echo " ===< TESTING >============================================================"
echo ""

TEST_OPTS="--data_path $data_path --model_path $RESULT_PATH/best_loss.hdf5 --save_path $RESULT_PATH --save_latent"

python test.py $TEST_OPTS


echo ""
echo " ===< VISUALIZATION >======================================================"
echo ""

random_sampling="30"

VIS_OPTS="--input_path $data_path --recon_path $RESULT_PATH/recons --save_path $RESULT_PATH/videos \
          --random_sample $random_sampling"

python visualize_input_recon.py $VIS_OPTS

# ()()
# ('') HAANJU.YOO
