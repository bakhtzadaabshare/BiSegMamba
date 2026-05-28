#!/bin/sh

DATASET_PATH=/data/BiSegMamba/data/DATASET_Acdc
CHECKPOINT_PATH=/data/BiSegMamba/pretrained_weights/acdc
export PYTHONPATH=.././
export RESULTS_FOLDER="$CHECKPOINT_PATH"
export BiSegMamba_preprocessed="$DATASET_PATH"/nnFormer_preprocessed
export BiSegMamba_raw_data_base="$DATASET_PATH"/nnFormer_raw

python ../BiSegMamba/run/run_training.py 3d_fullres BiSegMamba_trainer_acdc 1 0 -val