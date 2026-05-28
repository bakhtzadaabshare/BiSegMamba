#!/bin/sh

DATASET_PATH=/data/BiSegMamba/data/DATASET_Acdc

export PYTHONPATH=.././
export RESULTS_FOLDER=../latest_model_with_MSSM_v2
export BiSegMamba_preprocessed="$DATASET_PATH"/nnFormer_preprocessed
export BiSegMamba_raw_data_base="$DATASET_PATH"/nnFormer_raw

python ../BiSegMamba/run/run_training.py 3d_fullres BiSegMamba_trainer_acdc 1 0
