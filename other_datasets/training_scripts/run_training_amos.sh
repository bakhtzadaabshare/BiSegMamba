#!/bin/sh

DATASET_PATH=/data/BiSegMamba/data/DATASET_Amos

export PYTHONPATH=.././
export RESULTS_FOLDER=../output_amos_experiment_7_unetrup_decoder
export BiSegMamba_preprocessed="$DATASET_PATH"/BiSegMamba_preprocessed
export BiSegMamba_raw_data_base="$DATASET_PATH"/BiSegMamba_raw

python ../BiSegMamba/run/run_training.py 3d_fullres BiSegMamba_trainer_amos 9 0
