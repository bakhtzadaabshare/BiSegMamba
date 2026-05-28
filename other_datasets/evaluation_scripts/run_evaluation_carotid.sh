
DATASET_PATH=/data/BiSegMamba-experiments/data/DATASET_Carotid

export PYTHONPATH=.././
export RESULTS_FOLDER=../output_carotid_SwinUNETR
export BiSegMamba_preprocessed="$DATASET_PATH"/BiSegMamba_preprocessed
export BiSegMamba_raw_data_base="$DATASET_PATH"/BiSegMamba_raw

python ../BiSegMamba/run/run_training.py 3d_fullres BiSegMamba_trainer_carotid 2 0 -val