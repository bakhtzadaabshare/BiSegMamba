# BiSegMamba

Official PyTorch implementation of **BiSegMamba**, a Mamba-based framework for efficient 3D biomedical image segmentation.

This repository provides preprocessing, training, inference, and evaluation pipelines for volumetric medical image segmentation, with support for **BraTS 2023**, **AMOS**, **ACDC**, and **Carotid CTA** experiments.

> **Note:** This code is intended for academic research and reproducibility. Dataset files and clinical data are not included in this repository.

---

## Highlights

- **BiSegMamba model** for 3D medical image segmentation.
- **BraTS 2023 pipeline** with dataset renaming, preprocessing, training, prediction, and metric computation scripts.
- **nnFormer-style pipeline** for additional datasets including ACDC, AMOS, and Carotid CTA.
- **Sliding-window inference** for large 3D volumes.
- **Dataset-specific training and evaluation scripts** for easy reproduction.
- Modular code organization for extending the framework to new 3D segmentation datasets.

---

## Repository Structure

```text
BiSegMamba/
│
├── brats2023/
│   ├── 1_rename_mri_data_BraTS2023.py
│   ├── 2_preprocessing_BraTS2023.py
│   ├── 3_train.py
│   ├── 4_predict.py
│   ├── 5_compute_metrics.py
│   ├── light_training/
│   └── model_bisegmamba/
│
├── other_datasets/
│   ├── BiSegMamba/
│   │   ├── evaluation/
│   │   ├── experiment_planning/
│   │   ├── inference/
│   │   ├── network_architecture/
│   │   │   ├── acdc/
│   │   │   ├── amos/
│   │   │   └── carotid/
│   │   ├── preprocessing/
│   │   ├── run/
│   │   └── training/
│   │
│   ├── training_scripts/
│   │   ├── run_training_acdc.sh
│   │   ├── run_training_amos.sh
│   │   └── run_training_carotid.sh
│   │
│   └── evaluation_scripts/
│       ├── run_evaluation_acdc.sh
│       ├── run_evaluation_amos.sh
│       └── run_evaluation_carotid.sh
│
└── README.md
```

---

## Supported Datasets

| Dataset | Task | Pipeline |
|---|---|---|
| BraTS 2023 | Brain tumor segmentation | `brats2023/` |
| AMOS | Multi-organ abdominal segmentation | `other_datasets/` |
| ACDC | Cardiac segmentation | `other_datasets/` |
| Carotid CTA | Carotid artery segmentation | `other_datasets/` |

---

## Environment Installation

Create a new environment:

```bash
conda create -n bisegmamba python=3.10 -y
conda activate bisegmamba
```

Install PyTorch according to your CUDA version from the official PyTorch instructions. For example:

```bash
pip install torch torchvision torchaudio
```

Install commonly required packages:

```bash
pip install monai SimpleITK nibabel medpy numpy scipy scikit-image scikit-learn tqdm einops batchgenerators
```

If your model version uses Mamba/SSM operators, install the compatible packages for your CUDA/PyTorch version:

```bash
pip install causal-conv1d mamba-ssm
```

> The exact CUDA, PyTorch, `causal-conv1d`, and `mamba-ssm` versions should be kept consistent with your local GPU/server environment.

---

## Data Preparation

### BraTS 2023

Place the original BraTS 2023 training data in your preferred directory, then update the dataset path inside the BraTS scripts.

The BraTS pipeline follows this order:

```bash
cd brats2023
python 1_rename_mri_data_BraTS2023.py
python 2_preprocessing_BraTS2023.py
python 3_train.py
python 4_predict.py
python 5_compute_metrics.py --pred_name segmambav2
```

Before running, check and update the paths in the following files:

```text
brats2023/3_train.py
brats2023/4_predict.py
brats2023/5_compute_metrics.py
```

Important paths to verify include:

```text
data_dir
logdir
model_path
raw_data_dir
save_path
```

### ACDC, AMOS, and Carotid CTA

The non-BraTS datasets follow an nnFormer-style folder organization. Set the dataset root and output folders in the corresponding shell scripts:

```text
other_datasets/training_scripts/run_training_acdc.sh
other_datasets/training_scripts/run_training_amos.sh
other_datasets/training_scripts/run_training_carotid.sh
```

Each script defines variables such as:

```bash
DATASET_PATH=/path/to/dataset
export RESULTS_FOLDER=/path/to/output
export BiSegMamba_preprocessed="$DATASET_PATH"/BiSegMamba_preprocessed
export BiSegMamba_raw_data_base="$DATASET_PATH"/BiSegMamba_raw
```

Expected nnFormer-style dataset organization:

```text
DATASET_NAME/
├── BiSegMamba_raw/
│   └── TaskXXX_DatasetName/
│       ├── imagesTr/
│       ├── labelsTr/
│       ├── imagesTs/        # optional
│       └── dataset.json
│
└── BiSegMamba_preprocessed/
```

---

## Training

### Train on BraTS 2023

```bash
cd brats2023
python 3_train.py
```

The BraTS training script uses sliding-window validation and saves checkpoints under the configured `logdir`.

### Train on ACDC

```bash
cd other_datasets
bash training_scripts/run_training_acdc.sh
```

### Train on AMOS

```bash
cd other_datasets
bash training_scripts/run_training_amos.sh
```

### Train on Carotid CTA

```bash
cd other_datasets
bash training_scripts/run_training_carotid.sh
```

The training scripts call:

```bash
python ../BiSegMamba/run/run_training.py 3d_fullres <trainer_name> <task_id> <fold>
```

Example trainer names used in this repository:

```text
BiSegMamba_trainer_acdc
BiSegMamba_trainer_amos
BiSegMamba_trainer_carotid
```

---

## Inference

### BraTS 2023

Update the trained model path in:

```text
brats2023/4_predict.py
```

Then run:

```bash
cd brats2023
python 4_predict.py
```

Predictions are saved to the configured `prediction_results/` directory.

### ACDC, AMOS, and Carotid CTA

Inference entry files are located in:

```text
other_datasets/BiSegMamba/inference_acdc.py
other_datasets/BiSegMamba/inference_amos.py
other_datasets/BiSegMamba/inference_carotid.py
```

Update checkpoint paths and dataset paths inside the corresponding inference script before running.

---

## Evaluation

### BraTS 2023

After prediction, compute Dice and HD95 using:

```bash
cd brats2023
python 5_compute_metrics.py --pred_name segmambav2
```

### ACDC, AMOS, and Carotid CTA

Use the dataset-specific evaluation scripts:

```bash
cd other_datasets
bash evaluation_scripts/run_evaluation_acdc.sh
bash evaluation_scripts/run_evaluation_amos.sh
bash evaluation_scripts/run_evaluation_carotid.sh
```

---

## Important Notes

- The repository contains dataset-specific absolute paths in some scripts. Please replace them with your local paths before running.
- Clinical/private datasets such as Carotid CTA are not distributed with this repository.
- For fair comparison, use the same preprocessing, patch size, sliding-window overlap, and evaluation protocol reported in the paper.
- GPU memory requirements depend on patch size, batch size, model variant, and dataset resolution.

---

## TODO

- [ ] Release pretrained checkpoints.
- [ ] Add complete environment file.
- [ ] Add example dataset JSON templates.
- [ ] Add model architecture figure.

---

## Citation

If you find this repository useful, please cite our work:

```bibtex
@misc{zada2026bisegmambaefficientbidirectionaltrioriented,
      title={BiSegMamba: Efficient Bidirectional Tri-Oriented Mamba for 3D Medical Image Segmentation}, 
      author={Bakht Zada and Chao Tong and Qile Su and Shuai Zhang},
      year={2026},
      eprint={2605.30972},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2605.30972}, 
}
```

The citation will be updated after the arXiv/TMI version is available.

---

## Acknowledgement

This codebase is developed for research in 3D biomedical image segmentation. The implementation style and experimental pipeline are inspired by widely used medical segmentation repositories, including nnU-Net, nnFormer, MONAI, SegMamba, and UNETR++.

---

## Contact

For questions or collaboration, please open an issue in this repository or contact the authors at bakhtzada8c@gmail.com.
