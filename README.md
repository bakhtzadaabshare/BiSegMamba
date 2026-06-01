# BiSegMamba

[![arXiv](https://img.shields.io/badge/arXiv-2605.30972-b31b1b.svg)](https://arxiv.org/abs/2605.30972)
[![Code](https://img.shields.io/badge/Code-PyTorch-blue.svg)](https://github.com/bakhtzadaabshare/BiSegMamba)

Official PyTorch implementation of **BiSegMamba: Efficient Bidirectional Tri-Oriented Mamba for 3D Medical Image Segmentation**.

BiSegMamba is an efficient 3D medical image segmentation framework designed to balance **long-range volumetric context modeling**, **fine boundary preservation**, and **computational efficiency**. The model follows a compact-to-detail design that combines progressive compacting, multi-scale spatial mixing, bidirectional tri-oriented Mamba modeling, and adaptive directional fusion.

<p align="center">
  <img src="assets/bisegmamba_overview.png" width="850">
</p>

> **Note:** This repository is intended for academic research and reproducibility. Dataset files and private clinical data are not included.

---

## 🎉 News

- **2026.05:** BiSegMamba preprint is available on [arXiv](https://arxiv.org/pdf/2605.30972).
- **2026.05:** Code and pretrained checkpoints are released for reproducibility.

---

## 🗂️ Repository Structure

```text
BiSegMamba/
│
├── brats2023/                         # BraTS 2023 pipeline
│   ├── 1_rename_mri_data_BraTS2023.py
│   ├── 2_preprocessing_BraTS2023.py
│   ├── 3_train.py
│   ├── 4_predict.py
│   ├── 5_compute_metrics.py
│   ├── light_training/
│   └── model_bisegmamba/
│
├── other_datasets/                    # nnFormer-style pipeline for ACDC, AMOS, Carotid
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
├── assets/
│   └── bisegmamba_overview.png
│
└── README.md
```

---

##🧠 Model Implementations

The core BiSegMamba design is kept consistent across all datasets. However, small dataset-specific settings are used to match image resolution, anatomical scale, class number, and GPU-memory requirements. These settings were selected after a series of controlled experiments for each dataset.

| Dataset | Architecture file | Description |
|---|---|---|
| BraTS 2023 | `brats2023/model_bisegmamba/BiSegMamba.py` | BraTS-specific implementation for multi-modal brain tumor segmentation. |
| ACDC | `other_datasets/BiSegMamba/network_architecture/acdc/BiSegMamba_acdc.py` | ACDC-specific implementation for cardiac MRI segmentation. |
| AMOS-CT | `other_datasets/BiSegMamba/network_architecture/amos/BiSegMamba_amos.py` | AMOS-specific implementation for abdominal multi-organ CT segmentation. |
| Carotid CTA | `other_datasets/BiSegMamba/network_architecture/carotid/BiSegMamba_carotid.py` | Carotid-specific implementation for thin vascular structure segmentation. |

> These files are not different proposed methods. They are dataset-specific implementations of the same BiSegMamba framework.

---

## 📊 Supported Datasets

| Dataset | Task | Pipeline | Download / Access | Notes |
|---|---|---|---|---|
| BraTS 2023 | Brain tumor segmentation | `brats2023/` | [Official BraTS 2023 Synapse page](https://www.synapse.org/#!Synapse:syn51156910) | Users may need a Synapse account and must follow BraTS data-use terms. Public mirror links are also listed in the [SegMamba-V2 repository](https://github.com/ge-xing/SegMamba-V2). |
| ACDC | Cardiac MRI segmentation | `other_datasets/` | [Official ACDC dataset page](https://www.creatis.insa-lyon.fr/Challenge/acdc/databases.html) | Download the official training data and follow the challenge citation requirements. |
| AMOS-CT | Abdominal multi-organ CT segmentation | `other_datasets/` | [Official AMOS Zenodo page](https://zenodo.org/records/7262581) | This work uses the CT subset for abdominal multi-organ segmentation. |
| Carotid CTA | Carotid artery segmentation | `other_datasets/` | Not publicly available | Private in-house clinical dataset; not distributed with this repository. |

> Please download each public dataset from its official source and follow the corresponding license, citation, and data-use requirements.

---

## ⬇️ Pretrained Weights

All released pretrained checkpoints are provided in one Google Drive root directory:

**[Download BiSegMamba pretrained weights](https://drive.google.com/drive/folders/1UtqZvnGZnc2x2mll1PkhqnbOD_j1lJjN?usp=drive_link)**

Please download the checkpoint corresponding to the dataset you want to reproduce and update the checkpoint/output path in the relevant script.

| Dataset | Checkpoint location | Used for | Path to update |
|---|---|---|---|
| BraTS 2023 | BraTS checkpoint inside the Google Drive folder | Prediction and metric computation | `brats2023/4_predict.py` |
| ACDC | ACDC checkpoint inside the Google Drive folder | Validation prediction generation | `other_datasets/evaluation_scripts/run_evaluation_acdc.sh` |
| AMOS-CT | AMOS checkpoint inside the Google Drive folder | Validation prediction generation | `other_datasets/evaluation_scripts/run_evaluation_amos.sh` |
| Carotid CTA | Carotid checkpoint inside the Google Drive folder, if released | Validation prediction generation | `other_datasets/evaluation_scripts/run_evaluation_carotid.sh` |

> For non-BraTS datasets, the `RESULTS_FOLDER` variable in the evaluation script should point to the folder containing the trained/pretrained model checkpoint. The generated predictions are then evaluated using the metric scripts after setting the paths to the prediction folder and ground-truth labels.

---

## ⚙️ Environment Installation

Create a conda environment:

```bash
conda create -n bisegmamba python=3.10 -y
conda activate bisegmamba
```

Install PyTorch according to your CUDA version. For example:

```bash
pip install torch torchvision torchaudio
```

Install commonly required packages:

```bash
pip install monai SimpleITK nibabel medpy numpy scipy scikit-image scikit-learn tqdm einops batchgenerators
```

If your setup uses Mamba/SSM operators, install versions compatible with your CUDA and PyTorch environment:

```bash
pip install causal-conv1d mamba-ssm
```

> The exact CUDA, PyTorch, `causal-conv1d`, and `mamba-ssm` versions should be kept consistent with your GPU/server environment.

---

## Reproduction Workflow

BiSegMamba contains two reproduction pipelines:

1. **BraTS 2023 pipeline** in `brats2023/`.
2. **nnFormer-style pipeline** for ACDC, AMOS-CT, and Carotid CTA in `other_datasets/`.

The two pipelines have different script organization, so please follow the corresponding workflow below.

---

## 🔍 BraTS 2023 Workflow

The BraTS 2023 pipeline follows this order:

```text
1. Rename BraTS data
2. Preprocess BraTS data
3. Train BiSegMamba
4. Generate predictions
5. Compute metrics
```

### Step 1: Rename data

```bash
cd brats2023
python 1_rename_mri_data_BraTS2023.py
```

### Step 2: Preprocess data

```bash
python 2_preprocessing_BraTS2023.py
```

### Step 3: Train the model

```bash
python 3_train.py
```

### Step 4: Generate predictions

Set the checkpoint path and output path in:

```text
brats2023/4_predict.py
```

Then run:

```bash
python 4_predict.py
```

### Step 5: Compute metrics

Set the prediction path and ground-truth path in:

```text
brats2023/5_compute_metrics.py
```

Then run:

```bash
python 5_compute_metrics.py --pred_name bisegmamba
```

Important paths to verify in the BraTS scripts include:

```text
data_dir
logdir
model_path
raw_data_dir
save_path
```

---

## 🔍 ACDC, AMOS-CT, and Carotid CTA Workflow

For non-BraTS datasets, this repository follows an **nnFormer-style preprocessing and evaluation pipeline**. The full order is:

```text
1. Prepare the dataset in nnFormer-style format
2. Preprocess the dataset using nnFormer-style preprocessing
3. Train the model
4. Run evaluation to generate prediction_raw and prediction_raw_postprocessed
5. Run the metric scripts after setting prediction and label paths
```

### Step 1: Prepare nnFormer-style dataset folders

Set the dataset root and output folders in the corresponding scripts:

```text
other_datasets/training_scripts/run_training_acdc.sh
other_datasets/training_scripts/run_training_amos.sh
other_datasets/training_scripts/run_training_carotid.sh
```

The scripts define variables such as:

```bash
DATASET_PATH=/path/to/dataset
export RESULTS_FOLDER=/path/to/output
export BiSegMamba_preprocessed="$DATASET_PATH"/BiSegMamba_preprocessed
export BiSegMamba_raw_data_base="$DATASET_PATH"/BiSegMamba_raw
```

Expected nnFormer-style raw data organization:

```text
BiSegMamba_raw_data_base/
└── nnFormer_raw_data/
    └── TaskXXX_DatasetName/
        ├── imagesTr/
        ├── labelsTr/
        ├── imagesTs/        # optional
        └── dataset.json
```

Example task IDs used in this repository:

| Dataset | Task ID | Trainer |
|---|---:|---|
| ACDC | `1` | `BiSegMamba_trainer_acdc` |
| Carotid CTA | `2` | `BiSegMamba_trainer_carotid` |
| AMOS-CT | `9` | `BiSegMamba_trainer_amos` |

### Step 2: Preprocess using nnFormer-style preprocessing

After setting the environment variables and preparing the raw dataset folder, run preprocessing with the desired task ID:

```bash
cd other_datasets
export PYTHONPATH=.././
python BiSegMamba/experiment_planning/nnFormer_plan_and_preprocess.py -t <TASK_ID> --verify_dataset_integrity
```

Examples:

```bash
python BiSegMamba/experiment_planning/nnFormer_plan_and_preprocess.py -t 1 --verify_dataset_integrity   # ACDC
python BiSegMamba/experiment_planning/nnFormer_plan_and_preprocess.py -t 9 --verify_dataset_integrity   # AMOS-CT
python BiSegMamba/experiment_planning/nnFormer_plan_and_preprocess.py -t 2 --verify_dataset_integrity   # Carotid CTA
```

### Step 3: Train the model

Train ACDC:

```bash
cd other_datasets
bash training_scripts/run_training_acdc.sh
```

Train AMOS-CT:

```bash
cd other_datasets
bash training_scripts/run_training_amos.sh
```

Train Carotid CTA:

```bash
cd other_datasets
bash training_scripts/run_training_carotid.sh
```

The training scripts call:

```bash
python ../BiSegMamba/run/run_training.py 3d_fullres <trainer_name> <task_id> <fold>
```

### Step 4: Generate validation predictions

The evaluation scripts run the model in validation mode and generate prediction folders such as:

```text
prediction_raw/
prediction_raw_postprocessed/
```

Run ACDC evaluation/prediction generation:

```bash
cd other_datasets
bash evaluation_scripts/run_evaluation_acdc.sh
```

Run AMOS-CT evaluation/prediction generation:

```bash
cd other_datasets
bash evaluation_scripts/run_evaluation_amos.sh
```

Run Carotid CTA evaluation/prediction generation:

```bash
cd other_datasets
bash evaluation_scripts/run_evaluation_carotid.sh
```

Before running, make sure that `RESULTS_FOLDER` points to the folder containing the trained or pretrained checkpoint.

### Step 5: Compute metrics

After prediction generation, compute the corresponding metrics using the metric/inference scripts. Set the paths to:

```text
prediction_raw or prediction_raw_postprocessed
ground-truth label folder
output metric file/folder
```

The exact metric script should be configured according to the dataset. Please update all hard-coded paths before running.

> Important: In this repository, prediction generation and metric computation are separate steps for non-BraTS datasets. The evaluation shell scripts generate predictions, while the metric/inference scripts should be used afterward to compute Dice, HD95, or other dataset-specific metrics.

---

## 📌 Important Notes

- Replace all dataset-specific absolute paths in scripts with your local paths before running.
- Public datasets are not redistributed in this repository.
- The Carotid CTA dataset is private clinical data and is not publicly released.
- For fair comparison, use the same preprocessing, patch size, sliding-window overlap, and evaluation protocol reported in the paper.
- GPU memory requirements depend on patch size, batch size, model variant, and dataset resolution.

---

## 📝 Citation

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

---

## 🙏 Acknowledgement

This codebase is developed for research in 3D biomedical image segmentation. The implementation style and experimental pipeline are inspired by widely used medical segmentation repositories, including nnU-Net, nnFormer, MONAI, SegMamba, SegMamba-V2, and UNETR++.

---

## 📬 Contact

For questions, please open an issue in this repository or contact the authors at:

```text
bakhtzada8c@gmail.com
```
