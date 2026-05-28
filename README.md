# BiSegMamba

**BiSegMamba** is a deep learning-based biomedical image segmentation framework designed for 3D medical image segmentation tasks. The repository provides preprocessing, training, inference, and evaluation pipelines for multiple medical imaging datasets, including **BraTS 2023**, **ACDC**, **AMOS**, and **Carotid CTA**.

This codebase is intended for research and experimental use, especially for developing and evaluating advanced segmentation models for volumetric medical images.

---

## Highlights

- End-to-end pipeline for biomedical image segmentation
- Support for preprocessing, training, prediction, and evaluation
- Dataset-specific scripts for BraTS, ACDC, AMOS, and Carotid datasets
- Modular implementation of the BiSegMamba model
- Lightweight training framework for fast experimentation
- Support for 3D medical image formats such as NIfTI
- Easy to extend for other medical segmentation datasets

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
│   └── 5_compute_metrics.py
│
├── light_training/
│   ├── trainer.py
│   ├── launch.py
│   ├── prediction.py
│   ├── dataloading/
│   └── augmentation/
│
├── model_bisegmamba/
│   └── BiSegMamba.py
│
├── other_datasets/
│   ├── ACDC/
│   ├── AMOS/
│   ├── Carotid/
│   └── evaluation/
│
├── training_scripts/
│   └── training shell scripts
│
├── evaluation_scripts/
│   └── evaluation shell scripts
│
├── requirements.txt
├── README.md
└── LICENSE
