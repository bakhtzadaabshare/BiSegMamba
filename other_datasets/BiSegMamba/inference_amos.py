import argparse
import json
from pathlib import Path
from collections import defaultdict

import numpy as np
import SimpleITK as sitk
from medpy import metric


AMOS_LABELS = {
    1: "spleen",
    2: "right_kidney",
    3: "left_kidney",
    4: "gall_bladder",
    5: "esophagus",
    6: "liver",
    7: "stomach",
    8: "aorta",
    9: "postcava",
    10: "pancreas",
    11: "right_adrenal_gland",
    12: "left_adrenal_gland",
    13: "duodenum",
    14: "bladder",
    15: "prostate_uterus",
}


def read_nii(path: Path):
    img = sitk.ReadImage(str(path))
    arr = sitk.GetArrayFromImage(img)  # z, y, x
    spacing_xyz = img.GetSpacing()     # x, y, z
    spacing_zyx = spacing_xyz[::-1]    # z, y, x for numpy array
    return arr, spacing_zyx


def dice_score(pred_mask: np.ndarray, gt_mask: np.ndarray) -> float:
    pred_sum = pred_mask.sum()
    gt_sum = gt_mask.sum()

    if pred_sum + gt_sum == 0:
        return 1.0

    return 2.0 * np.logical_and(pred_mask, gt_mask).sum() / (pred_sum + gt_sum)


def hd95_score(pred_mask: np.ndarray, gt_mask: np.ndarray, voxelspacing=None) -> float:
    if pred_mask.sum() > 0 and gt_mask.sum() > 0:
        return float(metric.binary.hd95(pred_mask, gt_mask, voxelspacing=voxelspacing))
    elif pred_mask.sum() == 0 and gt_mask.sum() == 0:
        return 0.0
    else:
        # One is empty and the other is not. This is a failure case.
        # You can also return np.nan, but a large value is easier to notice.
        return 999.0


def normalize_case_name(path: Path) -> str:
    """
    Handles:
        amos_0001.nii.gz
        amos_0001_0000.nii.gz

    Returns:
        amos_0001
    """
    name = path.name
    if name.endswith(".nii.gz"):
        name = name[:-7]
    elif name.endswith(".nii"):
        name = name[:-4]

    if name.endswith("_0000"):
        name = name[:-5]

    return name


def collect_nii_files(folder: Path):
    files = sorted(folder.glob("*.nii.gz"))
    case_to_file = {}

    for f in files:
        case = normalize_case_name(f)
        if case in case_to_file:
            raise RuntimeError(f"Duplicate case name found: {case}\n{case_to_file[case]}\n{f}")
        case_to_file[case] = f

    return case_to_file


def evaluate_amos(
    pred_dir: str,
    label_dir: str,
    output_dir: str = None,
    only_cases_from_predictions: bool = True,
):
    pred_dir = Path(pred_dir)
    label_dir = Path(label_dir)

    if output_dir is None:
        output_dir = pred_dir
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    pred_files = collect_nii_files(pred_dir)
    label_files = collect_nii_files(label_dir)

    pred_cases = set(pred_files.keys())
    label_cases = set(label_files.keys())

    if only_cases_from_predictions:
        eval_cases = sorted(pred_cases & label_cases)
    else:
        eval_cases = sorted(label_cases)

    missing_labels = sorted(pred_cases - label_cases)
    missing_preds = sorted(label_cases - pred_cases)

    print(f"Prediction cases: {len(pred_cases)}")
    print(f"Label cases:      {len(label_cases)}")
    print(f"Eval cases:       {len(eval_cases)}")

    if missing_labels:
        print(f"WARNING: predictions without labels: {len(missing_labels)}")
        print(missing_labels[:20])

    if missing_preds:
        print(f"WARNING: labels without predictions: {len(missing_preds)}")
        print(missing_preds[:20])

    if len(eval_cases) == 0:
        raise RuntimeError("No matching prediction/label cases found.")

    per_class_dice = defaultdict(list)
    per_class_hd95 = defaultdict(list)
    per_case_results = {}

    txt_path = output_dir / "amos_dice_hd95.txt"
    csv_path = output_dir / "amos_dice_hd95_per_case.csv"
    json_path = output_dir / "amos_summary.json"

    with open(txt_path, "w") as fw, open(csv_path, "w") as csv_fw:
        header = ["case"]
        for c, name in AMOS_LABELS.items():
            header.append(f"dice_{name}")
            header.append(f"hd95_{name}")
        header.append("mean_dice")
        header.append("mean_hd95")
        csv_fw.write(",".join(header) + "\n")

        for case in eval_cases:
            pred_path = pred_files[case]
            label_path = label_files[case]

            print(f"Evaluating {case}")
            print(f"  pred : {pred_path.name}")
            print(f"  label: {label_path.name}")

            pred, pred_spacing = read_nii(pred_path)
            label, label_spacing = read_nii(label_path)

            if pred.shape != label.shape:
                raise RuntimeError(
                    f"Shape mismatch for {case}: pred {pred.shape}, label {label.shape}. "
                    "Your prediction was not restored to the original label space."
                )

            spacing = label_spacing

            case_dice = {}
            case_hd95 = {}

            fw.write("*" * 30 + "\n")
            fw.write(f"{case}\n")

            row = [case]

            for class_id, class_name in AMOS_LABELS.items():
                pred_mask = pred == class_id
                label_mask = label == class_id

                dsc = dice_score(pred_mask, label_mask)
                hd95 = hd95_score(pred_mask, label_mask, voxelspacing=spacing)

                per_class_dice[class_id].append(dsc)
                per_class_hd95[class_id].append(hd95)

                case_dice[class_name] = float(dsc)
                case_hd95[class_name] = float(hd95)

                fw.write(f"Dice_{class_name}: {dsc:.4f}\n")
                fw.write(f"HD95_{class_name}: {hd95:.4f}\n")

                row.append(f"{dsc:.6f}")
                row.append(f"{hd95:.6f}")

            mean_dice = float(np.mean(list(case_dice.values())))
            mean_hd95 = float(np.mean(list(case_hd95.values())))

            fw.write(f"Mean_Dice: {mean_dice:.4f}\n")
            fw.write(f"Mean_HD95: {mean_hd95:.4f}\n")

            row.append(f"{mean_dice:.6f}")
            row.append(f"{mean_hd95:.6f}")
            csv_fw.write(",".join(row) + "\n")

            per_case_results[case] = {
                "dice": case_dice,
                "hd95": case_hd95,
                "mean_dice": mean_dice,
                "mean_hd95": mean_hd95,
            }

        fw.write("\n" + "=" * 40 + "\n")
        fw.write("AMOS22 MEAN RESULTS\n")
        fw.write("=" * 40 + "\n")

        mean_dice_per_class = {}
        mean_hd95_per_class = {}

        for class_id, class_name in AMOS_LABELS.items():
            mean_dice = float(np.mean(per_class_dice[class_id]))
            mean_hd95 = float(np.mean(per_class_hd95[class_id]))

            mean_dice_per_class[class_name] = mean_dice
            mean_hd95_per_class[class_name] = mean_hd95

            fw.write(f"Mean_Dice_{class_name}: {mean_dice:.4f}\n")
            fw.write(f"Mean_HD95_{class_name}: {mean_hd95:.4f}\n")

        foreground_mean_dice = float(np.mean(list(mean_dice_per_class.values())))
        foreground_mean_hd95 = float(np.mean(list(mean_hd95_per_class.values())))

        fw.write("-" * 40 + "\n")
        fw.write(f"Foreground_Mean_Dice: {foreground_mean_dice:.4f}\n")
        fw.write(f"Foreground_Mean_HD95: {foreground_mean_hd95:.4f}\n")

    summary = {
        "num_predictions": len(pred_cases),
        "num_labels": len(label_cases),
        "num_evaluated_cases": len(eval_cases),
        "missing_labels_for_predictions": missing_labels,
        "missing_predictions_for_labels": missing_preds,
        "mean_dice_per_class": mean_dice_per_class,
        "mean_hd95_per_class": mean_hd95_per_class,
        "foreground_mean_dice": foreground_mean_dice,
        "foreground_mean_hd95": foreground_mean_hd95,
        "per_case": per_case_results,
    }

    with open(json_path, "w") as f:
        json.dump(summary, f, indent=4)

    print("\nDone.")
    print(f"Foreground mean Dice: {foreground_mean_dice:.4f}")
    print(f"Foreground mean HD95: {foreground_mean_hd95:.4f}")
    print(f"Saved TXT : {txt_path}")
    print(f"Saved CSV : {csv_path}")
    print(f"Saved JSON: {json_path}")


if __name__ == "__main__":

    raw_pred_dir = "/data/BiSegMamba/other_datasets/output_amos_experiment_7_unetrup_decoder/BiSegMamba/3d_fullres/Task009_Amos_CT/BiSegMamba_trainer_amos__nnFormerPlansv2.1/fold_0/validation_raw"
    postprocessed_pred_dir = "/data/BiSegMamba/other_datasets/output_amos_experiment_7_unetrup_decoder/BiSegMamba/3d_fullres/Task009_Amos_CT/BiSegMamba_trainer_amos__nnFormerPlansv2.1/fold_0/validation_raw_postprocessed"
    label_dir = "/data/BiSegMamba/data/amos22/labelsVa"
    output_dir_preprocessed = "/data/BiSegMamba/other_datasets/output_amos_experiment_7_unetrup_decoder/BiSegMamba/3d_fullres/Task009_Amos_CT/BiSegMamba_trainer_amos__nnFormerPlansv2.1/fold_0/preprocessed_evaluation"
    output_dir_postprocessed = "/data/BiSegMamba/other_datasets/output_amos_experiment_7_unetrup_decoder/BiSegMamba/3d_fullres/Task009_Amos_CT/BiSegMamba_trainer_amos__nnFormerPlansv2.1/fold_0/postprocessed_evaluation"

    evaluate_amos(
        pred_dir=raw_pred_dir,
        label_dir=label_dir,
        output_dir=output_dir_preprocessed,
        only_cases_from_predictions=True,  # for preprocessed, we want to evaluate all labels even if some predictions are missing
    )

    
    evaluate_amos(
        pred_dir=postprocessed_pred_dir,
        label_dir=label_dir,
        output_dir=output_dir_postprocessed,
        only_cases_from_predictions=True,
    )