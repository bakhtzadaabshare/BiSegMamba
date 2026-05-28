import os
import glob
import argparse
import numpy as np
import SimpleITK as sitk
from medpy.metric import binary


def read_nii(path):
    itk_img = sitk.ReadImage(path)
    spacing = np.array(itk_img.GetSpacing())  # x, y, z
    arr = sitk.GetArrayFromImage(itk_img)     # z, y, x
    return arr, spacing


def process_carotid_label(label, foreground_labels=None):
    """
    Carotid lumen segmentation:
        background = 0
        carotid/lumen = 1 or labels > 0

    If your dataset has multiple carotid labels, e.g. left/right carotid:
        --foreground_labels 1 2
    """
    if foreground_labels is None:
        return label > 0

    mask = np.zeros_like(label, dtype=bool)
    for cls in foreground_labels:
        mask |= (label == cls)
    return mask


def confusion_values(pred, gt):
    pred = pred.astype(bool)
    gt = gt.astype(bool)

    tp = np.logical_and(pred, gt).sum()
    fp = np.logical_and(pred, ~gt).sum()
    fn = np.logical_and(~pred, gt).sum()
    tn = np.logical_and(~pred, ~gt).sum()

    return tp, fp, fn, tn


def dice_score(pred, gt):
    tp, fp, fn, _ = confusion_values(pred, gt)
    denom = 2 * tp + fp + fn
    if denom == 0:
        return 1.0
    return 2 * tp / denom


def iou_score(pred, gt):
    tp, fp, fn, _ = confusion_values(pred, gt)
    denom = tp + fp + fn
    if denom == 0:
        return 1.0
    return tp / denom


def precision_score(pred, gt):
    tp, fp, _, _ = confusion_values(pred, gt)
    denom = tp + fp
    if denom == 0:
        return 1.0
    return tp / denom


def recall_score(pred, gt):
    tp, _, fn, _ = confusion_values(pred, gt)
    denom = tp + fn
    if denom == 0:
        return 1.0
    return tp / denom


def specificity_score(pred, gt):
    _, fp, _, tn = confusion_values(pred, gt)
    denom = tn + fp
    if denom == 0:
        return 1.0
    return tn / denom


def volume_metrics(pred, gt, spacing):
    voxel_volume = np.prod(spacing)  # mm^3

    pred_vol = pred.sum() * voxel_volume
    gt_vol = gt.sum() * voxel_volume

    abs_vol_diff = abs(pred_vol - gt_vol)

    if gt_vol == 0:
        rel_vol_diff = np.nan
    else:
        rel_vol_diff = abs_vol_diff / gt_vol

    return pred_vol, gt_vol, abs_vol_diff, rel_vol_diff

def make_class_mask(arr, class_ids):
    """
    Convert a label map into a binary mask for one or multiple class ids.
    Example:
        vessel:     class_ids=[1]
        plaque:     class_ids=[2]
        foreground: class_ids=[1, 2]
    """
    mask = np.zeros_like(arr, dtype=bool)
    for c in class_ids:
        mask |= (arr == c)
    return mask


def compute_all_metrics(pred_mask, label_mask, spacing):
    dsc = dice_score(pred_mask, label_mask)
    iou = iou_score(pred_mask, label_mask)
    prec = precision_score(pred_mask, label_mask)
    rec = recall_score(pred_mask, label_mask)
    spec = specificity_score(pred_mask, label_mask)
    asd = asd_score(pred_mask, label_mask, spacing)
    hd95 = hd95_score(pred_mask, label_mask, spacing)

    pred_vol, gt_vol, abs_vd, rel_vd = volume_metrics(
        pred_mask, label_mask, spacing
    )

    return {
        "dice": dsc,
        "iou": iou,
        "precision": prec,
        "recall": rec,
        "specificity": spec,
        "asd": asd,
        "hd95": hd95,
        "pred_volume": pred_vol,
        "gt_volume": gt_vol,
        "abs_volume_diff": abs_vd,
        "rel_volume_diff": rel_vd,
        "gt_positive": bool(label_mask.sum() > 0),
        "pred_positive": bool(pred_mask.sum() > 0),
    }

def hd95_score(pred, gt, spacing):
    pred = pred.astype(bool)
    gt = gt.astype(bool)

    if pred.sum() > 0 and gt.sum() > 0:
        voxelspacing = spacing[::-1]  # numpy is z,y,x
        return binary.hd95(pred, gt, voxelspacing=voxelspacing)

    if pred.sum() == 0 and gt.sum() == 0:
        return 0.0

    return np.nan


def asd_score(pred, gt, spacing):
    pred = pred.astype(bool)
    gt = gt.astype(bool)

    if pred.sum() > 0 and gt.sum() > 0:
        voxelspacing = spacing[::-1]  # numpy is z,y,x
        return binary.asd(pred, gt, voxelspacing=voxelspacing)

    if pred.sum() == 0 and gt.sum() == 0:
        return 0.0

    return np.nan


def keep_largest_component(mask):
    """
    Optional post-processing.
    Useful for carotid lumen if predictions contain many tiny false-positive islands.
    """
    mask_img = sitk.GetImageFromArray(mask.astype(np.uint8))
    cc = sitk.ConnectedComponent(mask_img)
    stats = sitk.LabelShapeStatisticsImageFilter()
    stats.Execute(cc)

    if len(stats.GetLabels()) == 0:
        return mask.astype(bool)

    largest_label = max(stats.GetLabels(), key=lambda l: stats.GetPhysicalSize(l))
    largest = sitk.BinaryThreshold(cc, largest_label, largest_label, 1, 0)
    return sitk.GetArrayFromImage(largest).astype(bool)


def evaluate_carotid(
    label_dir,
    pred_dir,
    output_dir,
    postprocess_largest_component=False
    ):
    os.makedirs(output_dir, exist_ok=True)

    label_list = sorted(glob.glob(os.path.join(label_dir, "*.nii.gz")))
    pred_list = sorted(glob.glob(os.path.join(pred_dir, "*.nii.gz")))

    label_dict = {os.path.basename(p): p for p in label_list}
    pred_dict = {os.path.basename(p): p for p in pred_list}

    common_names = sorted(set(label_dict.keys()) & set(pred_dict.keys()))
    missing_preds = sorted(set(label_dict.keys()) - set(pred_dict.keys()))
    extra_preds = sorted(set(pred_dict.keys()) - set(label_dict.keys()))

    print("Number of labels:", len(label_list))
    print("Number of predictions:", len(pred_list))
    print("Number of matched cases:", len(common_names))

    result_file = os.path.join(output_dir, "carotid_multiclass_inference_results.txt")
    csv_file = os.path.join(output_dir, "carotid_multiclass_inference_results.csv")

    classes = {
        "vessel": [1],
        "plaque": [2],
        "foreground": [1, 2],
    }

    metric_keys = [
        "dice",
        "iou",
        "precision",
        "recall",
        "specificity",
        "asd",
        "hd95",
        "pred_volume",
        "gt_volume",
        "abs_volume_diff",
        "rel_volume_diff",
    ]

    metrics = {
        class_name: {key: [] for key in metric_keys}
        for class_name in classes.keys()
    }

    # For plaque-specific clinical reporting
    plaque_gt_positive_dice = []
    plaque_gt_positive_hd95 = []
    plaque_gt_positive_asd = []
    plaque_negative_cases = 0
    plaque_false_positive_cases = 0
    plaque_false_positive_volumes = []

    skipped_cases = []

    with open(result_file, "w") as fw, open(csv_file, "w") as csv:
        fw.write("Carotid Vessel + Plaque Segmentation Inference Results\n")
        fw.write("=" * 80 + "\n")
        fw.write(f"Number of labels: {len(label_list)}\n")
        fw.write(f"Number of predictions: {len(pred_list)}\n")
        fw.write(f"Number of matched cases: {len(common_names)}\n")
        fw.write(f"Missing predictions: {missing_preds}\n")
        fw.write(f"Extra predictions ignored: {extra_preds}\n")
        fw.write(f"Largest-component postprocessing: {postprocess_largest_component}\n")
        fw.write("=" * 80 + "\n\n")

        csv.write(
            "case,class,dice,iou,precision,recall,specificity,asd_mm,hd95_mm,"
            "pred_volume_mm3,gt_volume_mm3,abs_volume_diff_mm3,rel_volume_diff,"
            "gt_positive,pred_positive\n"
        )

        for name in common_names:
            print("Evaluating:", name)

            label_path = label_dict[name]
            pred_path = pred_dict[name]

            label, spacing = read_nii(label_path)
            pred, _ = read_nii(pred_path)

            if pred.shape != label.shape:
                msg = f"Skipping {name}: shape mismatch pred={pred.shape}, label={label.shape}"
                print(msg)
                skipped_cases.append(msg)
                fw.write(msg + "\n")
                continue

            fw.write("*" * 80 + "\n")
            fw.write(f"Case: {name}\n")
            fw.write(f"GT unique labels: {np.unique(label)}\n")
            fw.write(f"Pred unique labels: {np.unique(pred)}\n")

            for class_name, class_ids in classes.items():
                label_mask = make_class_mask(label, class_ids)
                pred_mask = make_class_mask(pred, class_ids)

                # Only apply largest component to vessel/foreground, not plaque.
                # Plaque may naturally have multiple small components.
                if postprocess_largest_component and class_name in ["vessel", "foreground"]:
                    pred_mask = keep_largest_component(pred_mask)

                m = compute_all_metrics(pred_mask, label_mask, spacing)

                for key in metric_keys:
                    metrics[class_name][key].append(m[key])

                fw.write(f"\n[{class_name.upper()}]\n")
                fw.write(f"Dice: {m['dice']:.4f}\n")
                fw.write(f"IoU/mIoU: {m['iou']:.4f}\n")
                fw.write(f"Precision: {m['precision']:.4f}\n")
                fw.write(f"Recall/Sensitivity: {m['recall']:.4f}\n")
                fw.write(f"Specificity: {m['specificity']:.4f}\n")
                fw.write(
                    f"ASD_mm: {m['asd']:.4f}\n"
                    if not np.isnan(m["asd"]) else "ASD_mm: nan\n"
                )
                fw.write(
                    f"HD95_mm: {m['hd95']:.4f}\n"
                    if not np.isnan(m["hd95"]) else "HD95_mm: nan\n"
                )
                fw.write(f"Pred_volume_mm3: {m['pred_volume']:.4f}\n")
                fw.write(f"GT_volume_mm3: {m['gt_volume']:.4f}\n")
                fw.write(f"Abs_volume_diff_mm3: {m['abs_volume_diff']:.4f}\n")
                fw.write(
                    f"Rel_volume_diff: {m['rel_volume_diff']:.4f}\n"
                    if not np.isnan(m["rel_volume_diff"]) else "Rel_volume_diff: nan\n"
                )
                fw.write(f"GT_positive: {m['gt_positive']}\n")
                fw.write(f"Pred_positive: {m['pred_positive']}\n")

                csv.write(
                    f"{name},{class_name},"
                    f"{m['dice']:.6f},{m['iou']:.6f},{m['precision']:.6f},"
                    f"{m['recall']:.6f},{m['specificity']:.6f},"
                    f"{m['asd']:.6f},{m['hd95']:.6f},"
                    f"{m['pred_volume']:.6f},{m['gt_volume']:.6f},"
                    f"{m['abs_volume_diff']:.6f},{m['rel_volume_diff']:.6f},"
                    f"{m['gt_positive']},{m['pred_positive']}\n"
                )

                # Extra plaque-specific tracking
                if class_name == "plaque":
                    if m["gt_positive"]:
                        plaque_gt_positive_dice.append(m["dice"])
                        plaque_gt_positive_hd95.append(m["hd95"])
                        plaque_gt_positive_asd.append(m["asd"])
                    else:
                        plaque_negative_cases += 1
                        if m["pred_positive"]:
                            plaque_false_positive_cases += 1
                            plaque_false_positive_volumes.append(m["pred_volume"])
                        else:
                            plaque_false_positive_volumes.append(0.0)

            fw.write("\n")

        fw.write("\n" + "=" * 80 + "\n")
        fw.write("Summary\n")
        fw.write("=" * 80 + "\n")

        evaluated_cases = len(common_names) - len(skipped_cases)
        fw.write(f"Evaluated cases: {evaluated_cases}\n")
        fw.write(f"Skipped cases: {len(skipped_cases)}\n\n")

        for class_name in classes.keys():
            fw.write("-" * 80 + "\n")
            fw.write(f"{class_name.upper()} SUMMARY\n")
            fw.write("-" * 80 + "\n")

            for key in metric_keys:
                values = np.array(metrics[class_name][key], dtype=float)
                fw.write(f"Mean_{class_name}_{key}: {np.nanmean(values):.4f}\n")
                fw.write(f"Std_{class_name}_{key}: {np.nanstd(values):.4f}\n")

            fw.write("\n")

        fw.write("-" * 80 + "\n")
        fw.write("PLAQUE-SPECIFIC CLINICAL SUMMARY\n")
        fw.write("-" * 80 + "\n")

        plaque_fp_case_rate = (
            plaque_false_positive_cases / plaque_negative_cases
            if plaque_negative_cases > 0 else np.nan
        )

        fw.write(f"Plaque-positive GT cases: {len(plaque_gt_positive_dice)}\n")
        fw.write(f"Plaque-negative GT cases: {plaque_negative_cases}\n")
        fw.write(f"Plaque false-positive cases: {plaque_false_positive_cases}\n")
        fw.write(f"Plaque false-positive case rate: {plaque_fp_case_rate:.4f}\n")

        if len(plaque_gt_positive_dice) > 0:
            fw.write(
                f"Mean_plaque_Dice_on_GT_positive_cases: "
                f"{np.nanmean(plaque_gt_positive_dice):.4f}\n"
            )
            fw.write(
                f"Mean_plaque_HD95_on_GT_positive_cases: "
                f"{np.nanmean(plaque_gt_positive_hd95):.4f}\n"
            )
            fw.write(
                f"Mean_plaque_ASD_on_GT_positive_cases: "
                f"{np.nanmean(plaque_gt_positive_asd):.4f}\n"
            )

        if len(plaque_false_positive_volumes) > 0:
            fw.write(
                f"Mean_plaque_FP_volume_on_GT_negative_cases_mm3: "
                f"{np.nanmean(plaque_false_positive_volumes):.4f}\n"
            )

        if skipped_cases:
            fw.write("\nSkipped cases:\n")
            for case in skipped_cases:
                fw.write(case + "\n")

    print("Done.")
    print("Evaluated cases:", len(common_names) - len(skipped_cases))

    for class_name in classes.keys():
        dice_values = np.array(metrics[class_name]["dice"], dtype=float)
        print(f"Mean {class_name} Dice:", np.nanmean(dice_values))

    print("Plaque false-positive case rate:", plaque_fp_case_rate)
    print("Saved TXT to:", result_file)
    print("Saved CSV to:", csv_file)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--label_dir",
        default="/data/BiSegMamba/data/DATASET_Carotid/BiSegMamba_raw/nnFormer_raw_data/Task002_Carotid/labelsTr",
        help="Path to ground-truth carotid labels."
    )

    parser.add_argument(
        "--pred_dir",
        default="/data/BiSegMamba/other_datasets/output_carotid_SwinUNETR/BiSegMamba/3d_fullres/Task002_Carotid/BiSegMamba_trainer_carotid__nnFormerPlansv2.1/fold_0/validation_raw",
        help="Path to predicted carotid segmentation masks."
    )

    parser.add_argument(
        "--output_dir",
        default="/data/BiSegMamba/other_datasets/output_carotid_SwinUNETR/BiSegMamba/3d_fullres/Task002_Carotid/BiSegMamba_trainer_carotid__nnFormerPlansv2.1/fold_0/",
        help="Where to save evaluation results."
    )

    parser.add_argument(
        "--foreground_labels",
        nargs="+",
        type=int,
        default=[1, 2],
        help="Foreground label ids. Default: all labels > 0 are carotid foreground."
    )

    parser.add_argument(
        "--postprocess_largest_component",
        action="store_true",
        help="Keep only the largest connected component in prediction."
    )

    args = parser.parse_args()

    evaluate_carotid(
        label_dir=args.label_dir,
        pred_dir=args.pred_dir,
        output_dir=args.output_dir,
        postprocess_largest_component=args.postprocess_largest_component,
    )