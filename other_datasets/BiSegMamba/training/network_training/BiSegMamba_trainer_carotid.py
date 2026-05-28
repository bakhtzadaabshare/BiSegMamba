from collections import OrderedDict
from typing import Tuple

import numpy as np
import torch
from torch import nn
from torch.cuda.amp import autocast

from batchgenerators.utilities.file_and_folder_operations import *
from fvcore.nn import FlopCountAnalysis

from BiSegMamba.training.network_training.Trainer_amos_carotid import Trainer_amos_carotid
from BiSegMamba.training.data_augmentation.data_augmentation_moreDA import get_moreDA_augmentation
from BiSegMamba.training.loss_functions.deep_supervision import MultipleOutputLoss2
from BiSegMamba.training.dataloading.dataset_loading import unpack_dataset
from BiSegMamba.utilities.to_torch import maybe_to_torch, to_cuda
from BiSegMamba.utilities.nd_softmax import softmax_helper
from BiSegMamba.network_architecture.neural_network import SegmentationNetwork
from BiSegMamba.training.data_augmentation.default_data_augmentation import (
    default_2D_augmentation_params,
    default_3D_augmentation_params,
    get_patch_size,
)
from BiSegMamba.training.learning_rate.poly_lr import poly_lr
from BiSegMamba.network_architecture.initialization import InitWeights_He
from BiSegMamba.network_architecture.carotid.BiSegMamba_carotid import SegMamba


class BiSegMamba_trainer_carotid(Trainer_amos_carotid):
    """
    Carotid CTA vessel + plaque trainer.

    Label definition:
        0: background
        1: vessel / carotid artery
        2: plaque

    Dataset-specific motivation:
        - Plaque is very small compared with background and vessel.
        - Some cases are plaque-negative and should remain in training.
        - A deeper 3D patch is preferred to preserve vessel continuity.
        - Online validation emphasizes plaque Dice and plaque sensitivity.

    Recommended setup:
        Classes:    background + vessel + plaque = 3 channels
        Crop size:  [96, 128, 128]
        Batch size: 2
        Epochs:     1000
        Optimizer:  SGD + Nesterov momentum 0.99, lower LR for plaque stability
        Loss:       Dice + CE with deep supervision inherited from base trainer
    """

    def __init__(
        self,
        plans_file,
        fold,
        output_folder=None,
        dataset_directory=None,
        batch_dice=True,
        stage=None,
        unpack_data=True,
        deterministic=True,
        fp16=False,
    ):
        super().__init__(
            plans_file,
            fold,
            output_folder,
            dataset_directory,
            batch_dice,
            stage,
            unpack_data,
            deterministic,
            fp16,
        )

        self.max_num_epochs = 1000
        self.initial_lr = 5e-3
        self.pin_memory = True
        self.load_pretrain_weight = False

        # Plaque is extremely small and your first run showed plaque hallucination
        # on all plaque-negative validation cases. This weak penalty suppresses
        # class-2 probability only in patches where the GT contains no plaque.
        self.plaque_fp_loss_weight = 0.05

        self.load_plans_file()

        # Carotid CTA volumes are long in z/depth. This patch keeps more depth
        # than the AMOS [64, 160, 160] crop while keeping similar voxel count.
        self.crop_size = np.array([128, 128, 128])

        self.input_channels = self.plans["num_modalities"]

        # nnFormer/nnU-Net-v1 plans usually store foreground classes only.
        # For labels {0,1,2}, plans["num_classes"] should be 2, so +1 = 3.
        self.num_classes = self.plans["num_classes"] + 1

        if self.num_classes != 3:
            print(
                f"WARNING: Expected 3 output classes for carotid CTA "
                f"(background + vessel + plaque), but got {self.num_classes}. "
                f"Check dataset.json labels: 0 background, 1 vessel, 2 plaque."
            )

        self.conv_op = nn.Conv3d
        self.deep_supervision = False
        self.deep_supervision_scales = None
        self.ds_loss_weights = None

        self.online_eval_tp = []
        self.online_eval_fp = []
        self.online_eval_fn = []
        self.online_eval_plaque_neg_cases = []
        self.online_eval_plaque_fp_cases = []

        # In nnU-Net style dataloading this increases foreground-centered crops.
        # It is not class-specific, but it helps compared with fully random crops.
        # For true plaque-specific sampling, use a custom DataLoader that samples class 2.
        self.oversample_foreground_percent = 0.33

    def initialize(self, training=True, force_load_plans=False):
        if not self.was_initialized:
            maybe_mkdir_p(self.output_folder)

            if force_load_plans or self.plans is None:
                self.load_plans_file()

            # -------------------------------------------------------
            # Carotid CTA patch size
            # -------------------------------------------------------
            self.plans["plans_per_stage"][self.stage]["patch_size"] = np.array(
                [128, 128, 128]
            )
            self.crop_size = np.array([128, 128, 128])

            # -------------------------------------------------------
            # Safe anisotropic first downsampling.
            # First stage avoids reducing depth immediately, useful for
            # elongated vessels and small plaque structures.
            # -------------------------------------------------------
            self.plans["plans_per_stage"][self.stage]["pool_op_kernel_sizes"] = [
                [2, 2, 2],
                [2, 2, 2],
                [2, 2, 2],
                [2, 2, 2],
            ]

            self.plans["plans_per_stage"][self.stage]["batch_size"] = 2

            self.process_plans(self.plans)
            self.setup_DA_params()

            if self.deep_supervision:
                weights = np.array([1.0, 0.15, 0.0], dtype=np.float32)
                weights = weights / weights.sum()
                self.ds_loss_weights = weights
                self.loss = MultipleOutputLoss2(self.loss, self.ds_loss_weights)

            self.folder_with_preprocessed_data = join(
                self.dataset_directory,
                self.plans["data_identifier"] + "_stage%d" % self.stage,
            )

            if training:
                self.dl_tr, self.dl_val = self.get_basic_generators()

                if self.unpack_data:
                    print("unpacking dataset")
                    unpack_dataset(self.folder_with_preprocessed_data)
                    print("done")
                else:
                    print("INFO: Not unpacking data. Training may be slower.")

                num_threads = self.data_aug_params.get("num_threads")
                seeds_train = np.random.randint(0, 99999, size=num_threads)
                seeds_val = np.random.randint(
                    0, 99999, size=max(num_threads // 2, 1)
                )

                self.tr_gen, self.val_gen = get_moreDA_augmentation(
                    self.dl_tr,
                    self.dl_val,
                    self.data_aug_params["patch_size_for_spatialtransform"],
                    self.data_aug_params,
                    deep_supervision_scales=self.deep_supervision_scales
                    if self.deep_supervision
                    else None,
                    pin_memory=self.pin_memory,
                    use_nondetMultiThreadedAugmenter=False,
                    seeds_train=seeds_train,
                    seeds_val=seeds_val,
                )

                self.print_to_log_file(
                    "TRAINING KEYS:\n %s" % str(self.dataset_tr.keys()),
                    also_print_to_console=False,
                )
                self.print_to_log_file(
                    "VALIDATION KEYS:\n %s" % str(self.dataset_val.keys()),
                    also_print_to_console=False,
                )

            self.initialize_network()
            self.initialize_optimizer_and_scheduler()

            assert isinstance(self.network, (SegmentationNetwork, nn.DataParallel))

        else:
            self.print_to_log_file(
                "self.was_initialized is True, not running self.initialize again"
            )

        self.was_initialized = True

    def initialize_network(self):
        """
        Carotid CTA:
            input:  1 CT channel
            output: 3 channels = background + vessel + plaque
        """
        self.network = SegMamba(
            in_chans=self.input_channels,
            out_chans=self.num_classes,
            #do_ds=self.deep_supervision,
        )

        if torch.cuda.is_available():
            self.network.cuda()

        # ---------------------------------------------------------
        # Critical compatibility settings for validation/inference
        # ---------------------------------------------------------
        self.network.num_classes = int(self.num_classes)
        self.network.do_ds = self.deep_supervision
        self.network.inference_apply_nonlin = softmax_helper
        self.network.input_shape_must_be_divisible_by = np.array([16, 16, 16])

        try:
            n_parameters = sum(
                p.numel() for p in self.network.parameters() if p.requires_grad
            )

            input_res = (1, 96, 128, 128)  # C, D, H, W
            dummy_input = torch.ones(()).new_empty(
                (1, *input_res),
                dtype=next(self.network.parameters()).dtype,
                device=next(self.network.parameters()).device,
            )

            flops = FlopCountAnalysis(self.network, dummy_input)
            model_flops = flops.total()

            print("Model FLOPs:", model_flops)
            print(f"Total trainable parameters: {round(n_parameters * 1e-6, 2)} M")
            print(f"MAdds: {round(model_flops * 1e-9, 2)} G")

        except Exception as e:
            print(f"FLOPs computation skipped: {e}")

    def initialize_optimizer_and_scheduler(self):
        assert self.network is not None, "self.initialize_network must be called first"

        self.optimizer = torch.optim.SGD(
            self.network.parameters(),
            self.initial_lr,
            weight_decay=self.weight_decay,
            momentum=0.99,
            nesterov=True,
        )

        self.lr_scheduler = None

    def do_split(self):
        """
        Use a fixed split file.

        Recommended:
            splits_final.pkl with one split:
                train: 78 cases
                val:   20 cases

        Keep plaque-positive and plaque-negative cases in both train and val.
        """

        if self.fold == "all":
            tr_keys = val_keys = np.array(sorted(list(self.dataset.keys())))
        else:
            splits_file = join(self.dataset_directory, "splits_final.pkl")

            if not isfile(splits_file):
                raise RuntimeError(
                    "splits_final.pkl was not found in the preprocessed dataset folder:\n"
                    f"{splits_file}\n\n"
                    "For the carotid plaque dataset, use a fixed stratified split. "
                    "The validation set must include both plaque-positive and "
                    "plaque-negative cases."
                )

            self.print_to_log_file("Using splits from existing split file:", splits_file)
            splits = load_pickle(splits_file)
            self.print_to_log_file("The split file contains %d split(s)." % len(splits))

            if self.fold >= len(splits):
                raise RuntimeError(
                    f"Requested fold {self.fold}, but splits_final.pkl only contains "
                    f"{len(splits)} split(s). Use fold=0 for your fixed split."
                )

            tr_keys = np.array(sorted(splits[self.fold]["train"]))
            val_keys = np.array(sorted(splits[self.fold]["val"]))

        missing_train = [k for k in tr_keys if k not in self.dataset]
        missing_val = [k for k in val_keys if k not in self.dataset]

        if len(missing_train) > 0:
            raise RuntimeError(
                f"These training keys from splits_final.pkl are missing from dataset:\n"
                f"{missing_train[:20]}"
            )

        if len(missing_val) > 0:
            raise RuntimeError(
                f"These validation keys from splits_final.pkl are missing from dataset:\n"
                f"{missing_val[:20]}"
            )

        overlap = set(tr_keys) & set(val_keys)
        if len(overlap) > 0:
            raise RuntimeError(f"Train/validation overlap found: {sorted(list(overlap))[:20]}")

        self.dataset_tr = OrderedDict()
        for k in tr_keys:
            self.dataset_tr[k] = self.dataset[k]

        self.dataset_val = OrderedDict()
        for k in val_keys:
            self.dataset_val[k] = self.dataset[k]

        self.print_to_log_file(
            "Carotid plaque split: %d training cases, %d validation cases."
            % (len(tr_keys), len(val_keys))
        )
        self.print_to_log_file("Training cases:", tr_keys)
        self.print_to_log_file("Validation cases:", val_keys)

    def setup_DA_params(self):
        """
        Carotid CTA data augmentation.

        For crop size [96, 128, 128], with the current model outputs:
            out_main: [96, 128, 128]
            out_ds1:  [48, 32, 32]
            out_ds2:  [24, 16, 16]

        Therefore target scales:
            [1, 1, 1]
            [0.5, 0.25, 0.25]
            [0.25, 0.125, 0.125]
        """

        self.deep_supervision_scales = [
            [1, 1, 1],
            [0.25, 0.25, 0.25],
            [0.125, 0.125, 0.125],
        ]

        if self.threeD:
            self.data_aug_params = default_3D_augmentation_params

            # Plaques are small and boundary-sensitive. Use slightly milder
            # rotations than AMOS to avoid too much interpolation damage.
            self.data_aug_params["rotation_x"] = (
                -20.0 / 360 * 2.0 * np.pi,
                20.0 / 360 * 2.0 * np.pi,
            )
            self.data_aug_params["rotation_y"] = (
                -20.0 / 360 * 2.0 * np.pi,
                20.0 / 360 * 2.0 * np.pi,
            )
            self.data_aug_params["rotation_z"] = (
                -20.0 / 360 * 2.0 * np.pi,
                20.0 / 360 * 2.0 * np.pi,
            )

            if self.do_dummy_2D_aug:
                self.data_aug_params["dummy_2D"] = True
                self.print_to_log_file("Using dummy2d data augmentation")
                self.data_aug_params["elastic_deform_alpha"] = default_2D_augmentation_params[
                    "elastic_deform_alpha"
                ]
                self.data_aug_params["elastic_deform_sigma"] = default_2D_augmentation_params[
                    "elastic_deform_sigma"
                ]
                self.data_aug_params["rotation_x"] = default_2D_augmentation_params[
                    "rotation_x"
                ]
        else:
            self.do_dummy_2D_aug = False

            if max(self.patch_size) / min(self.patch_size) > 1.5:
                default_2D_augmentation_params["rotation_x"] = (
                    -15.0 / 360 * 2.0 * np.pi,
                    15.0 / 360 * 2.0 * np.pi,
                )

            self.data_aug_params = default_2D_augmentation_params

        self.data_aug_params["mask_was_used_for_normalization"] = self.use_mask_for_norm

        # Set scale range before get_patch_size so the enlarged patch is computed
        # from the actual augmentation range used for carotid training.
        self.data_aug_params["scale_range"] = (0.85, 1.25)
        self.data_aug_params["do_elastic"] = False

        if self.do_dummy_2D_aug:
            self.basic_generator_patch_size = get_patch_size(
                self.patch_size[1:],
                self.data_aug_params["rotation_x"],
                self.data_aug_params["rotation_y"],
                self.data_aug_params["rotation_z"],
                self.data_aug_params["scale_range"],
            )
            self.basic_generator_patch_size = np.array(
                [self.patch_size[0]] + list(self.basic_generator_patch_size)
            )
            patch_size_for_spatialtransform = self.patch_size[1:]
        else:
            self.basic_generator_patch_size = get_patch_size(
                self.patch_size,
                self.data_aug_params["rotation_x"],
                self.data_aug_params["rotation_y"],
                self.data_aug_params["rotation_z"],
                self.data_aug_params["scale_range"],
            )
            patch_size_for_spatialtransform = self.patch_size

        # Keep scale augmentation, but slightly narrower than AMOS because plaque
        # is very small and can be damaged by aggressive interpolation.
        self.data_aug_params["selected_seg_channels"] = [0]
        self.data_aug_params["patch_size_for_spatialtransform"] = patch_size_for_spatialtransform
        self.data_aug_params["num_cached_per_thread"] = 2

    def maybe_update_lr(self, epoch=None):
        if epoch is None:
            ep = self.epoch + 1
        else:
            ep = epoch

        self.optimizer.param_groups[0]["lr"] = poly_lr(
            ep,
            self.max_num_epochs,
            self.initial_lr,
            0.9,
        )

        self.print_to_log_file(
            "lr:",
            np.round(self.optimizer.param_groups[0]["lr"], decimals=6),
        )

    def plaque_false_positive_penalty(self, output, target):
        """
        Penalize plaque probability in patches where GT has no plaque.

        This directly targets the observed failure mode where the model predicts
        plaque in plaque-negative validation cases. It is intentionally weak so
        it does not suppress true plaque learning in plaque-positive patches.
        """

        output_main = output[0] if isinstance(output, (tuple, list)) else output
        target_main = target[0] if isinstance(target, (tuple, list)) else target

        # output_main: [B, C, D, H, W]
        # target_main can be [B, 1, D, H, W] or [B, D, H, W]
        if target_main.ndim == output_main.ndim:
            target_main = target_main[:, 0]

        target_main = target_main.long()

        if output_main.shape[1] <= 2:
            return torch.zeros((), device=output_main.device, dtype=output_main.dtype)

        plaque_prob = torch.softmax(output_main, dim=1)[:, 2]
        plaque_voxels_per_patch = (target_main == 2).flatten(1).sum(dim=1)
        plaque_negative_patch = plaque_voxels_per_patch == 0

        if plaque_negative_patch.sum() == 0:
            return torch.zeros((), device=output_main.device, dtype=output_main.dtype)

        return plaque_prob[plaque_negative_patch].mean()

    def run_iteration(self, data_generator, do_backprop=True, run_online_evaluation=False):
        data_dict = next(data_generator)
        data = data_dict["data"]
        target = data_dict["target"]

        data = maybe_to_torch(data)
        target = maybe_to_torch(target)

        if torch.cuda.is_available():
            data = to_cuda(data)
            target = to_cuda(target)

        self.optimizer.zero_grad()

        if self.fp16:
            with autocast():
                output = self.network(data)
                del data
                loss = self.loss(output, target)
                loss = loss + self.plaque_fp_loss_weight * self.plaque_false_positive_penalty(output, target)

            if do_backprop:
                self.amp_grad_scaler.scale(loss).backward()
                self.amp_grad_scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.network.parameters(), 12)
                self.amp_grad_scaler.step(self.optimizer)
                self.amp_grad_scaler.update()

        else:
            output = self.network(data)
            del data
            loss = self.loss(output, target)
            loss = loss + self.plaque_fp_loss_weight * self.plaque_false_positive_penalty(output, target)

            if do_backprop:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.network.parameters(), 12)
                self.optimizer.step()

        if run_online_evaluation:
            self.run_online_evaluation(output, target)

        del target
        return loss.detach().cpu().numpy()

    def on_epoch_end(self):
        super().on_epoch_end()

        continue_training = self.epoch < self.max_num_epochs

        if self.epoch == 100:
            if len(self.all_val_eval_metrics) > 0 and self.all_val_eval_metrics[-1] == 0:
                self.optimizer.param_groups[0]["momentum"] = 0.95
                self.network.apply(InitWeights_He(1e-2))
                self.print_to_log_file(
                    "At epoch 100, the carotid validation score was 0. "
                    "Momentum has been reduced to 0.95 and network weights were reinitialized."
                )

        return continue_training

    def run_training(self):
        self.maybe_update_lr(self.epoch)

        ds = getattr(self.network, "do_ds", None)

        if not hasattr(self.network, "do_ds"):
            self.network.do_ds = self.deep_supervision
            ds = self.deep_supervision

        if self.deep_supervision:
            self.network.do_ds = True
        else:
            self.network.do_ds = False

        ret = super().run_training()

        self.network.do_ds = ds
        return ret

    def run_online_evaluation(self, output, target):
        """
        Online validation for carotid CTA.

        Computes:
            - vessel Dice
            - plaque Dice
            - plaque sensitivity/recall
            - plaque false-positive cases among plaque-negative samples

        The best checkpoint metric is computed in finish_online_evaluation with
        plaque precision and a plaque false-positive penalty.
        """

        if isinstance(output, (tuple, list)):
            output = output[0]

        if isinstance(target, (tuple, list)):
            target = target[0]

        with torch.no_grad():
            num_classes = output.shape[1]

            if num_classes != 3:
                print(f"WARNING: Carotid trainer expected 3 output channels, got {num_classes}")

            output_seg = output.argmax(1)

            if target.ndim == output_seg.ndim + 1:
                target = target[:, 0]

            target = target.long()

            if not hasattr(self, "online_eval_tp"):
                self.online_eval_tp = []
            if not hasattr(self, "online_eval_fp"):
                self.online_eval_fp = []
            if not hasattr(self, "online_eval_fn"):
                self.online_eval_fn = []
            if not hasattr(self, "online_eval_plaque_neg_cases"):
                self.online_eval_plaque_neg_cases = []
            if not hasattr(self, "online_eval_plaque_fp_cases"):
                self.online_eval_plaque_fp_cases = []

            tp = []
            fp = []
            fn = []

            # foreground classes: 1=vessel, 2=plaque
            for c in range(1, num_classes):
                pred_c = output_seg == c
                target_c = target == c

                tp_c = torch.sum(pred_c & target_c).detach().cpu().numpy()
                fp_c = torch.sum(pred_c & ~target_c).detach().cpu().numpy()
                fn_c = torch.sum(~pred_c & target_c).detach().cpu().numpy()

                tp.append(tp_c)
                fp.append(fp_c)
                fn.append(fn_c)

            # Case-level plaque false positives for plaque-negative samples.
            # This matters because 36/98 cases are plaque-negative in your dataset.
            plaque_neg_cases = 0
            plaque_fp_cases = 0
            for b in range(target.shape[0]):
                target_has_plaque = torch.any(target[b] == 2)
                pred_has_plaque = torch.any(output_seg[b] == 2)
                if not target_has_plaque:
                    plaque_neg_cases += 1
                    if pred_has_plaque:
                        plaque_fp_cases += 1

            self.online_eval_tp.append(tp)
            self.online_eval_fp.append(fp)
            self.online_eval_fn.append(fn)
            self.online_eval_plaque_neg_cases.append(plaque_neg_cases)
            self.online_eval_plaque_fp_cases.append(plaque_fp_cases)

    def finish_online_evaluation(self):
        """
        Finish carotid online validation.

        Class index mapping in global_dc_per_class:
            index 0 -> vessel class 1
            index 1 -> plaque class 2

        Note: online evaluation is patch-based, so the plaque FP rate here is a
        patch-level warning signal. Use full-volume validation for final numbers.
        """

        if not hasattr(self, "online_eval_tp") or len(self.online_eval_tp) == 0:
            self.all_val_eval_metrics.append(0)
            return

        tp = np.sum(self.online_eval_tp, axis=0).astype(np.float64)
        fp = np.sum(self.online_eval_fp, axis=0).astype(np.float64)
        fn = np.sum(self.online_eval_fn, axis=0).astype(np.float64)

        global_dc_per_class = [
            2 * tp[c] / (2 * tp[c] + fp[c] + fn[c] + 1e-8)
            for c in range(len(tp))
        ]

        sensitivity_per_class = [
            tp[c] / (tp[c] + fn[c] + 1e-8)
            for c in range(len(tp))
        ]

        precision_per_class = [
            tp[c] / (tp[c] + fp[c] + 1e-8)
            for c in range(len(tp))
        ]

        vessel_dice = float(global_dc_per_class[0]) if len(global_dc_per_class) > 0 else 0.0
        plaque_dice = float(global_dc_per_class[1]) if len(global_dc_per_class) > 1 else 0.0
        vessel_sens = float(sensitivity_per_class[0]) if len(sensitivity_per_class) > 0 else 0.0
        plaque_sens = float(sensitivity_per_class[1]) if len(sensitivity_per_class) > 1 else 0.0
        vessel_precision = float(precision_per_class[0]) if len(precision_per_class) > 0 else 0.0
        plaque_precision = float(precision_per_class[1]) if len(precision_per_class) > 1 else 0.0

        plaque_neg_cases = int(np.sum(self.online_eval_plaque_neg_cases))
        plaque_fp_cases = int(np.sum(self.online_eval_plaque_fp_cases))
        plaque_fp_case_rate = (
            plaque_fp_cases / plaque_neg_cases if plaque_neg_cases > 0 else 0.0
        )

        # Plaque-aware checkpoint metric. Compared with the earlier score, this
        # adds plaque precision and penalizes plaque hallucination in plaque-negative
        # patches. This matches the observed failure mode: plaque FP case rate = 1.0.
        carotid_score = (
            0.25 * vessel_dice
            + 0.30 * plaque_dice
            + 0.20 * plaque_sens
            + 0.20 * plaque_precision
            - 0.15 * plaque_fp_case_rate
        )
        carotid_score = max(float(carotid_score), 0.0)

        self.print_to_log_file("Carotid online evaluation:")
        self.print_to_log_file(f"  Vessel Dice:       {vessel_dice:.5f}")
        self.print_to_log_file(f"  Plaque Dice:       {plaque_dice:.5f}")
        self.print_to_log_file(f"  Vessel Sensitivity:{vessel_sens:.5f}")
        self.print_to_log_file(f"  Plaque Sensitivity:{plaque_sens:.5f}")
        self.print_to_log_file(f"  Vessel Precision:  {vessel_precision:.5f}")
        self.print_to_log_file(f"  Plaque Precision:  {plaque_precision:.5f}")
        self.print_to_log_file(
            f"  Plaque FP patch rate on plaque-negative patches: "
            f"{plaque_fp_case_rate:.5f} ({plaque_fp_cases}/{plaque_neg_cases})"
        )
        self.print_to_log_file(f"  Carotid checkpoint score: {carotid_score:.5f}")

        self.all_val_eval_metrics.append(carotid_score)

        self.online_eval_tp = []
        self.online_eval_fp = []
        self.online_eval_fn = []
        self.online_eval_plaque_neg_cases = []
        self.online_eval_plaque_fp_cases = []

    def validate(
        self,
        do_mirroring: bool = False,
        use_sliding_window: bool = True,
        step_size: float = 0.5,
        save_softmax: bool = False,
        use_gaussian: bool = True,
        overwrite: bool = False,
        validation_folder_name: str = "validation_raw",
        debug: bool = False,
        all_in_gpu: bool = False,
        segmentation_export_kwargs: dict = None,
        run_postprocessing_on_folds: bool = True,
    ):
        # Disable mirroring by default for carotid CTA because left/right anatomy
        # and plaque location can be clinically meaningful. Enable only if your
        # original validation protocol requires test-time mirroring.
        do_mirroring = False
        save_softmax = False
        overwrite = False
        all_in_gpu = False

        ds = getattr(self.network, "do_ds", None)

        if hasattr(self.network, "do_ds"):
            self.network.do_ds = False
        else:
            self.network.do_ds = False
            ds = self.deep_supervision

        try:
            ret = super().validate(
                do_mirroring=do_mirroring,
                use_sliding_window=use_sliding_window,
                step_size=step_size,
                save_softmax=save_softmax,
                use_gaussian=use_gaussian,
                overwrite=overwrite,
                validation_folder_name=validation_folder_name,
                debug=debug,
                all_in_gpu=all_in_gpu,
                segmentation_export_kwargs=segmentation_export_kwargs,
                run_postprocessing_on_folds=run_postprocessing_on_folds,
            )
        finally:
            self.network.do_ds = ds

        return ret

    def predict_preprocessed_data_return_seg_and_softmax(
        self,
        data: np.ndarray,
        do_mirroring: bool = True,
        mirror_axes: Tuple[int] = None,
        use_sliding_window: bool = True,
        step_size: float = 0.5,
        use_gaussian: bool = True,
        pad_border_mode: str = "constant",
        pad_kwargs: dict = None,
        all_in_gpu: bool = False,
        verbose: bool = True,
        mixed_precision=True,
    ) -> Tuple[np.ndarray, np.ndarray]:

        ds = getattr(self.network, "do_ds", None)

        if hasattr(self.network, "do_ds"):
            self.network.do_ds = False

        ret = super().predict_preprocessed_data_return_seg_and_softmax(
            data,
            do_mirroring=do_mirroring,
            mirror_axes=mirror_axes,
            use_sliding_window=use_sliding_window,
            step_size=step_size,
            use_gaussian=use_gaussian,
            pad_border_mode=pad_border_mode,
            pad_kwargs=pad_kwargs,
            all_in_gpu=all_in_gpu,
            verbose=verbose,
            mixed_precision=mixed_precision,
        )

        if ds is not None and hasattr(self.network, "do_ds"):
            self.network.do_ds = ds
        return ret
