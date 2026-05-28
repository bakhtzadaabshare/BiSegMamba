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

# Use your abdomen model for AMOS.
# Change this import only if you create a separate AMOS model file.
from BiSegMamba.network_architecture.amos.BiSegMamba_amos import SegMamba


class BiSegMamba_trainer_amos(Trainer_amos_carotid):
    """
    AMOS2022 CT trainer.

    Recommended official-style setting:
        Training:   200 CT cases
        Validation: 100 CT cases
        Classes:    15 organs + background = 16 classes
        Crop size:  [64, 160, 160]
        Epochs:     1000
        Optimizer:  SGD + Nesterov momentum 0.99
        Loss:       Dice + CE with optional deep supervision

    Important:
        Use splits_final.pkl to enforce the official AMOS split.
        Do not use FLARE22_Tr_* / FLARETs_* split logic here.
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
        self.initial_lr = 1e-2
        self.pin_memory = True
        self.load_pretrain_weight = False

        self.load_plans_file()

        # AMOS benchmark-style crop size.
        self.crop_size = np.array([64, 160, 160])

        self.input_channels = self.plans["num_modalities"]

        # AMOS has 15 foreground organs + background = 16 classes.
        self.num_classes = self.plans["num_classes"] + 1

        if self.num_classes != 16:
            print(
                f"WARNING: Expected 16 classes for AMOS2022 "
                f"(background + 15 organs), but got {self.num_classes}. "
                f"Check dataset.json labels."
            )

        self.conv_op = nn.Conv3d
        self.deep_supervision = True
        self.deep_supervision_scales = None
        self.ds_loss_weights = None

    def initialize(self, training=True, force_load_plans=False):
        if not self.was_initialized:
            maybe_mkdir_p(self.output_folder)

            if force_load_plans or self.plans is None:
                self.load_plans_file()

            # -------------------------------------------------------
            # AMOS2022 CT crop size
            # -------------------------------------------------------
            self.plans["plans_per_stage"][self.stage]["patch_size"] = np.array(
                [64, 160, 160]
            )
            self.crop_size = np.array([64, 160, 160])

            # -------------------------------------------------------
            # Keep the same safe anisotropic planning style.
            # Deep supervision scales are manually defined below,
            # so we do not rely on automatic DS scale generation.
            # -------------------------------------------------------
            self.plans["plans_per_stage"][self.stage]["pool_op_kernel_sizes"] = [
                [1, 2, 2],
                [2, 2, 2],
                [2, 2, 2],
                [2, 2, 2],
            ]

            self.plans["plans_per_stage"][self.stage]["batch_size"] = 2

            self.process_plans(self.plans)
            self.setup_DA_params()

            # -------------------------------------------------------
            # Deep supervision loss
            # Model returns:
            #   output[0] = full resolution
            #   output[1] = D/2, H/4, W/4
            #   output[2] = D/4, H/8, W/8
            # -------------------------------------------------------
            if self.deep_supervision:
                # More stable than [1.0, 0.5, 0.25] for many-organ AMOS.
                weights = np.array([1.0, 0.3, 0.1], dtype=np.float32)
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
        AMOS2022 CT:
            input:  1 CT channel
            output: 16 channels = background + 15 organs
        """

        self.network = SegMamba(
            in_chans=self.input_channels,
            out_chans=self.num_classes,
            do_ds=self.deep_supervision,
        )

        if torch.cuda.is_available():
            self.network.cuda()

        self.network.inference_apply_nonlin = softmax_helper

        try:
            n_parameters = sum(
                p.numel() for p in self.network.parameters() if p.requires_grad
            )

            input_res = (1, 64, 160, 160)  # C, D, H, W
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
        AMOS2022 official split.

        Use the splits_final.pkl created during dataset preparation:
            train = official AMOS CT training cases
            val   = official AMOS CT validation cases

        For AMOS22-CT this should be:
            train: 200 cases
            val:   100 cases
        """

        if self.fold == "all":
            tr_keys = val_keys = np.array(sorted(list(self.dataset.keys())))
        else:
            splits_file = join(self.dataset_directory, "splits_final.pkl")

            if not isfile(splits_file):
                raise RuntimeError(
                    "splits_final.pkl was not found in the preprocessed dataset folder:\n"
                    f"{splits_file}\n\n"
                    "For AMOS2022, do not create a random 5-fold split. "
                    "Copy the official split file that you created during dataset preparation."
                )

            self.print_to_log_file("Using splits from existing split file:", splits_file)
            splits = load_pickle(splits_file)
            self.print_to_log_file("The split file contains %d split(s)." % len(splits))

            if self.fold >= len(splits):
                raise RuntimeError(
                    f"Requested fold {self.fold}, but splits_final.pkl only contains "
                    f"{len(splits)} split(s). For AMOS official split, use fold=0."
                )

            tr_keys = np.array(sorted(splits[self.fold]["train"]))
            #val_keys = np.array(sorted(splits[self.fold]["val"]))
            val_keys = ["amos_0377",
                            "amos_0128",
                            "amos_0364",
                            "amos_0070",
                            "amos_0189"]

        # Safety checks
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
            "AMOS2022 official split: %d training cases, %d validation cases."
            % (len(tr_keys), len(val_keys))
        )
        self.print_to_log_file("Training cases:", tr_keys)
        self.print_to_log_file("Validation cases:", val_keys)

    def setup_DA_params(self):
        """
        AMOS2022 data augmentation parameters.

        For crop size [64, 160, 160], your current model should produce:
            out_main: [64, 160, 160]
            out_ds1:  [32, 40, 40]
            out_ds2:  [16, 20, 20]

        Therefore target scales:
            [1, 1, 1]
            [0.5, 0.25, 0.25]
            [0.25, 0.125, 0.125]
        """

        self.deep_supervision_scales = [
            [1, 1, 1],
            [0.5, 0.25, 0.25],
            [0.25, 0.125, 0.125],
        ]

        if self.threeD:
            self.data_aug_params = default_3D_augmentation_params
            self.data_aug_params["rotation_x"] = (
                -30.0 / 360 * 2.0 * np.pi,
                30.0 / 360 * 2.0 * np.pi,
            )
            self.data_aug_params["rotation_y"] = (
                -30.0 / 360 * 2.0 * np.pi,
                30.0 / 360 * 2.0 * np.pi,
            )
            self.data_aug_params["rotation_z"] = (
                -30.0 / 360 * 2.0 * np.pi,
                30.0 / 360 * 2.0 * np.pi,
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

        self.data_aug_params["scale_range"] = (0.7, 1.4)
        self.data_aug_params["do_elastic"] = False
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
                    "At epoch 100, the mean foreground Dice was 0. "
                    "Momentum has been reduced to 0.95 and network weights were reinitialized."
                )

        return continue_training

    def run_training(self):
        self.maybe_update_lr(self.epoch)

        ds = self.network.do_ds

        if self.deep_supervision:
            self.network.do_ds = True
        else:
            self.network.do_ds = False

        ret = super().run_training()

        self.network.do_ds = ds
        return ret

    def run_online_evaluation(self, output, target):
        """
        AMOS2022 online evaluation.

        Computes Dice for 15 foreground classes:
            1 ... 15
        """

        if isinstance(output, (tuple, list)):
            output = output[0]

        if isinstance(target, (tuple, list)):
            target = target[0]

        with torch.no_grad():
            num_classes = output.shape[1]

            if num_classes != 16:
                print(f"WARNING: AMOS2022 expected 16 output channels, got {num_classes}")

            output_seg = output.argmax(1)

            if target.ndim == output_seg.ndim + 1:
                target = target[:, 0]

            target = target.long()

            if not hasattr(self, "online_eval_tp"):
                self.online_eval_tp = []
                self.online_eval_fp = []
                self.online_eval_fn = []

            tp = []
            fp = []
            fn = []

            for c in range(1, num_classes):
                pred_c = output_seg == c
                target_c = target == c

                tp_c = torch.sum(pred_c & target_c).detach().cpu().numpy()
                fp_c = torch.sum(pred_c & ~target_c).detach().cpu().numpy()
                fn_c = torch.sum(~pred_c & target_c).detach().cpu().numpy()

                tp.append(tp_c)
                fp.append(fp_c)
                fn.append(fn_c)

            self.online_eval_tp.append(tp)
            self.online_eval_fp.append(fp)
            self.online_eval_fn.append(fn)

    def finish_online_evaluation(self):
        """
        Finish AMOS2022 online Dice evaluation.

        Prints 15 foreground Dice values.
        """

        if not hasattr(self, "online_eval_tp") or len(self.online_eval_tp) == 0:
            self.all_val_eval_metrics.append(0)
            return

        tp = np.sum(self.online_eval_tp, axis=0)
        fp = np.sum(self.online_eval_fp, axis=0)
        fn = np.sum(self.online_eval_fn, axis=0)

        global_dc_per_class = [
            2 * tp[c] / (2 * tp[c] + fp[c] + fn[c] + 1e-8)
            for c in range(len(tp))
        ]

        self.print_to_log_file("Average global foreground Dice:", global_dc_per_class)
        self.print_to_log_file(
            "(interpret this as an estimate for the Dice of the different classes. This is not exact.)"
        )

        self.all_val_eval_metrics.append(np.mean(global_dc_per_class))

        self.online_eval_tp = []
        self.online_eval_fp = []
        self.online_eval_fn = []

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
        do_mirroring = False
        save_softmax = False
        overwrite = False
        all_in_gpu = False

        ds = self.network.do_ds
        self.network.do_ds = False

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

        ds = self.network.do_ds
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

        self.network.do_ds = ds
        return ret