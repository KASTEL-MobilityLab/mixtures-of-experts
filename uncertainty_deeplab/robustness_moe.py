import os
import csv
import warnings

import numpy as np
import torch
from torch.amp import autocast
from torch.utils.data import DataLoader
from tqdm import tqdm

from utils.robustness_helpers import corruptions
from dataloader.a2d2_loader import a2d2Loader
from utils.uncertainty_helpers import (
    get_confidences,
    compute_mutual_information,
    get_predictive_entropy,
)
from models import (
    BEST_CHECKPOINT_EXPERT_HIGHWAY,
    BEST_CHECKPOINT_EXPERT_URBAN,
    BEST_CHECKPOINT_MOE_SIMPLE_FINAL_CONV,
    BEST_CHECKPOINT_MOE_CLASSWISE,
    DATASET_ROOT,
    FINAL_CONV,
    GATELAYERSIMPLE,
    GATELAYERCLASSWISE,
    IMG_HEIGHT,
    IMG_WIDTH,
    LINEAR_FEATURES,
    NUM_CLASSES,
    OUTPUT_STRIDE,
)
from models.deeplab_moe import MoE
from utils.metrics import (
    clamp_and_log_values, segmentation_metrics_img, uncertainty_metrics_image, uncertainty_metrics_image_2)

# ---- global speed tweaks ----
torch.backends.cudnn.benchmark = True
torch.backends.cudnn.enabled = True
torch.backends.cuda.matmul.allow_tf32 = True  # enable TF32 on Ampere+


class CombinedMoEEvaluator:
    def __init__(self, model, modelname: str):
        self.device = torch.device('cuda:0')
        self.model = model.to(self.device)
        self.modelname = modelname

        root = os.path.join(DATASET_ROOT, 'all')
        test_set = a2d2Loader(root, split='test',
                              img_size=(IMG_HEIGHT, IMG_WIDTH))
        self.test_loader = DataLoader(
            test_set,
            batch_size=2,
            shuffle=False,
            num_workers=2,
            pin_memory=True,
        )

        self.seg_metrics = []
        self.unc_metrics_1 = []   # EVU
        self.pe_metrics_1 = []    # PE
        self.mi_metrics_1 = []    # MI
        self.unc_metrics_2 = []   # EVU curves
        self.pe_metrics_2 = []
        self.mi_metrics_2 = []

    @torch.inference_mode()
    def validation(self):
        print("Starting MOE evaluation")
        self.model.eval()
        img_counter = 0

        tbar = tqdm(self.test_loader, desc='Eval batches')
        for images, targets, _ in tbar:
            images = images.to(self.device, non_blocking=True)
            targets = targets.to(self.device, non_blocking=True)
            B = images.size(0)

            cpu_imgs = (
                images.cpu()
                      .permute(0, 2, 3, 1)
                      .mul(255)
                      .clip(0, 255)
                      .byte()
                      .numpy()
            )

            for corr_fn, corr_name in corruptions:
                for sev in [5]:
                    corrupted_np = np.stack([
                        corr_fn(im, severity=sev) for im in cpu_imgs
                    ], axis=0)
                    if np.any(np.isnan(corrupted_np)) or np.any(corrupted_np > 255) or np.any(corrupted_np < 0):
                        print(f"Invalid values in {corr_name} sev{sev}")
                        corrupted_np = np.clip(
                            corrupted_np, 0, 255).astype(np.uint8)
                    # Move to GPU in FP16
                    corrupted = (
                        torch.from_numpy(corrupted_np)
                             .permute(0, 3, 1, 2)
                             .float()
                             .div(255)
                             .to(self.device, non_blocking=True)
                    )

                    expert_outputs = []
                    with autocast("cuda"):
                        out = self.model(corrupted)
                        # e1 = self.model.expert1(
                        #     corrupted).float()
                        # e2 = self.model.expert2(corrupted).float()
                        expert_outputs.append(self.model.expert1(corrupted))
                        expert_outputs.append(self.model.expert2(corrupted))

                    preds = torch.argmax(out, dim=1)
                    # outs = torch.stack([e1, e2])
                    outs = torch.stack([out] + expert_outputs)
                    soft = torch.softmax(outs, dim=2)
                    mean_soft = torch.mean(soft, dim=0)
                    pe = get_predictive_entropy(mean_softmax=mean_soft)
                    mi = compute_mutual_information(
                        predictive_entropy=pe,
                        softmax_probs=soft,
                    )

                    max_e = torch.log(torch.tensor(
                        NUM_CLASSES, device=self.device))
                    scaled_pe = pe / max_e
                    scaled_mi = mi / max_e

                    pe_conf = 1 - scaled_pe
                    mi_conf = 1 - scaled_mi
                    confidences = get_confidences(preds, outs, corrupted)

                    pred_cpu = preds.cpu()
                    pe_conf_cpu = pe_conf.cpu()
                    mi_conf_cpu = mi_conf.cpu()
                    conf_cpu = confidences.cpu()
                    target_cpu = targets.cpu()

                    for b in range(B):
                        idx = img_counter + b
                        label = target_cpu[b]
                        pred = pred_cpu[b]

                        valid_mask = label != 255
                        t_valid = label[valid_mask]
                        p_valid = pred[valid_mask]

                        if len(t_valid) == 0:
                            print(f"Empty predictions: {corr_name} sev{sev}")
                            continue

                        conf_valid = conf_cpu[b][valid_mask]
                        pe_valid = pe_conf_cpu[b][valid_mask]
                        mi_valid = mi_conf_cpu[b][valid_mask]

                        # Apply confidence clamping
                        conf_clamped = clamp_and_log_values(
                            conf_valid.numpy())
                        pe_clamped = clamp_and_log_values(
                            pe_valid.numpy())
                        mi_clamped = clamp_and_log_values(
                            mi_valid.numpy())

                        try:
                            seg_m = segmentation_metrics_img(
                                t_valid, p_valid)
                            u_m = uncertainty_metrics_image(
                                t_valid, p_valid, conf_clamped)
                            pe_m = uncertainty_metrics_image(
                                t_valid, p_valid, pe_clamped)
                            mi_m = uncertainty_metrics_image(
                                t_valid, p_valid, mi_clamped)
                        except Exception as e:
                            print(
                                f"Metric error {corr_name} sev{sev}: {str(e)}")
                            seg_m = [np.nan]*7
                            u_m = pe_m = mi_m = [np.nan]*4

                        try:
                            t_valid_np = t_valid.numpy()
                            p_valid_np = p_valid.numpy()

                            u2_m = uncertainty_metrics_image_2(
                                t_valid_np, p_valid_np, conf_clamped)
                            pe2_m = uncertainty_metrics_image_2(
                                t_valid_np, p_valid_np, pe_clamped)
                            mi2_m = uncertainty_metrics_image_2(
                                t_valid_np, p_valid_np, mi_clamped)
                        except Exception as e:
                            print(
                                f"Extended metric error {corr_name} sev{sev}: {str(e)}")
                            u2_m = pe2_m = mi2_m = []

                        self.seg_metrics.append([idx, corr_name, sev] + seg_m)
                        self.unc_metrics_1.append([idx, corr_name, sev] + u_m)
                        self.pe_metrics_1.append([idx, corr_name, sev] + pe_m)
                        self.mi_metrics_1.append([idx, corr_name, sev] + mi_m)

                        for row in u2_m:
                            self.unc_metrics_2.append(
                                [idx, corr_name, sev] + row)
                        for row in pe2_m:
                            self.pe_metrics_2.append(
                                [idx, corr_name, sev] + row)
                        for row in mi2_m:
                            self.mi_metrics_2.append(
                                [idx, corr_name, sev] + row)

                    img_counter += B

            torch.cuda.empty_cache()
            self.save_results()

    def save_results(self):
        outdir = f"./perturbation/moe_classwise_sev5/"
        os.makedirs(outdir, exist_ok=True)
        files = [
            ("segmentation.csv", self.seg_metrics, [
                "image_index", "corruption_name", "severity_level",
                "iou", "m_iou", "fw_iou", "precision", "recall", "f1_score", "pixel_accuracy"
            ]),
            ("evu.csv", self.unc_metrics_1, [
                "image_index", "corruption_name", "severity_level", "mce", "ece", "brier", "nll"
            ]),
            ("pe.csv", self.pe_metrics_1, [
                "image_index", "corruption_name", "severity_level", "mce", "ece", "brier", "nll"
            ]),
            ("mi.csv", self.mi_metrics_1, [
                "image_index", "corruption_name", "severity_level", "mce", "ece", "brier", "nll"
            ]),
            ("evu_2.csv", self.unc_metrics_2, [
                "image_index", "corruption_name", "severity_level",
                "uncertainty_threshold", "p_accurate_given_certain",
                "p_uncertain_given_inaccurate", "pavpu"
            ]),
            ("pe_2.csv", self.pe_metrics_2, [
                "image_index", "corruption_name", "severity_level",
                "uncertainty_threshold", "p_accurate_given_certain",
                "p_uncertain_given_inaccurate", "pavpu"
            ]),
            ("mi_2.csv", self.mi_metrics_2, [
                "image_index", "corruption_name", "severity_level",
                "uncertainty_threshold", "p_accurate_given_certain",
                "p_uncertain_given_inaccurate", "pavpu"
            ]),
        ]
        for fname, data, header in files:
            with open(os.path.join(outdir, fname), 'w', newline='') as f:
                w = csv.writer(f)
                w.writerow(header)
                w.writerows(data)


if __name__ == "__main__":
    warnings.filterwarnings(
        "ignore",
        message="You are using `torch.load` with `weights_only=False`",
    )

    model = MoE(
        gate_type="classwise",
        with_conv=False,
        arch="deeplabv3plus",
        backbone="resnet101",
        output_stride=OUTPUT_STRIDE,
        num_classes=NUM_CLASSES,
        linear_features=LINEAR_FEATURES,
        checkpoint1=BEST_CHECKPOINT_EXPERT_HIGHWAY,
        checkpoint2=BEST_CHECKPOINT_EXPERT_URBAN,
    )

    ckpt = BEST_CHECKPOINT_MOE_CLASSWISE
    if not os.path.isfile(ckpt):
        raise RuntimeError(f"=> no checkpoint found at '{ckpt}'")
    print(f"Loading checkpoint from {ckpt}")
    state = torch.load(ckpt)
    model.load_state_dict(state["state_dict"])

    evaluator = CombinedMoEEvaluator(model, modelname="moe_classwise")
    evaluator.validation()
    evaluator.save_results()
