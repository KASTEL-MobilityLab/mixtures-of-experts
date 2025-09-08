import argparse
import os
import csv
import warnings
from collections import OrderedDict

import numpy as np
import torch
from torch.amp import autocast
from torch.utils.data import DataLoader
from tqdm import tqdm
import sys
import yaml
from PIL import ImageFile

from dataloader.a2d2_loader import get_dataloader, a2d2Loader
from models.deeplab_ensemble import Ensemble
from utils.metrics import Evaluator, segmentation_metrics_img, clamp_and_log_values
from utils.robustness_helpers import corruptions
from utils.uncertainty_helpers import (
    calculate_ece, calculate_mce, calculate_metrics,
    compute_mutual_information, get_predictive_entropy
)
from sklearn.metrics import brier_score_loss, log_loss

# Global speed tweaks
torch.backends.cudnn.benchmark = True
torch.backends.cudnn.enabled = True
torch.backends.cuda.matmul.allow_tf32 = True


class CombinedEnsembleEvaluator:
    """Combined Ensemble Evaluator with mIoU and corruption analysis"""

    def __init__(self, params):
        self.params = params
        self.is_cuda = len(params["gpu_ids"]) > 0
        self.device = torch.device('cuda', params["gpu_ids"][0]) \
            if self.is_cuda else torch.device('cpu')

        self.nclass = params['DATASET']['num_class']
        self.datasets = list(
            s for s in params["DATASET"]["dataset"].split(','))

        # Initialize data loaders
        self.test_loader, self.label_names, self.label_colors = get_dataloader(
            params, params['test_sets'], 'test')

        # Alternative loader for corruption analysis
        if 'DATASET_ROOT' in params:
            root = os.path.join(params['DATASET_ROOT'], 'all')
            test_set = a2d2Loader(root, split='test',
                                  img_size=(params['DATASET']['img_height'],
                                            params['DATASET']['img_width']))
            self.corruption_loader = DataLoader(
                test_set, batch_size=4, shuffle=False,
                num_workers=1, pin_memory=True
            )
        else:
            self.corruption_loader = self.test_loader

        # Initialize evaluator for mIoU calculation
        self.evaluator = Evaluator(self.nclass)

        # Initialize model based on expert type
        self.model = Ensemble(
            arch=self.params["MODEL"]["arch"],
            backbone=self.params["MODEL"]["backbone"],
            output_stride=self.params["MODEL"]["out_stride"],
            num_classes=self.nclass,
            checkpoint1=self.params["TEST"]["checkpoint_moe_expert_1"],
            checkpoint2=self.params["TEST"]["checkpoint_moe_expert_2"],
            ens_type=self.params["MODEL"]["ens_type"]
        )
        if self.is_cuda:
            self.model.expert1.to(self.device)
            self.model.expert2.to(self.device)

        # Metrics storage for corruption analysis
        self.seg_metrics = []
        self.pe_metrics_1 = []
        self.pe_metrics_2 = []
        self.mi_metrics_1 = []
        self.mi_metrics_2 = []

    @torch.inference_mode()
    def standard_validation(self):
        """Standard validation with mIoU calculation"""
        print("Starting standard validation")
        self.model.eval()
        self.evaluator.reset()
        tbar = tqdm(self.test_loader, desc="Standard validation")

        for i, sample in enumerate(tbar):
            image, target, label_path = sample
            image = image.to(self.device)
            target = target.to(self.device)

            with torch.no_grad():
                output = self.model(image)

            pred = output.data.cpu().numpy()
            target = target.cpu().numpy()
            pred = np.argmax(pred, axis=1)
            self.evaluator.add_batch(target, pred)

        # Calculate metrics
        acc = self.evaluator.pixel_accuracy()
        acc_class = self.evaluator.pixel_accuracy_class()
        m_iou = self.evaluator.mean_intersection_over_union(self.label_names)
        fw_iou = self.evaluator.frequency_weighted_intersection_over_union()

        print("Validation:")
        print("pAcc:{}, mAcc:{}, m_iou:{}, fwIoU: {}".format(
            acc, acc_class, m_iou, fw_iou))

        return acc, acc_class, m_iou, fw_iou

    @torch.inference_mode()
    def corruption_validation(self):
        """Corruption robustness validation for severities 1-5"""
        print("Starting corruption validation (ensemble)")
        self.model.eval()
        img_counter = 0

        tbar = tqdm(self.corruption_loader, desc='Corruption eval batches')
        for images, targets, _ in tbar:
            images = images.to(self.device, non_blocking=True)
            targets = targets.to(self.device, non_blocking=True)
            B = images.size(0)

            # Convert to CPU numpy for corruption functions
            cpu_imgs = (
                images.cpu().permute(0, 2, 3, 1)
                .mul(255).clip(0, 255).byte().numpy()
            )

            # Apply each corruption type and severity
            for corruption_fn, corr_name in corruptions:
                for severity in range(1, 6):  # for Severities 1-5
                    try:
                        if severity == 0:
                            # Use original uncorrupted images
                            corrupted_np = cpu_imgs.copy()
                        else:
                            # Apply corruption as usual
                            corrupted_np = np.stack([corruption_fn(im.copy(), severity=severity)
                                                    for im in cpu_imgs], axis=0)

                        corrupted = (
                            torch.from_numpy(corrupted_np)
                            .permute(0, 3, 1, 2).float().div(255)
                            .to(self.device)
                        )

                        # Get ensemble predictions
                        with autocast("cuda"):
                            out1 = self.model.expert1(corrupted)
                            out2 = self.model.expert2(corrupted)
                            # outputs = torch.stack([out1, out2], dim=0)
                            # softmax_probs = outputs.softmax(dim=2)
                            soft1 = torch.softmax(out1, dim=1)
                            soft2 = torch.softmax(out2, dim=1)
                            softmax_probs = torch.stack([soft1, soft2], dim=0)
                            mean_soft = softmax_probs.mean(dim=0)

                        # Calculate uncertainty metrics
                        pe = get_predictive_entropy(mean_softmax=mean_soft)
                        mi = compute_mutual_information(
                            predictive_entropy=pe,
                            softmax_probs=softmax_probs,
                        )
                        max_ent = torch.log(torch.tensor(
                            self.nclass, device=self.device))

                        # Confidence maps
                        pe_conf = 1.0 - pe / max_ent
                        mi_conf = 1.0 - mi / max_ent

                        pred = mean_soft.argmax(dim=1)
                        pred_cpu = pred.cpu().numpy()
                        target_cpu = targets.cpu().numpy()

                        # Process each image in batch
                        for b in range(B):
                            idx = img_counter + b
                            t_np = target_cpu[b]
                            p_np = pred_cpu[b]

                            valid_mask = t_np != 255
                            t_np = t_np[valid_mask]
                            p_np = p_np[valid_mask]

                            # Process PE and MI metrics
                            if pe_conf.nelement() > 0:
                                pe_conf_b = pe_conf[b][valid_mask].cpu(
                                ).numpy()
                                self._process_uncertainty(
                                    t_np, p_np,
                                    clamp_and_log_values(
                                        pe_conf_b, "PE confidence"),
                                    self.pe_metrics_1, self.pe_metrics_2,
                                    idx, corr_name, severity
                                )

                            if mi_conf.nelement() > 0:
                                mi_conf_b = mi_conf[b][valid_mask].cpu(
                                ).numpy()
                                self._process_uncertainty(
                                    t_np, p_np,
                                    clamp_and_log_values(
                                        mi_conf_b, "MI confidence"),
                                    self.mi_metrics_1, self.mi_metrics_2,
                                    idx, corr_name, severity
                                )

                            # Segmentation metrics
                            seg_m = segmentation_metrics_img(t_np, p_np)
                            self.seg_metrics.append(
                                [idx, corr_name, severity] + seg_m)

                    except Exception as e:
                        print(
                            f"Error processing {corr_name} severity {severity}: {str(e)}")
                        for b in range(B):
                            idx = img_counter + b
                            self.seg_metrics.append(
                                [idx, corr_name, severity] + [np.nan]*7)
                        continue

            img_counter += B

        torch.cuda.empty_cache()
        print("Corruption validation completed")

    def _process_uncertainty(self, t_np, p_np, conf, metrics_1,
                             metrics_2, idx, corr_name, severity):
        """Process uncertainty metrics for given predictions and confidence"""
        if len(t_np) == 0 or len(p_np) == 0:
            print(f"Empty predictions: {corr_name} sev{severity}")
            metrics_1.append(
                [idx, corr_name, severity, np.nan, np.nan, np.nan, np.nan])
            return

        try:
            # Apply valid mask and clamp confidence values
            valid_mask = t_np != 255
            flat_t = t_np[valid_mask].flatten()
            flat_p = p_np[valid_mask].flatten()
            flat_conf = clamp_and_log_values(
                conf[valid_mask].flatten(), "confidence")

            # Convert to tensors for metric calculation
            conf_t = torch.from_numpy(flat_conf)
            p_t = torch.from_numpy(flat_p)
            t_t = torch.from_numpy(flat_t)

            # Calculate metrics with error handling
            mce = calculate_mce(conf_t, p_t, t_t)
            ece = calculate_ece(conf_t, p_t, t_t)
            acc_map = (flat_p == flat_t).astype(np.uint8)
            brier = brier_score_loss(acc_map, flat_conf)

            try:
                nll = log_loss(acc_map, flat_conf, labels=[0, 1])
            except ValueError as e:
                if "Only one class" in str(e):
                    nll = np.nan
                    print(f"Single-class NLL error: {corr_name} sev{severity}")
                else:
                    raise

            # Store primary metrics
            metrics_1.append([
                idx, corr_name, severity,
                float(mce), float(ece), brier, nll
            ])

            # Threshold metrics with validation
            for th_i in range(1, 101, 2):
                thr = th_i / 100.0
                try:
                    p_acc_c, p_unc_inacc, pavpu = calculate_metrics(
                        target=t_t, predictions=p_t,
                        confidences=conf_t, uncertainty_threshold=thr
                    )
                except Exception as e:
                    print(f"Threshold {thr} failed: {str(e)}")
                    p_acc_c, p_unc_inacc, pavpu = np.nan, np.nan, np.nan

                metrics_2.append([
                    idx, corr_name, severity, thr,
                    float(p_acc_c), float(p_unc_inacc), float(pavpu)
                ])

        except Exception as e:
            print(f"Error processing {corr_name} sev{severity}: {str(e)}")
            metrics_1.append(
                [idx, corr_name, severity, np.nan, np.nan, np.nan, np.nan])

    def save_results(self, modelname="ensemble"):
        """Save all evaluation results to CSV files"""
        outdir = f"./perturbation/{modelname}"
        os.makedirs(outdir, exist_ok=True)

        metric_files = [
            ("segmentation.csv", self.seg_metrics, [
                "image_index", "corruption_name", "severity_level",
                "iou", "m_iou", "fw_iou", "precision",
                "recall", "f1_score", "pixel_accuracy"
            ]),
            ("pe.csv", self.pe_metrics_1, [
                "image_index", "corruption_name", "severity_level",
                "mce", "ece", "brier", "nll"
            ]),
            ("pe_2.csv", self.pe_metrics_2, [
                "image_index", "corruption_name", "severity_level",
                "uncertainty_threshold", "p_accurate_given_certain",
                "p_uncertain_given_inaccurate", "pavpu"
            ]),
            ("mi.csv", self.mi_metrics_1, [
                "image_index", "corruption_name", "severity_level",
                "mce", "ece", "brier", "nll"
            ]),
            ("mi_2.csv", self.mi_metrics_2, [
                "image_index", "corruption_name", "severity_level",
                "uncertainty_threshold", "p_accurate_given_certain",
                "p_uncertain_given_inaccurate", "pavpu"
            ])
        ]

        for fname, data, header in metric_files:
            path = os.path.join(outdir, fname)
            with open(path, 'w', newline="") as f:
                writer = csv.writer(f)
                writer.writerow(header)
                writer.writerows(data)

        print(f"Results saved to {outdir}")

    def run_full_evaluation(self):
        """Run both standard and corruption evaluations"""
        print("="*50)
        print("COMBINED ENSEMBLE EVALUATION")
        print("="*50)

        # Run standard validation
        # std_results = self.standard_validation()

        print("\n" + "="*50)

        # Run corruption validation
        self.corruption_validation()

        # Save results
        modelname = f"{self.params['MODEL']['expert']}"
        self.save_results(modelname)

        # return std_results
        return


if __name__ == "__main__":
    warnings.filterwarnings(
        "ignore",
        message="You are using `torch.load` with `weights_only=False`"
    )

    if len(sys.argv) != 2:
        print('\nPlease pass the desired param file for evaluation as an argument.\n'
              'e.g: params/params_ensemble.yaml')
    else:
        print('STARTING COMBINED EVALUATION WITH PARAM FILE: ',
              str(sys.argv[1]))
        with open(str(sys.argv[1]), 'r') as stream:
            try:
                params = yaml.safe_load(stream)
                params["gpu_ids"] = [0]  # Set GPU

                evaluator = CombinedEnsembleEvaluator(params)
                std_results = evaluator.run_full_evaluation()

                print("EVALUATION COMPLETED SUCCESSFULLY")

            except yaml.YAMLError as exc:
                print(f"YAML Error: {exc}")
            except Exception as exc:
                print(f"Evaluation Error: {exc}")
                raise
