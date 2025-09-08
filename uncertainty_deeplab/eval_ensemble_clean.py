import os
import csv
import warnings

from matplotlib import pyplot as plt
import numpy as np
import torch
from torch.amp import autocast
from models import IMG_HEIGHT, IMG_WIDTH
from tqdm import tqdm
import sys
import yaml

from dataloader.a2d2_loader import get_dataloader
from models.deeplab_ensemble import Ensemble
from utils.images_helpers import create_images
from utils.metrics import Evaluator, segmentation_metrics_img, clamp_and_log_values
from utils.uncertainty_helpers import (
    calculate_ece, calculate_mce, calculate_metrics,
    compute_mutual_information, get_predictive_entropy
)
from sklearn.metrics import brier_score_loss, log_loss

# Global speed tweaks
torch.backends.cudnn.benchmark = True
torch.backends.cudnn.enabled = True
torch.backends.cuda.matmul.allow_tf32 = True


class Severity0EnsembleEvaluator:
    """Fast Severity 0 (Clean Data) Ensemble Evaluator with Scientific Integrity"""

    def __init__(self, params):
        self.params = params
        self.is_cuda = len(params["gpu_ids"]) > 0
        self.device = torch.device('cuda', params["gpu_ids"][0]) \
            if self.is_cuda else torch.device('cpu')

        self.nclass = params['DATASET']['num_class']

        # Use test loader directly for clean evaluation
        self.test_loader, self.label_names, self.label_colors = get_dataloader(
            params, params['test_sets'], 'test')

        # Initialize evaluator for mIoU calculation
        self.evaluator = Evaluator(self.nclass)

        # Initialize ensemble model
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

        # Metrics storage - optimized for single corruption type
        self.seg_metrics = []
        self.pe_metrics_1 = []
        self.pe_metrics_2 = []
        self.mi_metrics_1 = []
        self.mi_metrics_2 = []

    @torch.inference_mode()
    def evaluate_clean_data(self):
        """Fast evaluation on clean (severity 0) data with scientific integrity"""
        print("Starting Severity 0 (Clean Data) Evaluation")
        print(
            f"Model: {self.params['MODEL']['expert']}_{self.params['MODEL']['arch']}")
        print(f"Device: {self.device}")

        self.model.eval()
        self.evaluator.reset()

        img_counter = 0
        max_entropy = torch.log(torch.tensor(self.nclass, device=self.device))

        tbar = tqdm(self.test_loader, desc='Clean data evaluation')
        for i, sample in enumerate(tbar):
            image, target, label_path = sample
            image = image.to(self.device, non_blocking=True)
            target = target.to(self.device, non_blocking=True)

            B = image.size(0)
            with autocast("cuda"):
                out1 = self.model.expert1(image)
                out2 = self.model.expert2(image)

                # Apply softmax to individual outputs first
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

            scaled_pe = pe / max_entropy
            scaled_mi = mi / max_entropy

            # Normalize uncertainty by maximum entropy
            pe_conf = 1.0 - scaled_pe
            mi_conf = 1.0 - scaled_mi

            # Get predictions
            pred = mean_soft.argmax(dim=1)
            pred_cpu = pred.cpu().numpy()
            target_cpu = target.cpu().numpy()

            # Add to evaluator for mIoU calculation
            self.evaluator.add_batch(target_cpu, pred_cpu)

            # Process each image in batch
            for b in range(B):
                idx = img_counter + b
                t_np = target_cpu[b]
                p_np = pred_cpu[b]

                # Apply valid mask (ignore label 255)
                valid_mask = t_np != 255
                t_valid = t_np[valid_mask]
                p_valid = p_np[valid_mask]

                if idx in [45]:  # highway, ambiguous at index 0, urban at index 45
                    # save raw input
                    img_np = image[b].permute(1, 2, 0).cpu().numpy()
                    img_np = (img_np - img_np.min()) / \
                        (img_np.max() - img_np.min())
                    os.makedirs("./images", exist_ok=True)
                    plt.imsave(
                        f"./images/{idx}_ensemble_input_urban.png", img_np)

                    # force 2D label
                    current_label = torch.from_numpy(t_np)
                    if current_label.dim() == 1:
                        current_label = current_label.view(
                            IMG_HEIGHT, IMG_WIDTH)

                    pred_tensor = torch.from_numpy(p_np).to(self.device)
                    label_tensor = torch.from_numpy(t_np).to(self.device)

                    # predictive entropy
                    create_images(
                        pred=pred_tensor,
                        label=label_tensor,
                        uncertainty=scaled_pe[b],
                        modelname=f"ensemble_urban_pe",
                        image_index=idx,
                        dropout_rate=0.0,
                        predictive_entropy=scaled_pe[b],
                        mutual_information=None,
                        image_path="./images/"
                    )
                    # mutual information
                    create_images(
                        pred=pred_tensor,
                        label=label_tensor,
                        uncertainty=scaled_mi[b],
                        modelname=f"ensemble_urban_mi",
                        image_index=idx,
                        dropout_rate=0.0,
                        predictive_entropy=None,
                        mutual_information=scaled_mi[b],
                        image_path="./images/"
                    )
                    return

                # Skip empty images
                if len(t_valid) == 0:
                    print(f"Warning: Empty valid pixels in image {idx}")
                    continue

                # Process PE uncertainty metrics
                if pe_conf.nelement() > 0:
                    pe_conf_b = pe_conf[b][valid_mask].cpu().numpy()
                    pe_conf_clamped = clamp_and_log_values(
                        pe_conf_b, "PE confidence")
                    self._process_uncertainty_fast(
                        t_valid, p_valid, pe_conf_clamped,
                        self.pe_metrics_1, self.pe_metrics_2,
                        idx, "clean", 0
                    )

                # Process MI uncertainty metrics
                if mi_conf.nelement() > 0:
                    mi_conf_b = mi_conf[b][valid_mask].cpu().numpy()
                    mi_conf_clamped = clamp_and_log_values(
                        mi_conf_b, "MI confidence")
                    self._process_uncertainty_fast(
                        t_valid, p_valid, mi_conf_clamped,
                        self.mi_metrics_1, self.mi_metrics_2,
                        idx, "clean", 0
                    )

                # Segmentation metrics per image
                try:
                    seg_m = segmentation_metrics_img(t_valid, p_valid)
                    self.seg_metrics.append([idx, "clean", 0] + seg_m)
                except Exception as e:
                    print(
                        f"Warning: Segmentation metrics failed for image {idx}: {str(e)}")
                    self.seg_metrics.append([idx, "clean", 0] + [np.nan]*7)

            img_counter += B

        # Calculate final segmentation metrics
        acc = self.evaluator.pixel_accuracy()
        acc_class = self.evaluator.pixel_accuracy_class()
        m_iou = self.evaluator.mean_intersection_over_union(
            self.label_names)
        fw_iou = self.evaluator.frequency_weighted_intersection_over_union()

        print(f"\nSeverity 0 (Clean Data) Results:")
        print(
            f"pAcc: {acc}, mAcc: {acc_class}, mIoU: {m_iou}, fwIoU: {fw_iou}")

        # Print uncertainty summary
        self._print_uncertainty_summary()

        torch.cuda.empty_cache()
        return acc, acc_class, m_iou, fw_iou

    def _process_uncertainty_fast(self, t_np, p_np, conf, metrics_1, metrics_2,
                                  idx, corr_name, severity):
        """Optimized uncertainty processing for single severity"""
        if len(t_np) == 0 or len(p_np) == 0:
            metrics_1.append([idx, corr_name, severity, np.nan,
                             np.nan, np.nan, np.nan, np.nan])
            return

        try:
            # Flatten arrays
            flat_t = t_np.flatten()
            flat_p = p_np.flatten()
            flat_conf = conf.flatten()

            # Calculate inaccuracies
            inaccuracies = np.sum(flat_p != flat_t)

            # Convert to tensors for metric calculation
            conf_t = torch.from_numpy(flat_conf)
            p_t = torch.from_numpy(flat_p)
            t_t = torch.from_numpy(flat_t)

            # Calculate calibration metrics
            try:
                mce = calculate_mce(conf_t, p_t, t_t)
                ece = calculate_ece(conf_t, p_t, t_t)
            except Exception as e:
                print(f"ECE/MCE calculation failed: {str(e)}")
                mce, ece = np.nan, np.nan

            # Calculate Brier score
            try:
                acc_map = (flat_p == flat_t).astype(np.uint8)
                brier = brier_score_loss(acc_map, flat_conf)
            except Exception as e:
                print(f"Brier score calculation failed: {str(e)}")
                brier = np.nan

            # Calculate NLL
            try:
                nll = log_loss(acc_map, flat_conf, labels=[0, 1])
            except ValueError as e:
                if "Only one class" in str(e):
                    nll = np.nan  # Handle perfect predictions
                else:
                    print(f"NLL calculation failed: {str(e)}")
                    nll = np.nan

            # Store primary metrics with inaccuracies
            metrics_1.append([
                idx, corr_name, severity, inaccuracies,
                float(mce) if not np.isnan(mce) else np.nan,
                float(ece) if not np.isnan(ece) else np.nan,
                float(brier) if not np.isnan(brier) else np.nan,
                float(nll) if not np.isnan(nll) else np.nan
            ])

            for th_i in range(1, 101, 2):
                thr = th_i / 100.0
                try:
                    p_acc_c, p_unc_inacc, pavpu = calculate_metrics(
                        target=t_t, predictions=p_t,
                        confidences=conf_t, uncertainty_threshold=thr
                    )

                    metrics_2.append([
                        idx, corr_name, severity, thr,
                        float(p_acc_c) if p_acc_c is not None else np.nan,
                        float(p_unc_inacc) if p_unc_inacc is not None else np.nan,
                        float(pavpu) if pavpu is not None else np.nan
                    ])
                except Exception as e:
                    print(f"Threshold {thr} calculation failed: {str(e)}")
                    metrics_2.append(
                        [idx, corr_name, severity, thr, np.nan, np.nan, np.nan])

        except Exception as e:
            print(f"Error processing uncertainty for image {idx}: {str(e)}")
            metrics_1.append([idx, corr_name, severity, np.nan,
                             np.nan, np.nan, np.nan, np.nan])

    def _print_uncertainty_summary(self):
        """Print summary of uncertainty metrics"""
        for metrics_list, metric_name in [
            (self.pe_metrics_1, "Predictive Entropy (PE)"),
            (self.mi_metrics_1, "Mutual Information (MI)")
        ]:
            if metrics_list:
                import pandas as pd
                df = pd.DataFrame(
                    metrics_list,
                    columns=["image_index", "corruption_name", "severity_level",
                             "inaccuracies", "mce", "ece", "brier", "nll"]
                )

                avg_ece = df["ece"].mean()
                avg_mce = df["mce"].mean()
                avg_brier = df["brier"].mean()
                avg_nll = df["nll"].mean()
                print(f"{metric_name} -> Avg ECE={avg_ece:.4f}, MCE={avg_mce:.4f}, "
                      f"Brier={avg_brier:.4f}, NLL={avg_nll:.4f}")

    def save_results(self, output_dir="./results/severity_0_ensemble"):
        """Save all evaluation results to CSV files"""
        os.makedirs(output_dir, exist_ok=True)

        metric_files = [
            ("segmentation.csv", self.seg_metrics, [
                "image_index", "corruption_name", "severity_level",
                "iou", "m_iou", "fw_iou", "precision",
                "recall", "f1_score", "pixel_accuracy"
            ]),
            ("pe.csv", self.pe_metrics_1, [
                "image_index", "corruption_name", "severity_level", "inaccuracies",
                "mce", "ece", "brier", "nll"
            ]),
            ("pe_2.csv", self.pe_metrics_2, [
                "image_index", "corruption_name", "severity_level",
                "uncertainty_threshold", "p_accurate_given_certain",
                "p_uncertain_given_inaccurate", "pavpu"
            ]),
            ("mi.csv", self.mi_metrics_1, [
                "image_index", "corruption_name", "severity_level", "inaccuracies",
                "mce", "ece", "brier", "nll"
            ]),
            ("mi_2.csv", self.mi_metrics_2, [
                "image_index", "corruption_name", "severity_level",
                "uncertainty_threshold", "p_accurate_given_certain",
                "p_uncertain_given_inaccurate", "pavpu"
            ])
        ]

        for fname, data, header in metric_files:
            if data:  # Only save if we have data
                path = os.path.join(output_dir, fname)
                with open(path, 'w', newline="") as f:
                    writer = csv.writer(f)
                    writer.writerow(header)
                    writer.writerows(data)
                print(f"Saved {len(data)} records to {fname}")
            else:
                print(f"Warning: No data for {fname}")

        print(f"\nAll results saved to {output_dir}")

    def run_evaluation(self):
        """Run complete severity 0 evaluation"""
        print("="*60)
        print("SEVERITY 0 (CLEAN DATA) ENSEMBLE EVALUATION")
        print("="*60)

        # Run clean data evaluation
        results = self.evaluate_clean_data()

        # Save results
        modelname = f"{self.params['MODEL']['expert']}_sev0_{self.params['test_sets']}_curr"
        output_dir = f"./results/{modelname}"
        self.save_results(output_dir)

        print("="*60)
        print("EVALUATION COMPLETED SUCCESSFULLY")
        print(f"Clean Data mIoU: {results[2]}")
        print("="*60)

        return results


if __name__ == "__main__":
    warnings.filterwarnings(
        "ignore",
        message="You are using `torch.load` with `weights_only=False`"
    )

    if len(sys.argv) != 2:
        print('\nUsage: python severity_0_ensemble.py <params_file>')
        print('Example: python severity_0_ensemble.py params/params_ensemble.yaml')
        sys.exit(1)

    params_file = sys.argv[1]
    print(f'STARTING SEVERITY 0 EVALUATION WITH PARAM FILE: {params_file}')

    try:
        with open(params_file, 'r') as stream:
            params = yaml.safe_load(stream)
            params["gpu_ids"] = [0]  # Set GPU

            evaluator = Severity0EnsembleEvaluator(params)
            results = evaluator.run_evaluation()

    except yaml.YAMLError as exc:
        print(f"YAML Error: {exc}")
        sys.exit(1)
    except Exception as exc:
        print(f"Evaluation Error: {exc}")
        raise
