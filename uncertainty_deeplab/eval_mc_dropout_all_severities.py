import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
import yaml
import os
import sys

from sklearn.metrics import brier_score_loss, log_loss
from utils.uncertainty_helpers import calculate_ece, calculate_mce, calculate_metrics, compute_mutual_information, get_predictive_entropy
from utils.metrics import Evaluator, clamp_and_log_values
from models.deeplab_modeling import _load_model, _load_model_mc, remap_and_load_state_dict, enable_dropout
from dataloader.a2d2_loader import get_dataloader
from utils.robustness_helpers import corruptions

# Global speed tweaks
torch.backends.cudnn.benchmark = True
torch.backends.cudnn.enabled = True
torch.backends.cuda.matmul.allow_tf32 = True


class CombinedMCDropoutEvaluator:
    """Evaluation Summarizer with MC-Dropout support, uncertainty metrics, and severity levels"""
    # pylint: disable=too-many-branches

    def __init__(self, params, dropout_rate=0.0, mc_samples=2):
        self.params = params
        self.dropout_rate = dropout_rate
        self.mc_samples = mc_samples
        self.is_cuda = len(params["gpu_ids"]) > 0
        self.device = torch.device('cuda', params["gpu_ids"][0]) \
            if self.is_cuda else torch.device('cpu')

        self.nclass = params['DATASET']['num_class']
        self.datasets = list(
            s for s in params["DATASET"]["dataset"].split(','))
        self.test_loader, self.label_names, self.label_colors = get_dataloader(
            params, params['test_sets'], 'test')

        self.evaluator = Evaluator(self.nclass)

        # Metrics storage - adding corruption and severity dimensions
        self.metrics_1_pe = []  # Per-image metrics PE
        self.metrics_1_mi = []  # Per-image metrics MI
        self.metrics_2_pe = []  # Threshold-based metrics PE
        self.metrics_2_mi = []  # Threshold-based metrics MI
        self.segmentation = []  # Segmentation metrics

        self._load_model()

    def _load_model(self):
        """Load model with dropout support"""
        if self.dropout_rate > 0:
            self.model = _load_model_mc(
                self.params["MODEL"]["arch"],
                self.params["MODEL"]["backbone"],
                self.nclass,
                output_stride=self.params["MODEL"]["out_stride"],
                pretrained_backbone=True,
                input_channels=3,
                dropout_rate=self.dropout_rate
            )
        else:
            self.model = _load_model(
                self.params["MODEL"]["arch"],
                self.params["MODEL"]["backbone"],
                self.nclass,
                output_stride=self.params["MODEL"]["out_stride"],
                pretrained_backbone=True,
                input_channels=3
            )

        self._load_checkpoint()

    def _load_checkpoint(self):
        """Load checkpoint for single model or MoE"""
        if self.params["TEST"]["checkpoint"] is not None:
            if not os.path.isfile(self.params["TEST"]["checkpoint"]):
                raise RuntimeError("=> no checkpoint found at '{}'".format(
                    self.params["TEST"]["checkpoint"]))

            print("Loading checkpoint from", self.params["TEST"]["checkpoint"])
            checkpoint = torch.load(self.params["TEST"]["checkpoint"])
            self.params["start_epoch"] = checkpoint["epoch"]

            # Handle potential state dict remapping for MC models
            if self.dropout_rate > 0:
                self.model = remap_and_load_state_dict(
                    self.model, checkpoint["state_dict"])
            else:
                self.model.load_state_dict(checkpoint["state_dict"])

            print("=> loaded checkpoint '{}' (epoch {})".format(
                self.params["TEST"]["checkpoint"], checkpoint["epoch"]))

            if self.is_cuda:
                if self.params["MODEL"]["expert"] == "moe":
                    self.model = self.model.to(self.device)
                    self.model.expert1.to(self.device)
                    self.model.expert2.to(self.device)
                else:
                    self.model = torch.nn.DataParallel(
                        self.model, device_ids=self.params["gpu_ids"])
                    self.model = self.model.to(self.device)
        else:
            raise RuntimeError("=> no checkpoint in input arguments")

    def validation_with_severity_and_uncertainty(self):
        """Validation with MC-Dropout support, uncertainty metrics, and severity levels"""
        print(
            f"Starting evaluation with dropout_rate={self.dropout_rate:.2f}, mc_samples={self.mc_samples}")
        self.model.eval()

        if self.dropout_rate > 0:
            enable_dropout(self.model)

        tbar = tqdm(self.test_loader, desc="\r")

        max_entropy = torch.log(torch.tensor(
            self.nclass, dtype=torch.float32, device=self.device))

        # Results storage for all severity levels
        all_evaluators = {}

        for i, sample in enumerate(tbar):
            image, target, label_path = sample
            image = image.to(self.device)
            target = target.to(self.device)

            if self.is_cuda:
                image, target = image.cuda(), target.cuda()

            cpu_imgs = (
                image.cpu().permute(0, 2, 3, 1)
                .mul(255).clip(0, 255).byte().numpy()
            )

            # Process each corruption and severity level
            for corruption_fn, corr_name in corruptions:
                for severity in range(1, 6):  # Severity levels 1-5
                    # Apply corruption with error handling
                    try:
                        corrupted_images = []
                        for im in cpu_imgs:
                            try:
                                corrupted_im = corruption_fn(
                                    im, severity=severity)
                                if corrupted_im is None:
                                    print(
                                        f"Warning: {corr_name} returned None for severity {severity}, using original image")
                                    corrupted_images.append(im)
                                else:
                                    corrupted_images.append(corrupted_im)
                            except Exception as e:
                                print(
                                    f"Error applying {corr_name} with severity {severity}: {str(e)}")
                                print(f"Using original image instead")
                                corrupted_images.append(im)
                        corrupted_np = np.stack(corrupted_images, axis=0)
                    except Exception as e:
                        print(
                            f"Failed to apply corruption {corr_name} with severity {severity}: {str(e)}")
                        print("Skipping this corruption/severity combination")
                        continue

                    # Convert back to tensor
                    corrupted_image = (
                        torch.from_numpy(corrupted_np)
                        .permute(0, 3, 1, 2).float().div(255)
                        .to(self.device, non_blocking=True)
                    )

                    # Initialize evaluator for this corruption/severity
                    key = f"{corr_name}_sev{severity}"
                    if key not in all_evaluators:
                        all_evaluators[key] = {
                            'evaluator': Evaluator(self.nclass),
                            'corruption': corr_name,
                            'severity': severity
                        }

                    with torch.no_grad():
                        outputs = []
                        for _ in range(self.mc_samples):
                            output = self.model(corrupted_image)
                            outputs.append(output)

                        # Stack outputs and compute softmax probabilities
                        output_tensor = torch.stack(outputs)
                        softmax_probs = torch.softmax(output_tensor, dim=2)
                        mean_softmax = torch.mean(softmax_probs, dim=0)

                        # Compute predictive entropy and mutual information
                        predictive_entropy = get_predictive_entropy(
                            mean_softmax=mean_softmax)
                        mutual_information = compute_mutual_information(
                            predictive_entropy, softmax_probs)

                        scaled_pe = predictive_entropy / max_entropy
                        scaled_mi = mutual_information / max_entropy
                        pe_conf = 1 - scaled_pe
                        mi_conf = 1 - scaled_mi

                        pred = torch.argmax(mean_softmax, dim=1).cpu().numpy()

                    target_np = target.cpu().numpy()
                    all_evaluators[key]['evaluator'].add_batch(target_np, pred)

                    # Process uncertainty metrics for each image in batch
                    batch_size = corrupted_image.size(0)
                    for j in range(batch_size):
                        image_index = i * batch_size + j
                        mask = (target_np[j] != 255)
                        label = target_np[j][mask]
                        prediction = pred[j][mask]

                        pe_conf_j = pe_conf[j].cpu().numpy()[mask]
                        mi_conf_j = mi_conf[j].cpu().numpy()[mask]

                        # Compute scalar metrics for both PE and MI
                        for metric_type, conf_map, store_1, store_2 in [
                            ("pe", pe_conf_j, self.metrics_1_pe, self.metrics_2_pe),
                            ("mi", mi_conf_j, self.metrics_1_mi, self.metrics_2_mi)
                        ]:
                            # Flatten arrays for metric calculations
                            flat_label = label.flatten()
                            flat_pred = prediction.flatten()
                            flat_conf = conf_map.flatten()

                            # Clamp and validate confidence values
                            flat_conf = clamp_and_log_values(
                                flat_conf, metric_type)

                            # Create tensors on CPU
                            conf_tensor = torch.from_numpy(flat_conf).float()
                            pred_tensor = torch.from_numpy(flat_pred)
                            label_tensor = torch.from_numpy(flat_label)

                            # Calculate inaccuracies
                            accuracy_map = (flat_pred == flat_label)
                            inaccuracies = np.sum(~accuracy_map)

                            # Compute metrics with error handling
                            ece = calculate_ece(
                                conf_tensor, pred_tensor, label_tensor)
                            mce = calculate_mce(
                                conf_tensor, pred_tensor, label_tensor)

                            try:
                                brier = brier_score_loss(
                                    accuracy_map, flat_conf)
                            except Exception:
                                brier = np.nan

                            try:
                                nll = log_loss(
                                    accuracy_map, flat_conf, labels=[0, 1])
                            except Exception:
                                nll = np.nan

                            # Store metrics_1 with corruption and severity
                            store_1.append([
                                inaccuracies,
                                self.dropout_rate,
                                float(ece) if not np.isnan(ece) else np.nan,
                                float(mce) if not np.isnan(mce) else np.nan,
                                brier,
                                nll,
                                corr_name,
                                severity
                            ])

                            # Threshold-based metrics
                            for th in range(1, 101, 2):
                                thr = th / 100.0
                                try:
                                    p_acc_c, p_unc_inacc, pavpu = calculate_metrics(
                                        target=label_tensor,
                                        predictions=pred_tensor,
                                        confidences=conf_tensor,
                                        uncertainty_threshold=thr
                                    )
                                except Exception as e:
                                    p_acc_c, p_unc_inacc, pavpu = np.nan, np.nan, np.nan

                                store_2.append([
                                    float(
                                        p_acc_c) if p_acc_c is not None else np.nan,
                                    float(
                                        p_unc_inacc) if p_unc_inacc is not None else np.nan,
                                    float(pavpu) if pavpu is not None else np.nan,
                                    self.dropout_rate,
                                    corr_name,
                                    severity,
                                    thr
                                ])

        # Compute segmentation metrics for each corruption/severity
        for key, result_data in all_evaluators.items():
            evaluator = result_data['evaluator']
            corruption = result_data['corruption']
            severity = result_data['severity']

            # Compute metrics using the same logic as original EvaluationSummarizer
            acc = evaluator.pixel_accuracy()
            acc_class = evaluator.pixel_accuracy_class()

            miou_res = evaluator.mean_intersection_over_union(self.label_names)
            # Handle tuple return from mIoU calculation
            if isinstance(miou_res, (tuple, list)):
                m_iou = float(miou_res[0])
            else:
                m_iou = float(miou_res)

            fw_iou = evaluator.frequency_weighted_intersection_over_union()

            print(f"\nValidation - {corruption} (Severity {severity}):")
            print("pAcc:{:.4f}, mAcc:{:.4f}, m_iou:{:.4f}, fwIoU:{:.4f}".format(
                acc, acc_class, m_iou, fw_iou))

            # Store segmentation metrics
            self.segmentation.append([
                corruption,
                severity,
                acc,
                acc_class,
                m_iou,
                fw_iou
            ])

        # Print summary statistics for uncertainty metrics
        for metrics_list, metric_name in zip(
            [self.metrics_1_pe, self.metrics_1_mi],
            ["PE", "MI"]
        ):
            if metrics_list:
                df = pd.DataFrame(
                    metrics_list,
                    columns=["inaccuracies", "dropout_rate", "ece", "mce",
                             "brier", "nll", "corruption", "severity"]
                )
                avg_ece = df["ece"].mean()
                avg_mce = df["mce"].mean()
                avg_brier = df["brier"].mean()
                avg_nll = df["nll"].mean()
                print(f"\n{metric_name} -> Avg ECE={avg_ece:.4f}, MCE={avg_mce:.4f}, "
                      f"Brier={avg_brier:.4f}, NLL={avg_nll:.4f}")

    def save_all_metrics(self, output_dir):
        """Save all metrics to CSV files"""
        os.makedirs(output_dir, exist_ok=True)

        # segmentation
        pd.DataFrame(
            self.segmentation,
            columns=["corruption", "severity", "pAcc", "mAcc", "mIoU", "fwIoU"]
        ).to_csv(os.path.join(output_dir, "segmentation.csv"), index=False)

        # pe_1
        pd.DataFrame(
            self.metrics_1_pe,
            columns=["inaccuracies", "dropout_rate", "ece", "mce",
                     "brier", "nll", "corruption", "severity"]
        ).to_csv(os.path.join(output_dir, "pe_1.csv"), index=False)

        # pe_2
        pd.DataFrame(
            self.metrics_2_pe,
            columns=["p_acc", "p_unc", "pavpu",
                     "dropout_rate", "corruption", "severity", "threshold"]
        ).to_csv(os.path.join(output_dir, "pe_2.csv"), index=False)

        # mi_1
        pd.DataFrame(
            self.metrics_1_mi,
            columns=["inaccuracies", "dropout_rate", "ece", "mce",
                     "brier", "nll", "corruption", "severity"]
        ).to_csv(os.path.join(output_dir, "mi_1.csv"), index=False)

        # mi_2
        pd.DataFrame(
            self.metrics_2_mi,
            columns=["p_acc", "p_unc", "pavpu",
                     "dropout_rate", "corruption", "severity", "threshold"]
        ).to_csv(os.path.join(output_dir, "mi_2.csv"), index=False)

        print(f"All metrics saved to {output_dir}")


def evaluate_dropout_rates_with_severity_and_uncertainty(params_file, output_dir='./mc_dropout_severity_uncertainty',
                                                         dropout_rates=None, mc_samples=2):
    """Evaluate model across different dropout rates and severity levels with uncertainty metrics"""
    if dropout_rates is None:
        dropout_rates = np.linspace(0, 1.0, 11)

    os.makedirs(output_dir, exist_ok=True)

    # Load parameters
    with open(params_file, 'r') as stream:
        params = yaml.safe_load(stream)
        params["gpu_ids"] = [0]

    # Evaluate each dropout rate
    for rate in dropout_rates:
        print(f"\n=== Evaluating dropout rate: {rate:.2f} ===")

        rate_output_dir = os.path.join(output_dir, f'dropout_{rate:.2f}')
        os.makedirs(rate_output_dir, exist_ok=True)

        try:
            evaluator = CombinedMCDropoutEvaluator(
                params, dropout_rate=rate, mc_samples=mc_samples)
            evaluator.validation_with_severity_and_uncertainty()
            evaluator.save_all_metrics(rate_output_dir)

        except Exception as e:
            print(f"Error at dropout_rate={rate}: {str(e)}")
            continue

    print(f"\nAll results saved to: {output_dir}")


def evaluate_single_dropout_with_severity_and_uncertainty(params_file, dropout_rate=0.1, mc_samples=2,
                                                          output_dir='./single_mc_dropout_severity_uncertainty'):
    """Evaluate single dropout rate across all severity levels with uncertainty metrics"""
    os.makedirs(output_dir, exist_ok=True)

    # Load parameters
    with open(params_file, 'r') as stream:
        params = yaml.safe_load(stream)
        params["gpu_ids"] = [0]

    print(f'STARTING SEVERITY EVALUATION WITH UNCERTAINTY METRICS')
    print(f'PARAM FILE: {params_file}')
    print(f'Dropout rate: {dropout_rate}, MC samples: {mc_samples}')
    print(f'Output directory: {output_dir}')

    evaluator = CombinedMCDropoutEvaluator(
        params, dropout_rate=dropout_rate, mc_samples=mc_samples)
    evaluator.validation_with_severity_and_uncertainty()
    evaluator.save_all_metrics(output_dir)

    return evaluator


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print('\nUsage:')
        print(
            'Single evaluation: python script.py params/params_moe.py [dropout_rate] [mc_samples] [output_dir]')
        print(
            'Severity sweep: python script.py params/params_moe.py --severity-sweep [output_dir] [mc_samples]')
        sys.exit(1)

    params_file = sys.argv[1]

    if len(sys.argv) > 2 and sys.argv[2] == '--severity-sweep':
        # Sweep across dropout rates with severity levels and uncertainty metrics
        output_dir = sys.argv[3] if len(
            sys.argv) > 3 else './perturbation/mc_dropout'
        mc_samples = int(sys.argv[4]) if len(sys.argv) > 4 else 2

        print(
            f'STARTING DROPOUT SEVERITY SWEEP WITH UNCERTAINTY METRICS')
        print(f'PARAM FILE: {params_file}')
        print(f'Output directory: {output_dir}')
        print(f'MC samples: {mc_samples}')

        evaluate_dropout_rates_with_severity_and_uncertainty(
            params_file, output_dir, mc_samples=mc_samples)

    else:
        # Single dropout rate with severity levels and uncertainty metrics
        dropout_rate = float(sys.argv[2]) if len(sys.argv) > 2 else 0.1
        mc_samples = int(sys.argv[3]) if len(sys.argv) > 3 else 2
        output_dir = sys.argv[4] if len(
            sys.argv) > 4 else './perturbation/mc_dropout'

        evaluate_single_dropout_with_severity_and_uncertainty(
            params_file, dropout_rate, mc_samples, output_dir)
