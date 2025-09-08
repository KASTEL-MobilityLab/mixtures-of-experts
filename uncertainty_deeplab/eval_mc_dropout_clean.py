import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from models import IMG_HEIGHT, IMG_WIDTH
from tqdm import tqdm
import yaml
import csv
import os
import sys

from sklearn.metrics import brier_score_loss, log_loss
from utils.images_helpers import create_images
from utils.uncertainty_helpers import calculate_ece, calculate_mce, calculate_metrics, compute_mutual_information, get_predictive_entropy
from utils.metrics import Evaluator, clamp_and_log_values
from models.deeplab_modeling import _load_model, _load_model_mc, remap_and_load_state_dict, enable_dropout
from dataloader.a2d2_loader import get_dataloader

# Global speed tweaks
torch.backends.cudnn.benchmark = True
torch.backends.cudnn.enabled = True
torch.backends.cuda.matmul.allow_tf32 = True


class EvaluationSummarizer:
    """Evaluation Summarizer with MC-Dropout support and uncertainty metrics"""
    # pylint: disable=too-many-branches

    def __init__(self, params, dropout_rate=0.1, mc_samples=2):
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

        # Per-image metrics (metrics_1)
        self.metrics_1_uncertainty = []
        self.metrics_1_pe = []
        self.metrics_1_mi = []

        # Threshold-based metrics (metrics_2)
        self.metrics_2_uncertainty = []
        self.metrics_2_pe = []
        self.metrics_2_mi = []

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

    def validation(self):
        """Validation with MC-Dropout support and uncertainty metrics"""
        print(
            f"Starting evaluation with dropout_rate={self.dropout_rate:.2f}, mc_samples={self.mc_samples}")
        self.model.eval()

        # Enable dropout for MC sampling if dropout_rate > 0
        if self.dropout_rate > 0:
            enable_dropout(self.model)

        self.evaluator.reset()
        self.uncertainty_metrics = []
        tbar = tqdm(self.test_loader, desc="\r")

        max_entropy = torch.log(torch.tensor(
            self.nclass, dtype=torch.float32, device=self.device))

        for i, sample in enumerate(tbar):
            image, target, label_path = sample
            image = image.to(self.device)
            target = target.to(self.device)

            if self.is_cuda:
                image, target = image.cuda(), target.cuda()

            with torch.no_grad():
                outputs = []
                for _ in range(self.mc_samples):
                    output = self.model(image)
                    outputs.append(output)

                # Stack outputs and compute softmax probabilities
                output_tensor = torch.stack(outputs)
                softmax_probs = torch.softmax(output_tensor, dim=2)
                mean_softmax = torch.mean(softmax_probs, dim=0)

                # Compute predictive entropy
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
            self.evaluator.add_batch(target_np, pred)

            batch_size = image.size(0)
            for j in range(batch_size):
                image_index = i * batch_size + j
                mask = (target_np[j] != 255)
                label = target_np[j][mask]
                prediction = pred[j][mask]

                if image_index in [0]:  # highway, ambiguous at index 0, urban at index 45
                    # save raw input
                    img_np = image[j].permute(1, 2, 0).cpu().numpy()
                    img_np = (img_np - img_np.min()) / \
                        (img_np.max() - img_np.min())
                    os.makedirs("./images", exist_ok=True)
                    plt.imsave(
                        f"./images/{image_index}_input.png", img_np)

                    # force 2D label
                    current_label = torch.from_numpy(target_np[j])
                    if current_label.dim() == 1:
                        current_label = current_label.view(
                            IMG_HEIGHT, IMG_WIDTH)

                    # predictive entropy
                    create_images(
                        pred=torch.from_numpy(pred[j]),
                        label=current_label,
                        uncertainty=scaled_pe[j],
                        modelname=f"mc_ambiguous_pe",
                        image_index=image_index,
                        dropout_rate=self.dropout_rate,
                        predictive_entropy=scaled_pe[j],
                        mutual_information=None,
                        image_path="./images/"
                    )
                    # mutual information
                    create_images(
                        pred=torch.from_numpy(pred[j]),
                        label=current_label,
                        uncertainty=scaled_mi[j],
                        modelname=f"mc_ambiguous_mi",
                        image_index=image_index,
                        dropout_rate=self.dropout_rate,
                        predictive_entropy=None,
                        mutual_information=scaled_mi[j],
                        image_path="./images/"
                    )
                    return

                pe_conf_j = pe_conf[j].cpu().numpy()[mask]
                mi_conf_j = mi_conf[j].cpu().numpy()[mask]

                # Compute scalar metrics
                for metric_type, conf_map, store_1, store_2 in [
                    ("pe", pe_conf_j, self.metrics_1_pe, self.metrics_2_pe),
                    ("mi", mi_conf_j, self.metrics_1_mi, self.metrics_2_mi)
                ]:
                    # Flatten arrays for metric calculations
                    flat_label = label.flatten()
                    flat_pred = prediction.flatten()
                    flat_conf = conf_map.flatten()

                    # Clamp and validate confidence values
                    flat_conf = clamp_and_log_values(flat_conf, metric_type)

                    # Create tensors on CPU
                    conf_tensor = torch.from_numpy(flat_conf).float()
                    pred_tensor = torch.from_numpy(flat_pred)
                    label_tensor = torch.from_numpy(flat_label)

                    # Calculate inaccuracies
                    accuracy_map = (flat_pred == flat_label)
                    inaccuracies = np.sum(~accuracy_map)

                    # Compute metrics with error handling
                    ece = calculate_ece(conf_tensor, pred_tensor, label_tensor)
                    mce = calculate_mce(conf_tensor, pred_tensor, label_tensor)

                    try:
                        brier = brier_score_loss(accuracy_map, flat_conf)
                    except Exception:
                        brier = np.nan

                    try:
                        nll = log_loss(accuracy_map, flat_conf, labels=[0, 1])
                    except Exception:
                        nll = np.nan

                    # Store metrics_1
                    store_1.append([
                        image_index,
                        self.dropout_rate,
                        float(ece) if not np.isnan(ece) else np.nan,
                        float(mce) if not np.isnan(mce) else np.nan,
                        brier,
                        nll,
                        inaccuracies,
                        0  # inference_time placeholder
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
                            image_index,
                            thr,
                            float(p_acc_c) if p_acc_c is not None else np.nan,
                            float(
                                p_unc_inacc) if p_unc_inacc is not None else np.nan,
                            float(pavpu) if pavpu is not None else np.nan,
                            self.dropout_rate,
                        ])

        # Compute segmentation metrics
        acc = self.evaluator.pixel_accuracy()
        acc_class = self.evaluator.pixel_accuracy_class()
        m_iou = self.evaluator.mean_intersection_over_union(
            self.label_names)
        fw_iou = self.evaluator.frequency_weighted_intersection_over_union()

        print("Validation:")
        print("pAcc:{}, mAcc:{}, m_iou:{}, fwIoU: {}".format(
            acc, acc_class, m_iou, fw_iou))

        for metrics_list, metric_name in zip(
            [self.metrics_1_pe, self.metrics_1_mi],
            ["PE", "MI"]
        ):
            if metrics_list:
                df = pd.DataFrame(
                    metrics_list,
                    columns=["image_index", "dropout_rate", "ece", "mce",
                             "brier", "nll", "inaccuracies", "inference_time"]
                )
                avg_ece = df["ece"].mean()
                avg_mce = df["mce"].mean()
                avg_brier = df["brier"].mean()
                avg_nll = df["nll"].mean()
                print(f"{metric_name} -> Avg ECE={avg_ece:.4f}, MCE={avg_mce:.4f}, "
                      f"Brier={avg_brier:.4f}, NLL={avg_nll:.4f}")


def get_summary_metrics(self):
    """Return summary metrics for this dropout rate"""
    summary = {}

    for metrics_list, metric_name in zip(
        [self.metrics_1_pe, self.metrics_1_mi],
        ["pe", "mi"]
    ):
        if metrics_list:
            df = pd.DataFrame(
                metrics_list,
                columns=["image_index", "dropout_rate", "ece", "mce",
                         "brier", "nll", "inaccuracies", "inference_time"]
            )

            summary[f"{metric_name}_avg_ece"] = df["ece"].mean()
            summary[f"{metric_name}_avg_mce"] = df["mce"].mean()
            summary[f"{metric_name}_avg_brier"] = df["brier"].mean()
            summary[f"{metric_name}_avg_nll"] = df["nll"].mean()

    return summary


def evaluate_dropout_rates(params_file, output_dir='./mc_dropout_rates',
                           dropout_rates=None, mc_samples=2):
    """Evaluate model across different dropout rates with uncertainty metrics"""
    if dropout_rates is None:
        dropout_rates = np.linspace(0, 1.0, 11)

    os.makedirs(output_dir, exist_ok=True)

    # Initialize CSV files
    for metric_name in ["pe", "mi"]:
        for suffix in ["1", "2"]:
            path = os.path.join(
                output_dir, f"{metric_name}_{suffix}.csv")
            if not os.path.exists(path):
                with open(path, 'w', newline='') as f:
                    writer = csv.writer(f)
                    if suffix == "1":
                        writer.writerow([
                            "image_index", "dropout_rate", "ece", "mce",
                            "brier", "nll", "inaccuracies", "inference_time"
                        ])
                    else:
                        writer.writerow([
                            "image_index", "uncertainty_threshold",
                            "p_accurate_given_certain", "p_uncertain_given_inaccurate",
                            "pavpu", "dropout_rate"
                        ])

    summary_results = []

    with open(params_file, 'r') as stream:
        params = yaml.safe_load(stream)
        params["gpu_ids"] = [0]

        for dropout_rate in tqdm(dropout_rates, desc="Dropout rates"):
            try:
                evaluator = EvaluationSummarizer(
                    params, dropout_rate=dropout_rate, mc_samples=mc_samples
                )
                evaluator.validation()

                summary_metrics = evaluator.get_summary_metrics()
                summary_row = {"dropout_rate": dropout_rate}
                summary_row.update(summary_metrics)
                summary_results.append(summary_row)

                # Append results to CSV
                for metrics_list, metric_name, suffix in zip(
                    [evaluator.metrics_1_pe, evaluator.metrics_1_mi,
                     evaluator.metrics_2_pe, evaluator.metrics_2_mi],
                    ["pe", "mi", "pe", "mi"],
                    ["1", "1", "2", "2"]
                ):
                    path = os.path.join(
                        output_dir, f"{metric_name}_{suffix}.csv")
                    with open(path, 'a', newline='') as f:
                        writer = csv.writer(f)
                        writer.writerows(metrics_list)

            except Exception as e:
                print(f"Error at dropout_rate={dropout_rate}: {str(e)}")
                continue

        if summary_results:
            summary_df = pd.DataFrame(summary_results)
            summary_path = os.path.join(output_dir, "dropout_summary.csv")
            summary_df.to_csv(summary_path, index=False)
            print(f"\nSummary results saved to: {summary_path}")

            # Print summary table
            print("\nSummary of Average Metrics Across Dropout Rates:")
            print("="*80)
            print(f"{'Dropout':<8} {'PE_ECE':<8} {'PE_MCE':<8} {'PE_Brier':<10} {'PE_NLL':<8} {'MI_ECE':<8} {'MI_MCE':<8} {'MI_Brier':<10} {'MI_NLL':<8}")
            print("-"*80)
            for _, row in summary_df.iterrows():
                print(f"{row['dropout_rate']:<8.2f} "
                      f"{row.get('pe_avg_ece', np.nan):<8.4f} "
                      f"{row.get('pe_avg_mce', np.nan):<8.4f} "
                      f"{row.get('pe_avg_brier', np.nan):<10.4f} "
                      f"{row.get('pe_avg_nll', np.nan):<8.4f} "
                      f"{row.get('mi_avg_ece', np.nan):<8.4f} "
                      f"{row.get('mi_avg_mce', np.nan):<8.4f} "
                      f"{row.get('mi_avg_brier', np.nan):<10.4f} "
                      f"{row.get('mi_avg_nll', np.nan):<8.4f}")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print('\nUsage:')
        print(
            'Single evaluation: python script.py params/params_moe.py [dropout_rate] [mc_samples]')
        print(
            'Multiple dropout rates: python script.py params/params_mc.py --sweep [output_dir] [mc_samples]')
        sys.exit(1)

    params_file = sys.argv[1]

    if len(sys.argv) > 2 and sys.argv[2] == '--sweep':
        # Sweep across dropout rates
        output_dir = sys.argv[3] if len(
            sys.argv) > 3 else './mc_dropout_ambiguous'
        mc_samples = int(sys.argv[4]) if len(sys.argv) > 4 else 2

        print(f'STARTING DROPOUT SWEEP WITH PARAM FILE: {params_file}')
        print(f'Output directory: {output_dir}')
        print(f'MC samples: {mc_samples}')

        evaluate_dropout_rates(params_file, output_dir, mc_samples=mc_samples)

    else:
        # Single evaluation
        dropout_rate = float(sys.argv[2]) if len(sys.argv) > 2 else 0.1
        mc_samples = int(sys.argv[3]) if len(sys.argv) > 3 else 2

        print(f'STARTING EVALUATION WITH PARAM FILE: {params_file}')
        print(f'Dropout rate: {dropout_rate}, MC samples: {mc_samples}')

        try:
            with open(params_file, 'r') as stream:
                params = yaml.safe_load(stream)
                params["gpu_ids"] = [0]
                evaluator = EvaluationSummarizer(
                    params, dropout_rate=dropout_rate, mc_samples=mc_samples)
                evaluator.validation()

                out_dir = './mc_dropout_rates'
                os.makedirs(out_dir, exist_ok=True)
                # per-image metrics (suffix "1")
                for metrics_list, metric_name in zip(
                    [evaluator.metrics_1_pe, evaluator.metrics_1_mi],
                    ["pe", "mi"]
                ):
                    if metrics_list:
                        df = pd.DataFrame(
                            metrics_list,
                            columns=[
                                "image_index", "dropout_rate",
                                "ece", "mce", "brier", "nll",
                                "inaccuracies", "inference_time"
                            ]
                        )
                        df.to_csv(
                            os.path.join(out_dir, f"{metric_name}_1.csv"),
                            index=False
                        )
                # threshold-based metrics (suffix "2")
                for metrics_list, metric_name in zip(
                    [evaluator.metrics_2_pe, evaluator.metrics_2_mi],
                    ["pe", "mi"]
                ):
                    if metrics_list:
                        df2 = pd.DataFrame(
                            metrics_list,
                            columns=[
                                "image_index", "uncertainty_threshold",
                                "p_accurate_given_certain",
                                "p_uncertain_given_inaccurate",
                                "pavpu", "dropout_rate"
                            ]
                        )
                        df2.to_csv(
                            os.path.join(out_dir, f"{metric_name}_2.csv"),
                            index=False
                        )

        except yaml.YAMLError as exc:
            print(exc)
