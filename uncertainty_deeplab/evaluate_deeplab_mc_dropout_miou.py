import os
import sys
import csv
import yaml
import torch
import numpy as np
from tqdm import tqdm

from dataloader.a2d2_loader import get_dataloader
from models.deeplab_modeling import _load_model, _load_model_mc, remap_and_load_state_dict, enable_dropout
from utils.metrics import Evaluator

# Global speed tweaks
torch.backends.cudnn.benchmark = True
torch.backends.cudnn.enabled = True
torch.backends.cuda.matmul.allow_tf32 = True


class EvaluationSummarizer:
    """Evaluation Summarizer with MC-Dropout support"""
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

        # Load model based on expert type
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
        """Validation with MC-Dropout support"""
        print(
            f"Starting evaluation with dropout_rate={self.dropout_rate:.2f}, mc_samples={self.mc_samples}")
        self.model.eval()

        # Enable dropout for MC sampling if dropout_rate > 0
        if self.dropout_rate > 0:
            enable_dropout(self.model)

        self.evaluator.reset()

        tbar = tqdm(self.test_loader, desc="\r")
        for i, sample in enumerate(tbar):
            image, target, label_path = sample
            image = image.to(self.device)
            target = target.to(self.device)

            if self.is_cuda:
                image, target = image.cuda(), target.cuda()

            with torch.no_grad():
                if self.dropout_rate > 0 and self.mc_samples > 1:
                    # Monte Carlo sampling
                    probs_stack = []
                    for _ in range(self.mc_samples):
                        output = self.model(image)
                        probs = torch.softmax(output, dim=1)
                        probs_stack.append(probs)

                    # Average predictions
                    probs_mean = torch.mean(
                        torch.stack(probs_stack, dim=0), dim=0)
                    pred = torch.argmax(probs_mean, dim=1).cpu().numpy()
                else:
                    # Single forward pass
                    output = self.model(image)
                    probs = torch.softmax(output, dim=1)
                    pred = torch.argmax(probs, dim=1).cpu().numpy()

            target = target.cpu().numpy()
            self.evaluator.add_batch(target, pred)

        # Compute metrics
        acc = self.evaluator.pixel_accuracy()
        acc_class = self.evaluator.pixel_accuracy_class()
        m_iou = self.evaluator.mean_intersection_over_union(
            self.label_names)
        fw_iou = self.evaluator.frequency_weighted_intersection_over_union()

        print("Validation:")
        print("pAcc, mAcc, m_iou, fwIoU".format(
            acc, acc_class, m_iou, fw_iou))

        return acc, acc_class, m_iou, fw_iou


def evaluate_dropout_rates(params_file, output_dir='./mc_dropout_all_miou',
                           dropout_rates=None, mc_samples=2):
    """Evaluate model across different dropout rates"""
    if dropout_rates is None:
        dropout_rates = np.linspace(0, 1.0, 11)

    os.makedirs(output_dir, exist_ok=True)
    summary_csv = os.path.join(output_dir, 'miou_summary.csv')

    # Create CSV header
    with open(summary_csv, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["dropout_rate", "pAcc", "mAcc", "mIoU", "fwIoU"])

    # Load parameters
    with open(params_file, 'r') as stream:
        params = yaml.safe_load(stream)
        params["gpu_ids"] = [0]

    # Evaluate each dropout rate
    for rate in dropout_rates:
        print(f"\n=== Evaluating dropout rate: {rate:.2f} ===")

        evaluator = EvaluationSummarizer(
            params, dropout_rate=rate, mc_samples=mc_samples)
        acc, acc_class, m_iou, fw_iou = evaluator.validation()

        # Log results
        with open(summary_csv, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([f"{rate}", f"{acc}", f"{acc_class}",
                             f"{m_iou}", f"{fw_iou}"])

    print(f"\nResults saved to: {summary_csv}")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print('\nUsage:')
        print(
            'Single evaluation: python script.py params/params_moe.py [dropout_rate] [mc_samples]')
        print(
            'Multiple dropout rates: python script.py params/params_moe.py --sweep [output_dir] [mc_samples]')
        sys.exit(1)

    params_file = sys.argv[1]

    if len(sys.argv) > 2 and sys.argv[2] == '--sweep':
        # Sweep across dropout rates
        output_dir = sys.argv[3] if len(
            sys.argv) > 3 else './results/mc_dropout_all_miou'
        mc_samples = int(sys.argv[4]) if len(sys.argv) > 4 else 2

        print(f'STARTING DROPOUT SWEEP WITH PARAM FILE: {params_file}')
        print(f'Output directory: {output_dir}')
        print(f'MC samples: {mc_samples}')

        evaluate_dropout_rates(params_file, output_dir, mc_samples=mc_samples)

    else:
        # Single evaluation
        dropout_rate = float(sys.argv[2]) if len(sys.argv) > 2 else 0.0
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

        except yaml.YAMLError as exc:
            print(exc)
