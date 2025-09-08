import os
import csv
import warnings
import sys
import yaml
import numpy as np
import torch
from tqdm import tqdm

from dataloader.a2d2_loader import get_dataloader
from models.deeplab_moe import MoE
from utils.robustness_helpers import corruptions
from utils.metrics import Evaluator

# Speed tweaks
torch.backends.cudnn.benchmark = True
torch.backends.cudnn.enabled = True
torch.backends.cuda.matmul.allow_tf32 = True


class MoESeverityEvaluator:
    def __init__(self, params):
        self.params = params
        self.device = torch.device(
            'cuda:0' if params.get('gpu_ids') else 'cpu')
        self.nclass = params['DATASET']['num_class']

        # Data loader
        self.test_loader, self.label_names, self.label_colors = get_dataloader(
            params, params['test_sets'], 'test')

        # Build model
        linear_feat = (params['DATASET']['img_height'] //
                       params['MODEL']['out_stride'] + 1)**2
        linear_feat = 3249  # hardcoded for A2D2
        self.model = MoE(
            arch=params['MODEL']['arch'],
            backbone=params['MODEL']['backbone'],
            output_stride=params['MODEL']['out_stride'],
            num_classes=self.nclass,
            linear_features=linear_feat,
            checkpoint1=params['MODEL']['checkpoint_moe_expert_1'],
            checkpoint2=params['MODEL']['checkpoint_moe_expert_2'],
            gate_type=params['MODEL']['gate'],
            with_conv=params['MODEL']['with_conv'],
            allow_gradient_flow=False
        )

        # Load checkpoint
        ckpt = params['TEST']['checkpoint']
        if not ckpt or not os.path.isfile(ckpt):
            raise RuntimeError(f"=> no checkpoint found at '{ckpt}'")
        print("Loading checkpoint from", ckpt)
        data = torch.load(ckpt)
        self.model.load_state_dict(data['state_dict'])
        self.model.to(self.device)
        self.model.expert1.to(self.device)
        self.model.expert2.to(self.device)

        # Storage for results
        self.results = []  # will store dicts with keys: corruption, severity, pixel_accuracy, mean_accuracy, mean_iou, fw_iou

    @torch.inference_mode()
    def evaluate_severity(self, severity_level: int, corr_fn, corr_name: str):
        print(f"Evaluating {corr_name}, severity level: {severity_level}")
        self.model.eval()
        evaluator = Evaluator(self.nclass)
        evaluator.reset()

        tbar = tqdm(self.test_loader, desc=f'{corr_name} Sev {severity_level}')
        for images, targets, _ in tbar:
            images = images.to(self.device, non_blocking=True)
            targets = targets.to(self.device, non_blocking=True)

            # apply corruption if provided
            if severity_level == 0 or corr_fn is None:
                corrupted = images
            else:
                cpu_imgs = (images.cpu().permute(
                    0, 2, 3, 1).mul(255).byte().numpy())
                corrupted_np = np.stack(
                    [corr_fn(im, severity=severity_level) for im in cpu_imgs], axis=0)
                corrupted_np = np.clip(corrupted_np, 0, 255).astype(np.uint8)
                corrupted = (torch.from_numpy(corrupted_np)
                             .permute(0, 3, 1, 2)
                             .float().div(255)
                             .to(self.device))

            # forward pass
            output = self.model(corrupted)
            pred = output.cpu().numpy()
            target = targets.cpu().numpy()
            pred = np.argmax(pred, axis=1)

            try:
                evaluator.add_batch(target, pred)
            except Exception as e:
                print(f"Error adding batch: {e}")
                continue

        # compute metrics
        try:
            pAcc = evaluator.pixel_accuracy()
            mAcc = evaluator.pixel_accuracy_class()
            mIoU, _ = evaluator.mean_intersection_over_union(self.label_names)
            fwIoU = evaluator.frequency_weighted_intersection_over_union()
        except Exception as e:
            print(
                f"Error calculating metrics for {corr_name} severity {severity_level}: {e}")
            pAcc = mAcc = mIoU = fwIoU = 0.0

        print(f"{corr_name} Sev {severity_level} -> pAcc: {pAcc:.4f}, mAcc: {mAcc:.4f}, mIoU: {mIoU:.4f}, fwIoU: {fwIoU:.4f}")
        return {
            'corruption': corr_name,
            'severity_level': severity_level,
            'pAcc': pAcc,
            'mAcc': mAcc,
            'mIoU': mIoU,
            'fwIoU': fwIoU
        }

    def evaluate_all(self):
        # Evaluate clean baseline (no corruption)
        baseline = self.evaluate_severity(0, None, 'no_corruption')
        self.results.append(baseline)
        torch.cuda.empty_cache()

        # Evaluate each corruption at severities 1-5
        for corr_fn, corr_name in corruptions:
            for sev in range(1, 6):
                res = self.evaluate_severity(sev, corr_fn, corr_name)
                self.results.append(res)
                torch.cuda.empty_cache()

        self.save()
        self.summary()

    def save(self):
        outdir = f"miou_evaluation/{self.params['MODEL'].get('name','moe_model')}"
        os.makedirs(outdir, exist_ok=True)
        path = os.path.join(outdir, 'miou_by_severity.csv')
        with open(path, 'w', newline='') as f:
            w = csv.writer(f)
            w.writerow(['corruption', 'severity_level',
                       'pAcc', 'mAcc', 'mIoU', 'fwIoU'])
            for r in self.results:
                w.writerow([
                    r['corruption'], r['severity_level'],
                    f"{r['pAcc']:.4f}", f"{r['mAcc']:.4f}",
                    f"{r['mIoU']:.4f}", f"{r['fwIoU']:.4f}"
                ])
        print(f"Saved results to {path}")

    def summary(self):
        print("\n--- Summary of MoE mIoU degradation ---")
        # baseline clean mIoU
        clean_iou = next(r['mIoU']
                         for r in self.results if r['corruption'] == 'no_corruption')
        for r in self.results:
            if r['corruption'] == 'no_corruption':
                continue
            drop = clean_iou - r['mIoU']
            pct = (drop / clean_iou * 100) if clean_iou else 0.0
            print(
                f"{r['corruption']} Severity {r['severity_level']}: mIoU {r['mIoU']:.4f}, drop {drop:.4f} ({pct:.1f}%)")


if __name__ == '__main__':
    if len(sys.argv) != 2:
        print("Usage: python evaluate_deeplab_moe_severity_by_corruption.py <config.yaml>")
        sys.exit(1)
    with open(sys.argv[1], 'r') as f:
        params = yaml.safe_load(f)
    params.setdefault('gpu_ids', [0])
    warnings.filterwarnings('ignore', message="You are using `torch.load`")
    evaluator = MoESeverityEvaluator(params)
    evaluator.evaluate_all()
