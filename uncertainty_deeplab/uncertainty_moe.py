import csv
import os

from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import brier_score_loss, log_loss
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from utils.images_helpers import create_images
from utils.uncertainty_helpers import calculate_ece, calculate_mce, calculate_metrics, compute_mutual_information, get_confidences, get_predictive_entropy
from dataloader.a2d2_loader import a2d2Loader
from models import BEST_CHECKPOINT_EXPERT_HIGHWAY, BEST_CHECKPOINT_EXPERT_URBAN, BEST_CHECKPOINT_EXPERT_ALL, BEST_CHECKPOINT_MOE_CLASSWISE, BEST_CHECKPOINT_MOE_CLASSWISE_FINAL_CONV, BEST_CHECKPOINT_MOE_SIMPLE, BEST_CHECKPOINT_MOE_SIMPLE_FINAL_CONV,  DATASET_ROOT, FINAL_CONV, GATELAYERCLASSWISE, GATELAYERSIMPLE, IMG_HEIGHT, IMG_WIDTH, LINEAR_FEATURES, NUM_CLASSES, OUTPUT_STRIDE
from models.deeplab_moe import MoE

import seaborn as sns


class Severity0MoEEvaluator:
    """Evaluation Summarizer"""

    def __init__(self, model, modelname: str):
        self.device = torch.device('cuda', 0)

        img_size = (IMG_HEIGHT, IMG_WIDTH)
        root = os.path.join(DATASET_ROOT, "ambiguous")
        test_set = a2d2Loader(root, split='test', img_size=img_size)
        self.test_loader = DataLoader(test_set, batch_size=4,
                                      shuffle=False, num_workers=2, pin_memory=False)
        self.label_names = test_set.label_names

        self.model = model
        self.modelname = modelname

        self.metrics_uncertainty = []
        self.metrics_uncertainty_2 = []
        self.metrics_pe = []
        self.metrics_pe_2 = []
        self.metrics_mi = []
        self.metrics_mi_2 = []
        self.metrics_gate = []
        self.metrics_gate_2 = []

    def validation(self):
        """Validation"""
        print("Starting evaluation")
        self.model.eval()
        tbar = tqdm(self.test_loader, desc="\r")
        processed_images = 0

        for i, sample in enumerate(tbar):
            image, target, label_path = sample
            image = image.to(self.device)
            target = target.to(self.device)

            image, target = image.cuda(), target.cuda()

            starter_1, ender_1 = torch.cuda.Event(
                enable_timing=True), torch.cuda.Event(enable_timing=True)
            starter_1.record(torch.cuda.current_stream(self.device))

            expert_outputs = []
            with torch.no_grad():
                output, gate_values = self.model(image, return_gate=True)
                expert_outputs.append(self.model.expert1(image))
                expert_outputs.append(self.model.expert2(image))

            ender_1.record(torch.cuda.current_stream(self.device))
            torch.cuda.synchronize()
            inference_time = starter_1.elapsed_time(ender_1)

            starter_2, ender_2 = torch.cuda.Event(
                enable_timing=True), torch.cuda.Event(enable_timing=True)
            starter_2.record(torch.cuda.current_stream(self.device))

            B, C, H, W = output.shape
            pred = torch.argmax(output, dim=1)
            if isinstance(gate_values, list):
                gate_probs = torch.stack(gate_values, dim=1)
                pixel_gate_probs = gate_probs.mean(dim=1)
                pixel_gate_probs = pixel_gate_probs.unsqueeze(
                    1).unsqueeze(2).expand(-1, H, W, -1)
            else:
                if gate_values.dim() == 2:  # [B, num_experts]
                    pixel_gate_probs = gate_values.unsqueeze(
                        1).unsqueeze(2).expand(-1, H, W, -1)
                else:  # Already [B, H, W, num_experts]
                    pixel_gate_probs = gate_values

            gate_entropy = - \
                (pixel_gate_probs * torch.log(pixel_gate_probs + 1e-12)).sum(-1)
            max_ent_gate = torch.log(torch.tensor(float(pixel_gate_probs.size(-1)),
                                                  device=gate_entropy.device))
            scaled_gate_entropy = gate_entropy / max_ent_gate

            output_tensor = torch.stack(expert_outputs)
            softmax_probs = torch.softmax(output_tensor, dim=2)
            mean_softmax = torch.mean(softmax_probs, dim=0)
            predictive_entropy = get_predictive_entropy(
                mean_softmax=mean_softmax)
            mutual_information = compute_mutual_information(
                predictive_entropy=predictive_entropy,
                softmax_probs=softmax_probs,
            )

            ender_2.record(torch.cuda.current_stream(self.device))
            torch.cuda.synchronize()

            max_entropy = torch.log(torch.tensor(
                NUM_CLASSES, dtype=torch.float32))
            scaled_pe = predictive_entropy / max_entropy
            scaled_mi = mutual_information / max_entropy

            pe_conf = 1 - scaled_pe
            mi_conf = 1 - scaled_mi
            gate_conf = 1.0 - scaled_gate_entropy
            evu_conf = get_confidences(pred, output_tensor, image)

            batch_size = image.size(0)
            for j in range(batch_size):
                image_index = batch_size * i + j
                label = target[j]
                prediction = pred[j]
                curr_pe_conf = pe_conf[j]
                curr_mi_conf = mi_conf[j]
                curr_evu_conf = evu_conf[j]

                # Create mask for valid pixels (exclude 255)
                mask = label != 255
                masked_label = label[mask]
                masked_pred = prediction[mask]
                masked_pe_conf = curr_pe_conf[mask]
                masked_mi_conf = curr_mi_conf[mask]
                masked_evu_conf = curr_evu_conf[mask]

                # Ensure all confidence values are in [0,1]
                masked_pe_conf = torch.clamp(masked_pe_conf, 0, 1)
                masked_mi_conf = torch.clamp(masked_mi_conf, 0, 1)
                masked_evu_conf = torch.clamp(masked_evu_conf, 0, 1)
                masked_gate_conf = torch.clamp(gate_conf[j][mask], 0, 1)

                # if image_index in [45]:  # highway, ambiguous at index 0, urban at index 45
                #     # save raw input
                #     img_np = image[j].permute(1, 2, 0).cpu().numpy()
                #     img_np = (img_np - img_np.min()) / \
                #         (img_np.max() - img_np.min())
                #     os.makedirs("./images", exist_ok=True)
                #     plt.imsave(
                #         f"./images/{image_index}_moe_input_urban.png", img_np)

                #     t_np = label.cpu().numpy()
                #     p_np = prediction.cpu().numpy()

                #     # force 2D label
                #     current_label = torch.from_numpy(t_np)
                #     if current_label.dim() == 1:
                #         current_label = current_label.view(
                #             IMG_HEIGHT, IMG_WIDTH)

                #     pred_tensor = torch.from_numpy(p_np).to(self.device)
                #     label_tensor = torch.from_numpy(t_np).to(self.device)

                #     # predictive entropy
                #     create_images(
                #         pred=pred_tensor,
                #         label=label_tensor,
                #         uncertainty=scaled_pe[j],
                #         modelname=f"moe_urban_pe",
                #         image_index=image_index,
                #         dropout_rate=0.0,
                #         predictive_entropy=scaled_pe[j],
                #         mutual_information=None,
                #         image_path="./images/"
                #     )
                #     # mutual information
                #     create_images(
                #         pred=pred_tensor,
                #         label=label_tensor,
                #         uncertainty=scaled_mi[j],
                #         modelname=f"moe_urban_mi",
                #         image_index=image_index,
                #         dropout_rate=0.0,
                #         predictive_entropy=None,
                #         mutual_information=scaled_mi[j],
                #         image_path="./images/"
                #     )
                #     return

                for value, metrics_list in zip(
                    [masked_evu_conf, masked_pe_conf,
                        masked_mi_conf, masked_gate_conf],
                        [self.metrics_uncertainty, self.metrics_pe, self.metrics_mi, self.metrics_gate]):
                    accuracy_map = masked_pred == masked_label
                    inaccuracies = (~accuracy_map).sum().item()
                    mce = calculate_mce(value, masked_pred, masked_label)
                    ece = calculate_ece(value, masked_pred, masked_label)
                    brier = brier_score_loss(
                        accuracy_map.cpu().numpy().astype(float), value.cpu().numpy())
                    nll = log_loss(accuracy_map.cpu().numpy().astype(float),
                                   value.cpu().numpy())
                    metrics_list.append(
                        [image_index, ece, mce, brier, nll, inaccuracies])

                for value, metrics_list in zip(
                    [masked_evu_conf, masked_pe_conf,
                        masked_mi_conf, masked_gate_conf],
                        [self.metrics_uncertainty_2, self.metrics_pe_2, self.metrics_mi_2, self.metrics_gate_2]):
                    uncertainty_values = 1.0 - value
                    for uncertainty_threshold in range(1, 101, 2):
                        uncertainty_threshold /= 100
                        p_accurate_given_certain, p_uncertain_given_inaccurate, pavpu = calculate_metrics(
                            masked_label, masked_pred, uncertainty_values, uncertainty_threshold)
                        metrics_list.append(
                            [image_index, uncertainty_threshold, p_accurate_given_certain, p_uncertain_given_inaccurate, pavpu])

    def save_results(self):
        for metrics_list, metric_name in zip([self.metrics_uncertainty, self.metrics_pe, self.metrics_mi, self.metrics_gate], ["uncertainty", "pe", "mi", "gate"]):
            path = f"./results/{self.modelname}"
            if not os.path.exists(path):
                os.makedirs(path, exist_ok=True)
            filename_1 = f"{path}/{metric_name}_1_per_image.csv"
            with open(filename_1, 'w') as f:
                writer = csv.writer(f)
                writer.writerow(["image_index", "ece", "mce",
                                 "brier", "nll", "inaccuracies"])
                writer.writerows(metrics_list)
        for metrics_list, metric_name in zip([self.metrics_uncertainty_2, self.metrics_pe_2, self.metrics_mi_2, self.metrics_gate_2], ["uncertainty", "pe", "mi", "gate"]):
            filename_2 = f"{path}/{metric_name}_2_per_image.csv"
            with open(filename_2, 'w') as f:
                writer = csv.writer(f)
                writer.writerow(["image_index", "uncertainty_threshold",
                                "p_accurate_given_certain", "p_uncertain_given_inaccurate", "pavpu"])
                writer.writerows(metrics_list)
        df_uncertainty = pd.DataFrame(
            self.metrics_uncertainty,
            columns=["image_index", "ece", "mce",
                     "brier", "nll", "inaccuracies"]
        )
        avg_ece = df_uncertainty["ece"].mean()
        avg_mce = df_uncertainty["mce"].mean()
        avg_brier = df_uncertainty["brier"].mean()
        avg_nll = df_uncertainty["nll"].mean()
        print(
            f"UNCERTAINTY -> Avg ECE={avg_ece}, MCE={avg_mce}, Brier={avg_brier}, NLL={avg_nll}")
        df_pe = pd.DataFrame(
            self.metrics_pe,
            columns=["image_index", "ece", "mce",
                     "brier", "nll", "inaccuracies"]
        )
        avg_ece = df_pe["ece"].mean()
        avg_mce = df_pe["mce"].mean()
        avg_brier = df_pe["brier"].mean()
        avg_nll = df_pe["nll"].mean()
        print(
            f"PE -> Avg ECE={avg_ece}, MCE={avg_mce}, Brier={avg_brier}, NLL={avg_nll}")
        df_mi = pd.DataFrame(
            self.metrics_mi,
            columns=["image_index", "ece", "mce",
                     "brier", "nll", "inaccuracies"]
        )
        avg_ece = df_mi["ece"].mean()
        avg_mce = df_mi["mce"].mean()
        avg_brier = df_mi["brier"].mean()
        avg_nll = df_mi["nll"].mean()
        print(
            f"MI -> Avg ECE={avg_ece}, MCE={avg_mce}, Brier={avg_brier}, NLL={avg_nll}")
        df_gate = pd.DataFrame(
            self.metrics_gate,
            columns=["image_index", "ece", "mce",
                     "brier", "nll", "inaccuracies"]
        )
        print(f"GATE -> Avg ECE={df_gate.ece.mean():.4f}, MCE={df_gate.mce.mean():.4f}, "
              f"Brier={df_gate.brier.mean():.4f}, NLL={df_gate.nll.mean():.4f}")


if __name__ == "__main__":
    device = torch.device('cuda', 0)
    models_to_test = [
        # {"name": "moe_simple_all",
        #  "model": MoE(
        #      gate_type="simple",
        #      with_conv=False,
        #      arch="deeplabv3plus",
        #      backbone="resnet101",
        #      output_stride=OUTPUT_STRIDE,
        #      num_classes=NUM_CLASSES,
        #      linear_features=LINEAR_FEATURES,
        #      checkpoint1=BEST_CHECKPOINT_EXPERT_HIGHWAY,
        #      checkpoint2=BEST_CHECKPOINT_EXPERT_URBAN,
        #  ),
        #  "checkpoint": BEST_CHECKPOINT_MOE_SIMPLE
        #  },
        # {
        #     "name": "moe_simple_final_conv_ambiguous",
        #     "model": MoE(
        #         gate_type="simple",
        #         with_conv=True,
        #         arch="deeplabv3plus",
        #         backbone="resnet101",
        #         output_stride=OUTPUT_STRIDE,
        #         num_classes=NUM_CLASSES,
        #         linear_features=LINEAR_FEATURES,
        #         checkpoint1=BEST_CHECKPOINT_EXPERT_HIGHWAY,
        #         checkpoint2=BEST_CHECKPOINT_EXPERT_URBAN,
        #     ),
        #     "checkpoint": BEST_CHECKPOINT_MOE_SIMPLE_FINAL_CONV
        # },
        # {
        #     "name": "moe_classwise_ambiguous",
        #     "model": MoE(
        #         gate_type="classwise",
        #         with_conv=False,
        #         arch="deeplabv3plus",
        #         backbone="resnet101",
        #         output_stride=OUTPUT_STRIDE,
        #         num_classes=NUM_CLASSES,
        #         linear_features=LINEAR_FEATURES,
        #         checkpoint1=BEST_CHECKPOINT_EXPERT_HIGHWAY,
        #         checkpoint2=BEST_CHECKPOINT_EXPERT_URBAN,
        #     ),
        #     "checkpoint": BEST_CHECKPOINT_MOE_CLASSWISE
        # },
        {
            "name": "moe_classwise_final_conv_ambiguous",
            "model": MoE(
                gate_type="classwise",
                with_conv=True,
                arch="deeplabv3plus",
                backbone="resnet101",
                output_stride=OUTPUT_STRIDE,
                num_classes=NUM_CLASSES,
                linear_features=LINEAR_FEATURES,
                checkpoint1=BEST_CHECKPOINT_EXPERT_HIGHWAY,
                checkpoint2=BEST_CHECKPOINT_EXPERT_URBAN,
            ),
            "checkpoint": BEST_CHECKPOINT_MOE_CLASSWISE_FINAL_CONV
        },
    ]

    device = torch.device('cuda', 0)

    for model_info in models_to_test:
        checkpoint_path = model_info["checkpoint"]
        modelname = model_info["name"]
        model = model_info["model"]
        if not os.path.isfile(checkpoint_path):
            raise RuntimeError(
                f"=> no checkpoint found at '{checkpoint_path}'")

        print(f"Loading checkpoint from {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path)
        model.load_state_dict(checkpoint["state_dict"])

        model = model.to(device)

        model.expert1.to(device)
        model.expert2.to(device)

        evaluator = Severity0MoEEvaluator(model, modelname)
        evaluator.validation()
        evaluator.save_results()
