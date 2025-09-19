"""Evaluation"""
import argparse
import os
from PIL import Image

import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image
from tqdm import tqdm
import sys
import yaml

from dataloader.dl_tools import make_data_loader_trainval, make_data_loader_test, decode_segmap
from models.deeplab_model_selection import get_model
from models.moe_shared_encoder import MoeWithSharedEncoder
from models.deeplab_shared_encoder import ExpertWithSharedEncoder
from utils.loss import SegmentationLosses
from utils.metrics import Evaluator

STORE_TO_CSV = True
DATA_PATH = "/disk/vanishing_data/pavlitsk/disagreements/deeplab_ambiguous/"

class EvaluationSummarizer:
    """Evaluation Summarizer"""
    # pylint: disable=too-many-branches
    def __init__(self, params):
        self.params = params

        # Define Dataloader
        kwargs = {"num_workers": 1, "pin_memory": True}

        (
            self.test_loader,
            self.nclass,
        ) = make_data_loader_test(params, **kwargs)

        _, self.model = get_model(params, num_classes=self.nclass)

        if isinstance(self.model, MoeWithSharedEncoder) or isinstance(self.model, ExpertWithSharedEncoder):
            # Default: 1 is urban, 2 is highway
            print("Decoder: ", self.params["decoder_for_eval"])
            self.decoder = self.params["decoder_for_eval"]
        else:
            self.decoder = None

        if params["cuda"]:
            self.model = torch.nn.DataParallel(self.model,
                                               device_ids=self.params["gpu_ids"])
            self.model = self.model.cuda()

        self.criterion = SegmentationLosses(
            weight=None, cuda=params["cuda"]).build_loss(mode=params["loss_type"])

        self.evaluator = Evaluator(self.nclass)

        # Resuming checkpoint
        if params["checkpoint"] is not None:
            if not os.path.isfile(params["checkpoint"]):
                raise RuntimeError("=> no checkpoint found at '{}'".format(
                    params["checkpoint"]))

            print("Loading checkpoint from", params["checkpoint"])
            checkpoint = torch.load(params["checkpoint"])
            params["start_epoch"] = checkpoint["epoch"]

            if params["cuda"]:
                self.model.module.load_state_dict(checkpoint["state_dict"])
            else:
                self.model.load_state_dict(checkpoint["state_dict"])
            print("=> loaded checkpoint '{}' (epoch {})".format(
                params["checkpoint"], checkpoint["epoch"]))

        else:
            raise RuntimeError("=> no checkpoint in input arguments")

    def validation(self):
        """Validation"""
        print("Starting evaluation")
        self.model.eval()
        self.evaluator.reset()
        # tbar = tqdm(self.val_loader, desc='\r')
        tbar = tqdm(self.test_loader, desc="\r")

        for i, sample in enumerate(tbar):
            image, target = sample["image"], sample["label"]
            if self.params["cuda"]:
                image, target = image.cuda(), target.cuda()
            with torch.no_grad():
                if self.decoder:
                    output = self.model(image, self.decoder)
                else:
                    output = self.model(image)

            target = target.cpu().numpy()
            target = np.array(target).astype(np.uint8)
            convert_to_img_and_save(target, i, "_gt.png")

            # moe prediction
            pred = output.data.cpu().numpy()
            moe_argmax = np.argmax(pred, axis=1).astype(np.uint8)
            convert_to_img_and_save(moe_argmax, i, "_moe.png")

            # first expert prediction

            output_expert1 = self.model.module.expert_1(image)
            output_expert1 = output_expert1.data.cpu().numpy()
            y1_argmax = np.argmax(output_expert1, axis=1).astype(np.uint8)
            convert_to_img_and_save(y1_argmax, i, "_expert1.png")

            output_expert2 = self.model.module.expert_2(image)
            output_expert2 = output_expert2.data.cpu().numpy()
            y2_argmax = np.argmax(output_expert2, axis=1).astype(np.uint8)
            convert_to_img_and_save(y2_argmax, i, "_expert2.png")


            expert_agree = np.equal(y1_argmax[0], y2_argmax[0])
            moe_agrees_with_expert_1 = np.equal(y1_argmax[0], moe_argmax[0])
            moe_agrees_with_expert_2 = np.equal(y2_argmax[0], moe_argmax[0])

            # both experts and moe agree
            perfect_cases_mask = np.logical_and(np.equal(y1_argmax[0], moe_argmax[0]),
                                                np.equal(y2_argmax[0], moe_argmax[0]),
                                                expert_agree)
            # pixels where moe chooses the same as expert 1
            normal_case_1_mask = np.logical_and(moe_agrees_with_expert_1, ~expert_agree)

            # pixels where moe chooses the same as expert 2
            normal_case_2_mask = np.logical_and(moe_agrees_with_expert_2, ~expert_agree)

            # pixels where moe chose differently from any of the experts
            critical_cases_mask = np.logical_and(~moe_agrees_with_expert_1, ~moe_agrees_with_expert_2)

            if STORE_TO_CSV:
                str_output = str(np.sum(perfect_cases_mask)) + "," + str(np.sum(normal_case_1_mask)) + "," + str(np.sum(normal_case_2_mask)) + "," + str(np.sum(critical_cases_mask))
                print(str_output)
                csvfile.write(str(i) + "," + str_output)
                csvfile.write('\n')

            red_ch = critical_cases_mask * 255
            green_ch = normal_case_1_mask * 255  # green - moe follows the first expert
            blue_ch = normal_case_2_mask * 255  # blue - moe follows the second expert
            alpha_ch = ~perfect_cases_mask * 100  # perfect case pixels are transparent, all others are half-transparent
            discrepancy_img = np.dstack((red_ch, green_ch, blue_ch, alpha_ch)).astype(np.uint8)

            #orig_img = F.interpolate(orig_img, size=(1920, 1208), mode="bilinear", align_corners=True)
            orig_img = np.squeeze(np.array(image.cpu().numpy()), axis = 0)
            orig_img = np.transpose(orig_img, axes=[1, 2, 0])
            # undo normalization
            orig_img *= (0.229, 0.224, 0.225)
            orig_img += (0.485, 0.456, 0.406)
            orig_img = np.uint8(orig_img*255)
            plt.imsave(DATA_PATH + str(i) + "_input.png", orig_img)
            orig_img_pil = Image.fromarray(orig_img)
            orig_img_pil = orig_img_pil.convert('RGBA')

            overlayed_img = Image.alpha_composite(orig_img_pil, Image.fromarray(np.uint8(discrepancy_img)))
            overlayed_img.save(DATA_PATH + str(i) + "_disagreements.png", "PNG")
            plt.imsave(DATA_PATH + str(i) + "_input.png", orig_img)


def convert_to_img_and_save(pred, i, file_suffix):
    pred = np.squeeze(pred, axis = 0)
    segmap = decode_segmap(pred, dataset="audi_split")
    plt.imsave(DATA_PATH + str(i) + "_" + file_suffix + ".png", segmap)

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print('\nPlease pass the desired param file for training as an argument.\n'
              'e.g: params/params_moe.py')
    else:
        print('STARTING EVALUATION WITH PARAM FILE: ', str(sys.argv[1]))
        with open(str(sys.argv[1]), 'r') as stream:
            try:
                params = yaml.safe_load(stream)
                if torch.cuda.is_available():
                    try:
                        params["gpu_ids"] = [int(s) for s in params["gpu_ids"].split(",")]
                    except ValueError as value_error:
                        raise ValueError(
                            "Argument --gpu_ids must be a\
                            comma-separated list of integers only"
                        ) from value_error

                params["checkname"] = "-".join(["deeplab", str(params["backbone"]), str(params["model"])])
                if "moe" in params["model"]:
                    params["checkname"] = "-".join([params["checkname"], params["gate_type"]])
                    gate_type = "with_conv" if params["with_conv"] else "without_conv"
                    params["checkname"] = "-".join([params["checkname"], gate_type])

                torch.manual_seed(1)

                for testset in params["test_datasets"]:
                    if STORE_TO_CSV:
                        csv_name = "disagreements_of_" + params['checkname'] + "_on_data_" + testset + ".csv"
                        csv_name = os.path.join("./", csv_name)
                        csvfile = open(csv_name, 'w')
                        csvfile.write("id_id,perfect_case,moe_agrees_with_expert_1,moe_agrees_with_expert_2,critical_case")
                        csvfile.write('\n')
                    print("Start evaluation for dataset ", testset)
                    params["test_dataset"] = testset
                    evaluator = EvaluationSummarizer(params)
                    evaluator.validation()

            except yaml.YAMLError as exc:
                print(exc)

