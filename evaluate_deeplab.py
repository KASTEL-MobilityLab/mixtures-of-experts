import argparse
import os
from collections import OrderedDict

import numpy as np
import torch
from tqdm import tqdm
import sys
import yaml

from dataloader.a2d2_loader import get_dataloader
from models.deeplab_moe import MoE
from models.deeplab_modeling import _load_model
from models.deeplab_ensemble import Ensemble
from utils.metrics import Evaluator
from utils.train_utils import colorize

class EvaluationSummarizer:
    """Evaluation Summarizer"""
    # pylint: disable=too-many-branches
    def __init__(self, params):
        self.params = params
        self.is_cuda = len(params["gpu_ids"]) > 0
        self.device = torch.device('cuda', params["gpu_ids"][0]) \
            if self.is_cuda else torch.device('cpu')

        self.nclass = params['DATASET']['num_class']
        self.datasets = list(s for s in params["DATASET"]["dataset"].split(','))
        self.test_loader, self.label_names, self.label_colors = get_dataloader(params, params['test_sets'], 'test')

        self.evaluator = Evaluator(self.nclass)

        if params["MODEL"]["expert"] == "moe":
            linear_feat = (params["DATASET"]["img_height"] // params["MODEL"]["out_stride"] + 1)**2
            linear_feat = 3249
            print("Linear features", linear_feat)
            self.model = MoE(
                arch=params["MODEL"]["arch"],
                backbone=params["MODEL"]["backbone"],
                output_stride=params["MODEL"]["out_stride"],
                num_classes=self.nclass,
                linear_features=linear_feat,
                checkpoint1=params["MODEL"]["checkpoint_moe_expert_1"], 
                checkpoint2=params["MODEL"]["checkpoint_moe_expert_2"],
                gate_type=params["MODEL"]["gate"],
                with_conv=params["MODEL"]["with_conv"],
                allow_gradient_flow = False
            )
            
            if params["TEST"]["checkpoint"] is not None:
                if not os.path.isfile(params["TEST"]["checkpoint"]):
                    raise RuntimeError("=> no checkpoint found at '{}'".format(
                        params["TEST"]["checkpoint"]))

                print("Loading checkpoint from", params["TEST"]["checkpoint"])
                checkpoint = torch.load(params["TEST"]["checkpoint"])
                params["start_epoch"] = checkpoint["epoch"]

                self.model.load_state_dict(checkpoint["state_dict"])
                print("=> loaded checkpoint '{}' (epoch {})".format(
                    params["TEST"]["checkpoint"], checkpoint["epoch"]))
                
                if self.is_cuda:
#                     self.model = torch.nn.DataParallel(self.model, device_ids=params["gpu_ids"])
                    self.model = self.model.to(self.device)
                    self.model.expert1.to(self.device)
                    self.model.expert2.to(self.device)

            else:
                raise RuntimeError("=> no checkpoint in input arguments")
            
        elif params["MODEL"]["expert"] == "ensemble":
            self.model = Ensemble(arch=params["MODEL"]["arch"], 
                                  backbone=params["MODEL"]["backbone"], 
                                  output_stride=self.params["MODEL"]["out_stride"],     
                                  num_classes=self.nclass, 
                                  checkpoint1=params["TEST"]["checkpoint_moe_expert_1"], 
                                  checkpoint2=params["TEST"]["checkpoint_moe_expert_2"], 
                                  ens_type=self.params["MODEL"]["ens_type"]
                                 )
            self.model.expert1.to(self.device)
            self.model.expert2.to(self.device)
        else:
            self.model = _load_model(params["MODEL"]["arch"], params["MODEL"]["backbone"], self.nclass, output_stride=self.params["MODEL"]["out_stride"], pretrained_backbone=True, input_channels=3)
            
            if params["TEST"]["checkpoint"] is not None:
                if not os.path.isfile(params["TEST"]["checkpoint"]):
                    raise RuntimeError("=> no checkpoint found at '{}'".format(
                        params["TEST"]["checkpoint"]))

                print("Loading checkpoint from", params["TEST"]["checkpoint"])
                checkpoint = torch.load(params["TEST"]["checkpoint"])
                params["start_epoch"] = checkpoint["epoch"]

                self.model.load_state_dict(checkpoint["state_dict"])
                print("=> loaded checkpoint '{}' (epoch {})".format(
                    params["TEST"]["checkpoint"], checkpoint["epoch"]))
                
                if self.is_cuda:
                    self.model = torch.nn.DataParallel(self.model, device_ids=params["gpu_ids"])
                    self.model = self.model.to(self.device)

            else:
                raise RuntimeError("=> no checkpoint in input arguments")

    def validation(self):
        """Validation"""
        print("Starting evaluation")
        self.model.eval()
        self.evaluator.reset()
        tbar = tqdm(self.test_loader, desc="\r")
        
        if 'delta_path' in params:
            delta = torch.load(params['delta_path'])
            print("Loading noise from", params['delta_path'])
            delta = delta.cpu()

        for i, sample in enumerate(tbar):
            image, target, label_path = sample
            image = image.to(self.device)
            target = target.to(self.device)
            
            if 'delta_path' in params:
                image = image + delta
            
            if self.is_cuda:
                image, target = image.cuda(), target.cuda()
            with torch.no_grad():
                output = self.model(image)
    
            pred = output.data.cpu().numpy()
            target = target.cpu().numpy()
            pred = np.argmax(pred, axis=1)
            self.evaluator.add_batch(target, pred)

        acc = self.evaluator.pixel_accuracy()
        acc_class = self.evaluator.pixel_accuracy_class()
        m_iou = self.evaluator.mean_intersection_over_union(self.label_names)
        fw_iou = self.evaluator.frequency_weighted_intersection_over_union()

        print("Validation:")
        print("pAcc:{}, mAcc:{}, m_iou:{}, fwIoU: {}".format(
            acc, acc_class, m_iou, fw_iou))


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print('\nPlease pass the desired param file for training as an argument.\n'
              'e.g: params/params_moe.py')
    else:
        print('STARTING EVALUATION WITH PARAM FILE: ', str(sys.argv[1]))
        with open(str(sys.argv[1]), 'r') as stream:
            try:
                params = yaml.safe_load(stream)
                params["gpu_ids"] = [0]
                evaluator = EvaluationSummarizer(params)
                evaluator.validation()

            except yaml.YAMLError as exc:
                print(exc)
