"""
DeepLab Ensemble
"""
import os
import torch
import torch.nn as nn
import torch.nn.functional as F

from models.deeplab_modeling import _load_model

class Ensemble(nn.Module):
    """Implement mixture of experts model"""
    #pylint: disable=too-many-arguments
    def __init__(
        self,
        arch="deeplabv3",
        backbone="resnet",
        output_stride=16,
        num_classes=21,
        checkpoint1=None,
        checkpoint2=None,
        ens_type="max",
        allow_gradient_flow=False
    ):
        super().__init__()
        self.allow_gradient_flow = allow_gradient_flow
        self.ens_type = ens_type
        self.num_classes = num_classes
        
        print("Initializing", self.ens_type, "ensemble")
        self.expert1 = _load_model(arch, backbone, self.num_classes, output_stride=output_stride, pretrained_backbone=True, input_channels=3)
        if not os.path.isfile(checkpoint1):
            raise RuntimeError(
                "=> No checkpoint found at '{}'".format(checkpoint1))
        checkpoint1_loaded = torch.load(checkpoint1)
        self.expert1.load_state_dict(checkpoint1_loaded["state_dict"])
        
        self.expert2 = _load_model(arch, backbone, self.num_classes, output_stride=output_stride, pretrained_backbone=True, input_channels=3)
        if not os.path.isfile(checkpoint2):
            raise RuntimeError(
                "=> No checkpoint found at '{}'".format(checkpoint2))
        checkpoint2_loaded = torch.load(checkpoint2)
        self.expert2.load_state_dict(checkpoint2_loaded["state_dict"])
        
        # --- freeze the expert model
        for param in self.expert1.parameters():
            param.requires_grad = False
        self.expert1.eval()
        for param in self.expert2.parameters():
            param.requires_grad = False
        self.expert2.eval()
        

    def forward(self, x_in):
        output_expert1 = self.expert1(x_in)
        output_expert2 = self.expert2(x_in)
        if self.ens_type=="max":
            output = torch.amax(torch.stack((output_expert1, output_expert2)), axis=0)
        elif self.ens_type=="mean":
            output = torch.mean(torch.stack((output_expert1, output_expert2)), axis=0)
        return output

    def get_train_params(self):
        params = list(self.parameters())
        return params

