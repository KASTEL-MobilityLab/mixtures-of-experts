"""
Mixture of experts model implementation.
Freezing experts after their checkpoints loaded.
Only the gate layer will be trained.
"""
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from models.deeplab_modeling import _load_model
from models.gate import make_gate_layer

class MoE(nn.Module):
    """Implement mixture of experts model"""
    #pylint: disable=too-many-arguments
    def __init__(
        self,
        arch="deeplabv3",
        backbone="resnet",
        output_stride=16,
        num_classes=21,
        linear_features=1200,
        checkpoint1=None,
        checkpoint2=None,
        gate_type="simple",
        with_conv=False,
        allow_gradient_flow=False
    ):
        super().__init__()
        self.with_conv = with_conv
        self.allow_gradient_flow = allow_gradient_flow
        self.num_classes = num_classes
        self.arch = arch
        
        self.expert1 = _load_model(arch, backbone, self.num_classes, output_stride=output_stride, pretrained_backbone=True, input_channels=3)
        if not os.path.isfile(checkpoint1):
            raise RuntimeError(
                "=> No checkpoint found at '{}'".format(checkpoint1))
        print("Loading expert1 from", checkpoint1)
        checkpoint1_loaded = torch.load(checkpoint1)
        self.expert1.load_state_dict(checkpoint1_loaded["state_dict"])
        
        self.expert2 = _load_model(arch, backbone, self.num_classes, output_stride=output_stride, pretrained_backbone=True, input_channels=3)
        if not os.path.isfile(checkpoint2):
            raise RuntimeError(
                "=> No checkpoint found at '{}'".format(checkpoint2))
        print("Loading expert2 from", checkpoint2)
        checkpoint2_loaded = torch.load(checkpoint2)
        self.expert2.load_state_dict(checkpoint2_loaded["state_dict"])
        
        # --- freeze the expert model
        for param in self.expert1.parameters():
            param.requires_grad = False
        self.expert1.eval()
        for param in self.expert2.parameters():
            param.requires_grad = False
        self.expert2.eval()
        
        self.gate = make_gate_layer(
            gate_type=gate_type,
            in_channels=256 * 2,  # channel number after aspp, concate
            in_features=linear_features,
            num_experts=2,
            num_cls=num_classes,
        )
        self.final_conv = (nn.Conv2d(
            num_classes, num_classes, 3, 1, 1, bias=False)
                           if with_conv else None)

    def forward_expert(self, expert, x):
        """expert model forward function, if model.train()
            then load checkpoint from experts"""
        
        # let gradient flow for adversarial attacks
        if not self.allow_gradient_flow:
            with torch.no_grad():
                input_shape = x.shape[-2:]
                features = expert.backbone(x)
                encoder_output = expert.classifier.aspp(features['out'])
                x = expert.classifier(features)
                x_out = F.interpolate(x, size=input_shape, mode='bilinear', align_corners=False)
        else:
            input_shape = x.shape[-2:]
            features = expert.backbone(x)
            encoder_output = expert.classifier.aspp(features['out'])
            x = expert.classifier(features)
            x_out = F.interpolate(x, size=input_shape, mode='bilinear', align_corners=False)

        return encoder_output, x_out
    
    
    def forward(self, x_in):
        x_encoder_1, out_1 = self.forward_expert(self.expert1, x_in)
        x_encoder_2, out_2 = self.forward_expert(self.expert2, x_in)
        # get weights, shape [batch_size, num_experts]
        weights = self.gate(x_encoder_1, x_encoder_2)
        if isinstance(weights, list):
            # class-wise gate
            weight1 = torch.cat([w[:, 0].reshape(-1, 1, 1, 1) for w in weights],
                           dim=1)
            weight2 = torch.cat([w[:, 1].reshape(-1, 1, 1, 1) for w in weights],
                           dim=1)
            out_1 *= weight1
            out_2 *= weight2
        else:
            out_1 *= weights[:, 0].reshape(-1, 1, 1, 1)
            out_2 *= weights[:, 1].reshape(-1, 1, 1, 1)

        if self.with_conv:
            out = self.final_conv.forward(out_1 + out_2)
        else:
            out = out_1 + out_2
        return out

    def get_train_params(self):
        """return trainable parameters"""
        params = list(self.gate.parameters())
        if self.with_conv:
            params += list(self.final_conv.parameters())
        return params

