"""DeeplabV3+ with shared encoder"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from modeling.sync_batchnorm.batchnorm import (
    SynchronizedBatchNorm2d,
)
from modeling.aspp import build_aspp
from modeling.decoder import build_decoder
from modeling.backbone import build_backbone
from modeling.deeplab import DeepLab


class MoeWithSharedEncoder(DeepLab):
    """DeepLabV3+ with shared encoder"""

    def __init__(
        self,
        backbone="resnet",
        output_stride=16,
        num_classes=21,
        sync_bn=True,
        freeze_bn=False,
    ):
        super().__init__()
        if backbone == "drn":
            output_stride = 8

        if sync_bn:
            BatchNorm = SynchronizedBatchNorm2d
        else:
            BatchNorm = nn.BatchNorm2d

        self.backbone = build_backbone(backbone, output_stride, BatchNorm)
        self.aspp = build_aspp(backbone, output_stride, BatchNorm)

        self.decoder_1 = build_decoder(num_classes, backbone, BatchNorm)
        self.decoder_2 = build_decoder(num_classes, backbone, BatchNorm)

        self.freeze_bn = freeze_bn

    def forward(self, input_x, decoder):
        out, low_level_feat = self.backbone(input_x)
        out = self.aspp(out)
        if decoder == 1:
            out = self.decoder_1(out, low_level_feat)
        elif decoder == 2:
            out = self.decoder_2(out, low_level_feat)
        else:
            raise RuntimeError(
                "Decoder index must be 1 or 2 but {} is given.".format(decoder)
            )
        out = F.interpolate(
            out, size=input_x.size()[2:], mode="bilinear", align_corners=True
        )

        return out

    def freeze_bn(self):
        for module in self.modules():
            if isinstance(module, SynchronizedBatchNorm2d):
                module.eval()
            elif isinstance(module, nn.BatchNorm2d):
                module.eval()

    def get_lr_params(self, modules):
        """Get learning rate params"""
        # pylint: disable=too-many-nested-blocks
        for mod in modules:
            for module in mod.named_modules():
                if self.freeze_bn:
                    if isinstance(module[1], nn.Conv2d):
                        for param in module[1].parameters():
                            if param.requires_grad:
                                yield param
                else:
                    if isinstance(
                        module[1],
                        (SynchronizedBatchNorm2d, nn.BatchNorm2d, nn.Conv2d),
                    ):
                        for param in module[1].parameters():
                            if param.requires_grad:
                                yield param

    def get_1x_lr_params(self):
        modules = [self.backbone]
        return self.get_lr_params(modules)

    def get_10x_lr_params(self):
        modules = [self.aspp, self.decoder_1, self.decoder_2]
        return self.get_lr_params(modules)


if __name__ == "__main__":
    model = MoeWithSharedEncoder(backbone="mobilenet", output_stride=16)
    model.eval()
    inp = torch.rand(1, 3, 513, 513)
    output = model(inp)
    print(output.size())
