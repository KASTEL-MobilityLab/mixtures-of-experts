"""
FRRN pytorch implementation
"""
from collections import OrderedDict
import torch
import torch.nn as nn
from models.frr_module import BasicBlock, FRRLayer, FRRN_PARAMS


class FRRNet(nn.Module):
    """Implementation table A of Full-Resolution Residual Networks """
    def __init__(self, in_channels=3, out_channels=19, table='A', efficient=True):
        super(FRRNet, self).__init__()
        pool_ch = FRRN_PARAMS[table]['pool_stream_channel']
        fuse_ch = FRRN_PARAMS[table]['pool_stream_channel']
        res_ch = FRRN_PARAMS[table]['res_stream_channel']
        encoder_setup = FRRN_PARAMS[table]['encoder']
        decoder_setup = FRRN_PARAMS[table]['decoder']
        # initial convolution layer
        self.first = nn.Sequential(
            OrderedDict([
                ('conv', nn.Conv2d(in_channels=in_channels, out_channels=pool_ch,
                                   kernel_size=5, padding=2)),
                ('bn', nn.BatchNorm2d(pool_ch)),
                ('relu', nn.ReLU()),
                ]))
        self.pool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.up = nn.UpsamplingBilinear2d(scale_factor=2)
        self.reslayers_in = nn.Sequential(*[BasicBlock(pool_ch, pool_ch, efficient=efficient)
                                            for _ in range(3)])
        self.divide = nn.Conv2d(in_channels=pool_ch, out_channels=res_ch, kernel_size=1)

        self.frrnlayer1 = FRRLayer(pool_ch, encoder_setup[0], res_ch, efficient)
        pool_ch = encoder_setup[0][1]
        self.frrnlayer2 = FRRLayer(pool_ch, encoder_setup[1], res_ch, efficient)
        pool_ch = encoder_setup[1][1]
        self.frrnlayer3 = FRRLayer(pool_ch, encoder_setup[2], res_ch, efficient)
        pool_ch = encoder_setup[2][1]
        self.frrnlayer4 = FRRLayer(pool_ch, encoder_setup[3], res_ch, efficient)
        pool_ch = encoder_setup[3][1]

        self.defrrnlayer1 = FRRLayer(pool_ch, decoder_setup[0], res_ch, efficient)
        pool_ch = decoder_setup[0][1]
        self.defrrnlayer2 = FRRLayer(pool_ch, decoder_setup[1], res_ch, efficient)
        pool_ch = decoder_setup[1][1]
        self.defrrnlayer3 = FRRLayer(pool_ch, decoder_setup[2], res_ch, efficient)
        pool_ch = decoder_setup[2][1]

        self.compress = nn.Conv2d(res_ch+pool_ch, fuse_ch, kernel_size=1)
        self.reslayers_out = nn.Sequential(*[BasicBlock(fuse_ch, fuse_ch, efficient=efficient)
                                             for _ in range(3)])
        self.out_conv = nn.Conv2d(fuse_ch, out_channels, 1)

    def forward(self, x_in):
        """forward function"""
        x_in = self.first(x_in)
        y_pool_s = self.reslayers_in(x_in)
        z_res_s = self.divide(y_pool_s)     # z for res stream
        y_pool_s = self.pool(y_pool_s)      # y for pooling stream
        y_pool_s, z_res_s = self.frrnlayer1(y_pool_s, z_res_s)
        y_pool_s = self.pool(y_pool_s)
        y_pool_s, z_res_s = self.frrnlayer2(y_pool_s, z_res_s)
        y_pool_s = self.pool(y_pool_s)
        y_pool_s, z_res_s = self.frrnlayer3(y_pool_s, z_res_s)
        y_pool_s = self.pool(y_pool_s)
        y_pool_s, z_res_s = self.frrnlayer4(y_pool_s, z_res_s)

        y_pool_s = self.up(y_pool_s)
        y_pool_s, z_res_s = self.defrrnlayer1(y_pool_s, z_res_s)
        y_pool_s = self.up(y_pool_s)
        y_pool_s, z_res_s = self.defrrnlayer2(y_pool_s, z_res_s)
        y_pool_s = self.up(y_pool_s)
        y_pool_s, z_res_s = self.defrrnlayer3(y_pool_s, z_res_s)
        y_pool_s = self.up(y_pool_s)
        refine = self.compress(torch.cat((y_pool_s, z_res_s), 1))
        out = self.reslayers_out(refine)
        out = self.out_conv(out)
        return out


if __name__ == '__main__':
    inp = torch.rand((1, 3, 480, 640))
    net = FRRNet().cuda()
    net = torch.nn.DataParallel(net, device_ids=[0])
    # out = net(inp)
    # print(out.size())
    import os
    from utils.saver import summary
    modelfile = os.path.join('model_functionsummary.txt')
    with open(modelfile, 'w') as f:
        f.write(str(net))
    modelfile = os.path.join('model_torchsummary.txt')
    with open(modelfile, 'w') as f:
        summary(net, (3, 48, 64), f)
