"""
Mixture of experts
"""
import torch
import torch.nn as nn
from models.frr_module import FRRN_PARAMS
from models.frrn import FRRNet

IN_FEATURES = 1200


class Ensemble(nn.Module):
    """Implementation ensemble"""
    def __init__(self, in_channels=3, out_channels=11,
                 checkpoints=[], ensemble_type="max"):
        super(Ensemble, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.checkpoints = checkpoints
        self.ens_type = ensemble_type
        print("Initiating a", self.ens_type, "ensemble")
        
        self.expert1 = FRRNet(in_channels, out_channels)
        self.expert2 = FRRNet(in_channels, out_channels)
        
        ckpt1 = torch.load(self.checkpoints[0])
        self.expert1.load_state_dict(ckpt1['state_dict'])
        for param in self.expert1.parameters():
            param.requires_grad = False
        self.expert1.eval()
        
        ckpt2 = torch.load(self.checkpoints[1])
        self.expert2.load_state_dict(ckpt2['state_dict'])
        for param in self.expert2.parameters():
            param.requires_grad = False
        self.expert2.eval()

        self.softmax = nn.Softmax(dim=1)


    def forward(self, x):
        out_1 = self.expert1(x)
        out_2 = self.expert2(x)
        if self.ens_type=="mean":
            out = torch.mean(torch.stack((out_1, out_2)), axis=0)
        elif self.ens_type=="max":
            out = torch.amax(torch.stack((out_1, out_2)), axis=0)
        return out

    def _get_train_params(self):
        params = list(self.parameters())
        return params


if __name__ == '__main__':
    checkpoints = ['../run/frrnA_highway/model_best.pth',
                   '../run/frrnA_urban/model_best.pth']
    inp = torch.rand((1, 3, 480, 640))
    net = Ensemble(checkpoints=checkpoints, gate='simple').cuda()
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
