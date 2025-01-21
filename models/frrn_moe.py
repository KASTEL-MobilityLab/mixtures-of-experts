"""
Mixture of experts
"""
import torch
import torch.nn as nn
from models.frr_module import FRRN_PARAMS
from models.frrn import FRRNet

IN_FEATURES = 1200

class GateLayer_simple(nn.Module):
    """simple gated layer"""
    def __init__(self, in_features=IN_FEATURES, num_experts=2):
        super(GateLayer_simple, self).__init__()
        in_channels = FRRN_PARAMS['A']['encoder'][-1][1]
        self.in_features = in_features
        self.conv1 = nn.Conv2d(in_channels*2, 1, kernel_size=3, stride=1, padding=1)
        self.relu = nn.ReLU()
        self.fc1 = nn.Linear(in_features, 128)
        self.fc2 = nn.Linear(128, num_experts)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x1, x2):
        """input features, output gate"""
        # pylint: disable-msg=invalid-name
        x = torch.cat((x1, x2), dim=1)
        bs = x.shape[0]
        x = self.conv1(x)
        x = self.relu(x)
        x = x.view(bs, -1)
        assert x.shape[1] == self.in_features, "input image resolution muss be 480x640."
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.softmax(x)
        return x


class GateLayer_classwise(nn.Module):
    """classwise gated layer"""
    def __init__(self, in_features=IN_FEATURES, num_experts=2, num_cls=11):
        # output gate shape [bs, num_experts, 1, 1]
        super(GateLayer_classwise, self).__init__()
        self.gates = nn.ModuleList([GateLayer_simple(in_features, num_experts)
                                    for _ in range(num_cls)])

    def forward(self, x1, x2):
        """input features, output gates"""
        # pylint: disable-msg=invalid-name
        gates = []
        for g in self.gates:
            gates.append(g(x1, x2))
        return gates


class MoE(nn.Module):
    """Implementation mixture of experts model"""
    def __init__(self, in_channels=3, out_channels=11, linear_features=IN_FEATURES,
                 checkpoints=[], gate='simple', with_conv=False, num_expert=2):
        super(MoE, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.checkpoints = checkpoints
        self.with_conv = with_conv
        self.expert = FRRNet(in_channels, out_channels) # added for backwards compatibility with the previously trained models
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
        if 'simple' in gate:
            self.gating = GateLayer_simple(linear_features, num_expert)
        elif 'classwise' in gate:
            self.gating = GateLayer_classwise(linear_features, num_expert, out_channels)
        else:
            raise NotImplementedError
        if with_conv:
            self.conv3x3 = nn.Conv2d(out_channels, out_channels, 3, 1, 1, bias=False)

    def forward_expert(self, expert, x):
        """expert model forward function"""
        # pylint: disable-msg=invalid-name
        

        #with torch.no_grad():
        x = expert.first(x)
        y = expert.reslayers_in(x)

        z = expert.divide(y)
        y = expert.pool(y)

        # pool 11, 12 and 13
        y, z = expert.frrnlayer1(y, z)

        # pool 21, 22, 23 and 24
        y = expert.pool(y)
        y, z = expert.frrnlayer2(y, z)
        #_feat, z_feat = y, z

        # pool 31 and 32
        y = expert.pool(y)
        y, z = expert.frrnlayer3(y, z)
        #y_feat, z_feat = y, z

        # pool 41 and 42
        y = expert.pool(y)
        y, z = expert.frrnlayer4(y, z)
        y_feat, z_feat = y, z

        # unpool1
        y = expert.up(y)
        y, z = expert.defrrnlayer1(y, z)

        # unpool2
        y = expert.up(y)
        y, z = expert.defrrnlayer2(y, z)

        # unpool3
        y = expert.up(y)
        y, z = expert.defrrnlayer3(y, z)

        # unpool 34
        y = expert.up(y)
        refine = expert.compress(torch.cat((y, z), 1))

        out = expert.reslayers_out(refine)
        final_out = expert.out_conv(out)
        return y_feat, z_feat, final_out

    def forward(self, x):
        # pylint: disable-msg=invalid-name

        y_1, _, out_1 = self.forward_expert(self.expert1, x)
        y_2, _, out_2 = self.forward_expert(self.expert2, x)

        weights = self.gating(y_1, y_2)
        if isinstance(weights, list):
            # classwise gating
            w1 = torch.cat([w[:, 0].reshape(-1, 1, 1, 1) for w in weights], dim=1)
            w2 = torch.cat([w[:, 1].reshape(-1, 1, 1, 1) for w in weights], dim=1)
            out_1 *= w1
            out_2 *= w2
        else:
            out_1 *= weights[:, 0].reshape(-1, 1, 1, 1)
            out_2 *= weights[:, 1].reshape(-1, 1, 1, 1)
        out = self.conv3x3(out_1+out_2) if self.with_conv else out_1+out_2
        return out

    def _get_train_params(self):
        params = list(self.gating.parameters())
        if self.with_conv:
            params += list(self.conv3x3.parameters())
        return params