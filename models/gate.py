"""
Gate layers of mixture of epxerts
"""
import torch
import torch.nn as nn


def make_gate_layer(
    gate_type,
    in_channels=256,
    in_features=1200,
    fc_features=128,
    num_experts=2,
    num_cls=11,
):
    """return gate layer depend on type of gate"""
    fc_features += 0 # Avoid pylint
    if "simple" in gate_type:
        return GateLayerSimple(
            in_channels=in_channels,  # channel number after aspp
            in_features=in_features,
            num_experts=num_experts,
        )
    if "classwise" in gate_type:
        return GateLayerClasswise(
            in_channels=in_channels,
            in_features=in_features,
            num_experts=num_experts,
            num_cls=num_cls,
        )

    raise NotImplementedError


class GateLayerSimple(nn.Module):
    """simple gated layer"""
    def __init__(self,
                 in_channels,
                 in_features=1200,
                 fc_features=128,
                 num_experts=2):
        super().__init__()
        self.in_features = in_features
        self.conv1 = nn.Conv2d(in_channels, 1, 3, 1, 1)
        self.relu = nn.ReLU()
        self.fc1 = nn.Linear(in_features, fc_features)
        self.fc2 = nn.Linear(fc_features, num_experts)
        self.softmax = nn.Softmax(dim=1)  # channel-wise

    def forward(self, x_in_1, x_in_2=None):
        """
        input features, x_in_i: [B, C, H, W]
        output gate: [B, num_experts]
        Note: in_features = CxHxW for fc-layer, the C will be 1 after self.conv1
        H=W=ceil(crop_size/16). For example, crop_size=901, then
        in_features = ceil(901/16) **2 = 57**2 = 3249
        """
        if x_in_2 is not None:
            x_in = torch.cat((x_in_1, x_in_2), dim=1)
        else:
            x_in = x_in_1
        batch_size = x_in.shape[0]
        x_in = self.conv1(x_in)
        x_in = self.relu(x_in)
        x_in = x_in.view(batch_size, -1)
        assert (
            x_in.shape[1] == self.in_features
        ), "Feature map dose not match fc-layer,\
            the crop_size muss be same with pre-defined."
        x_in = self.fc1(x_in)
        x_in = self.relu(x_in)
        x_in = self.fc2(x_in)
        gate_out = self.softmax(x_in)
        return gate_out


class GateLayerClasswise(nn.Module):
    """classwise gated layer"""
    def __init__(
        self,
        in_channels,
        in_features=1200,
        fc_features=128,
        num_experts=2,
        num_cls=11,
    ):
        # output gate shape [bs, num_experts, 1, 1]
        super().__init__()
        self.gates = nn.ModuleList([
            GateLayerSimple(in_channels, in_features, fc_features, num_experts)
            for _ in range(num_cls)
        ])

    def forward(self, x_in_1, x_in_2):
        """
        input features, x_in_i: [B, C, H, W]
        output gate list: [[B, num_experts], ...], len=num_cls
        """
        gates_out = []
        for gate_cls in self.gates:
            gates_out.append(gate_cls(x_in_1, x_in_2))
        return gates_out
