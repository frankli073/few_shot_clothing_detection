# File: protonet.py
# coding=utf-8
import torch.nn as nn

def conv_block(in_channels, out_channels):
    '''
    Returns a block conv-bn-relu-pool.
    '''
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, 3, padding=1),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(),
        nn.MaxPool2d(2)
    )

class ProtoNet(nn.Module):
    '''
    Model as described in the reference paper.
    '''
    def __init__(self, x_dim=3, hid_dim=64, z_dim=64):  # Adjust x_dim to 3 for RGB images
        super(ProtoNet, self).__init__()
        self.encoder = nn.Sequential(
            conv_block(x_dim, hid_dim),
            conv_block(hid_dim, hid_dim),
            conv_block(hid_dim, hid_dim),
            conv_block(hid_dim, z_dim),
        )

    def forward(self, x):
        x = self.encoder(x)
        return x.view(x.size(0), -1)  # Flatten the output
