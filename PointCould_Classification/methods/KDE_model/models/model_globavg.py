import torch
import torch.nn as nn


class PointTransformerCls(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        output_channels = cfg['num_class']
        d_grid = cfg['grid_dim']
        self.output_channels = output_channels
        self.grid_dim = cfg['grid_dim']
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax(dim=1)
        self.do = nn.Dropout(p=0.3)

        # convolution layer 1
        self.conv1 = nn.Conv3d(1, 32, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm3d(32)
        self.conv2 = nn.Conv3d(32, 32, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm3d(32)
        self.mp1 = nn.MaxPool3d(2)

        # convolution layer 2
        self.conv3 = nn.Conv3d(32, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn3 = nn.BatchNorm3d(64)
        self.conv4 = nn.Conv3d(64, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn4 = nn.BatchNorm3d(64)
        self.mp2 = nn.MaxPool3d(2)

        # convolution layer 3
        self.conv5 = nn.Conv3d(64, 128, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn5 = nn.BatchNorm3d(128)
        self.conv6 = nn.Conv3d(128, 128, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn6 = nn.BatchNorm3d(128)
        self.mp3 = nn.MaxPool3d(2)

        # convolution layer 4
        self.conv7 = nn.Conv3d(128, 256, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn7 = nn.BatchNorm3d(256)
        self.conv8 = nn.Conv3d(256, 256, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn8 = nn.BatchNorm3d(256)
        self.mp4 = nn.MaxPool3d(2)

        # convolution layer 5
        self.conv9 = nn.Conv3d(256, 512, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn9 = nn.BatchNorm3d(512)
        self.conv10 = nn.Conv3d(512, 512, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn10 = nn.BatchNorm3d(512)

        # global averaging
        self.conv11 = nn.Conv3d(512, output_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn11 = nn.BatchNorm3d(output_channels)
        self.gap = nn.AvgPool3d(int(d_grid/16))



        # fully connected layer 1
        self.linear = nn.Linear(output_channels, output_channels, bias=False)
        """self.linear1 = nn.Linear(512 * int((d_grid / 16) ** 3), 256, bias=False)

        # fully connected layer 1
        self.linear2 = nn.Linear(256, 128, bias=False)

        # fully connected layer 2
        self.linear3 = nn.Linear(128, output_channels)"""

    def forward(self, x):
        batch_size, grid_dim, _, _ = x.size()
        x = x.reshape((batch_size, 1, self.grid_dim, self.grid_dim, self.grid_dim)).float()  # B x 1 x N x N x N

        # whitening
        x = self.conv1(x)  # B x 32 x N x N x N
        norm = torch.norm(x, dim=1).reshape((batch_size, 1, grid_dim, grid_dim, grid_dim)) + 1e-9  # B x 1 x N x N x N
        x = x / norm  # B x 32 x N x N x N
        #x[x != x] = 0  # B x 32 x N x N x N
        x = self.relu(x)  # B x 32 x N x N x N

        # convolution layer 1
        #x = self.relu(self.bn1(self.conv1(x)))  # B x 32 x N x N x N
        x = self.bn2(self.conv2(x))  # B x 32 x N x N x N
        x = self.mp1(x)  # B x 32 x N/2 x N/2 x N/2

        # convolution layer 2
        x = self.relu(self.bn3(self.conv3(x)))  # B x 64 x N/2 x N/2 x N/2
        x = self.relu(self.bn4(self.conv4(x)))  # B x 64 x N/2 x N/2 x N/2
        x = self.mp2(x)  # B x 64 x N/4 x N/4 x N/4

        # convolution layer 3
        x = self.relu(self.bn5(self.conv5(x)))  # B x 128 x N/4 x N/4 x N/4
        x = self.relu(self.bn6(self.conv6(x)))  # B x 128 x N/4 x N/4 x N/4
        x = self.mp3(x)  # B x 64 x N/8 x N/8 x N/8

        # convolution layer 4
        x = self.relu(self.bn7(self.conv7(x)))  # B x 256 x N/8 x N/8 x N/8
        x = self.relu(self.bn8(self.conv8(x)))  # B x 256 x N/8 x N/8 x N/8
        x = self.mp4(x)  # B x 128 x N/16 x N/16 x N/16

        # convolution layer 5
        x = self.relu(self.bn9(self.conv9(x)))  # B x 512 x N/16 x N/16 x N/16
        x = self.relu(self.bn10(self.conv10(x)))  # B x 512 x N/16 x N/16 x N/16

        # global averaging
        x = self.relu(self.bn11(self.conv11(x)))  # B x C x N/16 x N/16 x N/16
        x = self.gap(x)  # B x C x 1 x 1 x 1
        x = x.reshape((batch_size, self.output_channels))  # B x C
        x = self.softmax(x)
        # fully connected layers
        """x_flat = x.reshape((batch_size, -1))  # B x (512 * (N/16)^3)
        x = self.relu(self.do(self.linear1(x_flat)))  # B x 256
        x = self.relu(self.do(self.linear2(x)))  # B x 128
        x = self.softmax(self.linear5(x))  # B x N_class"""

        return x
