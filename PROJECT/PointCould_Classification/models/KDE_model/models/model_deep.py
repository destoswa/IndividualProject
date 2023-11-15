import torch
import torch.nn as nn


class PointTransformerCls(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        output_channels = cfg['num_class']
        d_grid = cfg['grid_dim']
        self.grid_dim = cfg['grid_dim']
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax(dim=1)

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
        self.mp5 = nn.MaxPool3d(2)

        # convolution layer 6
        self.conv11 = nn.Conv3d(512, 1024, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn11 = nn.BatchNorm3d(1024)
        self.conv12 = nn.Conv3d(1024, 1024, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn12 = nn.BatchNorm3d(1024)
        self.mp6 = nn.MaxPool3d(2)

        # convolution layer 7
        self.conv13 = nn.Conv3d(1024, 2048, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn13 = nn.BatchNorm3d(2048)
        self.conv14 = nn.Conv3d(2048, 2048, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn14 = nn.BatchNorm3d(2048)

        """# fully connected layer 1
        self.linear1 = nn.Linear(512 * int((d_grid / 16) ** 3), 256, bias=False)
        self.do1 = nn.Dropout(p=0.3)"""
        # fully connected layer 1
        self.linear1 = nn.Linear(2048, 1024, bias=False)
        self.do = nn.Dropout(p=0.3)

        # fully connected layer 2
        self.linear2 = nn.Linear(1024, 512, bias=False)

        # fully connected layer 3
        self.linear3 = nn.Linear(512, 256, bias=False)

        # fully connected layer 4
        self.linear4 = nn.Linear(256, 128, bias=False)

        # fully connected layer 5
        self.linear5 = nn.Linear(128, output_channels)

    def forward(self, x):
        batch_size, grid_dim, _, _ = x.size()
        x = x.reshape((batch_size, 1, self.grid_dim, self.grid_dim, self.grid_dim)).float()  # B x 1 x N x N x N

        # convolution layer 1
        x = self.relu(self.bn1(self.conv1(x)))  # B x 32 x N x N x N
        """min = torch.min(x)
        max = torch.max(x)
        num = torch.sum(x[x != 0])"""
        #x = self.conv1(x)  # B x 32 x N x N x N
        # whitening
        """min = torch.min(x)
        max = torch.max(x)
        num = torch.sum(x[x != 0])"""
        #norm = torch.norm(x, dim=1).reshape((batch_size, 1, grid_dim, grid_dim, grid_dim))
        """min = torch.min(norm)
        max = torch.max(norm)
        num = torch.sum(norm[norm != 0])"""
        #x = x / norm
        #x[x != x] = 0
        """min = torch.min(x)
        max = torch.max(x)
        num = torch.sum(x[x != 0])"""
        #x = self.relu(x)
        x = self.relu(self.bn2(self.conv2(x)))  # B x 32 x N x N x N
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
        x = self.relu(self.bn9(self.conv9(x)))  # B x 256 x N/8 x N/8 x N/8
        x = self.relu(self.bn10(self.conv10(x)))  # B x 256 x N/8 x N/8 x N/8
        x = self.mp5(x)  # B x 128 x N/16 x N/16 x N/16

        # convolution layer 6
        x = self.relu(self.bn11(self.conv11(x)))  # B x 256 x N/8 x N/8 x N/8
        x = self.relu(self.bn12(self.conv12(x)))  # B x 256 x N/8 x N/8 x N/8
        x = self.mp6(x)  # B x 128 x N/16 x N/16 x N/16

        # convolution layer 7
        x = self.relu(self.bn13(self.conv13(x)))  # B x 512 x N/16 x N/16 x N/16
        x = self.relu(self.bn14(self.conv14(x)))  # B x 512 x N/16 x N/16 x N/16

        # fully connected layers
        x_flat = x.reshape((batch_size, -1))  # B x (512 * (N/16)^3)
        x = self.relu(self.do(self.linear1(x_flat)))  # B x 256
        x = self.relu(self.do(self.linear2(x)))  # B x 128
        x = self.relu(self.do(self.linear3(x)))  # B x 128
        x = self.relu(self.do(self.linear4(x)))  # B x 128
        x = self.softmax(self.linear5(x))  # B x N_class

        return x
