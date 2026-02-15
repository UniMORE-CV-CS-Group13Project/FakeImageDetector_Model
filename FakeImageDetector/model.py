import torch
import torch.nn as nn
import torch.nn.functional as F

class Block(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        # Branch 1: 3x3 -> 3x3 -> 3x3 -> 3x3
        self.branch1 = nn.Sequential(
            nn.Conv2d(
                in_channels=self.in_channels,
                out_channels=self.out_channels // 2,
                kernel_size=3,
                stride=1,
                padding=1,
                padding_mode="zeros", 
                bias=False,
            ),
            nn.BatchNorm2d(self.out_channels // 2),
            nn.GELU(),
            nn.Conv2d(
                in_channels=self.out_channels // 2,
                out_channels=self.out_channels // 2,
                kernel_size=3,
                stride=1,
                padding=1,
                padding_mode="zeros", 
                bias=False,
            ),
            nn.BatchNorm2d(self.out_channels // 2),
            nn.GELU(),
            nn.Conv2d(
                in_channels=self.out_channels // 2,
                out_channels=self.out_channels // 2,
                kernel_size=3,
                stride=1,
                padding=1,
                padding_mode="zeros", 
                bias=False,
            ),
            nn.BatchNorm2d(self.out_channels // 2),
            nn.GELU(),
            nn.Conv2d(
                in_channels=self.out_channels // 2,
                out_channels=self.out_channels // 2,
                kernel_size=3,
                stride=1,
                padding=1,
                padding_mode="zeros", 
                bias=False,
            ),
            nn.BatchNorm2d(self.out_channels // 2)
        )
        # Branch 2: 3x3 -> 3x3 (with dilation=2)
        self.branch2 = nn.Sequential(
            nn.Conv2d(
                in_channels=self.in_channels,
                out_channels=self.out_channels // 2,
                kernel_size=3,
                stride=1,
                padding=2,
                dilation=2,
                padding_mode="zeros", 
                bias=False,
            ),
            nn.BatchNorm2d(self.out_channels // 2),
            nn.GELU(),
            nn.Conv2d(
                in_channels=self.out_channels // 2,
                out_channels=self.out_channels // 2,
                kernel_size=3,
                stride=1,
                padding=2,
                dilation=2,
                padding_mode="zeros", 
                bias=False,
            ),
            nn.BatchNorm2d(self.out_channels // 2)
        )
        # Residual
        self.residual_connection = nn.Sequential(
            nn.Conv2d(
                in_channels=self.in_channels,
                out_channels=self.out_channels,
                kernel_size=1,
                stride=1,
                padding=0,
                padding_mode="zeros", 
                bias=False
            ),
            nn.BatchNorm2d(self.out_channels)
        )

    def forward(self, x):
        x1 = self.branch1(x)
        x2 = self.branch2(x)
        x_cat = torch.cat([x1, x2], dim=1)
        x = x_cat + self.residual_connection(x)
        out = F.gelu(x)
        return out


class GlobalPoolPyramidMLP(nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.mlp = nn.Sequential(
            nn.Linear(in_features=self.in_features, out_features=1024),
            nn.GELU(),
            nn.Dropout(0.50),
            nn.Linear(in_features=1024, out_features=512),
            nn.GELU(),
            nn.Dropout(0.25),
            nn.Linear(in_features=512, out_features=self.out_features)
        )
    
    def forward(self, x):
        x = self.mlp(x)
        return x


class CNN_Model(nn.Module):
    def __init__(self, in_channels, num_classes):
        super().__init__()
        self.in_channels = in_channels
        self.num_classes = num_classes
        # Start of the net and channel expansion
        self.first_bn = nn.BatchNorm2d(self.in_channels)
        self.conv_expansion = nn.Sequential(
            nn.Conv2d(
                in_channels=self.in_channels,
                out_channels=16,
                kernel_size=3,
                stride=1,
                padding=1,
                padding_mode="zeros", 
                bias=False,
            ),
            nn.BatchNorm2d(16)
        )
        # First residual Block
        self.res1_left_branch = nn.Sequential(
            nn.Conv2d(
                in_channels=16,
                out_channels=32,
                kernel_size=3,
                stride=1,
                padding=1,
                padding_mode="zeros", 
                bias=False
            ),
            nn.BatchNorm2d(32),
            nn.GELU(),
            nn.Conv2d(
                in_channels=32,
                out_channels=64,
                kernel_size=3,
                stride=1,
                padding=1,
                padding_mode="zeros", 
                bias=False
            ),
            nn.BatchNorm2d(64)
        )
        self.res1_right_branch = nn.Sequential(
            nn.Conv2d(
                in_channels=16,
                out_channels=64,
                kernel_size=1,
                stride=1,
                padding=0,
                padding_mode="zeros", 
                bias=False
            ),
            nn.BatchNorm2d(64)
        )
        # Blocks
        self.pool1 = nn.MaxPool2d(
            kernel_size=2,
            stride=2
        )
        self.block1 = Block(
            in_channels=64,
            out_channels=128
        )
        self.pool2 = nn.MaxPool2d(
            kernel_size=2,
            stride=2
        )
        self.block2 = Block(
            in_channels=128,
            out_channels=256
        )
        self.pool3 = nn.MaxPool2d(
            kernel_size=2,
            stride=2
        )
        self.block3 = Block(
            in_channels=256,
            out_channels=512
        )
        self.pool4 = nn.MaxPool2d(
            kernel_size=2,
            stride=2
        )
        self.block4 = Block(
            in_channels=512,
            out_channels=1024
        )
        self.pool5 = nn.MaxPool2d(
            kernel_size=2,
            stride=2
        )
        # Last residual Block
        self.res2_left_branch = nn.Sequential(
            nn.Conv2d(
                in_channels=1024,
                out_channels=1024,
                kernel_size=3,
                stride=1,
                padding=1,
                padding_mode="zeros", 
                bias=False
            ),
            nn.BatchNorm2d(1024),
            nn.GELU(),
            nn.Conv2d(
                in_channels=1024,
                out_channels=1024,
                kernel_size=3,
                stride=1,
                padding=1,
                padding_mode="zeros", 
                bias=False
            ),
            nn.BatchNorm2d(1024),
            nn.GELU(),
            nn.Conv2d(
                in_channels=1024,
                out_channels=1024,
                kernel_size=3,
                stride=1,
                padding=1,
                padding_mode="zeros", 
                bias=False
            ),
            nn.BatchNorm2d(1024)
        )
        self.res2_right_branch = nn.Sequential(
            nn.Conv2d(
                in_channels=1024,
                out_channels=1024,
                kernel_size=1,
                stride=1,
                padding=0,
                padding_mode="zeros", 
                bias=False
            ),
            nn.BatchNorm2d(1024)
        )
        # Final convolutional layers
        self.conv_compression = nn.Sequential(
            nn.Conv2d(
                in_channels=1024,
                out_channels=512,
                kernel_size=3,
                stride=1,
                padding=1,
                padding_mode="zeros", 
                bias=False,
            ),
            nn.BatchNorm2d(512),
            nn.GELU(),
            nn.Conv2d(
                in_channels=512,
                out_channels=256,
                kernel_size=3,
                stride=1,
                padding=1,
                padding_mode="zeros", 
                bias=False,
            ),
            nn.BatchNorm2d(256),
            nn.GELU(),
            nn.Conv2d(
                in_channels=256,
                out_channels=128,
                kernel_size=3,
                stride=1,
                padding=1,
                padding_mode="zeros", 
                bias=False,
            ),
            nn.BatchNorm2d(128)
        )
        # Global Average Pool Pyramid
        self.gap_1x1 = nn.AdaptiveAvgPool2d(1)
        self.gap_2x2 = nn.AdaptiveAvgPool2d(2)
        self.gap_4x4 = nn.AdaptiveAvgPool2d(4)
        # Global Max Pool Pyramid
        self.gmp_1x1 = nn.AdaptiveMaxPool2d(1)
        self.gmp_2x2 = nn.AdaptiveMaxPool2d(2)
        self.gmp_4x4 = nn.AdaptiveMaxPool2d(4)
        # Global Average Pooling Pyramid (MLP)
        self.gapp_mlp = GlobalPoolPyramidMLP(in_features=2688, out_features=256)
        # Global Max Pooling Pyramid (MLP)
        self.gmpp_mlp = GlobalPoolPyramidMLP(in_features=2688, out_features=256)
        # Classifier (MLP)
        self.classifier_mlp = nn.Sequential(
            nn.Linear(in_features=512, out_features=128),
            nn.GELU(),
            nn.Dropout(0.250),
            nn.Linear(in_features=128, out_features=32),
            nn.GELU(),
            nn.Dropout(0.125),
            nn.Linear(in_features=32, out_features=self.num_classes)
        )

    def forward(self, x):
        # Start of the network
        x = self.first_bn(x)
        x = self.conv_expansion(x)
        x = F.gelu(x)
        # First residual Block
        x_res1_left = self.res1_left_branch(x)
        x_res1_right = self.res1_right_branch(x)
        x = x_res1_left + x_res1_right
        x = F.gelu(x)
        # Cascade of Operational Blocks
        x = self.pool1(x)
        x = self.block1(x)
        x = self.pool2(x)
        x = self.block2(x)
        x = self.pool3(x)
        x = self.block3(x)
        x = self.pool4(x)
        x = self.block4(x)
        x = self.pool5(x)
        # Final residual Block
        x_res2_left = self.res2_left_branch(x)
        x_res2_right = self.res2_right_branch(x)
        x = x_res2_left + x_res2_right
        x = F.gelu(x)
        # Final Convolutions
        x = self.conv_compression(x)
        x = F.gelu(x)
        # Global Average Pool Pyramid
        x_gap1 = self.gap_1x1(x)
        x_gap1_flattened = torch.flatten(x_gap1, 1)
        x_gap2 = self.gap_2x2(x)
        x_gap2_flattened = torch.flatten(x_gap2, 1)
        x_gap4 = self.gap_4x4(x)
        x_gap4_flattened = torch.flatten(x_gap4, 1)
        x_gapp = torch.cat([x_gap1_flattened, x_gap2_flattened, x_gap4_flattened], dim=1)
        # Global Max Pool Pyramid
        x_gmp1 = self.gmp_1x1(x)
        x_gmp1_flattened = torch.flatten(x_gmp1, 1)
        x_gmp2 = self.gmp_2x2(x)
        x_gmp2_flattened = torch.flatten(x_gmp2, 1)
        x_gmp4 = self.gmp_4x4(x)
        x_gmp4_flattened = torch.flatten(x_gmp4, 1)
        x_gmpp = torch.cat([x_gmp1_flattened, x_gmp2_flattened, x_gmp4_flattened], dim=1)
        # Pyramids separate processing
        x_gapp = self.gapp_mlp(x_gapp)
        x_gmpp = self.gmpp_mlp(x_gmpp)
        # Pyramids results concatenation
        x = torch.cat([x_gapp, x_gmpp], dim=1)
        # Final Classifier
        out = self.classifier_mlp(x)
        return out