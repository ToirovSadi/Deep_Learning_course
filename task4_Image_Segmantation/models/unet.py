import torch
import torch.nn as nn

class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels, decode=False):
        super(DoubleConv, self).__init__()
        
        mid_channels = out_channels
        if decode:
            # for skip connections
            # example: in_channels=1024, out_channels=256
            # we will have: Conv2d(1024, 512) and Conv2d(512, 256)
            mid_channels = in_channels // 2
        
        self.conv0 = nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1)
        self.bn0 = nn.BatchNorm2d(mid_channels)
        
        self.conv1 = nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        
    def forward(self, x):
        x = torch.relu(self.bn0(self.conv0(x)))
        x = torch.relu(self.bn1(self.conv1(x)))
        
        return x

class UNet(nn.Module):
    def __init__(self, in_channels=3, out_channels=1):
        super(UNet, self).__init__()
        
        """UNet Architecture
            https://arxiv.org/pdf/1505.04597.pdf
            
            Three main parts:
                - Encoder
                - Bottleneck (Bridge)
                - Decoder
        """
        self.norm = nn.BatchNorm2d(in_channels)
        
        """Encoder"""
        self.MaxEn = nn.MaxPool2d(
            kernel_size=2,
            stride=2,
            return_indices=True,
        )
        self.e0 = DoubleConv(in_channels, 64)
        self.e1 = DoubleConv(64, 128)
        self.e2 = DoubleConv(128, 256)
        self.e3 = DoubleConv(256, 512)
        
        """Bottleneck"""
        self.bottleneck = nn.Sequential(
            DoubleConv(512, 1024),
            DoubleConv(1024, 512),
        )
        
        """Decoder"""
        self.MaxDe = nn.MaxUnpool2d(
            kernel_size=2,
            stride=2,
        )
        self.d3 = DoubleConv(1024, 256, decode=True)
        self.d2 = DoubleConv(512, 128, decode=True)
        self.d1 = DoubleConv(256, 64, decode=True)
        self.d0 = DoubleConv(128, 32, decode=True)
        
        self.output = nn.Conv2d(
            in_channels=32,
            out_channels=out_channels,
            kernel_size=3,
            padding=1,
        )
        
    def forward(self, x):
        x = self.norm(x) # normilize the input
        
        # x.shape (_, 3, 256, 256)
        """Encode"""
        s0 = self.e0(x)               # (_, 64, 256, 256) 
        x, ind0 = self.MaxEn(s0)      # (_, 64, 128, 128)
        
        s1 = self.e1(x)               # (_, 128, 128, 128)
        x, ind1 = self.MaxEn(s1)      # (_, 128, 64, 64)
        
        s2 = self.e2(x)               # (_, 256, 64, 64)
        x, ind2 = self.MaxEn(s2)      # (_, 256, 32, 32)
        
        s3 = self.e3(x)               # (_, 512, 32, 32)
        x, ind3 = self.MaxEn(s3)      # (_, 512, 16, 16)
        
        
        """Bottleneck"""
        x = self.bottleneck(x)        # (_, 512, 16, 16)
        
        
        """Decode"""
        x = self.MaxDe(x, ind3)       # (_, 512, 32, 32)
        x = self.d3(torch.cat([x, s3], axis=1))  # 2 * (_, 512, 32, 32) -> (_, 256, 32, 32)
        
        x = self.MaxDe(x, ind2)       # (_, 256, 64, 64)
        x = self.d2(torch.cat([x, s2], axis=1))  # 2 * (_, 256, 64, 64) -> (_, 128, 64, 64)
        
        x = self.MaxDe(x, ind1)       # (_, 128, 128, 128)
        x = self.d1(torch.cat([x, s1], axis=1))  # 2 * (_, 128, 128, 128) -> (_, 64, 128, 128)
        
        x = self.MaxDe(x, ind0)       # (_, 64, 256, 256)
        x = self.d0(torch.cat([x, s0], axis=1))   # 2 * (_, 64, 256, 256) -> (_, 32, 256, 256)
        
        x = self.output(x) # (_, 1, 256, 256)
        return x

if __name__ == '__main__':
    batch_size = 1 # to save memory
    in_channels = 3
    out_channels = 1
    image_size = (256, 256)
    
    x = torch.randn((batch_size, in_channels, *image_size))
    
    model = UNet(in_channels, out_channels)
    y = model(x)
    
    print(f"y.shape: {y.shape}")
    assert y.shape == (batch_size, out_channels, *image_size), "incorrect shape after forward prop."
