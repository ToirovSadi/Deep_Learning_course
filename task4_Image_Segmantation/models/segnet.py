import torch
import torch.nn as nn

class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DoubleConv, self).__init__()
        
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)


    def forward(self, x):
        x = torch.relu(self.bn1(self.conv1(x)))
        x = torch.relu(self.bn2(self.conv2(x)))
        return x


class SegNet(nn.Module):
    def __init__(self, in_channels=3, out_channels=1, **kwargs):
        super(SegNet, self).__init__(**kwargs)
        
        """SegNet Architecture
            https://arxiv.org/pdf/1511.00561.pdf
            
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
            return_indices=True, # to Unpool in corresponding layer
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
        self.d3 = DoubleConv(512, 256)
        self.d2 = DoubleConv(256, 128)
        self.d1 = DoubleConv(128, 64)
        self.d0 = DoubleConv(64, 32)
        
        self.output = nn.Conv2d(
            in_channels=32,
            out_channels=out_channels,
            kernel_size=3,
            padding=1,
        )
        
    
    def forward(self, x):
        x = self.norm(x) # normilize the input
        
        """Encode"""
        # (_, 3, 256, 256) -> (_, 64, 128, 128)
        x, ind0 = self.MaxEn(self.e0(x))
        
        # (_, 64, 128, 128) -> (_, 128, 64, 64)
        x, ind1 = self.MaxEn(self.e1(x))
        
        # (_, 128, 64, 64) -> (_, 256, 32, 32)
        x, ind2 = self.MaxEn(self.e2(x))
        
        # (_, 256, 32, 32) -> (_, 512, 16, 16)
        x, ind3 = self.MaxEn(self.e3(x))
        
        
        """Bottleneck"""
        x = self.bottleneck(x)
        
        
        """Decode"""
        # (_, 512, 16, 16) -> (_, 256, 32, 32)
        x = self.d3(self.MaxDe(x, ind3))
        
        # (_, 256, 32, 32) -> (_, 128, 64, 64)
        x = self.d2(self.MaxDe(x, ind2))
        
        # (_, 128, 64, 64) -> (_, 64, 128, 128)
        x = self.d1(self.MaxDe(x, ind1))
        
        # (_, 64, 128, 128) -> (_, 32, 256, 256)
        x = self.d0(self.MaxDe(x, ind0))
        
        x = self.output(x)
        return x
        
        
if __name__ == '__main__':
    batch_size = 1 # to save memory
    in_channels = 3
    out_channels = 1
    image_size = (256, 256)
    
    x = torch.randn((batch_size, in_channels, *image_size))
    
    model = SegNet(in_channels, out_channels)
    y = model(x)
    
    print(f"y.shape: {y.shape}")
    assert y.shape == (batch_size, out_channels, *image_size), "incorrect shape after forward prop."
