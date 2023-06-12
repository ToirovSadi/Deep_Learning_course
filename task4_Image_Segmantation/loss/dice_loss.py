import torch
import torch.nn as nn

class DiceLoss(nn.Module):
    def __init__(self):
        super(DiceLoss, self).__init__()

    def forward(self, output, target, smooth=1e-7):
        # use sigmoid to return probabilities
        output = torch.sigmoid(output)
        
        # flatten output and target
        output = output.view(-1)
        target = target.view(-1)
        
        intersection = (output * target).sum()
        dice = (2 * intersection + smooth) / (output.sum() + target.sum() + smooth)
        
        return 1 - dice


if __name__ == "__main__":
    criterion = DiceLoss()
    
    output = torch.randn(32, 1, 256, 256)
    target = (torch.rand(32, 1, 256, 256) > 0.5).type(torch.float32)
    
    loss = criterion(output, target)
    
    print(f"loss: {loss}")
