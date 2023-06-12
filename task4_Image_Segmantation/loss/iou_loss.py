import torch
import torch.nn as nn

class IoULoss(nn.Module):
    def __init__(self):
        super(IoULoss, self).__init__()
    
    def forward(self, output, target, smooth=1e-7):
        # use sigmoid to return probabilities
        output = torch.sigmoid(output)
        
        # flatten output and target
        output = output.view(-1)
        target = target.view(-1)
        
        intersection = (output * target).sum()
        total = (output + target).sum()
        union = total - intersection
        
        iou = (intersection + smooth) / (union + smooth)
        
        return 1 - iou


if __name__ == "__main__":
    criterion = IoULoss()
    
    output = torch.randn(32, 1, 256, 256)
    target = (torch.rand(32, 1, 256, 256) > 0.5).type(torch.float32)
    
    loss = criterion(output, target)
    
    print(f"loss: {loss}")
