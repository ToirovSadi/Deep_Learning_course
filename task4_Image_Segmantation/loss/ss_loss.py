import torch
import torch.nn as nn
from .bce_loss import BCELoss

class SSLoss(nn.Module):
    def __init__(self):
        """SSLLoss
            https://arxiv.org/pdf/1910.08711.pdf
            
        """
        super(SSLoss, self).__init__()

    def forward(self, output, target, smooth=1e-7):
        # flatten output and target
        output = output.view(-1)
        target = target.view(-1)
        
        output_norm = (output - output.mean() + smooth) / (output.std() + smooth)
        target_norm = (target - target.mean() + smooth) / (target.std() + smooth)
        
        corr = torch.abs(output_norm - target_norm)
        bce = BCELoss()(output, target)
        
        ssl = (corr * (corr > 0.1 * corr.max())) * bce
        
        return ssl.mean()


if __name__ == "__main__":
    criterion = SSLoss()
    
    output = torch.randn(32, 1, 256, 256)
    target = (torch.rand(32, 1, 256, 256) > 0.5).type(torch.float32)
    
    loss = criterion(output, target)
    
    print(f"loss: {loss}")
