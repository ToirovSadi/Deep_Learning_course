import torch
import torch.nn as nn
from torch.nn import BCEWithLogitsLoss as True_Loss

class BCELoss(nn.Module):
    def __init__(self, from_logits=True, sigmoid=True):
        super(BCELoss, self).__init__()
        self.from_logits = from_logits
        self.sigmoid = sigmoid
    
    
    def forward(self, output, target):
        # flatten output and target
        output = output.view(-1)
        target = target.view(-1)
        
        if self.from_logits is True:
            # optimization (work only with logits)
            bce = output - target * output + torch.log(1 + torch.exp(-output))
        else:
            if self.sigmoid:
                output = torch.sigmoid(output)
            
            bce = -(target * torch.log(output) + (1 - target) * torch.log(1 - output))
        
        
        return bce.mean()


if __name__ == "__main__":
    criterion = BCELoss()
    true_criterion = True_Loss()
    
    output = torch.randn(32, 1, 256, 256)
    target = (torch.rand(32, 1, 256, 256) > 0.5).type(torch.float32)
    
    loss = criterion(output, target)
    true_loss = true_criterion(output, target)
    
    print(f"loss: {loss}")
    print(f"true_loss: {true_loss}")
    assert torch.isclose(loss, true_loss), "loss is calculated incorrectly"
