import torch
import torch.nn as nn
from torchvision.ops.focal_loss import sigmoid_focal_loss

class FocalLoss(nn.Module):
    def __init__(self, from_logits=True):
        super(FocalLoss, self).__init__()
        self.from_logits = from_logits
        
    def forward(self, output, target, alpha=0.8, gamma=2):
        # flatten output and target
        output = output.view(-1)
        target = target.view(-1)
        
        prob = torch.sigmoid(output)
        
        focal = -(((1 - prob) ** gamma) * target * torch.log(prob) + \
                    (prob ** gamma) * (1 - target) * torch.log(1 - prob))
        
        if alpha >= 0:
            focal = focal * alpha
        
        return focal.mean()
    
if __name__ == "__main__":
    criterion = FocalLoss()
    
    output = torch.rand(32, 1, 256, 256)
    target = (torch.rand(32, 1, 256, 256) > 0.5).type(torch.float32)
    
    loss = criterion(output, target, alpha=-1)
    true_loss = sigmoid_focal_loss(output, target, reduction='mean', alpha=-1)
    
    print(f"loss: {loss}")
    print(f"true_loss: {true_loss}")
    
    assert torch.isclose(loss, true_loss), "loss is calculated incorreclty"