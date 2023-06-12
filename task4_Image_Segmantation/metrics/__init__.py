import torch

# or simply Intersection over Union (IoU)
def jaccard_index(output, target, smooth=1e-7):
    # use sigmoid to return probabilities
    output = torch.sigmoid(output)
    
    # flatten output and target
    output = output.view(-1)
    target = target.view(-1)
    
    intersection = (output * target).sum()
    total = (output + target).sum()
    union = total - intersection
    
    iou = (intersection + smooth) / (union + smooth)
    return iou.cpu().item()


def dice_coef(output, target, smooth=1e-7):
    # use sigmoid to return probabilities
    output = torch.sigmoid(output)
    
    # flatten output and target
    output = output.view(-1)
    target = target.view(-1)
    
    intersection = (output * target).sum()
    dice = (2 * intersection + smooth) / (output.sum() + target.sum() + smooth)
    
    return dice.cpu().item()
