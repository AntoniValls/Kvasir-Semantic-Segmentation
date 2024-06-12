# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:light
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.16.2
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# +
import torch

# Dice coeficient an IoU score
def iou_score_old(output, target):
    smooth = 1e-5

    if torch.is_tensor(output):
        output = torch.sigmoid(output).data.cpu().numpy()
    if torch.is_tensor(target):
        target = target.data.cpu().numpy()
    output_ = output > 0.5
    target_ = target > 0.5
    intersection = (output_ & target_).sum()
    union = (output_ | target_).sum()

    return (intersection + smooth) / (union + smooth)

def iou_score(output, target):

    smooth = 1e-5
    output = torch.sigmoid(output).view(-1).data.cpu().numpy()
    target = target.view(-1).data.cpu().numpy()

    intersection = (output * target).sum()
    union = output.sum() + target.sum() - intersection

    return (intersection + smooth) / (union + smooth)


def dice_coef(output, target):
    smooth = 1e-5

    output = torch.sigmoid(output).view(-1).data.cpu().numpy()
    target = target.view(-1).data.cpu().numpy()
    intersection = (output * target).sum()

    return (2. * intersection + smooth) / (output.sum() + target.sum() + smooth)

def pixel_accuracy(output, target):
    
    # Apply sigmoid to get probabilities and then threshold at 0.5 to get binary predictions
    output = torch.sigmoid(output)  # Convert logits to probabilities
    output = (output > 0.5).float() # Convert probabilities to binary (0 or 1)
    
    # Flatten the tensors to make them 1-dimensional arrays
    output = output.view(-1).data.cpu().numpy()
    target = target.view(-1).data.cpu().numpy()
    
    # Calculate the number of correct predictions
    correct = (output == target).sum()
    
    # Calculate the total number of pixels
    total = target.size
    
    # Calculate pixel accuracy
    pixel_accuracy = correct / total
    
    return pixel_accuracy
    
class AverageMeter(object):
    
    def __init__(self):
        self.reset()
    
    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
    
    def update(self, val, n=1):
        self.val = val
        self.sum += val*n
        self.count+=n
        self.avg = self.sum/self.count

def count_parameters(model):
  return sum(p.numel() for p in model.parameters() if p.requires_grad)
