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
import torch.nn as nn
import torch.nn.functional as F

class BCEDiceLoss(nn.Module): # not working well (negative values)
    
    def __init__(self):
        super().__init__()
    
    def forward(self,input,target):
        # Permute target to match input's shape if needed
        if target.size() != input.size():
        # Permute target from [batch_size, height, width, channels] to [batch_size, channels, height, width]
            target = target.permute(0, 3, 1, 2)
            
        bce = F.binary_cross_entropy_with_logits(input,target) 
        smooth = 1e-5
        input = torch.sigmoid(input)
        num = target.size(0)
        input = input.view(num,-1)
        target = target.view(num,-1)
        intersection = (input * target)
        dice = (2. * intersection.sum(1) + smooth) / (input.sum(1) + target.sum(1) + smooth)
        dice = 1 - dice.sum() / num
        return (0.5 * bce + dice)
        
class DiceLoss(nn.Module):

    def __init__(self):
        super().__init__()
    
    def forward(self, input, target):

        smooth = 1e-5
        
        # Flatten the input and target tensors
        input = input.view(-1)
        target = target.view(-1)
        
        # Apply sigmoid activation to the input tensor
        input = torch.sigmoid(input)
        
        # Compute the Dice coefficient
        intersection = (input * target).sum()
        dice = (2. * intersection + smooth) / (input.sum() + target.sum() + smooth)
        
        # Compute the Dice loss
        dice_loss = 1 - dice
        
        return dice_loss

class IoULoss(nn.Module):
    def __init__(self):
        super(IoULoss, self).__init__()
    
    def forward(self, input, target):
        smooth = 1e-5

        # Apply sigmoid to get probabilities if not already done
        if input.dtype != torch.float:
            input = torch.sigmoid(input)

        # Flatten the tensors
        input = input.view(-1)
        target = target.view(-1)

        intersection = (input * target).sum()
        union = input.sum() + target.sum() - intersection

        iou = (intersection + smooth) / (union + smooth)
        iou_loss = 1 - iou
        return iou_loss