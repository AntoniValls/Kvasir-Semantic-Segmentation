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

import torch
from src.utils import iou_score, dice_coef, pixel_accuracy, AverageMeter
from collections import OrderedDict
import numpy as np
from tqdm import tqdm

# Training phase
def train(config, model, iterator, criterion, optimizer, deep_supervision=False, device="cpu"):
    avg_meters = {'loss':AverageMeter(),'iou':AverageMeter(), 'dice_coef':AverageMeter(), 'accuracy':AverageMeter()}

    # tqdm object
    pbar = tqdm(total=len(iterator))
    
    # Train mode
    model.train()
        
    for input,target in iterator:
        input = input.to(device)
        target = target.float()
        target = target.to(device)

        # Deep supervision application
        if deep_supervision:
            outputs = model(input)
            loss = 0
            for output in outputs:
                loss += criterion(output,target)
            loss /= len(outputs)
            iou = iou_score(outputs[-1],target)
            dice_score = dice_coef(outputs[-1],target)
            accuracy = pixel_accuracy(outputs[-1],target)
            
        else:
            output = model(input)
            loss = criterion(output,target)
            iou = iou_score(output,target)
            dice_score = dice_coef(output,target)
            accuracy = pixel_accuracy(output,target)

        # Set gradients to zero
        optimizer.zero_grad()

        # Backpropagation
        loss.backward()

        # Apply optimizer
        optimizer.step()

        # Update the loss, iou and dice scores
        avg_meters['loss'].update(loss.item(), input.size(0))
        avg_meters['iou'].update(iou, input.size(0))
        avg_meters['dice_coef'].update(dice_score, input.size(0))
        avg_meters['accuracy'].update(accuracy, input.size(0))

        # Update the tqdm bar
        postfix = OrderedDict([
            ('loss',avg_meters['loss'].avg),
            ('iou', avg_meters['iou'].avg),
            ('dice_coef', avg_meters['dice_coef'].avg),
            ('accuracy', avg_meters['accuracy'].avg)
        ])
        pbar.set_postfix(postfix)
        pbar.update(1)
        
    pbar.close()

    return OrderedDict([
        ('loss', avg_meters['loss'].avg),
        ('iou', avg_meters['iou'].avg),
        ('dice_coef', avg_meters['dice_coef'].avg),
        ('accuracy', avg_meters['accuracy'].avg)
    ])


# Valudation/Testing phase
def evaluate(config, model, iterator, criterion, deep_supervision=False, device="cpu"):
    avg_meters = {'loss': AverageMeter(),'iou': AverageMeter(),'dice_coef':AverageMeter(), 'accuracy':AverageMeter()}

    # Switch to evaluate mode
    model.eval()

    # Do not compute the gradients
    with torch.no_grad():

        # tqdm object
        pbar = tqdm(total=len(iterator))
        
        for input, target in iterator:
            input = input.to(device)
            target = target.float()
            target = target.to(device)

            # Deep supervision application
            if deep_supervision:
                outputs = model(input)
                loss = 0
                for output in outputs:
                    loss += criterion(output, target)
                loss /= len(outputs)
                iou = iou_score(outputs[-1], target)
                dice_score = dice_coef(outputs[-1],target)
                accuracy = pixel_accuracy(outputs[-1],target)
            else:
                output = model(input)
                loss = criterion(output, target)
                iou = iou_score(output, target)
                dice_score = dice_coef(output,target)
                accuracy = pixel_accuracy(output,target)

            # Update the loss, iou and dice scores
            avg_meters['loss'].update(loss.item(), input.size(0))
            avg_meters['iou'].update(iou, input.size(0))
            avg_meters['dice_coef'].update(dice_score, input.size(0))
            avg_meters['accuracy'].update(accuracy, input.size(0))

            # Update the tqdm bar
            postfix = OrderedDict([
                ('loss', avg_meters['loss'].avg),
                ('iou', avg_meters['iou'].avg),
                ('dice_coef',avg_meters['dice_coef'].avg),
                ('accuracy', avg_meters['accuracy'].avg)
            ])
            pbar.set_postfix(postfix)
            pbar.update(1)
            
        pbar.close()

    return OrderedDict([
        ('loss', avg_meters['loss'].avg),
        ('iou', avg_meters['iou'].avg),
        ('dice_coef', avg_meters['dice_coef'].avg),
        ('accuracy', avg_meters['accuracy'].avg)
    ])

def predict(model, iterator, deep_supervision=False, device="cpu"):
    output_array = []
    # Switch to evaluate mode
    model.eval()

    # Do not compute the gradients
    with torch.no_grad():
        
        for input, target in iterator:
            input = input.to(device)

            # Deep supervision application
            if deep_supervision:
                outputs = model(input)
                output_array.append(outputs[-1:])
            else:
                output = model(input)
                output_array.append(output)


    return output_array