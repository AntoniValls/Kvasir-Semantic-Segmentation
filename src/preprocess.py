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
import os
import json
from PIL import Image
import numpy as np
from torch.utils.data import Dataset
import torchvision.transforms as transforms



def load_data(path, is_mask=False):
    if is_mask:
        files = os.listdir(path)
        return {filename[:-4]: np.array(Image.open(os.path.join(path, filename))) for filename in files}
    else:
        with open(path, 'r') as f:
            return json.load(f)

class KvasirDataset(Dataset):
    def __init__(self, images, masks, transforms=None, device='cpu'):
        self.images = images
        self.masks = masks
        self.transforms = transforms
        self.device = device

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = self.images[idx]
        mask = self.masks[idx]

        # Convert mask values from 255 to 1
        mask = (mask / 255).astype(np.float32)

        if self.transforms:
            image = self.transforms(image)

        return image, mask




