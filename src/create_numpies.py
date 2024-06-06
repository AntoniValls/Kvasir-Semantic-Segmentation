import os
import numpy as np
from PIL import Image
from tqdm import tqdm

def create_np_images(path, size = (256, 256)):
    # Path to the directory contains images
    images_dir = path
    
    # List all image filenames in the directory
    image_filenames = os.listdir(images_dir)
    
    # Create an empty list to store the images
    images = []
    
    # Load images as NumPy arrays and store them in the list
    for filename in tqdm(image_filenames, desc="Generating images numpy arrays"):
        image_path = os.path.join(images_dir, filename)
        if os.path.isfile(image_path):  # Check if it's a file, not a directory (to avoid the notebook checkpoints)
            image = Image.open(image_path)
    
            target_size = size
            image = image.resize(target_size, Image.LANCZOS)
            
            # Convert image to array
            image_array = np.array(image)
            images.append(image_array)
    
    # Convert the list of arrays to a single NumPy array
    images = np.array(images)
    np.save("data/images.npy", images)
    return images

def create_np_masks(path, size = (256, 256)):
    # Path to the directory containing masks in your Google Drive
    masks_dir = path
    
    # List all mask filenames in the directory
    mask_filenames = os.listdir(masks_dir)
    
    # Create an empty list to store the masks
    masks = []
    
    # Load masks as NumPy arrays and store them in the list
    for filename in tqdm(mask_filenames, desc="Generating masks numpy arrays"):
        mask_path = os.path.join(masks_dir, filename)
        if os.path.isfile(mask_path):  # Check if it's a file, not a directory (to avoid the notebook checkpoints)
            mask = Image.open(mask_path)
        
            # Resize masks to the same size as images
            target_size = size
            mask = mask.resize(target_size, Image.LANCZOS)
        
            # Convert mask to array
            mask_array = np.array(mask)
        
            masks.append(mask_array)
    
    # Convert the list of arrays to a single NumPy array
    masks = np.array(masks)
    np.save("data/masks.npy", masks)
    return masks
    
