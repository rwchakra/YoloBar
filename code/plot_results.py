# Imports
import os 
import json 
import tqdm
import numpy as np 
from PIL import Image

# PyTorch Imports
import torch 

# Project Imports
from data_utilities import draw_results



# Directories
data_dir = "data"
json_dir = os.path.join(data_dir, "json", "challenge")
images_dir = os.path.join(data_dir, "processed", "train")
masks_dir = os.path.join(data_dir, "masks", "train")
results_vis_dir = os.path.join("results", "visualisations", "train")
if not os.path.isdir(results_vis_dir):
    os.makedirs(results_vis_dir)


# JSON file
json_file = os.path.join(json_dir, "train_challenge.json")

# Open JSON file
with open(json_file, 'r') as j:

    # Load JSON contents
    json_data = json.loads(j.read())


# Get the images
images = list(json_data.keys())

# Create a label dict with bboxes and masks
label_dict = json_data.copy()

# Iterate through images
for img_fname in tqdm.tqdm(images):

    # Get image data
    pil_image = Image.open(os.path.join(images_dir, img_fname)).convert('RGB')
    npy_image = np.asarray(pil_image)

    # Get bounding boxes data
    bboxes = label_dict[img_fname]['boxes']

    # Get masks data
    masks = label_dict[img_fname]['masks']
    masks = [np.asarray(Image.open(os.path.join(masks_dir, img_fname.split('.')[0], m)).convert("L")) for m in masks]
    masks = [(m > 0).astype(np.uint8) for m in masks]


    # Visualisation of results
    vis_results = draw_results(image=npy_image, masks=masks, bboxes=bboxes)

    # Convert this result to NumPy
    vis_results = vis_results.permute(1, 2, 0)
    vis_results = vis_results.numpy()
    # print(vis_results.max(), vis_results.min(), vis_results.shape)

    # Convert into PIL
    pil_vis_results = Image.fromarray(vis_results)
    pil_vis_results.save(os.path.join(results_vis_dir, img_fname))


print("Finished.")
