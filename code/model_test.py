# Imports
import os
import numpy as np

# PyTorch Imports
import torch
from torch.utils.data import DataLoader

# Project Imports
from data_utilities import get_transform, collate_fn, LoggiPackageDataset
from model_utilities import LoggiBarcodeDetectionModel, evaluate, visum2022score


# Constant variables
IMG_SIZE = 1024
SAVED_MODEL = os.path.join("results", "models", "visum2022.pt")

# The DATA_DIR and PREDICTIONS_DIR are important to validate your submission; do not modify these variables
DATA_DIR = "data"
PREDICTIONS_DIR = "predictions"
DEVICE = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

# Create test set and test loader with transforms
test_transforms = get_transform(data_augment=False, img_size=IMG_SIZE)
test_set = LoggiPackageDataset(data_dir=DATA_DIR, training=False, transforms=test_transforms)
test_loader = DataLoader(test_set, batch_size=4, num_workers=1, shuffle=False, collate_fn=collate_fn)


# Load model
model = LoggiBarcodeDetectionModel(min_img_size=IMG_SIZE, max_img_size=IMG_SIZE)
checkpoint = torch.load(SAVED_MODEL, map_location=DEVICE)
model.load_state_dict(checkpoint['model_state_dict'])
model.to(DEVICE)


# Get all the metric results on test set
eval_results = evaluate(model, test_loader, DEVICE)

# Get the bounding-boxes results (for VISUM2022 Score)
bbox_results = eval_results.coco_eval['bbox']
bbox_map = bbox_results.stats[0]

# Get the segmentation results (for VISUM2022 Score)
segm_results = eval_results.coco_eval['segm']
segm_map = segm_results.stats[0]

# Compute the VISUM2022 Score
visum_score = visum2022score(bbox_map, segm_map)

# Print mAP values
print(f"Detection mAP: {np.round(bbox_map, 4)}")
print(f"Segmentation mAP: {np.round(segm_map, 4)}")
print(f"VISUM Score: {np.round(visum_score, 4)}")


# Save visum_score into a metric.txt file in the PREDICTIONS_DIR
with open(os.path.join(PREDICTIONS_DIR, "metric.txt"), "w") as m:
    m.write(f"{visum_score}")
