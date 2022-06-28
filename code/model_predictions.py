# Imports
import os
import argparse
import json
import tqdm
import numpy as np
from PIL import Image

# PyTorch Imports
import torch
from torch.utils.data import DataLoader
from torchvision.ops import nms

# Project Imports
from data_utilities import get_transform, LoggiPackageDataset
from model_utilities import LoggiBarcodeDetectionModel



# CLI Arguments
parser = argparse.ArgumentParser(description = 'description')
parser.add_argument('--img_size', type = int, default = 1024, help="new size for img resize transform.")
args = parser.parse_args()


# Constant variables
NMS_THRESHOLD = 0.1
SAVED_MODEL = os.path.join("results", "models", "visum2022.pt")
DATA_DIR = "data"
DEVICE = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')


# Directories
if not os.path.isdir(os.path.join("results", "predictions")):
    os.makedirs(os.path.join("results", "predictions"))


# Create test set with transforms
test_transforms = get_transform(training=False, data_augment=False, img_size=args.img_size)
test_set = LoggiPackageDataset(data_dir="data", training=False, transforms=test_transforms)
test_loader = DataLoader(test_set, batch_size=1, shuffle=False)


# Load model
model = torch.load(SAVED_MODEL)
model.to(DEVICE)

# Put the model in evaluation mode
model.eval()


# Create dict() to append predictions
predictions = dict()


# Go through test set
for image, target, image_fname in tqdm.tqdm(test_loader):

    # Add a dict to bounding-boxes and scores predictions
    predictions[image_fname[0]] = dict()
    predictions[image_fname[0]]['boxes'] = list()
    predictions[image_fname[0]]['scores'] = list()
    predictions[image_fname[0]]['masks'] = list()


    # Get predictions
    with torch.no_grad():
        prediction = model(image.to(DEVICE), target)

    boxes = prediction[0]['boxes'].cpu()
    scores = prediction[0]['scores'].cpu()
    masks = prediction[0]['masks'].cpu()
    # print(f"Shape of Masks: {masks.shape}")
    # print(f"Length of Masks: {len(masks)}")

    # NMS
    nms_indices = nms(boxes, scores, NMS_THRESHOLD)
    nms_boxes = boxes[nms_indices].tolist()
    nms_scores = scores[nms_indices].tolist()
    nms_masks = masks[nms_indices]


    # If there are no detections there is no need to include that entry in the predictions
    if len(nms_boxes) > 0:
        i = 0
        for bb, score, msk in zip(nms_boxes, nms_scores, nms_masks):
            
            # Bouding-boxes
            # predictions.append([image_fname[0], list(bb), score])
            predictions[image_fname[0]]['boxes'].append(list(bb))
            
            # Scores
            predictions[image_fname[0]]['scores'].append(score)

            # Masks
            msk_fname = f"{i}.jpg"
            predictions[image_fname[0]]['masks'].append(msk_fname)

            # Save masks into directory
            msk_ = np.squeeze(a=msk.detach().cpu().numpy().copy(), axis=0)
            pil_mask = Image.fromarray(msk_).convert("L")

            if not os.path.isdir(os.path.join("results", "predictions", "masks", image_fname[0].split('.')[0])):
                os.makedirs(os.path.join("results", "predictions", "masks", image_fname[0].split('.')[0]))
            
            pil_mask.save(os.path.join("results", "predictions", "masks", image_fname[0].split('.')[0], msk_fname))

            # Update i (idx)
            i += 1


# Save this into a JSON of predictions
json_object = json.dumps(predictions, indent=4)

with open(os.path.join("results", "predictions", "predictions.json"), "w") as j:
    j.write(json_object)
