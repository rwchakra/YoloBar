# Imports
import os
import json
import tqdm

# PyTorch Imports
import torch
from torch.utils.data import DataLoader
from torchvision.ops import nms

# Project Imports
from data_utilities import get_transform, LoggiPackageDataset
from model_utilities import LoggiBarcodeDetectionModel



# Constant variables
NMS_THRESHOLD = 0.1
SAVED_MODEL = os.path.join("results", "models", "visum2022.pt")
DATA_DIR = "data"
DEVICE = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')


# Create test set with transforms
test_transforms = get_transform(training=False, data_augment=False)
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

    
    with torch.no_grad():
        prediction = model(image.to(DEVICE), target)

    boxes = prediction[0]['boxes'].cpu()
    scores = prediction[0]['scores'].cpu()

    nms_indices = nms(boxes, scores, NMS_THRESHOLD)

    nms_boxes = boxes[nms_indices].tolist()
    nms_scores = scores[nms_indices].tolist()
        
    # if there are no detections there is no need to include that entry in the predictions
    if len(nms_boxes) > 0:
        for bb, score in zip(nms_boxes, nms_scores):
            # predictions.append([image_fname[0], list(bb), score])
            predictions[image_fname[0]] = [list(bb), score]



# Save this into a JSON of predictions
json_object = json.dumps(predictions, indent=4)

# Writing to good_quality_imgs.json
if not os.path.isdir(os.path.join("results", "predictions")):
    os.makedirs(os.path.join("results", "predictions"))

with open(os.path.join("results", "predictions", "predictions.json"), "w") as j:
    j.write(json_object)
