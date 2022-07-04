# Imports
import sys
import os
import numpy as np

# Project Imports
from metrics_utilities import compute_mAP_metrics_from_files, visum2022score


# TODO: Change data paths according to Eduardo's editions
# Directories and File Paths
DATA_DIR = "data"
PREDICTIONS_DIR = "predictions"

# JSON Files
GROUNDTRUTH_JSON_PATH = os.path.join(DATA_DIR, "json", "challenge", "test_challenge.json")
PREDICTIONS_JSON_PATH = os.path.join("results", PREDICTIONS_DIR, "predictions.json")

# Masks Direcotries
GROUNDTRUTH_MASKS_PATH = os.path.join(DATA_DIR, "masks", "test")
PREDICTIONS_MASKS_PATH = os.path.join("results", PREDICTIONS_DIR, "masks")

# IoU Range
IOU_RANGE = np.arange(0.5, 1, 0.05)


# Compute Evaluation metrics
bboxes_mAP, bboxes_APs, masks_mAP, masks_APs = compute_mAP_metrics_from_files(groundtruth_json=GROUNDTRUTH_JSON_PATH, groundtruth_dir=GROUNDTRUTH_MASKS_PATH, predictions_json=PREDICTIONS_JSON_PATH, predictions_dir=PREDICTIONS_MASKS_PATH)

# Print bounding-boxes mAP
print("bboxes_mAP:{:.4f}".format(bboxes_mAP))
for ap_metric, iou in zip(bboxes_APs, IOU_RANGE):
    print("\tAP at IoU level [{:.2f}]: {:.4f}".format(iou, ap_metric))


# Print masks AP
print("masks_mAP:{:.4f}".format(masks_mAP))
for ap_metric, iou in zip(masks_APs, IOU_RANGE):
    print("\tAP at IoU level [{:.2f}]: {:.4f}".format(iou, ap_metric))


# Print VISUM2022 Score
visum_score = visum2022score(bboxes_mAP=bboxes_mAP, masks_mAP=masks_mAP)
print(f"VISUM2022 Score: {visum_score}")
