# Imports
import os
import numpy as np
import copy
import json

# Project Imports
from metrics_utilities import compute_mAP_from_files


# Directories and File Paths
DATA_DIR = "data"
PREDICTIONS_DIR = "predictions"

# JSON Files
GROUNDTRUTH_JSON_PATH = os.path.join(DATA_DIR, "json", "challenge", "test_challenge.json")
PREDICTIONS_JSON_PATH = os.path.join("results", PREDICTIONS_DIR, "predictions.json")


# Compute Evaluation metrics
mAP, AP = compute_mAP_from_files(groundtruth_json=GROUNDTRUTH_JSON_PATH, predictions_json=PREDICTIONS_JSON_PATH)
# mAP, AP = compute_mAP_from_files(groundtruth_json=GROUNDTRUTH_JSON_PATH, predictions_json=GROUNDTRUTH_JSON_PATH)
# mAP, AP = compute_mAP_from_files(groundtruth_json=PREDICTIONS_JSON_PATH, predictions_json=PREDICTIONS_JSON_PATH)
print("mAP:{:.4f}".format(mAP))
for ap_metric, iou in zip(AP, np.arange(0.5, 1, 0.05)):
    print("\tAP at IoU level [{:.2f}]: {:.4f}".format(iou, ap_metric))
