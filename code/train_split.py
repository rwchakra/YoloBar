# Some basic setup:
# Setup detectron2 logger
import detectron2
from detectron2.utils.logger import setup_logger
setup_logger()

# import some common libraries
import numpy as np
import matplotlib.pyplot as plt
import os, json, cv2, random
#from google.colab.patches import cv2_imshow

# import some common detectron2 utilities
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog, DatasetCatalog
from detectron2.data.datasets import register_coco_instances

ROOT_DIR = "../data_participants/coco/"
# ANNO_PATH = "/home/sinan/datasets/instance_segm/train_coco_20210212"
train_json_file = os.path.join(ROOT_DIR ,"train_annotations.json")
train_img_dir = os.path.join(ROOT_DIR ,"processed/train")
val_json_file = os.path.join(ROOT_DIR ,"val_annotations.json")

register_coco_instances("ct", {}, train_json_file, train_img_dir)
coco_metadata_train = MetadataCatalog.get("ct")
dataset_train = DatasetCatalog.get("ct")

register_coco_instances("coco_val", {}, val_json_file, train_img_dir)
coco_metadata_val = MetadataCatalog.get("coco_val")
dataset_dicts_val = DatasetCatalog.get("coco_val")

print("bla")