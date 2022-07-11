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

ROOT_DIR = "../data_participants"
train_json_file = os.path.join(ROOT_DIR,"coco/annotations.json")
train_img_dir = os.path.join(ROOT_DIR ,"processed/train")
register_coco_instances("coco_val", {}, train_json_file, train_img_dir)
coco_metadata = MetadataCatalog.get("coco_val")
dataset_dicts = DatasetCatalog.get("coco_val")

c = 0
for d in dataset_dicts:
    img = cv2.imread(d["file_name"])
    visualizer = Visualizer(img[:, :, ::-1], metadata=coco_metadata, scale=1.5)
    vis = visualizer.draw_dataset_dict(d)
    img_id = d['file_name'].split('/')[-1]
    cv2.imwrite('../data_participants/coco/annot_data/'+img_id, vis.get_image()[:, :, ::-1])
    print("Done: ", c+1)
    c += 1
