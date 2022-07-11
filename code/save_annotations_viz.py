# Some basic setup:
# Setup detectron2 logger
import detectron2
from detectron2.utils.logger import setup_logger
setup_logger()
from sklearn.model_selection import train_test_split

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
# train_img_dir = os.path.join(ROOT_DIR ,"processed/train")
# register_coco_instances("coco_val", {}, train_json_file, train_img_dir)
# coco_metadata = MetadataCatalog.get("coco_val")
# dataset_dicts = DatasetCatalog.get("coco_val")
#
# c = 0
# for d in dataset_dicts:
#     img = cv2.imread(d["file_name"])
#     visualizer = Visualizer(img[:, :, ::-1], metadata=coco_metadata, scale=1.5)
#     vis = visualizer.draw_dataset_dict(d)
#     img_id = d['file_name'].split('/')[-1]
#     cv2.imwrite('../data_participants/coco/annot_data/'+img_id, vis.get_image()[:, :, ::-1])
#     print("Done: ", c+1)
#     c += 1

with open(train_json_file) as f:
    data = json.load(f)

train_data = {}
val_data = {}

train_data['info'] = data['info']
train_data['categories'] = data['categories']
train_data['licenses'] = data['licenses']
train_data['type'] = 'instances'
train_data['images'] = []
train_data['annotations'] = []

val_data['info'] = data['info']
val_data['categories'] = data['categories']
val_data['licenses'] = data['licenses']
val_data['type'] = 'instances'
val_data['images'] = []
val_data['annotations'] = []

images = random.sample(data['images'], len(data['images']))
train_img = images[0:900]
train_data['images'] = train_img
train_ann = []
val_ann = []
for img in train_img:
    i = img['id']
    ann = [j for j in data['annotations'] if i == j['image_id']]
    train_ann.append(ann)

train_data['annotations'] = train_ann
val_img = images[900:]
val_data['images'] = val_img
for img in val_img:
    i = img['id']
    ann = [j for j in data['annotations'] if i == j['image_id']]
    val_ann.append(ann)

val_data['annotations'] = val_ann


with open(ROOT_DIR+'/coco/annotations_train.json', "w") as f:
    json.dump(train_data, f)

with open(ROOT_DIR+'/coco/annotations_val.json', "w") as f:
    json.dump(val_data, f)