import os
import sys
import json
import glob
from typing import Dict, Any
from datetime import datetime
from PIL import Image
import cv2
import matplotlib.pyplot as plt
import numpy as np
import random

def read_train_json_file(json_file_path):
# Load the json
    print('Loading json file...')
    with open(json_file_path) as json_file:
        train_json = json.load(json_file)
    return train_json

MASK_ROOT_PATH = "../data_participants/masks/train/"
IMG_ROOT_PATH = "../data_participants/processed/train/"
TRAIN_JSON_PATH= "../data_participants/challenge/train_challenge.json"

train_json_dict = read_train_json_file(TRAIN_JSON_PATH)
# Create a list with all the images' filenames
image_keys = list(train_json_dict.keys())

random.shuffle(image_keys)
image_keys_val = image_keys[:299]
now = datetime.now()

data = dict(
        info=dict(
            description=None,
            url=None,
            version=None,
            year=now.year,
            contributor=None,
            date_created=now.strftime("%Y-%m-%d %H:%M:%S.%f"),
        ),
        licenses=[dict(url=None, id=0, name=None,)],
        images=[
            # license, url, file_name, height, width, date_captured, id
        ],
        type="instances",
        annotations=[
            # segmentation, area, iscrowd, image_id, bbox, category_id, id
        ],
        categories=[
            # supercategory, id, name
        ],
    )

data["categories"].append(
            dict(supercategory=None, id=1, name='barcode',)
        )

# Function: Convert bounding box to COCO notation
def convert_bbox_to_coco(bbox, reverse=False):

    if not reverse:
        # Our notation has the format [x, y, x+w, y+h]
        # In COCO, the notation has the format [x_min, y_min, width, height]
        x_min, y_min, width, height = bbox[0], bbox[1], (
            bbox[2] - bbox[0]), (bbox[3] - bbox[1])

        # We then create a list with these entries
        converted_bbox = [x_min, y_min, width, height]

    else:
        # We assume we receive the data in the COCO format
        # The notation has the format [x_min, y_min, width, height]
        x_min, y_min, width, height = bbox[0], bbox[1], bbox[2], bbox[3]

        # We then convert it to our notation [x, y, x+w, y+h]
        converted_bbox = [x_min, y_min, x_min + width, y_min + height]

    return converted_bbox

for image_id, image_key in enumerate(image_keys[299:]):
    tmp_train_anno_dict = train_json_dict[image_key]
    tmp_labels = tmp_train_anno_dict['labels']
    tmp_boxes = tmp_train_anno_dict['boxes']
    tmp_masks = tmp_train_anno_dict['masks']

    # We have to convert all the bounding boxes to COCO notation before augmentation
    coco_bboxes = [convert_bbox_to_coco(b) for b in tmp_boxes]
    # print(tmp_labels,tmp_boxes,coco_bboxes,tmp_masks)
    img_base = image_key.split('.')[0]
    for label, bbox, mask_name in zip(tmp_labels, coco_bboxes, tmp_masks):
        # Open the image and (to be sure) we convert it to RGB

        tmp_mask_path = os.path.join(MASK_ROOT_PATH + img_base, mask_name)
        mask_img_open = cv2.imread(tmp_mask_path, 0)
        mask_img = (mask_img_open > int(mask_img_open.max() / 2)).astype(np.uint8)
        img_height = mask_img_open.shape[0]
        img_width = mask_img_open.shape[1]
        contours, _ = cv2.findContours(mask_img, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
        polygons = []
        for object in contours:
            coords = []

            for point in object:
                coords.append(int(point[0][0]))
                coords.append(int(point[0][1]))
            c_area = cv2.contourArea(object)
            polygons.append(coords)
        print(len(polygons))
        data["annotations"].append(
            dict(
                id=len(data["annotations"]),
                image_id=image_id,
                category_id=1,
                segmentation=polygons,
                area=c_area,
                bbox=bbox,
                iscrowd=0,
            )
        )

    data["images"].append(
        dict(
            license=0,
            url=None,
            file_name=image_key,
            height=img_height,
            width=img_width,
            date_captured=None,
            id=image_id,
        )
    )

JSON_SAVE_PATH = "../data_participants/coco/"
os.makedirs(JSON_SAVE_PATH,exist_ok=True)
out_ann_file = os.path.join(JSON_SAVE_PATH,"train_annotations.json")
with open(out_ann_file, "w") as f:
        json.dump(data, f)

now = datetime.now()

data_val= dict(
        info=dict(
            description=None,
            url=None,
            version=None,
            year=now.year,
            contributor=None,
            date_created=now.strftime("%Y-%m-%d %H:%M:%S.%f"),
        ),
        licenses=[dict(url=None, id=0, name=None,)],
        images=[
            # license, url, file_name, height, width, date_captured, id
        ],
        type="instances",
        annotations=[
            # segmentation, area, iscrowd, image_id, bbox, category_id, id
        ],
        categories=[
            # supercategory, id, name
        ],
    )

data_val["categories"].append(
            dict(supercategory=None, id=1, name='barcode',)
        )

for image_id, image_key in enumerate(image_keys_val):
    tmp_train_anno_dict = train_json_dict[image_key]
    tmp_labels = tmp_train_anno_dict['labels']
    tmp_boxes = tmp_train_anno_dict['boxes']
    tmp_masks = tmp_train_anno_dict['masks']

    # We have to convert all the bounding boxes to COCO notation before augmentation
    coco_bboxes = [convert_bbox_to_coco(b) for b in tmp_boxes]
    # print(tmp_labels,tmp_boxes,coco_bboxes,tmp_masks)
    img_base = image_key.split('.')[0]
    for label, bbox, mask_name in zip(tmp_labels, coco_bboxes, tmp_masks):
        # Open the image and (to be sure) we convert it to RGB

        tmp_mask_path = os.path.join(MASK_ROOT_PATH + img_base, mask_name)
        mask_img_open = cv2.imread(tmp_mask_path, 0)
        mask_img = (mask_img_open > int(mask_img_open.max() / 2)).astype(np.uint8)
        img_height = mask_img_open.shape[0]
        img_width = mask_img_open.shape[1]
        contours, _ = cv2.findContours(mask_img, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
        polygons = []
        for object in contours:
            coords = []

            for point in object:
                coords.append(int(point[0][0]))
                coords.append(int(point[0][1]))
            c_area = cv2.contourArea(object)
            polygons.append(coords)
        # print(len(polygons))
        data_val["annotations"].append(
            dict(
                id=len(data_val["annotations"]),
                image_id=image_id,
                category_id=1,
                segmentation=polygons,
                area=c_area,
                bbox=bbox,
                iscrowd=0,
            )
        )

    data_val["images"].append(
        dict(
            license=0,
            url=None,
            file_name=image_key,
            height=img_height,
            width=img_width,
            date_captured=None,
            id=image_id,
        )
    )

out_ann_file = os.path.join(JSON_SAVE_PATH,"val_annotations.json")
with open(out_ann_file, "w") as f:
        json.dump(data_val, f)