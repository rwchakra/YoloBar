# Imports
import os
import numpy as np
import pandas as pd
import copy
import csv
import sys



# Function: Compute Intersection over Union (IoU)
def IOU(box1, box2):
    # coordinates for the intersection box (if it exists)
    xmin_inter = max(box1[0], box2[0])
    ymin_inter = max(box1[1], box2[1])
    xmax_inter = min(box1[2], box2[2])
    ymax_inter = min(box1[3], box2[3])

    # calculate area of intersection rectangle
    inter_area = max(0, xmax_inter - xmin_inter + 1) * max(0, ymax_inter - ymin_inter + 1)
 
    # calculate area of actual and predicted boxes
    area1 = (box1[2] - box1[0] + 1) * (box1[3] - box1[1] + 1)
    area2 = (box2[2] - box2[0] + 1) * (box2[3] - box2[1] + 1)
 
    # computing intersection over union
    iou = inter_area / float(area1 + area2 - inter_area)
 
    # return the intersection over union value
    return iou



# Function: Compute AP, from https://towardsdatascience.com/breaking-down-mean-average-precision-map-ae462f623a52#1a59
def compute_AP(preds_list, labels_dict, iou_level=0.5):
    precision = list()
    recall = list()
    # copy the labels dict since we gonna edit it
    labels_dict = copy.deepcopy(labels_dict)
    # Initialize True positives and False positives to zero
    TP = FP = 0
    # The total number of positives = (TP + FN) which is constant if we fix labels_dict
    Total_positives = sum([len(x[1]) for x in labels_dict.items()])
    # sort predictions by their score
    preds_list = sorted(preds_list, key=lambda x:x[3], reverse=True)
    for seq, frame, box, score in preds_list:
        if (seq, frame) not in labels_dict.keys():
            FP += 1
            continue

        possible_matches = labels_dict[seq, frame]
        if len(possible_matches) == 0:
            FP+=1
            continue        

        ious_elems = [(IOU(box, x), x) for x in possible_matches]
        ious_elems = sorted(ious_elems, key=lambda x:x[0], reverse=True)
        #top_match
        iou, elem = ious_elems[0]
        if iou >= iou_level:
            TP += 1
            labels_dict[seq, frame].remove(elem)
            precision.append(TP/(TP+FP))
            recall.append(TP/Total_positives)
        else:
            FP += 1
    
    # max to the right
    max_to_the_right = 0
    for i, x in enumerate(precision[::-1]):
        if x < max_to_the_right:
            precision[-i-1] = max_to_the_right
        elif x > max_to_the_right:
            max_to_the_right = x

    # integrate rectangles to obtain Average Precision
    prev_recall = 0
    area = 0
    for p, r in zip(precision, recall):
        area += p * (r - prev_recall)
        prev_recall = r
    return area



# Function: Compute mAP
def compute_mAP(preds, labels):
    labels_dict = dict()

    for row in labels:
        if len(row) == 2:
            continue
        labels_dict[int(row[0]), int(row[1])] = row[2]
    for p in preds:
        p[0] = int(p[0])  # seq
        p[1] = int(p[1])  # frame
    # return mAP
    APs = [compute_AP(preds, labels_dict, iou) for iou in np.arange(0.5, 1.0, 0.05)]
    return np.mean(APs), APs



# Function: Compute mAP from files
def compute_mAP_from_files(preds_file, labels_file):
    labels = list()
    # convert labels to the right format (dict of lists) [seq, frame] -> [box1, box2, ...]
    with open(labels_file, newline='') as csvfile:
        reader = csv.reader(csvfile, delimiter=';')
        for i, row in enumerate(reader):
            if i == 0 or len(row) == 2:
                continue
            labels.append([int(row[0]), int(row[1]), eval(row[2])])

    preds = list()
    # conver predictions to the right format (list) [seq, frame, [x, y, x+w, y+h], score]
    with open(preds_file, newline='') as csvfile:
        reader = csv.reader(csvfile, delimiter=';')
        for i, row in enumerate(reader):
            if row == "seq;frame;label;score".split(";"):
                continue
            preds.append([int(row[0]), int(row[1]), eval(row[2]), float(row[3])])
    return compute_mAP(preds, labels)



# Run this to test file
if __name__=="__main__":
    mAP, AP = compute_mAP_from_files("predictions.csv", "/home/master/dataset/test/labels.csv")
    print("mAP:{:.4f}".format(mAP))
    for ap_metric, iou in zip(AP, np.arange(0.5, 1, 0.05)):
        print("\tAP at IoU level [{:.2f}]: {:.4f}".format(iou, ap_metric))
