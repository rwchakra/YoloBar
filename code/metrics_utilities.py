# Imports
import os
import numpy as np
import copy
import json


# TODO: Add IoU for Segmentation Masks
# Check: https://github.com/matterport/Mask_RCNN/issues/2440



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
    
    # Create two lists for precision and recall values
    precision = list()
    recall = list()
    
    # Copy the labels dict since we gonna edit it
    labels_dict = copy.deepcopy(labels_dict)
    
    # Initialize true positives (TP) and false positives (FP) to zero
    TP = FP = 0
    
    # The total number of positives = (TP + FN) which is constant if we fix labels_dict
    total_positives = sum([len(x[1]) for x in labels_dict.items()])
    

    # Sort predictions by their score
    preds_list = sorted(preds_list, key=lambda x:x[2], reverse=True)
    
    # Go through predictions
    for image_fname, box, score in preds_list:
        
        # Check if we have a prediction for all test images
        if image_fname not in labels_dict.keys():
            FP += 1
            continue


        # Check if we have possible matches
        possible_matches = labels_dict[image_fname]
        if len(possible_matches) == 0:
            FP+=1
            continue        


        # Compute IoU for all the boxes
        ious_elems = [(IOU(box, x), x) for x in possible_matches]
        ious_elems = sorted(ious_elems, key=lambda x:x[0], reverse=True)
        
        # Select the top_match
        iou, elem = ious_elems[0]
        
        # Check the IoU threshold
        if iou >= iou_level:
            TP += 1
            labels_dict[image_fname].remove(elem)
            precision.append(TP/(TP+FP))
            recall.append(TP/total_positives)
        
        else:
            FP += 1
    

    # TODO: (Study this!) Max to the right
    max_to_the_right = 0
    for i, x in enumerate(precision[::-1]):
        if x < max_to_the_right:
            precision[-i-1] = max_to_the_right
        elif x > max_to_the_right:
            max_to_the_right = x


    # Integrate rectangles to obtain Average Precision
    prev_recall = 0
    area = 0
    
    for p, r in zip(precision, recall):
        area += p * (r - prev_recall)
        prev_recall = r
    

    return area



# Function: Compute mAP
def compute_mAP(preds, labels):
    labels_dict = dict()
    # print(preds)
    # print(labels)

    # Create labels_dict {image_fname:boxes}
    for row in labels:
        labels_dict[row[0]] = row[1]
    
    # return mAP
    APs = [compute_AP(preds, labels_dict, iou) for iou in np.arange(0.5, 1.0, 0.05)]
    
    return np.mean(APs), APs



# Function: Compute mAP from files
def compute_mAP_from_files(preds_file, labels_file):
    
    # Convert labels
    labels = list()
    
    # Convert labels to the right format (dict of lists) [image_fname] -> [box1, box2, ...]
    # Open train JSON file
    with open(labels_file, 'r') as j:

        # Load JSON contents
        json_data = json.loads(j.read())
            
    for key, value in json_data.items():
        # TODO: Define a label system based on JSON, that will be used in the competition
        labels.append([key, [value[0]]])


    # Convert prediction
    preds = list()
    
    # Convert predictions to the right format (list) [image_fname, [x, y, x+w, y+h], score]
    with open(preds_file, 'r') as j:
        # Load JSON contents
        json_data = json.loads(j.read())
            
    for key, value in json_data.items():
        preds.append([key, value[0], value[1]])
    
    
    return compute_mAP(preds, labels)



# Run this to test file
if __name__=="__main__":
    mAP, AP = compute_mAP_from_files(os.path.join("results", "predictions", "predictions.json"), os.path.join("results", "predictions", "predictions.json"))
    print("mAP:{:.4f}".format(mAP))
    for ap_metric, iou in zip(AP, np.arange(0.5, 1, 0.05)):
        print("\tAP at IoU level [{:.2f}]: {:.4f}".format(iou, ap_metric))
