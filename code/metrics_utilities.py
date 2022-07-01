# Imports
import os
import numpy as np
import copy
import json
from PIL import Image

# Acknowledgements
# Kudos to: https://github.com/matterport/Mask_RCNN/issues/2440
# Kudos to: https://kharshit.github.io/blog/2019/09/20/evaluation-metrics-for-object-detection-and-segmentation


# IoU (Intersection over Union) for Segmentation Masks
# Function: Computes IoU (Intersection over Union) score for two binary masks
def IOU_mask(predicted_mask, groundtruth_mask, iou_type=2):

    # Note: Both predicted_masks and groundtruth_masks are NumPy arrays

    assert iou_type in (1, 2), f"Parameter iou_type must be either 1 or 2; {iou_type} is not a valid value."


    # First, we must be sure that we have predicted masks:
    if len(predicted_mask) == 0:
        return 0
    
    # If yes, we may compute the IoU
    else:
        
        # Create copy of the masks for further processing
        mask1 = predicted_mask.copy()
        mask2 = groundtruth_mask.copy()
        

        # Compute IoU
        if iou_type == 1:
            intersection = np.sum((mask1 + mask2) > 1)
            union = np.sum((mask1 + mask2) > 0)
            iou_score = intersection / float(union)
            # print(f"IoU Type 1 : {iou_score}.")
        
        elif iou_type == 2:
            intersection = np.logical_and(mask1, mask2)
            union = np.logical_or(mask1, mask2)
            iou_score = np.sum(intersection) / np.sum(union)
            # print(f"IoU Type 2 : {iou_score}.")


        return iou_score



# Function: Compute Boundin-Boxes AP
# Adapted from: https://towardsdatascience.com/breaking-down-mean-average-precision-map-ae462f623a52#1a59
def compute_masks_AP(predictions_data, predictions_dir, groundtruth_data, groundtruth_dir, iou_level=0.5):
    
    # Create two lists for precision and recall values
    precision = list()
    recall = list()
    
    # Copy the labels dict since we are going to edit it
    groundtruth_data = copy.deepcopy(groundtruth_data)
    
    # Initialize true positives (TP) and false positives (FP) to zero
    TP = FP = 0
    
    # The total number of positives = (TP + FN) which is constant if we fix labels_dict
    total_positives = sum([len(groundtruth_data[img_fname]['masks']) for img_fname in groundtruth_data.keys()])
    # print(f"Total Positives: {total_positives}")

    
    # Go through predictions
    for image_fname in predictions_data.keys():
        for mask in predictions_data[image_fname]['masks']:
        
            # Check if we have a prediction for all test images
            if image_fname not in groundtruth_data.keys():
                FP += 1
                continue


            # Check if we have possible matches
            possible_matches = groundtruth_data[image_fname]['masks']
            if len(possible_matches) == 0:
                FP += 1
                continue        


            # Compute IoU for all the bounding-boxes
            ious_elems = list()
            # ious_elems = [(IOU_mask(mask, x), x) for x in possible_matches]
            for x_idx, x in enumerate(possible_matches):
                
                # Open predicted mask
                mask_ = Image.open(os.path.join(predictions_dir, image_fname.split('.')[0], mask)).convert('L')
                mask_ = np.asarray(mask_.copy(), dtype=np.uint8)
                # print(f"Mask, Max {mask_.max()} Min {mask_.min()}")

                # Open ground-truth mask
                x_ = Image.open(os.path.join(groundtruth_dir, image_fname.split('.')[0], x)).convert('L')
                x_ = np.asarray(x_.copy(), dtype=np.uint8)
                # print(f"X, Max {x_.max()} Min {x_.min()}")

                # Add IoU to ious_elems
                ious_elems.append((IOU_mask(mask_, x_), x, x_idx))
                
            
            
            # Sort IoUs by value
            ious_elems = sorted(ious_elems, key=lambda x:x[0], reverse=True)


            # Select the top_match
            iou, elem, elem_idx = ious_elems[0]
            
            # Check the IoU threshold
            if iou >= iou_level:
                TP += 1
                
                # Remove the used element, so we don't repeat it in next iteration
                # groundtruth_data[image_fname]['masks'].remove(elem)
                groundtruth_data[image_fname]['masks'].pop(elem_idx)
                precision.append(TP/(TP+FP))
                recall.append(TP/total_positives)
            
            else:
                FP += 1
    

    # Max to the right
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



# IoU (Intersection over Union) for Bounding Boxes
# Function: Compute Intersection over Union (IoU)
def IOU(box1, box2):
    # Coordinates for the intersection box (if it exists)
    xmin_inter = max(box1[0], box2[0])
    ymin_inter = max(box1[1], box2[1])
    xmax_inter = min(box1[2], box2[2])
    ymax_inter = min(box1[3], box2[3])

    # Calculate area of intersection rectangle
    inter_area = max(0, xmax_inter - xmin_inter + 1) * max(0, ymax_inter - ymin_inter + 1)
 
    # Calculate area of actual and predicted boxes
    area1 = (box1[2] - box1[0] + 1) * (box1[3] - box1[1] + 1)
    area2 = (box2[2] - box2[0] + 1) * (box2[3] - box2[1] + 1)
 
    # Computing intersection over union
    iou = inter_area / float(area1 + area2 - inter_area)
 
    # Return the intersection over union value
    return iou



# Function: Compute Bounding-Boxes AP
# Source: https://towardsdatascience.com/breaking-down-mean-average-precision-map-ae462f623a52#1a59
def compute_bboxes_AP(predictions_data, groundtruth_data, iou_level=0.5):
    
    # Create two lists for precision and recall values
    precision = list()
    recall = list()
    
    # Copy the groundtruth_data since we gonna edit it
    groundtruth_data = copy.deepcopy(groundtruth_data)
    
    # Initialize true positives (TP) and false positives (FP) to zero
    TP = FP = 0
    
    # The total number of positives = (TP + FN) which is constant if we fix labels_dict
    total_positives = sum([len(groundtruth_data[img_fname]['boxes']) for img_fname in groundtruth_data.keys()])
    # print(f"Total Positives: {total_positives}")

    
    # Go through predictions
    for image_fname in predictions_data.keys():
        for box in predictions_data[image_fname]['boxes']:
        
            # Check if we have a prediction for all test images
            if image_fname not in groundtruth_data.keys():
                FP += 1
                continue


            # Check if we have possible matches
            possible_matches = groundtruth_data[image_fname]['boxes']
            if len(possible_matches) == 0:
                FP += 1
                continue        


            # Compute IoU for all the bounding-boxes
            ious_elems = [(IOU(box, x), x) for x in possible_matches]
            ious_elems = sorted(ious_elems, key=lambda x:x[0], reverse=True)


            # Select the top_match
            iou, elem = ious_elems[0]
            
            # Check the IoU threshold
            if iou >= iou_level:
                TP += 1
                
                # Removed the used element so we don't used it again
                groundtruth_data[image_fname]['boxes'].remove(elem)
                
                precision.append(TP/(TP+FP))
                recall.append(TP/total_positives)
            
            else:
                FP += 1
    

    # Max to the right
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
def compute_mAP_metrics(predictions_data, predictions_dir, groundtruth_data, groundtruth_dir, iou_range=np.arange(0.5, 1.0, 0.05)):
    
    # Get bounding-boxes APs and mAP
    bboxes_APs = [compute_bboxes_AP(predictions_data, groundtruth_data, iou) for iou in iou_range]
    bboxes_mAP = np.mean(bboxes_APs)

    # Get masks APs and mAP
    masks_APs = [compute_masks_AP(predictions_data, predictions_dir, groundtruth_data, groundtruth_dir, iou) for iou in iou_range]
    masks_mAP = np.mean(masks_APs)
    
    return bboxes_mAP, bboxes_APs, masks_mAP, masks_APs



# Function: Compute mAP from files
def compute_mAP_metrics_from_files(groundtruth_json, groundtruth_dir, predictions_json, predictions_dir):
    
    # Convert labels to the right format (dict of lists) [image_fname] -> [box1, box2, ...]
    # Open train JSON file
    with open(groundtruth_json, 'r') as j:

        # Load JSON contents
        groundtruth_data = json.loads(j.read())

    
    # Convert predictions to the right format (list) [image_fname, [x, y, x+w, y+h], score]
    with open(predictions_json, 'r') as j:
        
        # Load JSON contents
        predictions_data = json.loads(j.read())
    
    
    return compute_mAP_metrics(predictions_data=predictions_data, predictions_dir=predictions_dir, groundtruth_data=groundtruth_data, groundtruth_dir=groundtruth_dir)



# Function: Compute VISUM 2022 Competition Metric
def visum2022score(bboxes_mAP, masks_mAP, bboxes_mAP_weight=0.5):

    # Compute masks_mAP_weight from bboxes_mAP_weight
    masks_mAP_weight = 1-bboxes_mAP_weight

    # Compute score, i.e., score = 0.5*bboxes_mAP + 0.5*masks_mAP
    score = (bboxes_mAP_weight * bboxes_mAP) + (masks_mAP_weight * masks_mAP)

    return score



# Run this to test file
if __name__=="__main__":

    # IoU Range
    iou_range = np.arange(0.5, 1, 0.05)
    
    # Directories
    # groundtruth_data = predictions_data = os.path.join("data", "json", "challenge", "test_challenge.json")
    groundtruth_data = predictions_data = os.path.join("results", "predictions", "predictions.json")
    
    # gt_masks_dir = pred_masks_dir = os.path.join("data", "masks", "test")
    gt_masks_dir = pred_masks_dir = os.path.join("results", "predictions", "masks")

    # Bounding-boxes AP
    bboxes_mAP, bboxes_APs, masks_mAP, masks_APs = compute_mAP_metrics_from_files(groundtruth_data, gt_masks_dir, predictions_data, pred_masks_dir)
    print("bboxes_mAP:{:.4f}".format(bboxes_mAP))
    for ap_metric, iou in zip(bboxes_APs, iou_range):
        print("\tAP at IoU level [{:.2f}]: {:.4f}".format(iou, ap_metric))


    # Masks AP
    print("masks_mAP:{:.4f}".format(masks_mAP))
    for ap_metric, iou in zip(masks_APs, iou_range):
        print("\tAP at IoU level [{:.2f}]: {:.4f}".format(iou, ap_metric))

    
    # Print VISUM2022 Score
    visum_score = visum2022score(bboxes_mAP=bboxes_mAP, masks_mAP=masks_mAP)
    print(f"VISUM2022 Score: {visum_score}")
