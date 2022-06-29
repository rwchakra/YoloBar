# Imports
import os
import numpy as np
import copy
import json


# Acknowledgements
# Kudos to: https://github.com/matterport/Mask_RCNN/issues/2440
# Kudos to: https://kharshit.github.io/blog/2019/09/20/evaluation-metrics-for-object-detection-and-segmentation


# IoU (Intersection over Union) for Segmentation Masks
# Function: Merge n masks into a single mask
def merged_mask(masks):
    """
    param masks: NumPy array
    """
    
    # Get the number of predicted masks (the number of elements in the list)
    n_masks = len(masks)

    # Squeze axis=1 (channels) of the masks
    masks_ = np.squeeze(a=masks.copy(), axis=1)
    

    # We must have at least one mask
    if n > 0:        
        
        # Create an array for the final merged mask
        merged_mask = np.zeros((masks.shape[0], masks.shape[1]))
        
        # Iterate through all the masks 
        for i in range(n):
            
            # And add them in the merged mask
            # merged_mask += masks[..., i]
            merged_mask += masks[i, :, :]
        

        # Create uint8 array
        merged_mask = np.asarray(merged_mask, dtype=np.uint8)
        
        return merged_mask
    
    # return masks[:,:,0]



# Function: Computes IoU (Intersection over Union) score for two binary masks
def IOU_mask(pred_masks, gt_masks, iou_type=1):
    """
    param pred_masks: NumPy array
    param gt_masks: NumPy array
    param iou_type: Integer (Note: Results are same for both types.)
    return iou score:
    """

    assert iou_type in (1, 2), f"Parameter iou_type must be either 1 or 2; {iou_type} is not a valid value."

    # First, we must be sure that we have predicted masks:
    if len(pred_maks) == 0:
        return 0
    
    # If yes, we may compute the IoU
    else:

        # Apply the function to merge all the predicted and ground-truth masks
        mask1 = merged_mask(pred_masks)
        mask2 = merged_mask(gt_masks)
        

        # Compute IoU
        if iou_type == 1:
            intersection = np.sum((mask1 + mask2) > 1)
            union = np.sum((mask1 + mask2) > 0)
            iou_score = intersection / float(union)
            print(f"IoU Type 1 : {iou_score}.")
        
        elif iou_type == 2:
            intersection = np.logical_and(mask1, mask2)#*
            union = np.logical_or(mask1, mask2)# +
            iou_score = np.sum(intersection) / np.sum(union)
            print(f"IoU Type 2 : {iou_score}.")
        
        return iou_score



# IoU (Intersection over Union) for Bounding Boxes
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
def compute_AP(predictions_data, groundtruth_data, iou_level=0.5):
    
    # Create two lists for precision and recall values
    precision = list()
    recall = list()
    
    # Copy the labels dict since we gonna edit it
    # labels_dict = copy.deepcopy(labels_dict)
    groundtruth_data = copy.deepcopy(groundtruth_data)
    # predictions_data = copy.deepcopy(predictions_data)
    
    # Initialize true positives (TP) and false positives (FP) to zero
    TP = FP = 0
    
    # The total number of positives = (TP + FN) which is constant if we fix labels_dict
    # TODO: Erase uppon review 
    # Note: {image_fname:boxes}
    # print(f"GT Data Items: {groundtruth_data.items()}")
    # print(f"GT Data Keys: {groundtruth_data.keys()}")
    # total_positives = sum([len(x[1]) for x in labels_dict.items()])
    total_positives = sum([len(groundtruth_data[img_fname]['boxes']) for img_fname in groundtruth_data.keys()])
    # print(f"Total Positives: {total_positives}")
    

    # TODO: Erase uppon review
    # TODO: Sort predictions by their score (no need to do this)
    # preds_list = sorted(preds_list, key=lambda x:x[2], reverse=True)
    
    # Go through predictions
    # for image_fname, box, score in preds_list:
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
                # labels_dict[image_fname].remove(elem)
                groundtruth_data[image_fname]['boxes'].remove(elem)
                # groundtruth_data[image_fname]['boxes'].pop(elem_idx)
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
def compute_mAP(predictions_data, groundtruth_data):
    
    # TODO: Erase uppon review
    # labels_dict = dict()
    # print(preds)
    # print(labels)

    # Create labels_dict {image_fname:boxes}
    # for row in labels:
        # labels_dict[row[0]] = row[1]
    
    # return mAP
    APs = [compute_AP(predictions_data, groundtruth_data, iou) for iou in np.arange(0.5, 1.0, 0.05)]
    
    return np.mean(APs), APs



# Function: Compute mAP from files
def compute_mAP_from_files(groundtruth_json, predictions_json):
    
    # TODO: Erase uppon review
    # Convert labels
    # labels = list()
    
    # Convert labels to the right format (dict of lists) [image_fname] -> [box1, box2, ...]
    # Open train JSON file
    with open(groundtruth_json, 'r') as j:

        # Load JSON contents
        groundtruth_data = json.loads(j.read())
    

    # TODO: Erase uppon review
    # for key, value in json_data.items():
        # TODO: Define a label system based on JSON, that will be used in the competition
        # labels.append([key, [value[0]]])


    # TODO: Erase uppon review
    # Convert prediction
    # preds = list()
    
    # Convert predictions to the right format (list) [image_fname, [x, y, x+w, y+h], score]
    with open(predictions_json, 'r') as j:
        
        # Load JSON contents
        predictions_data = json.loads(j.read())

    # TODO: Erase uppon review   
    # for key, value in json_data.items():
        # preds.append([key, value[0], value[1]])
    
    
    return compute_mAP(predictions_data=predictions_data, groundtruth_data=groundtruth_data)



# Run this to test file
if __name__=="__main__":
    mAP, AP = compute_mAP_from_files(os.path.join("results", "predictions", "predictions.json"), os.path.join("results", "predictions", "predictions.json"))
    print("mAP:{:.4f}".format(mAP))
    for ap_metric, iou in zip(AP, np.arange(0.5, 1, 0.05)):
        print("\tAP at IoU level [{:.2f}]: {:.4f}".format(iou, ap_metric))
