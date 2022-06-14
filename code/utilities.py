# Imports
import os
import numpy as np 
import cv2



# Function: Convert Labelbox coordinates into format [ymin, xmin, ymax, xmax]
def convert_lbox_coords(coords_list):

    # Create xx and yy temporary lists
    xx = list()
    yy = list()


    # Go through the list
    for point in coords_list:
        xx.append(point['x'])
        yy.append(point['y'])
    

    # Convert into new format
    ymin, ymax = min(yy), max(yy)
    xmin, xmax = min(xx), max(xx)


    return ymin, xmin, ymax, xmax



# Function: Blur sensitive area
def apply_blurring(img_array, kernel_size=(15, 15)):

    # Randomly get the blurring type
    blur_type = np.random.choice(["average", "gaussian", "median"])

    if blur_type == "average":
        blurred = cv2.blur(img_array, kernel_size)
    
    elif blur_type == "gaussian":
        blurred = cv2.GaussianBlur(img_array, kernel_size, 0)
    
    elif blur_type == "median":
        k = kernel_size[0]
        blurred = cv2.medianBlur(img_array, k)


    return blurred
