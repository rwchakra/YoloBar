# Imports
import os
import numpy as np 



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


    # New coordinates
    new_coords = [ymin, xmin, ymax, xmax]


    return new_coords
