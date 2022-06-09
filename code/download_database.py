# Imports
import os
import json
import numpy as np 
import pandas as pd 
from PIL import Image



# Directories and Paths
code = 'code'
data = 'data'
json_file = os.path.join(data, 'export-2022-06-09T10_25_16.167Z.json')


# Open JSON file
with open(json_file, 'r') as j:

    # Load JSON contents
    json_data = json.loads(j.read())


# Uncomment if you need some sanity check prints
# for key, value in json_data[0].items():
    # print(f"Key: {key} | Value: {value}")



# Go through all the images of the JSON
for data_point in json_data:

    # Image file
    image_file = data_point["Labeled Data"]

    # Labels
    labels = data_point["Label"]

    # Filename
    filename = data_point["External ID"]


    print(f"URL: {image_file}")
    print(f"Labels: {labels}")
    print(f"Filename: {filename}")
