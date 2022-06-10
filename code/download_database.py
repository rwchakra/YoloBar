# Imports
import os
import json
import wget
import tqdm
from urllib import request
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



# TODO: Erase uppon review
# Uncomment if you need some sanity check prints
# for key, value in json_data[0].items():
    # print(f"Key: {key} | Value: {value}")



# Create a data dictionary (that will be saved later into a JSON)
data_dict = dict()



# Go through all the images of the JSON
for data_point in tqdm.tqdm(json_data):

    # Image and Filename
    image_url = data_point["Labeled Data"]
    image_filename = data_point["External ID"]


    # Labels: Objects and Classification
    labels = data_point["Label"]
    l_objects = labels["objects"]
    l_classification = labels["classifications"][0]["answer"]["value"]

    # We want the 'good_quality' images
    if l_classification.lower() == "good_quality":
        
        # Create a folder to save images (if it does not exist)
        if not os.path.isdir(os.path.join(data, "raw")):
            os.makedirs(os.path.join(data, "raw"))
        
        # Download the image to this folder
        # response = wget.download(image_url, os.path.join(data, "raw", image_filename))
        response = request.urlretrieve(image_url, os.path.join(data, "raw", image_filename))

        # Add data point to the data dictionary
        data_dict[image_filename] = {"barcode":list(), "invoice":list(), "recipient":list(), "sender":list()}

        # Go through all the objects
        for obj in l_objects:

            # Access the object type ("barcode", "invoice", "recipient", "sender")...
            # ... and append it to the corresponding list
            data_dict[image_filename][obj["value"]].append(obj["polygon"])


        # TODO: Erase uppon review
        # print(f"URL: {image_url}")
        # print(f"Filename: {image_filename}")
        # print(f"Labels: {labels}")
        # print(f"Objects: {l_objects}")
        # print(f"Classifications: {l_classification}")

    # Convert the data dictionary into a JSON and dump it to a file
    json_object = json.dumps(data_dict, indent=4)

    # Writing to good_quality_imgs.json
    with open(os.path.join(data, "good_quality_imgs.json"), "w") as j:
        j.write(json_object)


print("Finished.")
