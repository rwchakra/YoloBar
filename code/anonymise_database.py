# Imports
import os
import json
import tqdm
import numpy as np 
from PIL import Image

# Project Imports
from utilities import convert_lbox_coords, apply_blurring


# Random seed
np.random.seed = 42


# Directories and Paths
code_dir = "code"
data_dir = "data"
raw_dir = os.path.join(data_dir, "raw")
json_dir = os.path.join(data_dir, "json")
train_json = os.path.join(json_dir, "train.json")



# Open train JSON file
with open(train_json, 'r') as j:

    # Load JSON contents
    json_data = json.loads(j.read())



# Get the trai images
train_images = json_data.keys()
# print(f"Train images: {train_images}")


# Iterate through training images
for image_filename in tqdm.tqdm(train_images):
    
    # Open image
    pil_img = Image.open(os.path.join(raw_dir, image_filename))
    img_array = np.asarray(pil_img)
    # print(f"Shape of the image array: {img_array.shape}")

    # Image array to process
    img_array_proc = np.copy(img_array)


    # Get image annotations
    img_annotations = json_data[image_filename]
    barcodes = img_annotations['barcode']
    invoices = img_annotations['invoice']
    senders = img_annotations['sender']
    recipients = img_annotations['recipient']
    # print(f"Length of the Invoices: {len(invoices)}")
    # print(f"Length of the Recipients: {len(recipients)}")

    # Objects to anonymise
    objs_anon = [invoices, recipients, senders]

    # Go through these objects
    for labels in objs_anon:

        # Iterate through each list
        for coords_list in labels:
            
            # Get 2D indices
            ymin, xmin, ymax, xmax = convert_lbox_coords(coords_list)

            # Get this slice (it will be changed)
            slice_to_anon = img_array_proc[int(ymin):int(ymax), int(xmin):int(xmax)].copy()

            # Apply anonymisation operation
            slice_anon = apply_blurring(slice_to_anon)

            # Apply to image array
            img_array_proc[int(ymin):int(ymax), int(xmin):int(xmax)] = slice_anon
    

    # Fix barcodes (in case it is affected by the blurring operation)
    # Create a new fresh copy of the image array
    img_array_tmp = np.copy(img_array)
    
    # Objects to keep
    objs_keep = [barcodes]
    
    # Go through these objects
    for labels in objs_keep:
        
        # Iterate through each list
        for coords_list in labels:
                
            # Get 2D indices
            ymin, xmin, ymax, xmax = convert_lbox_coords(coords_list)

            # Get the barcode slices
            bcode_slice = img_array_tmp[int(ymin):int(ymax), int(xmin):int(xmax)].copy()

            # Apply to processed image array
            img_array_proc[int(ymin):int(ymax), int(xmin):int(xmax)] = bcode_slice


    # Re-convert image array into PIL Image
    new_pil_img = Image.fromarray(img_array_proc)

    # Save this into new folder
    if not os.path.isdir(os.path.join(data_dir, "processed", "train")):
        os.makedirs(os.path.join(data_dir, "processed", "train"))
    
    new_pil_img.save(os.path.join(data_dir, "processed", "train", image_filename))


print("Finished.")
