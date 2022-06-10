# Imports
import os
import json
import tqdm
import numpy as np 

# Sklearn Imports
from sklearn.model_selection import train_test_split

# Project Imports
from utilities import convert_lbox_coords



# Directories and Files
data = "data"
json_file = os.path.join(data, "good_quality_imgs.json")


# Open JSON file
with open(json_file, 'r') as j:

    # Load JSON contents
    json_data = json.loads(j.read())

print(f"Number of images in JSON: {len(json_data)}")


# Read the raw-files directory
raw_images = [i for i in os.listdir(os.path.join(data, "raw")) if not i.startswith('.')]
print(f"Number of images in the folder: {len(raw_images)}")



# Divide data into train and test...
# ... since train will be anonimised, but test won't
_y = np.zeros_like(np.array(raw_images))
train_images, test_images, _, _ = train_test_split(raw_images, _y, test_size=0.30, random_state=42)
print(f"Shape of train subset: {np.shape(train_images)}")
print(f"Shape of test subset: {np.shape(test_images)}")


# Create data dictionaries for train and test subsets
train_dict = dict()
test_dict = dict()


# Iterate through subsets and the corresponding data dictionary
for subset, subset_dict, split in zip([train_images, test_images], [train_dict, test_dict], ["train", "test"]):

    # Go through the images of the subset
    for image_filename in tqdm.tqdm(subset):
        subset_dict[image_filename] = {"barcode":list(), "invoice":list(), "recipient":list(), "sender":list()}

        # Get data from JSON
        annotations = json_data[image_filename]

        # Barcode data
        barcodes = annotations["barcode"]
        for coords_list in barcodes:
            subset_dict[image_filename]["barcode"].append(convert_lbox_coords(coords_list))

        # Invoice data
        invoices = annotations["invoice"]
        for coords_list in invoices:
            subset_dict[image_filename]["invoice"].append(convert_lbox_coords(coords_list))

        # Recipient data
        recipients = annotations["recipient"]
        for coords_list in recipients:
            subset_dict[image_filename]["recipient"].append(convert_lbox_coords(coords_list))

        # Sender data
        senders = annotations["sender"]
        for coords_list in senders:
            subset_dict[image_filename]["sender"].append(convert_lbox_coords(coords_list))
    

    # Convert this information into JSON and dump it to a file
    json_object = json.dumps(subset_dict, indent=4)

    # Writing to split.json
    if not os.path.isdir(os.path.join(data, "json")):
        os.makedirs(os.path.join(data, "json"))

    with open(os.path.join(data, "json", f"{split}.json"), "w") as j:
        j.write(json_object)



print("Finished")
