# Imports
import os
import json
import tqdm
import numpy as np 
from PIL import Image
import cv2


# Random seed
np.random.seed = 42


# Directories and Paths
code_dir = "code"
data_dir = "data"
raw_dir = os.path.join(data_dir, "raw")
processed_dir = os.path.join(data_dir, "processed")
json_dir = os.path.join(data_dir, "json")
train_json = os.path.join(json_dir, "train.json")
test_json = os.path.join(json_dir, "test.json")


# Open Train JSON file
with open(train_json, 'r') as tr:

    # Load JSON contents
    train_data = json.loads(tr.read())

print(f"Number of images in Train Set: {len(train_data)}")


# Open Test JSON file
with open(test_json, 'r') as ts:

    # Load JSON contents
    test_data = json.loads(ts.read())

print(f"Number of images in Test Set: {len(test_data)}")




# Start by processing the train images
print("Processing train images")

# Create a challenge train dictionary 
challenge_train_dict = dict()

for image_fname in tqdm.tqdm(train_data.keys()):

    # Add the image_fname as key
    challenge_train_dict[image_fname] = dict()
    challenge_train_dict[image_fname]["boxes"] = list()
    challenge_train_dict[image_fname]["labels"] = list()
    challenge_train_dict[image_fname]["masks"] = list()


    # Open image
    pil_image = Image.open(os.path.join(processed_dir, "train", image_fname)).convert('RGB')
    npy_image = np.asarray(pil_image)

    # Get the barcodes
    barcodes = train_data[image_fname]["barcode"]

    # Iterate through barcodes
    for idx, barcode in enumerate(barcodes):
        # print(barcode)

        # Create a black image with the same shape
        mask = np.zeros_like(npy_image)

        # Convert the barcode notation into contour notation
        # ymin, xmin, ymax, xmax = barcode[0], barcode[1], barcode[2], barcode[3]
        contours = [np.array([[point['x'], point['y']] for point in barcode], dtype=np.int32)]
        for cnt in contours:
            cv2.drawContours(mask, [cnt], 0, (255, 255, 255), -1)


        # Obtain a single-channel image
        mask = np.mean(mask, axis=2)
        # print(f"Mask Shape: {mask.shape}, Max: {np.max(mask)}, Min: {np.min(mask)}")

        # Convert this mask into PIL
        pil_mask = Image.fromarray(mask).convert("L")

        # Save mask
        if not os.path.isdir(os.path.join(data_dir, "masks", "train", image_fname.split('.')[0])):
            os.makedirs(os.path.join(data_dir, "masks", "train", image_fname.split('.')[0]))
        
        pil_mask.save(os.path.join(data_dir, "masks", "train", image_fname.split('.')[0], f"{idx}.jpg"))


        # Update challenge train dictionary with all the fields expected by the Mask-RCNN
        # boxes (``FloatTensor[N, 4]``): the ground-truth boxes in ``[x1, y1, x2, y2]`` format, with ``0 <= x1 < x2 <= W`` and ``0 <= y1 < y2 <= H``.
        vesMask = (mask > 0).astype(np.uint8)
        x, y, w, h = cv2.boundingRect(vesMask)
        challenge_train_dict[image_fname]["boxes"].append([x, y, x+w, y+h])

        # labels (Int64Tensor[N]): the class label for each ground-truth box
        challenge_train_dict[image_fname]["labels"].append(1)

        # masks (UInt8Tensor[N, H, W]): the segmentation binary masks for each instance
        challenge_train_dict[image_fname]["masks"].append(f"{idx}.jpg")


# Convert this information into JSON and dump it to a file
json_object = json.dumps(challenge_train_dict, indent=4)

# Writing to split.json
if not os.path.isdir(os.path.join(data_dir, "json", "challenge")):
    os.makedirs(os.path.join(data_dir, "json", "challenge"))

with open(os.path.join(data_dir, "json", "challenge", "train_challenge.json"), "w") as j:
    j.write(json_object)



# Process test images
print("Processing test images...")

# Create a challenge test dictionary 
challenge_test_dict = dict()

for image_fname in tqdm.tqdm(test_data.keys()):

    # Add the image_fname as key
    challenge_test_dict[image_fname] = dict()
    challenge_test_dict[image_fname]["boxes"] = list()
    challenge_test_dict[image_fname]["labels"] = list()
    challenge_test_dict[image_fname]["masks"] = list()

    # Open image
    pil_image = Image.open(os.path.join(raw_dir, image_fname)).convert('RGB')
    npy_image = np.asarray(pil_image)

    # Get the barcodes
    barcodes = test_data[image_fname]["barcode"]

    # Iterate through barcodes
    for idx, barcode in enumerate(barcodes):
        # print(barcode)

        # Create a black image with the same shape
        mask = np.zeros_like(npy_image)

        # Convert the barcode notation into contour notation
        # ymin, xmin, ymax, xmax = barcode[0], barcode[1], barcode[2], barcode[3]
        # contours = [np.array([[xmin, ymax], [xmin, ymin], [xmax, ymin], [xmax, ymax]], dtype=np.int32)]
        contours = [np.array([[point['x'], point['y']] for point in barcode], dtype=np.int32)]
        for cnt in contours:
            cv2.drawContours(mask, [cnt], 0, (255, 255, 255), -1)


        # Obtain a single-channel image
        mask = np.mean(mask, axis=2)
        # print(f"Mask Shape: {mask.shape}, Max: {np.max(mask)}, Min: {np.min(mask)}")

        # Convert this mask into PIL
        pil_mask = Image.fromarray(mask).convert("L")

        # Save mask
        if not os.path.isdir(os.path.join(data_dir, "masks", "test", image_fname.split('.')[0])):
            os.makedirs(os.path.join(data_dir, "masks", "test", image_fname.split('.')[0]))
        
        pil_mask.save(os.path.join(data_dir, "masks", "test", image_fname.split('.')[0], f"{idx}.jpg"))

        # Update challenge train dictionary with all the fields expected bay the Mask-RCNN
        # boxes (``FloatTensor[N, 4]``): the ground-truth boxes in ``[x1, y1, x2, y2]`` format, with ``0 <= x1 < x2 <= W`` and ``0 <= y1 < y2 <= H``.
        vesMask = (mask > 0).astype(np.uint8)
        x, y, w, h = cv2.boundingRect(vesMask)
        challenge_test_dict[image_fname]["boxes"].append([x, y, x+w, y+h])

        # labels (Int64Tensor[N]): the class label for each ground-truth box
        challenge_test_dict[image_fname]["labels"].append(1)

        # masks (UInt8Tensor[N, H, W]): the segmentation binary masks for each instance
        challenge_test_dict[image_fname]["masks"].append(f"{idx}.jpg")


# Convert this information into JSON and dump it to a file
json_object = json.dumps(challenge_test_dict, indent=4)

# Writing to split.json
if not os.path.isdir(os.path.join(data_dir, "json", "challenge")):
    os.makedirs(os.path.join(data_dir, "json", "challenge"))

with open(os.path.join(data_dir, "json", "challenge", "test_challenge.json"), "w") as j:
    j.write(json_object)



print("Finished.")
