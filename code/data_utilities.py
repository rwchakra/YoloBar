# Imports
import os
import json
import numpy as np
from PIL import Image

# PyTorch Imports
import torch
from torch.utils.data import Dataset, DataLoader
import torchvision



# Class: Loggi Package Barcode Detection Dataset
class LoggiPackageDataset(Dataset):
    def __init__(self, data_dir="data", training=True, transforms=None):
        
        # Initialise variables
        self.data_dir = data_dir
        self.transforms = transforms

        # Load JSON file in the data directory
        json_file = os.path.join(self.data_dir, "json", "train.json")

        # Open JSON file
        with open(json_file, 'r') as j:

            # Load JSON contents
            json_data = json.loads(j.read())


        # Create a list with all the images' filenames
        self.images = list(json_data.keys())

        # Add the "json_data" variable to the class variables
        self.label_dict = json_data.copy()


    # Method: __getitem__
    def __getitem__(self, idx):
        
        # Get image data
        image_fname = self.images[idx]
        img_path = os.path.join(self.data_dir, "processed", "train", image_fname)
        image = Image.open(img_path).convert('RGB')


        # Get label data
        boxes = self.label_dict[image_fname]['barcode']
        
        # Number of barcodes
        n_objs = len(boxes)

        # Bounding boxes
        if n_objs > 0:
            boxes = torch.as_tensor(boxes, dtype=torch.float32)
        else:
            boxes = torch.zeros((0, 4), dtype=torch.float32)
        
        # TODO: Labels (do we need this?)
        labels = torch.ones((n_objs,), dtype=torch.int64)
        
        # TODO: Image Index (do we need this?)
        image_idx = torch.tensor([idx])
        
        # TODO: Area (do we need this?)
        area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])
        
        # TODO: Suppose all instances are not crowd (do we need this?)
        iscrowd = torch.zeros((n_objs,), dtype=torch.int64)        


        # Target (dictionary)
        target = dict()
        target["boxes"] = boxes
        target["labels"] = labels
        target["image_idx"] = image_idx
        target["area"] = area
        target["iscrowd"] = iscrowd


        # TODO: Review transforms to image data and labels (albumentations?)
        if self.transforms:
            image = self.transforms(image)
        
        return image, target


    # Method: __len__
    def __len__(self):
        return len(self.images)



# Run this file to test the Dataset class
if __name__=="__main__":
    
    # Create a LoggiPackageDataset instance
    train_transforms = torchvision.transforms.Compose([torchvision.transforms.ToTensor()])
    train_set = LoggiPackageDataset(data_dir="data", transforms=train_transforms)

    # Create a DataLoader
    train_loader = DataLoader(dataset=train_set)

    # Iterate through DataLoader
    for images, targets in train_loader:
        print(targets)
