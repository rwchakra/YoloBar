# Imports
import os
import json
import random
import numpy as np
from PIL import Image

# PyTorch Imports
import torch
from torch.utils.data import Dataset, DataLoader
import torchvision
from torchvision.transforms import functional as F



# Function: collate_fn
def collate_fn(batch):
    return tuple(zip(*batch))



# Function: Convert bounding box to COCO notation
def convert_bbox_to_coco(bbox, reverse=False):

    # Our notation has the format [x, y, x+w, y+h]
    # In COCO, the notation has the format [x_min, y_min, width, height]
    x_min, y_min, width, height = bbox[0], bbox[1], (bbox[2]-bbox[0]), (bbox[3]-bbox[1])

    # We then create a list


    return



# Class: (Homebrew) Compose
class Compose(object):
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, image, target):
        for t in self.transforms:
            image, target = t(image, target)
        
        return image, target



# Class: (Homebrew) RandomHorizontalFlip
class RandomHorizontalFlip(object):
    def __init__(self, prob):
        self.prob = prob

    def __call__(self, image, target):
        if random.random() < self.prob:
            height, width = image.shape[-2:]
            image = image.flip(-1)
            
            # Boxes
            bbox = target["boxes"]
            bbox[:, [0, 2]] = width - bbox[:, [2, 0]]
            target["boxes"] = bbox

            # Masks
            target["masks"] = target["masks"].flip(-1)

        return image, target



# Class: (Homebrew) ToTensor
class ToTensor(object):
    def __call__(self, image, target):
        image = F.to_tensor(image)
        
        return image, target



# Function: Create a Compose of data transforms (for training)
def get_transform(training=True, data_augment=True):
    
    # During training (training and validation sets)
    if training:
        transforms = []

        # Converts the image, a PIL image, into a PyTorch Tensor
        transforms.append(ToTensor())
        
        # Apply data augmentation during training
        if data_augment:
            # In this case, we are applying a RandomHorizontalFlip
            transforms.append(RandomHorizontalFlip(0.5))
        

        return Compose(transforms)
    
    
    # During test (test set)
    else:

        return torchvision.transforms.Compose([torchvision.transforms.ToTensor()])



# Class: Loggi Package Barcode Detection Dataset
class LoggiPackageDataset(Dataset):
    def __init__(self, data_dir="data", training=True, transforms=None):
        
        # Initialise variables
        self.data_dir = data_dir
        self.transforms = transforms
        self.training = training

        # Load JSON file in the data directory
        if self.training:
            json_file = os.path.join(self.data_dir, "json", "challenge", "train_challenge.json")
        
        else:
            json_file = os.path.join(self.data_dir, "json", "challenge", "test_challenge.json")


        # Open JSON file
        with open(json_file, 'r') as j:

            # Load JSON contents
            json_data = json.loads(j.read())


        # Create a list with all the images' filenames
        self.images = list(json_data.keys())

        # Add the "json_data" variable to the class variables
        self.label_dict = json_data.copy() if self.training else None


    # Method: __getitem__
    def __getitem__(self, idx):

        # Mode
        if self.training:
        
            # Get image data
            image_fname = self.images[idx]
            img_path = os.path.join(self.data_dir, "processed", "train", image_fname)
            image = Image.open(img_path).convert('RGB')


            # Get annotation data
            # Boxes
            boxes = self.label_dict[image_fname]['boxes']
            
            # Number of Boxes
            n_objs = len(boxes)

            
            if n_objs > 0:
                boxes = torch.as_tensor(boxes, dtype=torch.float32)
            else:
                boxes = torch.zeros((0, 4), dtype=torch.float32)

            
            # Labels
            labels = self.label_dict[image_fname]['labels']
            labels = torch.as_tensor(labels, dtype=torch.int64)

            # Masks
            masks = self.label_dict[image_fname]['masks']
            masks = [np.asarray(Image.open(os.path.join(self.data_dir, "masks", "train", image_fname.split('.')[0], m)).convert("L")) for m in masks]
            masks = [(m > 0).astype(np.uint8) for m in masks]
            masks = torch.as_tensor(masks, dtype=torch.uint8)
            
            # Image Index
            image_id = torch.tensor([idx])
            
            # Area (do we need this?)
            area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])
            
            # Suppose all instances are not crowd (do we need this?)
            iscrowd = torch.zeros((n_objs,), dtype=torch.int64)        


            # Target (dictionary)
            target = dict()
            target["boxes"] = boxes
            target["labels"] = labels
            target["masks"] = masks
            target["area"] = area
            target["iscrowd"] = iscrowd
            target["image_id"] = image_id


            # Apply transforms to both image and target
            if self.transforms:
                image, target = self.transforms(image, target)
            

            return image, target, image_fname
        


        else:

            # Get image data
            image_fname = self.images[idx]
            img_path = os.path.join(self.data_dir, "raw", image_fname)
            image = Image.open(img_path).convert('RGB')

            # For model purposes
            target = list()

            if self.transforms:
                image = self.transforms(image)
            

            return image, target, image_fname


    # Method: __len__
    def __len__(self):
        return len(self.images)



# Run this file to test the Dataset class
if __name__ == "__main__":
    
    # Create a LoggiPackageDataset instance
    train_transforms = get_transform()
    train_set = LoggiPackageDataset(data_dir="data", transforms=train_transforms)

    # Create a DataLoader
    batch_size = 5
    train_loader = DataLoader(dataset=train_set, batch_size=batch_size, collate_fn=collate_fn)

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    
    # Iterate through DataLoader
    for images, targets, image_fname in train_loader:
        
        # Get images
        images_ = list(image.to(device) for image in images)
        targets_ = [{k: v.to(device) for k, v in t.items()} for t in targets]

        # print(images_, targets_)
        
        for i in range(batch_size):
            print(f"i: {i}")
            print(f"Images (shape, min, max): {images_[i].shape}, {torch.min(images[i])}, {torch.max(images[i])}")
            print(f"Boxes (shape): {targets_[i]['boxes'].shape}")
            print(f"Labels (shape): {targets_[i]['labels'].shape}")
            print(f"Masks (shape, min, max): {targets_[i]['masks'].shape}, {torch.min(targets_[i]['masks'])}, {torch.max(targets_[i]['masks'])}")
            print(f"Image Filename: {image_fname}")
            print(f"Area of the objects: {targets_[i]['area']}")
            print(f"Is crowd: {targets_[i]['iscrowd']}")

        break
