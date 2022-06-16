# Imports
import os
import json
import random
import numpy as np
from PIL import Image, ImageColor, ImageDraw, ImageFont
import albumentations as A 
import cv2
from typing import Any, BinaryIO, List, Optional, Tuple, Union

# PyTorch Imports
import torch
from torch.utils.data import Dataset, DataLoader
import torchvision
from torchvision.transforms import functional as F



# Torchvision Utils Source Code
# Source: https://pytorch.org/vision/stable/_modules/torchvision/utils.html
@torch.no_grad()
def draw_bounding_boxes(
    image: torch.Tensor,
    boxes: torch.Tensor,
    labels: Optional[List[str]] = None,
    colors: Optional[Union[List[Union[str, Tuple[int, int, int]]], str, Tuple[int, int, int]]] = None,
    fill: Optional[bool] = False,
    width: int = 1,
    font: Optional[str] = None,
    font_size: int = 10,
) -> torch.Tensor:

    """
    Draws bounding boxes on given image.
    The values of the input image should be uint8 between 0 and 255.
    If fill is True, Resulting Tensor should be saved as PNG image.

    Args:
        image (Tensor): Tensor of shape (C x H x W) and dtype uint8.
        boxes (Tensor): Tensor of size (N, 4) containing bounding boxes in (xmin, ymin, xmax, ymax) format. Note that
            the boxes are absolute coordinates with respect to the image. In other words: `0 <= xmin < xmax < W` and
            `0 <= ymin < ymax < H`.
        labels (List[str]): List containing the labels of bounding boxes.
        colors (color or list of colors, optional): List containing the colors
            of the boxes or single color for all boxes. The color can be represented as
            PIL strings e.g. "red" or "#FF00FF", or as RGB tuples e.g. ``(240, 10, 157)``.
            By default, random colors are generated for boxes.
        fill (bool): If `True` fills the bounding box with specified color.
        width (int): Width of bounding box.
        font (str): A filename containing a TrueType font. If the file is not found in this filename, the loader may
            also search in other directories, such as the `fonts/` directory on Windows or `/Library/Fonts/`,
            `/System/Library/Fonts/` and `~/Library/Fonts/` on macOS.
        font_size (int): The requested font size in points.

    Returns:
        img (Tensor[C, H, W]): Image Tensor of dtype uint8 with bounding boxes plotted.
    """

    if not isinstance(image, torch.Tensor):
        raise TypeError(f"Tensor expected, got {type(image)}")
    elif image.dtype != torch.uint8:
        raise ValueError(f"Tensor uint8 expected, got {image.dtype}")
    elif image.dim() != 3:
        raise ValueError("Pass individual images, not batches")
    elif image.size(0) not in {1, 3}:
        raise ValueError("Only grayscale and RGB images are supported")

    num_boxes = boxes.shape[0]

    if labels is None:
        labels: Union[List[str], List[None]] = [None] * num_boxes  # type: ignore[no-redef]
    elif len(labels) != num_boxes:
        raise ValueError(
            f"Number of boxes ({num_boxes}) and labels ({len(labels)}) mismatch. Please specify labels for each box."
        )

    if colors is None:
        colors = _generate_color_palette(num_boxes)
    elif isinstance(colors, list):
        if len(colors) < num_boxes:
            raise ValueError(f"Number of colors ({len(colors)}) is less than number of boxes ({num_boxes}). ")
    else:  # colors specifies a single color for all boxes
        colors = [colors] * num_boxes

    colors = [(ImageColor.getrgb(color) if isinstance(color, str) else color) for color in colors]

    # Handle Grayscale images
    if image.size(0) == 1:
        image = torch.tile(image, (3, 1, 1))

    ndarr = image.permute(1, 2, 0).cpu().numpy()
    img_to_draw = Image.fromarray(ndarr)
    img_boxes = boxes.to(torch.int64).tolist()

    if fill:
        draw = ImageDraw.Draw(img_to_draw, "RGBA")
    else:
        draw = ImageDraw.Draw(img_to_draw)

    txt_font = ImageFont.load_default() if font is None else ImageFont.truetype(font=font, size=font_size)

    for bbox, color, label in zip(img_boxes, colors, labels):  # type: ignore[arg-type]
        if fill:
            fill_color = color + (100,)
            draw.rectangle(bbox, width=width, outline=color, fill=fill_color)
        else:
            draw.rectangle(bbox, width=width, outline=color)

        if label is not None:
            margin = width + 1
            draw.text((bbox[0] + margin, bbox[1] + margin), label, fill=color, font=txt_font)

    return torch.from_numpy(np.array(img_to_draw)).permute(2, 0, 1).to(dtype=torch.uint8)



@torch.no_grad()
def draw_segmentation_masks(
    image: torch.Tensor,
    masks: torch.Tensor,
    alpha: float = 0.8,
    colors: Optional[Union[List[Union[str, Tuple[int, int, int]]], str, Tuple[int, int, int]]] = None,
) -> torch.Tensor:

    """
    Draws segmentation masks on given RGB image.
    The values of the input image should be uint8 between 0 and 255.

    Args:
        image (Tensor): Tensor of shape (3, H, W) and dtype uint8.
        masks (Tensor): Tensor of shape (num_masks, H, W) or (H, W) and dtype bool.
        alpha (float): Float number between 0 and 1 denoting the transparency of the masks.
            0 means full transparency, 1 means no transparency.
        colors (color or list of colors, optional): List containing the colors
            of the masks or single color for all masks. The color can be represented as
            PIL strings e.g. "red" or "#FF00FF", or as RGB tuples e.g. ``(240, 10, 157)``.
            By default, random colors are generated for each mask.

    Returns:
        img (Tensor[C, H, W]): Image Tensor, with segmentation masks drawn on top.
    """

    if not isinstance(image, torch.Tensor):
        raise TypeError(f"The image must be a tensor, got {type(image)}")
    elif image.dtype != torch.uint8:
        raise ValueError(f"The image dtype must be uint8, got {image.dtype}")
    elif image.dim() != 3:
        raise ValueError("Pass individual images, not batches")
    elif image.size()[0] != 3:
        raise ValueError("Pass an RGB image. Other Image formats are not supported")
    if masks.ndim == 2:
        masks = masks[None, :, :]
    if masks.ndim != 3:
        raise ValueError("masks must be of shape (H, W) or (batch_size, H, W)")
    if masks.dtype != torch.bool:
        raise ValueError(f"The masks must be of dtype bool. Got {masks.dtype}")
    if masks.shape[-2:] != image.shape[-2:]:
        raise ValueError("The image and the masks must have the same height and width")

    num_masks = masks.size()[0]
    if colors is not None and num_masks > len(colors):
        raise ValueError(f"There are more masks ({num_masks}) than colors ({len(colors)})")

    if colors is None:
        colors = _generate_color_palette(num_masks)

    if not isinstance(colors, list):
        colors = [colors]
    if not isinstance(colors[0], (tuple, str)):
        raise ValueError("colors must be a tuple or a string, or a list thereof")
    if isinstance(colors[0], tuple) and len(colors[0]) != 3:
        raise ValueError("It seems that you passed a tuple of colors instead of a list of colors")

    out_dtype = torch.uint8

    colors_ = []
    for color in colors:
        if isinstance(color, str):
            color = ImageColor.getrgb(color)
        colors_.append(torch.tensor(color, dtype=out_dtype))

    img_to_draw = image.detach().clone()
    # TODO: There might be a way to vectorize this
    for mask, color in zip(masks, colors_):
        img_to_draw[:, mask] = color[:, None]

    out = image * (1 - alpha) + img_to_draw * alpha
    return out.to(out_dtype)



# Function: collate_fn
def collate_fn(batch):
    return tuple(zip(*batch))



# Function: Convert bounding box to COCO notation
def convert_bbox_to_coco(bbox, reverse=False):

    if not reverse:
        # Our notation has the format [x, y, x+w, y+h]
        # In COCO, the notation has the format [x_min, y_min, width, height]
        x_min, y_min, width, height = bbox[0], bbox[1], (bbox[2]-bbox[0]), (bbox[3]-bbox[1])

        # We then create a list with these entries
        converted_bbox = [x_min, y_min, width, height]
    
    else:
        # We assume we receive the data in the COCO format
        # The notation has the format [x_min, y_min, width, height]
        x_min, y_min, width, height = bbox[0], bbox[1], bbox[2], bbox[3]


        # We then convert it to our notation [x, y, x+w, y+h]
        converted_bbox = [x_min, y_min, x_min+width, y_min+height]


    return converted_bbox



# Function: Convert COCO notation to bounding box
def convert_coco_to_bbox(bbox):

    return convert_bbox_to_coco(bbox, reverse=True)




# Function: Create a Compose of data transforms (for training)
def get_transform(training, data_augment):

    # Assert conditions
    assert training in (True, False), f"The 'training' parameter should be a boolean (True, False). You entered {training}."
    assert data_augment in (True, False), f"The 'data_augment' parameter should be a boolean (True, False). You entered {data_augment}."

    # Initialise transforms to None
    transforms = None
    
    # During training (training and validation sets)
    if training:
        
        if data_augment:
            transforms = A.Compose([
                A.HorizontalFlip(p=0.5)
            ], bbox_params=A.BboxParams(format='coco', label_fields=['bbox_classes']))


        return transforms
    
    
    # During test (test set)
    else:

        return transforms



# Visualisation tools
def draw_results(image, masks, bboxes):

    # The values of the input image should be uint8 between 0 and 255
    image = torch.as_tensor(image.copy(), dtype=torch.uint8)
    image = image.permute(2, 0, 1)
    # Tensor of shape (num_masks, H, W) or (H, W) and dtype bool.
    masks = torch.as_tensor(masks, dtype=torch.bool)

    # Get image with mask on top
    colors = ["blue" for i in range(len(masks))]
    image_w_mask = draw_segmentation_masks(image=image, masks=masks, colors=colors)


    # Now, let's design the bounding boxes
    # The values of the input image should be uint8 between 0 and 255
    image_w_mask = torch.as_tensor(image_w_mask, dtype=torch.uint8)
    
    # Bounding boxes
    # Tensor of size (N, 4) containing bounding boxes in (xmin, ymin, xmax, ymax) format. 
    # Note that the boxes are absolute coordinates with respect to the image. 
    # In other words: 0 <= xmin < xmax < W and 0 <= ymin < ymax < H.
    labels = ['barcode' for i in range(len(bboxes))]
    bboxes = torch.as_tensor(bboxes, dtype=torch.float32)

    # Get the final image
    image_results = draw_bounding_boxes(image=image_w_mask, boxes=bboxes, labels=labels, colors=colors)


    return image_results



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
            image = np.asarray(image)


            # Get annotation data
            # Boxes
            bboxes = self.label_dict[image_fname]['boxes']
            
            
            # Labels
            bbox_classes = self.label_dict[image_fname]['labels']
            

            # Masks
            masks = self.label_dict[image_fname]['masks']
            masks = [np.asarray(Image.open(os.path.join(self.data_dir, "masks", "train", image_fname.split('.')[0], m)).convert("L")) for m in masks]
            masks = [(m > 0).astype(np.uint8) for m in masks]


            # Apply transforms to both image and target
            if self.transforms:

                # We have to convert all the bounding boxes to COCO notation before augmentation
                bboxes = [convert_bbox_to_coco(b) for b in bboxes]

                # Apply transforms
                transformed = self.transforms(image=image, masks=masks, bboxes=bboxes, bbox_classes=bbox_classes)
                
                # Get image
                image = transformed["image"]

                # Get masks
                masks = transformed["masks"]

                # Get bounding boxes
                bboxes = transformed["bboxes"]
                # We must convert into our notation again
                bboxes = [convert_coco_to_bbox(c) for c in bboxes]

                # Get labels
                bbox_classes = transformed["bbox_classes"]
            


            # Convert to Tensors
            image = F.to_tensor(image)
            masks = torch.as_tensor(masks, dtype=torch.uint8)
            labels = torch.as_tensor(bbox_classes, dtype=torch.int64)
            boxes = torch.as_tensor(bboxes, dtype=torch.float32)

            # Area
            area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])

            # Suppose all instances are not crowd
            n_objs = len(bboxes)
            iscrowd = torch.zeros((n_objs,), dtype=torch.int64)

            # Image Index
            image_id = torch.tensor([idx])


            # Build the target dictionary for the model
            target = dict()
            target["boxes"] = boxes
            target["labels"] = labels
            target["masks"] = masks
            target["area"] = area
            target["iscrowd"] = iscrowd
            target["image_id"] = image_id
            

        else:

            # Get image data
            image_fname = self.images[idx]
            img_path = os.path.join(self.data_dir, "raw", image_fname)
            image = Image.open(img_path).convert('RGB')
            image = np.asarray(image)

            # For model purposes
            target = list()


            # Check if we have transforms
            if self.transforms:
                image = self.transforms(image)
            

            # Convert to Tensors
            image = F.to_tensor(image)
            


        return image, target, image_fname


    # Method: __len__
    def __len__(self):
        return len(self.images)



# Run this file to test the Dataset class
if __name__ == "__main__":
    
    # Create a LoggiPackageDataset instance
    # Test different transforms
    split = 'test'
    # Train
    if split == 'train':
        transforms = get_transform(training=True, data_augment=True)
    
    elif split == 'validation':
        transforms = get_transform(training=True, data_augment=False)
    
    elif split == 'test':
        transforms = get_transform(training=False, data_augment=False)
    
    
    # Create a dataset
    print(f"Transforms: {transforms}")
    dataset = LoggiPackageDataset(data_dir="data", transforms=transforms)

    # Create a DataLoader
    batch_size = 5
    dataloader = DataLoader(dataset=dataset, batch_size=batch_size, collate_fn=collate_fn)

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    
    # Iterate through DataLoader
    for images, targets, image_fname in dataloader:
        
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
