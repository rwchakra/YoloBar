# Imports
import os
import tqdm
import numpy as np
import argparse

# PyTorch Imports
import torch
import torch.utils.data

# Project Imports
from data_utilities import get_transform, collate_fn, LoggiPackageDataset
from model_utilities import LoggiBarcodeDetectionModel
from metrics_utilities import compute_mAP



# Random seeds
torch.manual_seed(42)

# Args for training
parser = argparse.ArgumentParser(description = 'description')
parser.add_argument('--batch_size', type = int, default = 1)
parser.add_argument('--num_epochs', type = int, default = 1)
parser.add_argument('--img_size', type = int, default = 1024, help="new size for img resize transform.")
args = parser.parse_args()


# Directories
DATA_DIR = "data"
SAVE_MODEL_DIR = "results/models"
if not os.path.isdir(SAVE_MODEL_DIR):
    os.makedirs(SAVE_MODEL_DIR)



# Prepare data
# First, we create two train sets with different transformations (we will use the one w/out transforms as validation set)
dataset = LoggiPackageDataset(data_dir=DATA_DIR, training=True, transforms=get_transform(training=True, data_augment=True, img_size=args.img_size))
dataset_notransforms = LoggiPackageDataset(data_dir=DATA_DIR, training=True, transforms=get_transform(training=False, data_augment=False, img_size=args.img_size))

# Split the dataset into train and validation sets
indices = torch.randperm(len(dataset)).tolist()
# Train Set: 1000 samples
train_set = torch.utils.data.Subset(dataset, indices[:-299])
# Validation Set: 299 samples
val_set = torch.utils.data.Subset(dataset_notransforms, indices[-299:])

# DataLoaders
# Define batch size
BATCH_SIZE = args.batch_size

# Train loader
train_loader = torch.utils.data.DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True, num_workers=4, collate_fn=collate_fn)

# Validation loader
val_loader = torch.utils.data.DataLoader(val_set, batch_size=BATCH_SIZE, shuffle=False, num_workers=4, collate_fn=collate_fn)


# Define DEVICE (GPU or CPU)
DEVICE = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
print(f"Using device: {DEVICE}")


# Define model
model = LoggiBarcodeDetectionModel()

# Print model summary
model_summary = model.summary()

# Put model into DEVICE
model.to(DEVICE)


# Define an optimizer
model_params = [p for p in model.parameters() if p.requires_grad]
optimizer = torch.optim.SGD(model_params, lr=0.005, momentum=0.9, weight_decay=0.0005)


# Define the number of epochs
NUM_EPOCHS = args.num_epochs


# Start the training and validation loops
for epoch in range(NUM_EPOCHS):

    # Epoch
    print(f"Epoch: {epoch+1}/{NUM_EPOCHS}")

    # Training Phase
    print("Training Phase")
    model.train()


    # Go through train loader
    for images, targets, _ in tqdm.tqdm(train_loader):
        
        # Load data
        images = list(image.to(DEVICE) for image in images)
        targets_ = [{k: v.to(DEVICE) for k, v in t.items()} for t in targets]


        # Compute loss
        loss_dict = model(images, targets_)
        losses = sum(loss for loss in loss_dict.values())
        loss_value = losses.item()

        # Print loss values
        print(f"Loss Classifier: {loss_dict['loss_classifier'].item()}")
        print(f"Loss Box Regression: {loss_dict['loss_box_reg'].item()}")
        print(f"Loss Mask: {loss_dict['loss_mask'].item()}")
        print(f"Loss Objectness: {loss_dict['loss_objectness'].item()}")
        print(f"Loss RPN Box Regression: {loss_dict['loss_rpn_box_reg'].item()}")


        # Optimise models parameters
        optimizer.zero_grad()
        losses.backward()
        optimizer.step()


    

    # Validation Phase
    print("Validation Phase")
    model.eval()

    # Create lists for ground-truth and predictions
    ground_truth = list()
    predictions = list()

    with torch.no_grad():
        # Go through validation loader
        for images, targets, image_fnames in tqdm.tqdm(val_loader):

            # Load data
            images = list(img.to(DEVICE) for img in images)
            targets_ = [{k: v.to(DEVICE) for k, v in t.items()} for t in targets]
            image_fnames_ = [f for f in image_fnames]
            outputs = model(images, targets)


            # Add to ground truth list
            for out, t, fname in zip(outputs, targets_, image_fnames_):
                gt_boxes = list()
                gt_masks = list()

                for bb, mask in zip(t["boxes"], t["masks"]):
                    gt_boxes.append(list(bb.detach().cpu().numpy()))
                    print(f'Masks shape: {t["masks"].shape}')
                    gt_masks.append(mask.detach().cpu().numpy())
                
                ground_truth.append([fname, gt_boxes, gt_masks])


                for bb, score in zip(out["boxes"], out["scores"]):
                    predictions.append([fname, list(bb.detach().cpu().numpy()), float(score.detach().cpu())])
    

    # Compute validation metrics
    mAP, AP = compute_mAP(predictions, ground_truth)
    print("mAP:{:.3f}".format(mAP))


    for ap_metric, iou in zip(AP, np.arange(0.5, 1, 0.05)):
        print("\tAP at IoU level [{:.2f}]: {:.3f}".format(iou, ap_metric))



    torch.save(model, os.path.join(SAVE_MODEL_DIR, "visum2022.pt"))

    print(f"Model successfully saved at {os.path.join(SAVE_MODEL_DIR, 'visum2022.pt')}")


print("Finished.")
