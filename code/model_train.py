# Imports
import os
import tqdm
import numpy as np
import argparse
from PIL import Image

# PyTorch Imports
import torch
import torch.utils.data

# Project Imports
from data_utilities import get_transform, collate_fn, LoggiPackageDataset
from model_utilities import LoggiBarcodeDetectionModel
from metrics_utilities import compute_mAP_metrics, visum2022score



# Random seeds
torch.manual_seed(42)

# TODO: Remove uppon review args for training
# parser = argparse.ArgumentParser(description = 'description')
# parser.add_argument('--batch_size', type = int, default = 1)
# parser.add_argument('--num_epochs', type = int, default = 1)
# parser.add_argument('--img_size', type = int, default = 1024, help="new size for img resize transform.")
# args = parser.parse_args()


# Constant variables
BATCH_SIZE = 1
NUM_EPOCHS = 1
IMG_SIZE = 1024
IOU_RANGE = np.arange(0.5, 1, 0.05)
VAL_MAP_FREQ = 1

# Directories
DATA_DIR = "data_participants"
SAVE_MODEL_DIR = "results/models"
if not os.path.isdir(SAVE_MODEL_DIR):
    os.makedirs(SAVE_MODEL_DIR)



# Prepare data
# First, we create two train sets with different transformations (we will use the one w/out transforms as validation set)
dataset = LoggiPackageDataset(data_dir=DATA_DIR, training=True, transforms=get_transform(training=True, data_augment=True, img_size=IMG_SIZE))
dataset_notransforms = LoggiPackageDataset(data_dir=DATA_DIR, training=True, transforms=get_transform(training=True, data_augment=False, img_size=IMG_SIZE))

# Split the dataset into train and validation sets
indices = torch.randperm(len(dataset)).tolist()
# Train Set: 1000 samples
train_set = torch.utils.data.Subset(dataset, indices[:-299])
# Validation Set: 299 samples
val_set = torch.utils.data.Subset(dataset_notransforms, indices[-299:])

# DataLoaders
# Train loader
train_loader = torch.utils.data.DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True, num_workers=4, collate_fn=collate_fn)

# Validation loader
val_loader = torch.utils.data.DataLoader(val_set, batch_size=BATCH_SIZE, shuffle=False, num_workers=4, collate_fn=collate_fn)


# Define DEVICE (GPU or CPU)
DEVICE = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
print(f"Using device: {DEVICE}")


# Define model
model = LoggiBarcodeDetectionModel(min_img_size=IMG_SIZE, max_img_size=IMG_SIZE)

# Print model summary
model_summary = model.summary()

# Put model into DEVICE
model.to(DEVICE)


# Define an optimizer
model_params = [p for p in model.parameters() if p.requires_grad]
optimizer = torch.optim.SGD(model_params, lr=0.005, momentum=0.9, weight_decay=0.0005)


# Start the training and validation loops
for epoch in range(NUM_EPOCHS):

    # Epoch
    print(f"Epoch: {epoch+1}/{NUM_EPOCHS}")

    # Training Phase
    print("Training Phase")
    model.train()

    # Initialize lists of losses for tracking
    losses_classifier = list()
    losses_box_reg = list()
    losses_mask = list()
    losses_objectness = list()
    losses_rpn_box_reg = list()
    losses_ = list()


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
        losses_classifier.append(loss_dict['loss_classifier'].item())
        losses_box_reg.append(loss_dict['loss_box_reg'].item())
        losses_mask.append(loss_dict['loss_mask'].item())
        losses_objectness.append(loss_dict['loss_objectness'].item())
        losses_rpn_box_reg.append(loss_dict['loss_rpn_box_reg'].item())
        losses_.append(loss_value)


        # Optimise models parameters
        optimizer.zero_grad()
        losses.backward()
        optimizer.step()
    

    # Print loss values
    print(f"Loss Classifier: {np.sum(losses_classifier) / len(train_set)}")
    print(f"Loss Box Regression: {np.sum(losses_box_reg) / len(train_set)}")
    print(f"Loss Mask: {np.sum(losses_mask) / len(train_set)}")
    print(f"Loss Objectness: {np.sum(losses_objectness) / len(train_set)}")
    print(f"Loss RPN Box Regression: {np.sum(losses_rpn_box_reg) / len(train_set)}")
    print(f"Loss: {np.sum(losses_) / len(train_set)}")


    

    if (epoch % VAL_MAP_FREQ == 0 and epoch > 0) or (epoch == NUM_EPOCHS - 1):
        # Validation Phase
        print("Validation Phase")
        model.eval()

        # Create dictionaries for ground-truth and predictions
        groundtruth_data = dict()
        predictions_data = dict()

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

                    # Create dictionaries for groundtruth data
                    groundtruth_data[fname] = dict()
                    groundtruth_data[fname]['boxes'] = list()
                    groundtruth_data[fname]['scores'] = list()
                    groundtruth_data[fname]['masks'] = list()

                    # i = 0
                    for bb, mask in zip(t["boxes"], t["masks"]):
                        # Bounding-boxes
                        groundtruth_data[fname]['boxes'].append(list(bb.detach().cpu().numpy()))
                        
                        # Masks
                        # msk_fname = f"{i}.jpg"
                        # groundtruth_data[fname]['masks'].append(msk_fname)
                        # print(f'Masks shape: {t["masks"].shape}')

                        # Save masks into directory
                        msk_ = mask.detach().cpu().numpy().copy()
                        groundtruth_data[fname]['masks'].append(msk_)
                        
                        # pil_mask = Image.fromarray(msk_).convert("L")
                        
                        # Save into temporary directory
                        # if not os.path.isdir(os.path.join("results", "validation", "masks", "gt", fname.split('.')[0])):
                            # os.makedirs(os.path.join("results", "validation", "masks", "gt", fname.split('.')[0]))
                
                        # pil_mask.save(os.path.join("results", "validation", "masks",  "gt", fname.split('.')[0], msk_fname))

                        # Update i (idx)
                        # i += 1


                    # Create dictionaries for predictions data
                    predictions_data[fname] = dict()
                    predictions_data[fname]['boxes'] = list()
                    predictions_data[fname]['scores'] = list()
                    predictions_data[fname]['masks'] = list()

                    # j = 0
                    for bb, mask, score in zip(out["boxes"], out["masks"], out["scores"]):
                        # Bounding-boxes
                        predictions_data[fname]['boxes'].append(list(bb.detach().cpu().numpy()))

                        # Scores
                        predictions_data[fname]['scores'].append(float(score.detach().cpu()))

                        # Masks
                        # msk_fname = f"{i}.jpg"
                        # predictions_data[fname]['masks'].append(msk_fname)

                        # Save masks into directory
                        msk_ = np.squeeze(a=mask.detach().cpu().numpy().copy(), axis=0)
                        predictions_data[fname]['masks'].append(msk_)
                        # pil_mask = Image.fromarray(msk_).convert("L")
                        
                        # Save into temporary directory
                        # if not os.path.isdir(os.path.join("results", "validation", "masks", "pred", fname.split('.')[0])):
                            # os.makedirs(os.path.join("results", "validation", "masks", "pred", fname.split('.')[0]))
                
                        # pil_mask.save(os.path.join("results", "validation", "masks",  "pred", fname.split('.')[0], msk_fname))

                        # Update j (idx)
                        # j += 1
        

        # TODO: Erase uppon review
        # Compute validation metrics
        # predictions_dir = os.path.join("results", "validation", "masks",  "pred")
        # groundtruth_dir = os.path.join("results", "validation", "masks",  "gt")

        # bboxes_mAP, bboxes_APs, masks_mAP, masks_APs = compute_mAP_metrics(predictions_data, groundtruth_data, predictions_dir, groundtruth_dir)
        bboxes_mAP, bboxes_APs, masks_mAP, masks_APs = compute_mAP_metrics(predictions_data, groundtruth_data)
        

        # Bounding-boxes mAP
        print("Bounding-boxes mAP:{:.3f}".format(bboxes_mAP))
        for ap_metric, iou in zip(bboxes_APs, IOU_RANGE):
            print("\tBounding-boxes AP at IoU level [{:.2f}]: {:.3f}".format(iou, ap_metric))
        
        # Masks mAP
        print("Masks mAP:{:.3f}".format(masks_mAP))
        for ap_metric, iou in zip(masks_APs, IOU_RANGE):
            print("\tMasks AP at IoU level [{:.2f}]: {:.3f}".format(iou, ap_metric))


        # Compute VISUM SCORE 2022
        score = visum2022score(bboxes_mAP=bboxes_mAP, masks_mAP=masks_APs)



    torch.save(model, os.path.join(SAVE_MODEL_DIR, "visum2022.pt"))

    print(f"Model successfully saved at {os.path.join(SAVE_MODEL_DIR, 'visum2022.pt')}")


print("Finished.")
