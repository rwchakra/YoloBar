# Imports
import os
from tqdm import tqdm
from PIL import Image

# PyTorch Imports
import torch

# Project Imports
from data_utilities import get_transform, LoggiPackageDataset, draw_results
from model_utilities import LoggiBarcodeDetectionModel



# Constant variables
PLOT_PREDICTIONS = True
IMG_SIZE = 1024
SAVED_MODEL = os.path.join("results", "models", "visum2022.pt")
DATA_DIR = "data_participants"
DEVICE = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')


# Directory to save visualisations
results_vis_dir = os.path.join("results", "visualisations", "val")
if not os.path.isdir(results_vis_dir):
    os.makedirs(results_vis_dir)


# Create Dataset
dataset_notransforms = LoggiPackageDataset(data_dir=DATA_DIR, training=True, transforms=get_transform(data_augment=False, img_size=IMG_SIZE))

# Split the dataset into train and validation sets
indices = torch.randperm(len(dataset_notransforms)).tolist()
# Validation Set: 299 samples
val_set = torch.utils.data.Subset(dataset_notransforms, indices[-299:])


# Choose if you want to plot prediction
if PLOT_PREDICTIONS:
    # Load model
    model = LoggiBarcodeDetectionModel(min_img_size=IMG_SIZE, max_img_size=IMG_SIZE)

    checkpoint = torch.load(SAVED_MODEL, map_location=DEVICE)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(DEVICE)
    model.eval()

with torch.no_grad():
    for image, target in tqdm(val_set):
        img_fname = target['image_fname']

        # Ground-truth
        # Visualisation of results
        vis_results_gt = draw_results(image=image * 255, masks=target['masks'], bboxes=target['boxes'])

        # Convert this result to NumPy
        vis_results_gt = vis_results_gt.permute(1, 2, 0).numpy()

        # Convert into PIL
        pil_vis_results_gt = Image.fromarray(vis_results_gt)
        pil_vis_results_gt.save(os.path.join(results_vis_dir, img_fname.split('.')[0] + "_gt.png"))

        # Predictions
        if PLOT_PREDICTIONS:
            gpu_image = image.to(DEVICE)

            # Add batch dimensions
            gpu_image = gpu_image.unsqueeze(0)

            # Feed image to the model
            outputs = model(gpu_image)[0]
            pred_bboxes = outputs['boxes'].cpu()
            pred_masks = outputs['masks'].cpu().squeeze(1)
            pred_scores = list(map(str, outputs['scores'].cpu().numpy()))

            # Visualisation of results
            vis_results_pred = draw_results(image=image * 255, masks=pred_masks, bboxes=pred_bboxes, scores=pred_scores)

            # Convert this result to NumPy
            vis_results_pred = vis_results_pred.permute(1, 2, 0).numpy()

            # Convert into PIL
            pil_vis_results_pred = Image.fromarray(vis_results_pred)
            pil_vis_results_pred.save(os.path.join(results_vis_dir, img_fname.split('.')[0] + "_pred.png"))
