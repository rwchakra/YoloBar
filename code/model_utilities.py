# Imports

# PyTorch Imports
import torch
import torchvision
from torchvision.models.detection import MaskRCNN
from torchvision.models.detection.anchor_utils import AnchorGenerator
from torchinfo import summary
from tqdm import tqdm
from coco_eval import CocoEvaluator, convert_to_coco_api

# Model: LoggiBarcodeDetectionModel


class LoggiBarcodeDetectionModel(torch.nn.Module):
    def __init__(self, min_img_size=800, max_img_size=1333, nr_classes=2, backbone="mobilenet_v2", backbone_pretrained=True):
        super(LoggiBarcodeDetectionModel, self).__init__()

        # Init variables
        self.nr_classes = nr_classes
        self.backbone = backbone
        self.backbone_pretrained = backbone_pretrained

        # Select the backbone
        if self.backbone == "mobilenet_v2":
            # Source: https://pytorch.org/vision/stable/_modules/torchvision/models/detection/mask_rcnn.html#maskrcnn_resnet50_fpn
            # Load a pre-trained model for classification and return only the features
            backbone_ = torchvision.models.mobilenet_v2(
                pretrained=self.backbone_pretrained).features
            # MaskRCNN needs to know the number of output channels in a backbone
            # For mobilenet_v2, it's 1280, so we need to add it here
            backbone_.out_channels = 1280

            # Let's make the RPN generate 5 x 3 anchors per spatial location, with 5 different sizes and 3 different aspect ratios
            # We have a Tuple[Tuple[int]] because each feature map could potentially have different sizes and aspect ratios
            anchor_generator = AnchorGenerator(
                sizes=((32, 64, 128, 256, 512),), aspect_ratios=((0.5, 1.0, 2.0),))

            # Let's define what are the feature maps that we will use to perform the region of interest cropping, as well as the size of the crop after rescaling
            # If your backbone returns a Tensor, featmap_names is expected to be ['0']
            # More generally, the backbone should return an OrderedDict[Tensor], and in featmap_names you can choose which feature maps to use
            roi_pooler = torchvision.ops.MultiScaleRoIAlign(
                featmap_names=['0'], output_size=7, sampling_ratio=2)
            mask_roi_pooler = torchvision.ops.MultiScaleRoIAlign(
                featmap_names=['0'], output_size=14, sampling_ratio=2)

            # Put the pieces together inside a MaskRCNN model
            self.model = MaskRCNN(backbone_, min_size=min_img_size, max_size=max_img_size, num_classes=2,
                                  rpn_anchor_generator=anchor_generator, box_roi_pool=roi_pooler, mask_roi_pool=mask_roi_pooler)

        # You can add your backbones here...
        # elif self.backbone == "your_backbone_name"

        return

    def forward(self, inputs, targets=None):

        # Compute outputs
        if targets and self.training:
            outputs = self.model(inputs, targets)

        else:
            outputs = self.model(inputs)

        return outputs

    def summary(self):

        return summary(self.model, (1, 3, 1024, 1024))


@torch.inference_mode()
def evaluate(model, data_loader, device):
    print("Evaluating...")
    n_threads = torch.get_num_threads()
    torch.set_num_threads(1)
    cpu_device = torch.device("cpu")
    model.eval()

    # coco = get_coco_api_from_dataset(data_loader.dataset)
    coco = convert_to_coco_api(data_loader.dataset)
    iou_types = ['bbox', 'segm']
    coco_evaluator = CocoEvaluator(coco, iou_types)

    for images, targets in tqdm(data_loader):
        images = list(img.to(device) for img in images)

        if torch.cuda.is_available():
            torch.cuda.synchronize()
        outputs = model(images)

        outputs = [{k: v.to(cpu_device) for k, v in t.items()}
                   for t in outputs]

        res = {target["image_id"].item(): output for target,
               output in zip(targets, outputs)}
        coco_evaluator.update(res)

    coco_evaluator.synchronize_between_processes()

    # accumulate predictions from all images
    coco_evaluator.accumulate()
    coco_evaluator.summarize()
    torch.set_num_threads(n_threads)
    return coco_evaluator


# Function: Compute VISUM 2022 Competition Metric
def visum2022score(bboxes_mAP, masks_mAP, bboxes_mAP_weight=0.5):

    # Compute masks_mAP_weight from bboxes_mAP_weight
    masks_mAP_weight = 1 - bboxes_mAP_weight

    # Compute score, i.e., score = 0.5*bboxes_mAP + 0.5*masks_mAP
    score = (bboxes_mAP_weight * bboxes_mAP) + (masks_mAP_weight * masks_mAP)

    return score


# Run this file to test the LoggiBarcodeDetectionModel class
if __name__ == "__main__":

    # Model
    model = LoggiBarcodeDetectionModel()

    # Print model summary
    model_summary = summary(model, (1, 3, 1024, 1024))
