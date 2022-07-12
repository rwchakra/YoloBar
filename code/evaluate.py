import torch
print(torch.__version__, torch.cuda.is_available())
# assert torch.__version__.startswith("1.7")
# Some basic setup:
# Setup detectron2 logger
import detectron2
from detectron2.utils.logger import setup_logger

# import some common libraries
import numpy as np
import matplotlib.pyplot as plt
import os, json, cv2, random
#from google.colab.patches import cv2_imshow

# import some common detectron2 utilities
from detectron2.engine import DefaultTrainer
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog, DatasetCatalog
from detectron2.data.datasets import register_coco_instances


from detectron2.evaluation import COCOEvaluator, inference_on_dataset
from detectron2.data import build_detection_test_loader

from coco_utils import save_data_coco_style

setup_logger()

def register_test_set(test_dir_root):
    # json_file = os.path.join(
    #             self.data_dir, "json", "challenge", "test_challenge.json")
    #         self.imgs_path = os.path.join(self.data_dir, "raw")
    #         self.masks_path = os.path.join(self.data_dir, "masks", "test")
    ROOT_DIR = "/data/"
    # ANNO_PATH = "/home/sinan/datasets/instance_segm/train_coco_20210212"
    train_json_file = os.path.join(ROOT_DIR,"coco/test_annotations.json")
    train_img_dir = os.path.join(ROOT_DIR ,"raw")
    register_coco_instances("coco_val", {}, train_json_file, train_img_dir)
# register_coco_instances("coco_val", {}, val_json_file, val_img_dir)

def update_cfg_file(cfg):
    cfg = get_cfg()
    cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"))
    cfg.DATASETS.TRAIN = ("coco_val",)
    # cfg.DATASETS.TEST = ("coco_val",)
    cfg.DATASETS.TEST = ()
    cfg.DATALOADER.NUM_WORKERS = 1
    # cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml")  # Let training initialize from model zoo
    cfg.SOLVER.IMS_PER_BATCH = 4
    cfg.TEST.EVAL_PERIOD = 1000
    cfg.SOLVER.CHECKPOINT_PERIOD = 1000
    cfg.SOLVER.BASE_LR = 0.00025  # pick a good LR
    cfg.SOLVER.MAX_ITER = 2500 #100000 #10000    # 300 iterations seems good enough for this toy dataset; you will need to train longer for a practical dataset
    cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 32   # faster, and good enough for this toy dataset (default: 512)
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = 2 #3  # only has one class (ballon). (see https://detectron2.readthedocs.io/tutorials/datasets.html#update-the-config-for-new-datasets)
    # NOTE: this config means the number of classes, but a few popular unofficial tutorials incorrect uses num_classes+1 here.
    cfg.INPUT.MIN_SIZE_TRAIN = (1024, 1024)
    cfg.INPUT.MAX_SIZE_TRAIN = 1024
    cfg.INPUT.MIN_SIZE_TEST = 1024
    cfg.INPUT.MAX_SIZE_TEST = 1024
    cfg.MODEL.WEIGHTS = os.path.join('./models', "model_final.pth")
    #cfg.MODEL.WEIGHTS = os.path.join(cfg.OUTPUT_DIR, "model_final.pth")  # path to the model we just trained
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.2   # set a custom testing threshold
    cfg.MODEL.ROI_HEADS.NMS_THRESH_TEST = 0.5
    return cfg
   
def get_predictor():
    predictor = DefaultPredictor(update_cfg_file())
    return predictor

def get_trainer_model(cfg):
    trainer = DefaultTrainer(cfg)
    trainer.resume_or_load(resume=True)
    return trainer

def evaluate_model():
    cfg = update_cfg_file()
    trainer =get_trainer_model(cfg)
    evaluator = COCOEvaluator("coco_test", ("bbox", "segm"), False, output_dir=cfg.OUTPUT_DIR )
    val_loader = build_detection_test_loader(cfg, "coco_test")
    res = inference_on_dataset(trainer.model, val_loader, evaluator)
    print(res)
    # print(inference_on_dataset(trainer.model, val_loader, evaluator))
def main():
    print('Saving data in coco format...')
    save_data_coco_style()
    print('Evaluating the moodel...')
    evaluate_model()
    
if __name__ == "__main__":
    main()