# import torch, detectron2
import sys
# sys.path.append('/disk2/transformer')
sys.path.append('/disk2/transformer/efficientdet1')
from detectron2.utils.logger import setup_logger
setup_logger()

# import some common libraries
import numpy as np
import os, json, cv2, random
# from google.colab.patches import cv2_imshow

# import some common detectron2 utilities
# from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor, default_argument_parser
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog, DatasetCatalog

import efficientdet1.model_inspect1 as effi
# from efficientdet import inference
import tensorflow.compat.v1 as tf
# import parser1
from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession
config = ConfigProto()
config.gpu_options.allow_growth = True
session = InteractiveSession(config=config)

# foto = '/disk2/datasets/vis/000001.jpg'
foto = '/disk2/datasets/PennFudanPed/PNGImages/FudanPed00001.png'
im = cv2.imread(foto)
cfg = get_cfg()
# cfg.merge_from_file("configs/COCO-Detection/solo-effi.yaml")
# cfg.merge_from_file("configs/COCO-InstanceSegmentation/effi_mask_rcnn.yaml")
# cfg.MODEL.WEIGHTS = 'output_effimask/test1/model_0004999.pth'

# cfg.merge_from_file("configs/COCO-InstanceSegmentation/effi-mask_sam.yaml")
# cfg.MODEL.WEIGHTS = 'output_effisam/test1/model_0004999.pth'

cfg.merge_from_file("configs/PennFudanPed-InstanceSegmentation/effi_mask_rcnn.yaml")
cfg.MODEL.WEIGHTS = 'output_effimask_penn/model_final.pth'

args = default_argument_parser().parse_args()
# print("Command Line Args:", args)
if 'effi' in cfg.MODEL.BACKBONE.NAME:
    tf.disable_eager_execution()

    inspector = effi.ModelInspector(
        model_name=args.model_name,
        logdir=args.logdir,
        tensorrt=args.tensorrt,
        use_xla=args.xla,
        ckpt_path=args.ckpt_path,
        export_ckpt=args.export_ckpt,
        saved_model_dir=args.saved_model_dir,
        tflite_path=args.tflite_path,
        batch_size=args.batch_size,
        hparams=args.hparams)

predictor = DefaultPredictor(cfg, inspector)
outputs = predictor(im)
v = Visualizer(im[:, :, ::-1], MetadataCatalog.get(cfg.DATASETS.TRAIN[0]), scale=1.2)
out = v.draw_instance_predictions(outputs["instances"].to("cpu"))
# cv2.imshow('Image', )
# filename = '/disk2/datasets/vis/prediction_effi_detectron.jpg'
# filename = '/disk2/datasets/vis/prediction_effimask.jpg'
filename = '/disk2/datasets/vis/prediction_effimask.jpg'
cv2.imwrite(filename, out.get_image()[:, :, ::-1])