import numpy as np
import fvcore.nn.weight_init as weight_init
import torch
import torch.nn.functional as F
from torch import nn
from torchvision import transforms

from detectron2.layers import (
    CNNBlockBase,
    Conv2d,
    DeformConv,
    ModulatedDeformConv,
    ShapeSpec,
    get_norm,
)

from .backbone import Backbone
from .build import BACKBONE_REGISTRY

import sys

sys.path.append('/disk2/transformer')
sys.path.append('/disk2/transformer/efficientdet1')
# import efficientdet.model_inspect1 as effi
from efficientdet1 import inference
# import tensorflow.compat.v1 as tf
import math
from detectron2.structures import Boxes, Instances
from typing import List

from detectron2.utils.visualizer import show_image_from_tensor

coco_id_mapping = {1: 1, 2: 2, 3: 3, 4: 4, 5: 5, 6: 6, 7: 7, 8: 8, 9: 9, 10: 10, 11: 11, 13: 12, 14: 13, 15: 14, 16: 15,
                   17: 16, 18: 17, 19: 18, 20: 19, 21: 20, 22: 21, 23: 22, 24: 23, 25: 24, 27: 25, 28: 26, 31: 27,
                   32: 28, 33: 29, 34: 30, 35: 31, 36: 32, 37: 33, 38: 34, 39: 35, 40: 36, 41: 37, 42: 38, 43: 39,
                   44: 40, 46: 41, 47: 42, 48: 43, 49: 44, 50: 45, 51: 46, 52: 47, 53: 48, 54: 49, 55: 50, 56: 51,
                   57: 52, 58: 53, 59: 54, 60: 55, 61: 56, 62: 57, 63: 58, 64: 59, 65: 60, 67: 61, 70: 62, 72: 63,
                   73: 64, 74: 65, 75: 66, 76: 67, 77: 68, 78: 69, 79: 70, 80: 71, 81: 72, 82: 73, 84: 74, 85: 75,
                   86: 76, 87: 77, 88: 78, 89: 79, 90: 80}



def add_to_dict(d, k, v):
    if k in d.keys():
        d[k].append(v)
    elif k not in d.keys():
        d[k] = [v]
        # d[k].append(v)

    return d


class EfficientDet_with_detections(Backbone):

    def __init__(self, inspector, driver, out_features=None, num_classes=None):
        super().__init__()

        model_feats = [{'p2': 24, 'p3': 64, 'p4': 64, 'p5': 64, 'p6': 64, 'p7': 64},
                       {'p2': 24, 'p3': 88, 'p4': 88, 'p5': 88, 'p6': 88, 'p7': 88},
                       {'p2': 24, 'p3': 112, 'p4': 112, 'p5': 112, 'p6': 112, 'p7': 112},
                       {'p2': 32, 'p3': 160, 'p4': 160, 'p5': 160, 'p6': 160, 'p7': 160},
                       {'p2': 32, 'p3': 224, 'p4': 224, 'p5': 224, 'p6': 224, 'p7': 224},
                       {'p2': 40, 'p3': 288, 'p4': 288, 'p5': 288, 'p6': 288, 'p7': 288},
                       {'p2': 40, 'p3': 384, 'p4': 384, 'p5': 384, 'p6': 384, 'p7': 384},
                       {'p2': 40, 'p3': 384, 'p4': 384, 'p5': 384, 'p6': 384, 'p7': 384}]
        self.num_classes = num_classes

        self._out_feature_strides = {'p2': 4, 'p3': 8, 'p4': 16, 'p5': 32, 'p6': 64, 'p7': 128}
        # self._out_feature_channels = {'p2': 40, 'p3': 384, 'p4': 384, 'p5': 384, 'p6': 384, 'p7': 384}
        # self._out_feature_channels = {'p2': 32, 'p3': 160, 'p4': 160, 'p5': 160, 'p6': 160, 'p7': 160}
        self._out_feature_channels = model_feats[int(inspector.model_name[-1])]

        self._out_features = out_features

        self.stage_names = ['p2', 'p3', 'p4', 'p5', 'p6', 'p7']
        self.inspector = inspector
        self.driver = driver

        # self.preprocessing = transforms.ToPILImage()
        # self.postprocessing = transforms.ToTensor()

    def forward(self, x, training):
        # show_image_from_tensor(x[0].cpu(), 'original')
        # print(x.shape)
        ## det structure = [[image_ids, xmin, ymin, xmax, ymax, nmsed_scores, classes]]
        # feats structure = [features, det]
        feats, detections = self.inspector.saved_model_inference_for_transformer(x.cuda(), self.driver,
                                                                                 self.stage_names)

        # show_image_from_tensor(feats['p2'][0][0].unsqueeze(0), 'output_from_effi')
        dim = x.shape[3] / x.shape[2]
        # print(len(feats['p2']))
        batch_size = len(feats['p2'])
        feats_output = {}
        if dim <= 1:
            for i in self.stage_names:
                if i in self._out_features:
                    out_dim = feats[i][0].shape
                    for n in range(batch_size):
                        feats_output = add_to_dict(feats_output, i, feats[i][n][:, :, 0:math.ceil(out_dim[-1] * dim)])
                        if n == batch_size - 1:
                            feats_output[i] = torch.stack(feats_output[i]).cuda()
                            feats.pop(i)

                        # feats[i] = torch.stack([feats[i][n][:, :, 0:math.ceil(out_dim[-1] * dim)]]).cuda()
                else:
                    feats.pop(i)


        else:
            for i in self._out_features:
                if i in self._out_features:
                    out_dim = feats[i][0].shape
                    for n in range(batch_size):
                        feats_output = add_to_dict(feats_output, i, feats[i][n][:, :, 0:math.ceil(out_dim[-1] * dim)])
                        if n == batch_size - 1:
                            feats_output[i] = torch.stack(feats_output[i]).cuda()
                            feats.pop(i)
                    # feats[i] = torch.stack([feats[i][0][:, 0:math.ceil(out_dim[-1] / dim), :]]).cuda()
                else:
                    feats.pop(i)

        results: List[Instances] = []
        # print(detections[:, 0])
        if training:
            for n in range(x.shape[0]):
                img_det = np.where(detections[:, 0] == n)
                # print(detections[img_det][:, 1:5])
                # print(detections[img_det])
                # quit()
                coco_clases = []
                for i in detections[img_det][:, 6]:
                    coco_clases.append(coco_id_mapping[i]-1)

                boxes = Boxes(torch.from_numpy(detections[img_det][:, 1:5]).cuda())
                scores_per_img = torch.from_numpy(detections[img_det][:, 5]).cuda()
                clses = torch.from_numpy(np.asarray(coco_clases)).cuda()
                image_size = x.shape[2:4]
                res = Instances(image_size)
                res.proposal_boxes = boxes
                res.objectness_logits = scores_per_img
                res.gt_classes = clses
                # res.pred_classes = clses
                results.append(res)
        else:
            for n in range(x.shape[0]):
                img_det = np.where(detections[:, 0] == n)
                # print(detections[img_det][:, 1:5])
                # print(detections[img_det])
                # quit()
                coco_clases = []
                for i in detections[img_det][:, 6]:
                    coco_clases.append(coco_id_mapping[i]-1)

                boxes = Boxes(torch.from_numpy(detections[img_det][:, 1:5]).cuda())
                scores_per_img = torch.from_numpy(detections[img_det][:, 5]).cuda()
                clses = torch.from_numpy(np.asarray(coco_clases)).to(torch.int64).cuda()
                image_size = x.shape[2:4]
                res = Instances(image_size)
                res.pred_boxes = boxes
                res.scores = scores_per_img
                res.pred_classes = clses
                results.append(res)

        # for i in self.stage_names:
        # # out_dim = feats[i][0].shape
        #         # print(out_dim)
        #     feats[i] = torch.stack(feats[i]).cuda()
        # print('ok')
        # print(results)
        # quit()
        # print(feats_output['p3'].shape)
        # quit()

        return feats_output, results

    def output_shape(self):
        return {
            name: ShapeSpec(
                channels=self._out_feature_channels[name], stride=self._out_feature_strides[name]
            )
            for name in self._out_features
        }


@BACKBONE_REGISTRY.register()
def build_efficientDet_with_detecttions_backbone(cfg, input_shape, inspector=None):
    driver = inference.ServingDriver(
        inspector.model_name,
        inspector.ckpt_path,
        batch_size=inspector.batch_size,
        use_xla=inspector.use_xla,
        model_params=inspector.model_config.as_dict())

    # driver.load(inspector.saved_model_dir)

    return EfficientDet_with_detections(inspector, driver, cfg.MODEL.BACKBONE.OUT_FEATURES)
