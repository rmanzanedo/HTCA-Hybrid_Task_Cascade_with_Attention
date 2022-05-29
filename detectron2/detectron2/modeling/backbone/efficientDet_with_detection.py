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

        self.num_classes = num_classes

        self._out_feature_strides = {'p2': 4, 'p3': 8, 'p4': 16, 'p5': 32, 'p6': 64, 'p7': 128}
        # self._out_feature_channels = {'p2': 40, 'p3': 384, 'p4': 384, 'p5': 384, 'p6': 384, 'p7': 384}
        self._out_feature_channels = {'p2': 32, 'p3': 160, 'p4': 160, 'p5': 160, 'p6': 160, 'p7': 160}


        self._out_features = out_features

        self.stage_names = ['p2', 'p3', 'p4', 'p5', 'p6', 'p7']
        self.inspector = inspector
        self.driver = driver

        # self.preprocessing = transforms.ToPILImage()
        # self.postprocessing = transforms.ToTensor()

    def forward(self, x, training):
        # show_image_from_tensor(x[0].cpu(), 'original')
        # print(torch.max(x), torch.min(x))
        ## det structure = [[image_ids, xmin, ymin, xmax, ymax, nmsed_scores, classes]]
        # feats structure = [features, det]
        feats, detections = self.inspector.saved_model_inference_for_transformer(x.cuda(), self.driver, self.stage_names)

        # show_image_from_tensor(feats['p2'][0][0].unsqueeze(0), 'output_from_effi')
        dim = x.shape[3]/x.shape[2]
        # print(len(feats['p2']))
        batch_size = len(feats['p2'])
        feats_output={}
        if dim <= 1:
            for i in self.stage_names:
                if i in self._out_features:
                    out_dim = feats[i][0].shape
                    for n in range(batch_size):
                        feats_output = add_to_dict(feats_output, i, feats[i][n][:, :, 0:math.ceil(out_dim[-1] * dim)])
                        if n == batch_size-1:
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
                        if n == batch_size-1:
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
                boxes = Boxes(torch.from_numpy(detections[img_det][:, 1:5]).cuda())
                scores_per_img = torch.from_numpy(detections[img_det][:, 5]).cuda()
                clses = torch.from_numpy(detections[img_det][:, 6]).cuda()
                image_size = x.shape[2:4]
                res = Instances(image_size)
                res.proposal_boxes = boxes
                res.objectness_logits = scores_per_img
                res.gt_classes = clses
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
def build_efficientDet_with_detecttions_backbone(cfg, input_shape, inspector = None):

    driver = inference.ServingDriver(
        inspector.model_name,
        inspector.ckpt_path,
        batch_size=inspector.batch_size,
        use_xla=inspector.use_xla,
        model_params=inspector.model_config.as_dict())

    driver.load(inspector.saved_model_dir)

    return EfficientDet_with_detections(inspector, driver, cfg.MODEL.BACKBONE.OUT_FEATURES)