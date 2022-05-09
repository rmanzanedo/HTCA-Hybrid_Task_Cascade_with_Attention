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
sys.path.append('/disk2/transformer/efficientdet')
# import efficientdet.model_inspect1 as effi
from efficientdet import inference
# import tensorflow.compat.v1 as tf
import math

from detectron2.utils.visualizer import show_image_from_tensor


class EfficientDet(Backbone):

    def __init__(self, inspector, driver, out_features=None, num_classes=None):
        super().__init__()

        self.num_classes = num_classes

        self._out_feature_strides = {'p2': 4, 'p3': 8, 'p4': 16, 'p5': 32, 'p6': 64, 'p7': 128}
        # self._out_feature_channels = {'p2': 40, 'p3': 384, 'p4': 384, 'p5': 384, 'p6': 384, 'p7': 384}
        self._out_feature_channels = {'p2': 32, 'p3': 160, 'p4': 160, 'p5': 160, 'p6': 160, 'p7': 160}
        self._out_features = ['p2', 'p3', 'p4', 'p5']

        self.stage_names = ['p2', 'p3', 'p4', 'p5']
        self.inspector = inspector
        self.driver = driver

        # self.preprocessing = transforms.ToPILImage()
        # self.postprocessing = transforms.ToTensor()

    def forward(self, x):
        # show_image_from_tensor(x[0].cpu(), 'original')
        # print(torch.max(x), torch.min(x))
        feats = self.inspector.saved_model_inference_for_transformer(x.cuda(), self.driver, self.stage_names)

        show_image_from_tensor(feats['p2'][0][0].unsqueeze(0), 'output_from_effi')
        dim = x.shape[3]/x.shape[2]
        # print(dim)
        if dim <= 1:
            for i in self.stage_names:
                out_dim = feats[i][0].shape
                # print(out_dim)

                # if i == 'p2':
                #     feat1 = feats[i][0][:, :, 0:math.ceil(out_dim[-1] * dim)]
                #     # print(feat1.shape, math.ceil(out_dim[-1] * dim))
                #     show_image_from_tensor(feat1[0].unsqueeze(0), 'output_from_effi')
                #     del feat1
                feats[i] = torch.stack([feats[i][0][:, :, 0:math.ceil(out_dim[-1] * dim)]]).cuda()


        else:
            for i in self.stage_names:
                out_dim = feats[i][0].shape
                # print(out_dim)

                # if i == 'p2':
                #     feat1 = feats[i][0][:, 0:math.ceil(out_dim[-1] / dim), :]
                #     # print(feat1.shape, math.ceil(out_dim[-1] / dim))
                #     show_image_from_tensor(feat1[0].unsqueeze(0), 'output_from_effi')
                #     del feat1
                feats[i] = torch.stack([feats[i][0][:, 0:math.ceil(out_dim[-1] / dim), :]]).cuda()
        # print('ok')



        # for i in self.stage_names:
        # # out_dim = feats[i][0].shape
        #         # print(out_dim)
        #     feats[i] = torch.stack(feats[i]).cuda()
        # print('ok')


        return feats

    def output_shape(self):
        return {
            name: ShapeSpec(
                channels=self._out_feature_channels[name], stride=self._out_feature_strides[name]
            )
            for name in self._out_features
        }


@BACKBONE_REGISTRY.register()
def build_efficientDet_backbone(cfg, input_shape, inspector = None):

    driver = inference.ServingDriver(
        inspector.model_name,
        inspector.ckpt_path,
        batch_size=inspector.batch_size,
        use_xla=inspector.use_xla,
        model_params=inspector.model_config.as_dict())

    driver.load(inspector.saved_model_dir)

    return EfficientDet(inspector, driver)