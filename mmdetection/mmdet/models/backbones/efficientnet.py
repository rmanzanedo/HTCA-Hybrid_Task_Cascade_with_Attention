# import numpy as np
# import fvcore.nn.weight_init as weight_init
import torch
# import torch.nn.functional as F
# from torch import nn
# from torchvision import transforms
#
# import warnings
#
# import torch.utils.checkpoint as cp
# from mmcv.cnn import build_conv_layer, build_norm_layer, build_plugin_layer
from mmcv.runner import BaseModule
# from torch.nn.modules.batchnorm import _BatchNorm

from ..builder import BACKBONES
# from ..utils import ResLayer


import sys
sys.path.append('/disk2/htc')
sys.path.append('/disk2/htc/efficientdet')
# import efficientdet.model_inspect1 as effi
from efficientdet import inference
# import tensorflow.compat.v1 as tf
import math

@BACKBONES.register_module()
class EfficientDet(BaseModule):

    def __init__(self, inspector=None, out_features=None, num_classes=None, style='pytorch',num_stages=4, init_cfg=None):
        super(EfficientDet, self).__init__(init_cfg)

        driver = inference.ServingDriver(
            inspector.model_name,
            inspector.ckpt_path,
            batch_size=inspector.batch_size,
            use_xla=inspector.use_xla,
            model_params=inspector.model_config.as_dict())

        driver.load(inspector.saved_model_dir)

        # self.num_classes = num_classes

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

        feats = self.inspector.saved_model_inference_for_transformer(x,self.driver, self.stage_names)

        feats1 = ()

        dim = x.shape[3] / x.shape[2]
        # print(type(feats['p2'][0]))
        if dim < 1:
            for i in self.stage_names:
                out_dim = feats[i][0].shape
                # print(out_dim)
                feats1 += (torch.stack([feats[i][0][:, :, 0:math.ceil(out_dim[-2] * dim)]]).cuda(), )
                feats.pop(i)
        elif dim > 1:
            for i in self.stage_names:
                out_dim = feats[i][0].shape
                # print(out_dim)
                feats1 += (torch.stack([feats[i][0][:, 0:math.ceil(out_dim[-1] / dim), :]]).cuda(), )
                feats.pop(i)
        else:
            for i in self.stage_names:
                out_dim = feats[i][0].shape
                # print(out_dim)
                feats1 += (torch.stack(feats[i]).cuda(), )
                feats.pop(i)

        # for i in self.stage_names:
        #     # feats[i] = torch.stack(feats[i]).cuda()
        #
        #     feats1 += (torch.stack(feats[i]).cuda(),)
        #     feats.pop(i)
        # print('ok')

        return feats1




def build_efficientDet_backbone(cfg, inspector=None):

    driver = inference.ServingDriver(
        inspector.model_name,
        inspector.ckpt_path,
        batch_size=inspector.batch_size,
        use_xla=inspector.use_xla,
        model_params=inspector.model_config.as_dict())

    driver.load(inspector.saved_model_dir)

    return EfficientDet(inspector, driver)