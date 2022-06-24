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
from ..utils.visualize import show_image_from_tensor


import sys
sys.path.append('/disk2/htc')
sys.path.append('/disk2/htc/efficientdet')
# import efficientdet.model_inspect1 as effi
from efficientdet import inference
# import tensorflow.compat.v1 as tf
import math


def add_to_dict(d, k, v):
    if k in d.keys():
        d[k].append(v)
    elif k not in d.keys():
        d[k] = [v]
    return d

@BACKBONES.register_module()
class EfficientDet(BaseModule):

    def __init__(self, inspector=None, out_features=None, num_classes=None, style='pytorch',num_stages=4, init_cfg=None):
        super(EfficientDet, self).__init__(init_cfg)

        model_feats = [{'p2': 24, 'p3': 64, 'p4': 64, 'p5': 64, 'p6': 64, 'p7': 64},
                       {'p2': 24, 'p3': 88, 'p4': 88, 'p5': 88, 'p6': 88, 'p7': 88},
                       {'p2': 24, 'p3': 112, 'p4': 112, 'p5': 112, 'p6': 112, 'p7': 112},
                       {'p2': 32, 'p3': 160, 'p4': 160, 'p5': 160, 'p6': 160, 'p7': 160},
                       {'p2': 32, 'p3': 224, 'p4': 224, 'p5': 224, 'p6': 224, 'p7': 224},
                       {'p2': 40, 'p3': 288, 'p4': 288, 'p5': 288, 'p6': 288, 'p7': 288},
                       {'p2': 40, 'p3': 384, 'p4': 384, 'p5': 384, 'p6': 384, 'p7': 384},
                       {'p2': 40, 'p3': 384, 'p4': 384, 'p5': 384, 'p6': 384, 'p7': 384}]

        driver = inference.ServingDriver(
            inspector.model_name,
            inspector.ckpt_path,
            batch_size=inspector.batch_size,
            use_xla=inspector.use_xla,
            model_params=inspector.model_config.as_dict())

        driver.load(inspector.saved_model_dir)

        # self.num_classes = num_classes

        self._out_feature_strides = {'p2': 4, 'p3': 8, 'p4': 16, 'p5': 32, 'p6': 64, 'p7': 128}
        self._out_feature_channels = model_feats[int(inspector.model_name[-1])]
        self._out_features = out_features

        self.stage_names = ['p2', 'p3', 'p4', 'p5']
        self.inspector = inspector
        self.driver = driver

        # self.preprocessing = transforms.ToPILImage()
        # self.postprocessing = transforms.ToTensor()

    def forward(self, x):
        # show_image_from_tensor(x[0].cpu(), 'original')
        print('input shape: {}'.format(x.shape))
        feats = self.inspector.saved_model_inference_for_transformer(x,self.driver, self.stage_names)

        feats1 = ()
        # show_image_from_tensor(feats['p2'][0][0].unsqueeze(0), 'output_from_effi')
        dim = x.shape[3] / x.shape[2]
        batch_size = len(feats['p2'])


        feats_output = {}
        scale_output = []
        if dim <= 1:
            for i in self.stage_names:
                if i in self._out_features:
                    out_dim = feats[i][0].shape
                    scale_output.append(out_dim[-1] / x.shape[2])

                    for n in range(batch_size):
                        feats_output = add_to_dict(feats_output, i, feats[i][n][:, :, 0:math.ceil(out_dim[-1] * dim)])
                        if n == batch_size - 1:
                            feats1 += (torch.stack(feats_output[i]).cuda(), )
                            feats.pop(i)
                else:
                    feats.pop(i)

        else:
            for i in self._out_features:
                if i in self._out_features:
                    out_dim = feats[i][0].shape
                    scale_output.append(out_dim[-1] / x.shape[3])
                    for n in range(batch_size):
                        feats_output = add_to_dict(feats_output, i, feats[i][n][:, 0:math.ceil(out_dim[-1] / dim), :])
                        if n == batch_size - 1:
                            feats1 += (torch.stack(feats_output[i]).cuda(), )
                            feats.pop(i)
                else:
                    feats.pop(i)

        # show_image_from_tensor(feats1[0][0][0].unsqueeze(0).cpu(), 'output_from_effi_1')
        del feats_output

        return feats1, scale_output




def build_efficientDet_backbone(cfg, inspector=None):

    driver = inference.ServingDriver(
        inspector.model_name,
        inspector.ckpt_path,
        batch_size=inspector.batch_size,
        use_xla=inspector.use_xla,
        model_params=inspector.model_config.as_dict())

    driver.load(inspector.saved_model_dir)

    return EfficientDet(inspector, driver)