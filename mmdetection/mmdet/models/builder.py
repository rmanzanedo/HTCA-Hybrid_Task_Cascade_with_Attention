# Copyright (c) OpenMMLab. All rights reserved.
import warnings

from mmcv.cnn import MODELS as MMCV_MODELS
from mmcv.utils import Registry

MODELS = Registry('models', parent=MMCV_MODELS)

BACKBONES = MODELS
NECKS = MODELS
ROI_EXTRACTORS = MODELS
SHARED_HEADS = MODELS
HEADS = MODELS
LOSSES = MODELS
DETECTORS = MODELS


def build_backbone(cfg, inspector=None):
    """Build backbone."""

    if inspector is not None:
        return BACKBONES.build1(inspector, cfg, default_args=dict(inspector=inspector))
    return BACKBONES.build(cfg)


def build_neck(cfg, inspector=None):
    """Build neck."""
    if inspector is not None:
        return NECKS.build(cfg, inspector)
    return NECKS.build(cfg)


def build_roi_extractor(cfg):
    """Build roi extractor."""
    return ROI_EXTRACTORS.build(cfg)


def build_shared_head(cfg):
    """Build shared head."""
    return SHARED_HEADS.build(cfg)


def build_head(cfg):
    """Build head."""
    return HEADS.build(cfg)


def build_loss(cfg):
    """Build loss."""
    return LOSSES.build(cfg)


def build_detector(cfg, train_cfg=None, test_cfg=None, inspector=None):
    """Build detector."""
    if train_cfg is not None or test_cfg is not None:
        warnings.warn(
            'train_cfg and test_cfg is deprecated, '
            'please specify them in model', UserWarning)
    assert cfg.get('train_cfg') is None or train_cfg is None, \
        'train_cfg specified in both outer field and model field '
    assert cfg.get('test_cfg') is None or test_cfg is None, \
        'test_cfg specified in both outer field and model field '
    ######################################### Mi codigo  ###################################################
    # if inspector is None:
    #     return DETECTORS.build(
    #         cfg, default_args=dict(train_cfg=train_cfg, test_cfg=test_cfg))
    # else:
    #     return DETECTORS.build1(
    #         cfg, default_args=dict(train_cfg=train_cfg, test_cfg=test_cfg, inspector=inspector))

    ############################################Original ########################################

    return DETECTORS.build(
        cfg, default_args=dict(train_cfg=train_cfg, test_cfg=test_cfg, inspector=inspector))