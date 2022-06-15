# -*- coding: utf-8 -*-
# Copyright (c) Facebook, Inc. and its affiliates.
# Modified by Bowen Cheng from https://github.com/facebookresearch/detectron2/blob/main/tools/analyze_model.py
import sys
sys.path.append('/disk2/transformer')
sys.path.append('/disk2/transformer/efficientdet')
sys.path.append('/disk2/transformer/detectron2')

import logging
import numpy as np
from collections import Counter
import tqdm
from fvcore.nn import flop_count_table  # can also try flop_count_str

from detectron2.checkpoint import DetectionCheckpointer
from detectron2.config import CfgNode, LazyConfig, get_cfg, instantiate
from detectron2.data import build_detection_test_loader
from detectron2.engine import default_argument_parser
from detectron2.modeling import build_model, build_model1
from detectron2.projects.deeplab import add_deeplab_config
from detectron2.utils.analysis import (
    FlopCountAnalysis,
    activation_count_operators,
    parameter_count_table,
)
from detectron2.utils.logger import setup_logger

# fmt: off
import os
import sys
sys.path.insert(1, os.path.join(sys.path[0], '..'))
# fmt: on

from mask2former import add_maskformer2_config


import efficientdet.model_inspect1 as effi
# from efficientdet import inference
import tensorflow.compat.v1 as tf
# import parser1

from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession
config = ConfigProto()
config.gpu_options.allow_growth = True
session = InteractiveSession(config=config)

logger = logging.getLogger("detectron2")


def setup(args):
    if args.config_file.endswith(".yaml"):
        cfg = get_cfg()
        add_deeplab_config(cfg)
        add_maskformer2_config(cfg)
        cfg.merge_from_file(args.config_file)
        cfg.DATALOADER.NUM_WORKERS = 0
        cfg.merge_from_list(args.opts)
        cfg.freeze()
    else:
        cfg = LazyConfig.load(args.config_file)
        cfg = LazyConfig.apply_overrides(cfg, args.opts)
    setup_logger(name="fvcore")
    setup_logger()
    return cfg


def do_flop(cfg, inspector = None):
    if isinstance(cfg, CfgNode):
        data_loader = build_detection_test_loader(cfg, cfg.DATASETS.TEST[0])
        # model = build_model(cfg)
        model = build_model1(cfg, inspector)
        DetectionCheckpointer(model).load(cfg.MODEL.WEIGHTS)
    else:
        data_loader = instantiate(cfg.dataloader.test)
        model = instantiate(cfg.model)
        model.to(cfg.train.device)
        DetectionCheckpointer(model).load(cfg.train.init_checkpoint)
    model.eval()

    counts = Counter()
    total_flops = []
    for idx, data in zip(tqdm.trange(args.num_inputs), data_loader):  # noqa
        if args.use_fixed_input_size and isinstance(cfg, CfgNode):
            import torch
            crop_size = cfg.INPUT.CROP.SIZE[0]
            data[0]["image"] = torch.zeros((3, crop_size, crop_size))
        flops = FlopCountAnalysis(model, data)
        if idx > 0:
            flops.unsupported_ops_warnings(False).uncalled_modules_warnings(False)
        counts += flops.by_operator()
        total_flops.append(flops.total())

    logger.info("Flops table computed from only one input sample:\n" + flop_count_table(flops))
    logger.info(
        "Average GFlops for each type of operators:\n"
        + str([(k, v / (idx + 1) / 1e9) for k, v in counts.items()])
    )
    logger.info(
        "Total GFlops: {:.1f}±{:.1f}".format(np.mean(total_flops) / 1e9, np.std(total_flops) / 1e9)
    )


def do_activation(cfg, inspector = None):
    if isinstance(cfg, CfgNode):
        data_loader = build_detection_test_loader(cfg, cfg.DATASETS.TEST[0])
        model = build_model(cfg)
        DetectionCheckpointer(model).load(cfg.MODEL.WEIGHTS)
    else:
        data_loader = instantiate(cfg.dataloader.test)
        model = instantiate(cfg.model)
        model.to(cfg.train.device)
        DetectionCheckpointer(model).load(cfg.train.init_checkpoint)
    model.eval()

    counts = Counter()
    total_activations = []
    for idx, data in zip(tqdm.trange(args.num_inputs), data_loader):  # noqa
        count = activation_count_operators(model, data)
        counts += count
        total_activations.append(sum(count.values()))
    logger.info(
        "(Million) Activations for Each Type of Operators:\n"
        + str([(k, v / idx) for k, v in counts.items()])
    )
    logger.info(
        "Total (Million) Activations: {}±{}".format(
            np.mean(total_activations), np.std(total_activations)
        )
    )


def do_parameter(cfg, inspector = None):
    if isinstance(cfg, CfgNode):
        model = build_model(cfg)
    else:
        model = instantiate(cfg.model)
    logger.info("Parameter Count:\n" + parameter_count_table(model, max_depth=5))


def do_structure(cfg, inspector = None):
    if isinstance(cfg, CfgNode):
        model = build_model(cfg)
    else:
        model = instantiate(cfg.model)
    logger.info("Model Structure:\n" + str(model))


if __name__ == "__main__":
    parser = default_argument_parser(
        epilog="""
Examples:
To show parameters of a model:
$ ./analyze_model.py --tasks parameter \\
    --config-file ../configs/COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_1x.yaml
Flops and activations are data-dependent, therefore inputs and model weights
are needed to count them:
$ ./analyze_model.py --num-inputs 100 --tasks flop \\
    --config-file ../configs/COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_1x.yaml \\
    MODEL.WEIGHTS /path/to/model.pkl
"""
    )
    parser.add_argument(
        "--tasks",
        choices=["flop", "activation", "parameter", "structure"],
        required=True,
        nargs="+",
    )
    parser.add_argument(
        "-n",
        "--num-inputs",
        default=100,
        type=int,
        help="number of inputs used to compute statistics for flops/activations, "
        "both are data dependent.",
    )
    parser.add_argument(
        "--use-fixed-input-size",
        action="store_true",
        help="use fixed input size when calculating flops",
    )
    # parser.add_argument("--model_name", default="efficientdet-d7", help="Model.",)
    # parser.add_argument("--logdir", default="log", help="log directory.")
    # parser.add_argument("--runmode", default="saved_model_infer", help="Run mode: {freeze, bm, dry}")
    # parser.add_argument("--trace_filename", default=None, help="Trace file name.")
    #
    # parser.add_argument("--threads", default=0, help="Number of threads.")
    # parser.add_argument("--bm_runs", default=10, help="Number of benchmark runs.")
    # parser.add_argument("--tensorrt", default=None, help="TensorRT mode: {None, FP32, FP16, INT8}")
    # parser.add_argument("--delete_logdir", default=True, help="Whether to delete logdir.")
    # parser.add_argument("--freeze", default=False, help="Freeze graph.")
    # parser.add_argument("--xla", default=False, help="Run with xla optimization.")
    # parser.add_argument("--batch_size", default=1, help="Batch size for inference.")
    #
    # parser.add_argument("--ckpt_path", default=None, help="checkpoint dir used for eval.")
    # parser.add_argument("--export_ckpt", default=None, help="Path for exporting new models.")
    #
    # parser.add_argument("--hparams", default="",
    #                     help="Comma separated k=v pairs of hyperparameters or a module containing attributes to use as hyperparameters.")
    # # For saved model.
    # parser.add_argument("--saved_model_dir", default="/disk2/transformer/efficientdet/saved_model_only_feats/",
    #                     help="Folder path for saved model.")
    # parser.add_argument("--tflite_path", default=None, help="Path for exporting tflite file.")

    args = parser.parse_args()
    assert not args.eval_only
    assert args.num_gpus == 1

    '''setup efficientdet'''
    if 'effi' in args.config_file:
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
    else:
        inspector = None

    cfg = setup(args)

    for task in args.tasks:
        {
            "flop": do_flop,
            "activation": do_activation,
            "parameter": do_parameter,
            "structure": do_structure,
        }[task](cfg, inspector)
