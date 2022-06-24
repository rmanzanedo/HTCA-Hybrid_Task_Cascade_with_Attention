# Copyright (c) OpenMMLab. All rights reserved.
import argparse

import numpy as np
import torch
from mmcv import Config, DictAction

from mmdet.models import build_detector

import sys
sys.path.append('/disk2/htc')
sys.path.append('/disk2/htc/efficientdet')
import efficientdet.model_inspect1 as effi
# from efficientdet import inference
import tensorflow.compat.v1 as tf
# import parser1

from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession
config = ConfigProto()
config.gpu_options.allow_growth = True
session = InteractiveSession(config=config)

try:
    from mmcv.cnn import get_model_complexity_info
except ImportError:
    raise ImportError('Please upgrade mmcv to >0.6.2')


def parse_args():
    parser = argparse.ArgumentParser(description='Train a detector')
    parser.add_argument('config', help='train config file path')
    parser.add_argument(
        '--shape',
        type=int,
        nargs='+',
        default=[1280, 800],
        help='input image size')
    parser.add_argument(
        '--cfg-options',
        nargs='+',
        action=DictAction,
        help='override some settings in the used config, the key-value pair '
        'in xxx=yyy format will be merged into config file. If the value to '
        'be overwritten is a list, it should be like key="[a,b]" or key=a,b '
        'It also allows nested list/tuple values, e.g. key="[(a,b),(c,d)]" '
        'Note that the quotation marks are necessary and that no white space '
        'is allowed.')
    parser.add_argument(
        '--size-divisor',
        type=int,
        default=32,
        help='Pad the input image, the minimum size that is divisible '
        'by size_divisor, -1 means do not pad the image.')

    parser.add_argument("--model_name", default="efficientdet-d7", help="Model.", )
    parser.add_argument("--logdir", default="log", help="log directory.")
    parser.add_argument("--runmode", default="saved_model_infer", help="Run mode: {freeze, bm, dry}")
    parser.add_argument("--trace_filename", default=None, help="Trace file name.")

    parser.add_argument("--threads", default=0, help="Number of threads.")
    parser.add_argument("--bm_runs", default=10, help="Number of benchmark runs.")
    parser.add_argument("--tensorrt", default=None, help="TensorRT mode: {None, FP32, FP16, INT8}")
    parser.add_argument("--delete_logdir", default=True, help="Whether to delete logdir.")
    parser.add_argument("--freeze", default=False, help="Freeze graph.")
    parser.add_argument("--xla", default=False, help="Run with xla optimization.")
    parser.add_argument("--batch_size", default=1, help="Batch size for inference.")

    parser.add_argument("--ckpt_path", default=None, help="checkpoint dir used for eval.")
    parser.add_argument("--export_ckpt", default=None, help="Path for exporting new models.")

    parser.add_argument("--hparams", default="",
                        help="Comma separated k=v pairs of hyperparameters or a module containing attributes to use as hyperparameters.")
    # For saved model.
    parser.add_argument("--saved_model_dir", default="/disk2/transformer/efficientdet/saved_model_only_feats/",
                        help="Folder path for saved model.")
    parser.add_argument("--tflite_path", default=None, help="Path for exporting tflite file.")

    args = parser.parse_args()
    return args


def main():

    args = parse_args()

    if len(args.shape) == 1:
        h = w = args.shape[0]
    elif len(args.shape) == 2:
        h, w = args.shape
    else:
        raise ValueError('invalid input shape')
    orig_shape = (3, h, w)
    divisor = args.size_divisor
    if divisor > 0:
        h = int(np.ceil(h / divisor)) * divisor
        w = int(np.ceil(w / divisor)) * divisor

    input_shape = (3, h, w)

    cfg = Config.fromfile(args.config)
    if args.cfg_options is not None:
        cfg.merge_from_dict(args.cfg_options)

    if 'effi' in args.config:
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

    model = build_detector(
        cfg.model,
        train_cfg=cfg.get('train_cfg'),
        test_cfg=cfg.get('test_cfg'),
        inspector= inspector)
    if torch.cuda.is_available():
        model.cuda()
    model.eval()

    if hasattr(model, 'forward_dummy'):
        model.forward = model.forward_dummy
    else:
        raise NotImplementedError(
            'FLOPs counter is currently not currently supported with {}'.
            format(model.__class__.__name__))

    flops, params = get_model_complexity_info(model, input_shape)
    split_line = '=' * 30

    if divisor > 0 and \
            input_shape != orig_shape:
        print(f'{split_line}\nUse size divisor set input shape '
              f'from {orig_shape} to {input_shape}\n')
    print(f'{split_line}\nInput shape: {input_shape}\n'
          f'Flops: {flops}\nParams: {params}\n{split_line}')
    print('!!!Please be cautious if you use the results in papers. '
          'You may need to check if all ops are supported and verify that the '
          'flops computation is correct.')


if __name__ == '__main__':
    main()
