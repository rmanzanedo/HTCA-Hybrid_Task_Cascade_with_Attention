from __future__ import absolute_import
import argparse


def arg_parse():
    parser = argparse.ArgumentParser(description='all together')

    # Datasets parameters
    parser.add_argument('--seg_dir', type=str, default='./efficientdet2/VOCdevkit/VOC2012',
                        help="root path to data directory")
    parser.add_argument('--img_dir', type=str, default='./efficientdet2/VOCdevkit/VOC2012',
                        help="root path to data directory")
    parser.add_argument('--train_set',type=int, default=2)
    parser.add_argument('--val_set', type=int, default=8)
    parser.add_argument('--workers', default=1, type=int,
                        help="number of data loading workers (default: 4)")

    # training parameters
    parser.add_argument('--gpu', default=0, type=int,
                        help='gpu device ids for CUDA_VISIBLE_DEVICES')
    parser.add_argument('--epoch', default=120, type=int,
                        help="num of validation iterations")
    parser.add_argument('--val_epoch', default=1, type=int,
                        help="num of validation iterations")
    parser.add_argument('--train_batch', default=10, type=int,
                        help="train batch size")
    parser.add_argument('--test_batch', default=16, type=int,
                        help="test batch size")
    parser.add_argument('--lr', default=0.02, type=float,
                        help="initial learning rate")
    parser.add_argument('--weight-decay', default=0.0005, type=float,
                        help="initial learning rate")
    parser.add_argument('--vis', default=True, type=bool, help="show a pic comparing pred vs gt")
    parser.add_argument('--checkpoint', default='segmentation/checkpoint_coco_bg_person', help='trained model')
    parser.add_argument('--load_pretrained', type=bool, default=True)

    # resume trained model
    parser.add_argument('--resume', type=str, default='log/best_model.pth.tar',
                        help="path to the trained model")
    # others
    parser.add_argument('--save_dir', type=str, default='log')
    parser.add_argument('--random_seed', type=int, default=999)

    #efficiendet

    parser.add_argument('--model_name', default='efficientdet2-d7', help='Model.')
    parser.add_argument('--logdir', default='log', help='log directory.')
    parser.add_argument('--runmode', default='saved_model_infer', help='Run mode: {freeze, bm, dry}')
    parser.add_argument('--trace_filename', default=None, help='Trace file name.')

    parser.add_argument('--threads', default=0, help='Number of threads.')
    parser.add_argument('--bm_runs', default=10, help='Number of benchmark runs.')
    parser.add_argument('--tensorrt', default=None, help='TensorRT mode: {None, FP32, FP16, INT8}')
    parser.add_argument('--delete_logdir', default=True, help='Whether to delete logdir.')
    parser.add_argument('--freeze', default=False, help='Freeze graph.')
    parser.add_argument('--xla', default=False, help='Run with xla optimization.')
    parser.add_argument('--batch_size', default=1, help='Batch size for inference.')

    parser.add_argument('--ckpt_path', default=None, help='checkpoint dir used for eval.')
    parser.add_argument('--export_ckpt', default=None, help='Path for exporting new models.')

    parser.add_argument('--hparams', default='',
                        help='Comma separated k=v pairs of hyperparameters or a module containing attributes to use as hyperparameters.')

    parser.add_argument('--input_image', default='seg', help='Input image path for inference.')
    parser.add_argument('--output_image_dir', default='serve_image_out', help='Output dir for inference.')

    # For video.
    parser.add_argument('--input_video', default=None, help='Input video path for inference.')
    parser.add_argument('--output_video', default=None, help='Output video path. If None, play it online instead.')

    # For visualization.
    parser.add_argument('--line_thickness', default=None, help='Line thickness for box.')
    parser.add_argument('--max_boxes_to_draw', default=None, help='Max number of boxes to draw.')
    parser.add_argument('--min_score_thresh', default=None, help='Score threshold to show box.')

    # For saved model.
    parser.add_argument('--saved_model_dir', default='efficientdet2/saved_model/efficientdet2-d7_frozen.pb',
                        help='Folder path for saved model.')
    parser.add_argument('--tflite_path', default=None, help='Path for exporting tflite file.')



    parser.add_argument('--dataset', default='coco', type=str)
    parser.add_argument('--change_lr', default=5, type=str,help="training dataset")
    parser.add_argument('--epoch_eval',type=str, default='6')
    # own visualization
    parser.add_argument('--pattern_vis', default='../vis/*.jpg', type=str)
    parser.add_argument('--vis_batch', type=int, default=8)

    args = parser.parse_args()

    return args
