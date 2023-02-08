import argparse
import os

import cv2
import mmcv
import torch
import tqdm
from mmcv import Config, DictAction
from mmcv.cnn import fuse_conv_bn
from mmcv.parallel import MMDataParallel, MMDistributedDataParallel
from mmcv.runner import get_dist_info, init_dist, load_checkpoint
# from mmdet.core import wrap_fp16_model
from mmdet.datasets import build_dataset


def parse_args():
    parser = argparse.ArgumentParser(description='pcan test model')
    parser.add_argument('--config',
                        default=r'D:\Pycharm_Programms_New\PCAN\pcan\configs\segtrack-frcnn_r50_fpn_12e_bdd10k_fixed_pcan.py',
                        help='test config file path')
    parser.add_argument('--checkpoint',
                        default=r'D:\Pycharm_Programms_New\PCAN\pcan\pcan_pretrained_model.pth',
                        help='checkpoint file')
    parser.add_argument('--out', help='output result file')
    parser.add_argument(
        '--fuse-conv-bn',
        action='store_true',
        help='Whether to fuse conv and bn, this will slightly increase'
        'the inference speed')
    parser.add_argument(
        '--format-only',
        action='store_true',
        help='Format the output results without perform evaluation. It is'
        'useful when you want to format the result to a specific format and '
        'submit it to the test server')
    parser.add_argument('--webcam', default=True, action='store_true', help='use webcam')
    parser.add_argument('--eval', type=str, nargs='+', help='eval types')
    parser.add_argument('--show', action='store_true', help='show results')
    parser.add_argument(
        '--show-dir', help='directory where painted images will be saved')
    parser.add_argument(
        '--gpu-collect',
        action='store_true',
        help='whether to use gpu to collect results.')
    parser.add_argument(
        '--tmpdir',
        help='tmp directory used for collecting results from multiple '
        'workers, available when gpu-collect is not specified')
    parser.add_argument(
        '--show_score_thr', default=0.3, type=float, help='output result file')
    parser.add_argument(
        '--cfg-options',
        nargs='+',
        action=DictAction,
        help='override some settings in the used config, the key-value pair '
        'in xxx=yyy format will be merged into config file.')
    parser.add_argument(
        '--eval-options',
        nargs='+',
        action=DictAction,
        help='custom options for evaluation, the key-value pair in xxx=yyy '
        'format will be kwargs for dataset.evaluate() function')
    parser.add_argument(
        '--launcher',
        choices=['none', 'pytorch', 'slurm', 'mpi'],
        default='none',
        help='job launcher')
    parser.add_argument('--local_rank', type=int, default=0)
    args = parser.parse_args()
    if 'LOCAL_RANK' not in os.environ:
        os.environ['LOCAL_RANK'] = str(args.local_rank)
    return args


def main():
    args = parse_args()

    assert args.out or args.eval or args.format_only or args.show \
        or args.show_dir or args.webcam, \
        ('Please specify at least one operation (save/eval/format/show the '
         'results / save the results) with the argument "--out", "--eval"'
         ', "--format-only", "--show" or "--show-dir"')

    if args.eval and args.format_only:
        raise ValueError('--eval and --format_only cannot be both specified')

    if args.out is not None and not args.out.endswith(('.pkl', '.pickle')):
        raise ValueError('The output file must be a pkl file.')

    cfg = Config.fromfile(args.config)

    if args.cfg_options is not None:
        cfg.merge_from_dict(args.cfg_options)

    if cfg.get('USE_MMDET', False):
        from mmdet.apis import multi_gpu_test, single_gpu_test
        from mmdet.models import build_detector as build_model
        from mmdet.datasets import build_dataloader
    else:
        from pcan.apis import multi_gpu_test, single_gpu_test
        from pcan.models import build_model
        from pcan.datasets import build_dataloader

    # set cudnn_benchmark
    if cfg.get('cudnn_benchmark', False):
        torch.backends.cudnn.benchmark = True
    cfg.model.pretrained = None
    cfg.data.test.test_mode = True

    # init distributed env first, since logger depends on the dist info.
    if args.launcher == 'none':
        distributed = False
    else:
        distributed = True
        init_dist(args.launcher, **cfg.dist_params)

    # build the dataloader
    dataset = build_dataset(cfg.data.test)
    data_loader = build_dataloader(
        dataset,
        samples_per_gpu=1,
        workers_per_gpu=cfg.data.workers_per_gpu,
        dist=distributed,
        shuffle=False)

    # build the model and load checkpoint
    model = build_model(cfg.model, train_cfg=None, test_cfg=None)
    # fp16_cfg = cfg.get('fp16', None)
    # if fp16_cfg is not None:
    #     wrap_fp16_model(model)
    checkpoint = load_checkpoint(model, args.checkpoint, map_location='cpu')

    if args.fuse_conv_bn:
        model = fuse_conv_bn(model)

    if 'CLASSES' in checkpoint['meta']:
        model.CLASSES = checkpoint['meta']['CLASSES']

    model.eval()
    cam = cv2.VideoCapture(0)


    for vis in tqdm.tqdm(model(cam)):
        cv2.imshow('img', vis)
        if cv2.waitKey(1) == 27:
            break  # esc to quit
    cam.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
