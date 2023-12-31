import sys
import time
import os.path as osp
import argparse
import torch
import torch.nn as nn

import torchreid
from torchreid.utils import (
    Logger, check_isfile, set_random_seed, collect_env_info,
    resume_from_checkpoint, load_pretrained_weights, compute_model_complexity
)

from default_config import (
    imagedata_kwargs, optimizer_kwargs, videodata_kwargs, engine_run_kwargs,
    get_default_config, lr_scheduler_kwargs
)

from models.resnet_ms import resnet50_fc512, resnet50_fc512_ms12_a0d2, resnet50_fc512_ms12_a0d1, resnet50_fc512_ms12_a0d3
from models.resnet_ms import resnet50_fc512_ms1_a0d1, resnet50_fc512_ms123_a0d1, resnet50_fc512_ms1234_a0d1, resnet50_fc512_ms23_a0d1, resnet50_fc512_ms14_a0d1
from models.resnet_csu import resnet50_fc512_csu12_a0d2, resnet50_fc512_csu12_a0d1, resnet50_fc512_csu12_a0d3, resnet50_fc512_csu12_a0d5, resnet50_fc512_csu012345_a0d3
from models.resnet_csu import resnet50_fc512_csu01_a0d1, resnet50_fc512_csu12_a0d1, resnet50_fc512_csu23_a0d1, resnet50_fc512_csu34_a0d1, resnet50_fc512_csu45_a0d1
from models.resnet_csu import resnet50_fc512_csu012_a0d1, resnet50_fc512_csu123_a0d1, resnet50_fc512_csu234_a0d1, resnet50_fc512_csu345_a0d1
from models.resnet_csu import resnet50_fc512_csu0123_a0d1,resnet50_fc512_csu1234_a0d1, resnet50_fc512_csu2345_a0d1
from models.resnet_csu import resnet50_fc512_csu01234_a0d1,resnet50_fc512_csu12345_a0d1, resnet50_fc512_csu012345_a0d1
from models.resnet_dsu import uresnet50_dsu
from models.resnet_pada import uresnet50_pada

from models.resnet_ms2 import resnet50_fc512_ms12_a0d1_domprior
from models.resnet_db import resnet50_fc512_db12

from models.osnet_ms import osnet_x1_0, osnet_x1_0_ms23_a0d1, osnet_x1_0_ms23_a0d2, osnet_x1_0_ms23_a0d3
from models.osnet_csu import osnet_x1_0_csu12_a0d1, osnet_x1_0_csu23_a0d1, osnet_x1_0_csu34_a0d1, osnet_x1_0_csu012345_a0d1
from models.osnet_csu import osnet_x1_0_csu123_a0d1, osnet_x1_0_csu234_a0d1, osnet_x1_0_csu1234_a0d1, osnet_x1_0_csu12345_a0d1
from models.osnet_dsu import osnet_x1_0_dsu
from models.osnet_pada import osnet_x1_0_pada


from models.osnet_ms2 import osnet_x1_0_ms23_a0d1_domprior
from models.osnet_db import osnet_x1_0_db23


def build_datamanager(cfg):
    if cfg.data.type == 'image':
        return torchreid.data.ImageDataManager(**imagedata_kwargs(cfg))
    else:
        return torchreid.data.VideoDataManager(**videodata_kwargs(cfg))


def build_engine(cfg, datamanager, model, optimizer, scheduler):
    if cfg.data.type == 'image':
        if cfg.loss.name == 'softmax':
            engine = torchreid.engine.ImageSoftmaxEngine(
                datamanager,
                model,
                optimizer=optimizer,
                scheduler=scheduler,
                use_gpu=cfg.use_gpu,
                label_smooth=cfg.loss.softmax.label_smooth
            )

        else:
            engine = torchreid.engine.ImageTripletEngine(
                datamanager,
                model,
                optimizer=optimizer,
                margin=cfg.loss.triplet.margin,
                weight_t=cfg.loss.triplet.weight_t,
                weight_x=cfg.loss.triplet.weight_x,
                scheduler=scheduler,
                use_gpu=cfg.use_gpu,
                label_smooth=cfg.loss.softmax.label_smooth
            )

    else:
        if cfg.loss.name == 'softmax':
            engine = torchreid.engine.VideoSoftmaxEngine(
                datamanager,
                model,
                optimizer=optimizer,
                scheduler=scheduler,
                use_gpu=cfg.use_gpu,
                label_smooth=cfg.loss.softmax.label_smooth,
                pooling_method=cfg.video.pooling_method
            )

        else:
            engine = torchreid.engine.VideoTripletEngine(
                datamanager,
                model,
                optimizer=optimizer,
                margin=cfg.loss.triplet.margin,
                weight_t=cfg.loss.triplet.weight_t,
                weight_x=cfg.loss.triplet.weight_x,
                scheduler=scheduler,
                use_gpu=cfg.use_gpu,
                label_smooth=cfg.loss.softmax.label_smooth
            )

    return engine


def reset_config(cfg, args):
    if args.root:
        cfg.data.root = args.root
    if args.sources:
        cfg.data.sources = args.sources
    if args.targets:
        cfg.data.targets = args.targets
    if args.transforms:
        cfg.data.transforms = args.transforms


def main():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        '--config-file', type=str, default='', help='path to config file'
    )
    parser.add_argument(
        '-s',
        '--sources',
        type=str,
        nargs='+',
        help='source datasets (delimited by space)'
    )
    parser.add_argument(
        '-t',
        '--targets',
        type=str,
        nargs='+',
        help='target datasets (delimited by space)'
    )
    parser.add_argument(
        '--transforms', type=str, nargs='+', help='data augmentation'
    )
    parser.add_argument(
        '--root', type=str, default='', help='path to data root'
    )
    parser.add_argument(
        'opts',
        default=None,
        nargs=argparse.REMAINDER,
        help='Modify config options using the command-line'
    )
    args = parser.parse_args()

    cfg = get_default_config()
    cfg.use_gpu = torch.cuda.is_available()
    if args.config_file:
        cfg.merge_from_file(args.config_file)
    reset_config(cfg, args)
    cfg.merge_from_list(args.opts)
    set_random_seed(cfg.train.seed)

    log_name = 'test.log' if cfg.test.evaluate else 'train.log'
    log_name += time.strftime('-%Y-%m-%d-%H-%M-%S')
    sys.stdout = Logger(osp.join(cfg.data.save_dir, log_name))

    print('Show configuration\n{}\n'.format(cfg))
    print('Collecting env info ...')
    print('** System info **\n{}\n'.format(collect_env_info()))

    if cfg.use_gpu:
        torch.backends.cudnn.benchmark = True

    datamanager = build_datamanager(cfg)

    model_factory = {
        'resnet50_fc512': resnet50_fc512,
        'osnet_x1_0': osnet_x1_0,
        # mixstyle models
        'resnet50_fc512_ms12_a0d1': resnet50_fc512_ms12_a0d1,
        'resnet50_fc512_ms12_a0d2': resnet50_fc512_ms12_a0d2,
        'resnet50_fc512_ms12_a0d3': resnet50_fc512_ms12_a0d3,
        'resnet50_fc512_ms12_a0d1_domprior': resnet50_fc512_ms12_a0d1_domprior,
        'osnet_x1_0_ms23_a0d1': osnet_x1_0_ms23_a0d1,
        'osnet_x1_0_ms23_a0d2': osnet_x1_0_ms23_a0d2,
        'osnet_x1_0_ms23_a0d3': osnet_x1_0_ms23_a0d3,
        'osnet_x1_0_ms23_a0d1_domprior': osnet_x1_0_ms23_a0d1_domprior,
        # ablation
        'resnet50_fc512_ms1_a0d1': resnet50_fc512_ms1_a0d1,
        'resnet50_fc512_ms123_a0d1': resnet50_fc512_ms123_a0d1,
        'resnet50_fc512_ms1234_a0d1': resnet50_fc512_ms1234_a0d1,
        'resnet50_fc512_ms14_a0d1': resnet50_fc512_ms14_a0d1,
        'resnet50_fc512_ms23_a0d1': resnet50_fc512_ms23_a0d1,
        # dropblock models
        'resnet50_fc512_db12': resnet50_fc512_db12,
        'osnet_x1_0_db23': osnet_x1_0_db23,
        # csu model
        'resnet50_fc512_csu01_a0d1':resnet50_fc512_csu01_a0d1,
        'resnet50_fc512_csu12_a0d1':resnet50_fc512_csu12_a0d1,
        'resnet50_fc512_csu23_a0d1':resnet50_fc512_csu23_a0d1,
        'resnet50_fc512_csu34_a0d1':resnet50_fc512_csu34_a0d1,
        'resnet50_fc512_csu45_a0d1':resnet50_fc512_csu45_a0d1,
        'resnet50_fc512_csu012_a0d1':resnet50_fc512_csu012_a0d1,
        'resnet50_fc512_csu123_a0d1':resnet50_fc512_csu123_a0d1,
        'resnet50_fc512_csu234_a0d1':resnet50_fc512_csu234_a0d1,
        'resnet50_fc512_csu345_a0d1':resnet50_fc512_csu345_a0d1,
        'resnet50_fc512_csu0123_a0d1':resnet50_fc512_csu0123_a0d1,
        'resnet50_fc512_csu1234_a0d1':resnet50_fc512_csu1234_a0d1,
        'resnet50_fc512_csu2345_a0d1':resnet50_fc512_csu2345_a0d1,
        'resnet50_fc512_csu01234_a0d1':resnet50_fc512_csu01234_a0d1,
        'resnet50_fc512_csu12345_a0d1':resnet50_fc512_csu12345_a0d1,
        'resnet50_fc512_csu012345_a0d1':resnet50_fc512_csu012345_a0d1,
        
        'resnet50_fc512_csu23_a0d1': resnet50_fc512_csu23_a0d1,
        'resnet50_fc512_csu12_a0d2': resnet50_fc512_csu12_a0d2,
        'resnet50_fc512_csu12_a0d1': resnet50_fc512_csu12_a0d1,
        'resnet50_fc512_csu12_a0d3': resnet50_fc512_csu12_a0d3,
        'resnet50_fc512_csu12_a0d5': resnet50_fc512_csu12_a0d5,
        'osnet_x1_0_csu12_a0d1': osnet_x1_0_csu12_a0d1,
        'osnet_x1_0_csu23_a0d2': osnet_x1_0_csu23_a0d1,
        'osnet_x1_0_csu34_a0d1': osnet_x1_0_csu34_a0d1,
        'osnet_x1_0_csu123_a0d1': osnet_x1_0_csu123_a0d1,
        'osnet_x1_0_csu234_a0d2': osnet_x1_0_csu234_a0d1,
        'osnet_x1_0_csu1234_a0d1': osnet_x1_0_csu1234_a0d1,
        'osnet_x1_0_csu12345_a0d1': osnet_x1_0_csu12345_a0d1,
        'osnet_x1_0_csu012345_a0d1': osnet_x1_0_csu012345_a0d1,
        # Comparation with DSU
        'osnet_x1_0_dsu': osnet_x1_0_dsu,
        'osnet_x1_0_pada': osnet_x1_0_pada,
        'uresnet50_dsu': uresnet50_dsu,
        'uresnet50_pada': uresnet50_pada,

    }

    print('Building model: {}'.format(cfg.model.name))
    model = model_factory[cfg.model.name](
        num_classes=datamanager.num_train_pids,
        loss=cfg.loss.name,
        pretrained=cfg.model.pretrained,
        use_gpu=cfg.use_gpu
    )
    num_params, flops = compute_model_complexity(
        model, (1, 3, cfg.data.height, cfg.data.width)
    )
    print('Model complexity: params={:,} flops={:,}'.format(num_params, flops))

    if cfg.model.load_weights and check_isfile(cfg.model.load_weights):
        load_pretrained_weights(model, cfg.model.load_weights)

    if cfg.use_gpu:
        # model = nn.DataParallel(model).cuda()
        model = model.cuda()

    optimizer = torchreid.optim.build_optimizer(model, **optimizer_kwargs(cfg))
    scheduler = torchreid.optim.build_lr_scheduler(
        optimizer, **lr_scheduler_kwargs(cfg)
    )

    if cfg.model.resume and check_isfile(cfg.model.resume):
        cfg.train.start_epoch = resume_from_checkpoint(
            cfg.model.resume, model, optimizer=optimizer, scheduler=scheduler
        )

    print(
        'Building {}-engine for {}-reid'.format(cfg.loss.name, cfg.data.type)
    )
    engine = build_engine(cfg, datamanager, model, optimizer, scheduler)
    engine.run(**engine_run_kwargs(cfg))


if __name__ == '__main__':
    main()
