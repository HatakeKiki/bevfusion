import argparse
import copy
import os
import random
import time

import numpy as np
import torch
from mmcv import Config
from torchpack import distributed as dist
from torchpack.environ import auto_set_run_dir, set_run_dir
from torchpack.utils.config import configs

from mmdet3d.apis import train_model
from mmdet3d.datasets import build_dataset
from mmdet3d.models import build_model
from mmdet3d.utils import get_root_logger, convert_sync_batchnorm, recursive_eval
from pathlib import Path
import warnings
warnings.filterwarnings("ignore", category=UserWarning)
# warnings.filterwarnings("ShapelyDeprecationWarning", category=UserWarning)      

def main():

    parser = argparse.ArgumentParser()
    parser.add_argument("config", metavar="FILE", help="config file")
    parser.add_argument("--run-dir", metavar="DIR", help="run directory")
    parser.add_argument('--auto-resume', action='store_true', default=False, 
                        help='resume from the latest checkpoint automatically')
    parser.add_argument("--no_validate", action='store_true', default=False)
    parser.add_argument("--not_dist", action='store_true', default=False)
    parser.add_argument("--sync_bn", action='store_true', default=False)
    args, opts = parser.parse_known_args()
    
    if not args.not_dist:
        dist.init()

    if args.config.split('.')[-1] == 'py':
        cfg = Config.fromfile(args.config)
    elif args.config.split('.')[-1] == 'yaml':
        configs.load(args.config, recursive=True)
        configs.update(opts)
        cfg = Config(recursive_eval(configs), filename=args.config)

    torch.backends.cudnn.benchmark = cfg.cudnn_benchmark
    torch.cuda.set_device(dist.local_rank())
    # torch.autograd.set_detect_anomaly(True)

    if args.run_dir is None:
        args.run_dir = auto_set_run_dir()
    else:
        set_run_dir(args.run_dir)
    cfg.run_dir = args.run_dir
    cfg.auto_resume = args.auto_resume

    ckpt_dir = args.run_dir + '/ckpt'
    ckpt_dir = Path(ckpt_dir)
    if not ckpt_dir.exists():
        ckpt_dir.mkdir(exist_ok=False, parents=True)

    # dump config
    cfg.dump(os.path.join(cfg.run_dir, "configs." + args.config.split('.')[-1]))

    # init the logger before other steps
    timestamp = time.strftime("%Y%m%d_%H%M%S", time.localtime())
    log_file = os.path.join(cfg.run_dir, f"{timestamp}.log")
    logger = get_root_logger(log_file=log_file)

    # log some basic info
    logger.info(f"Config:\n{cfg.pretty_text}")

    # set random seeds
    if cfg.seed is not None:
        logger.info(
            f"Set random seed to {cfg.seed}, "
            f"deterministic mode: {cfg.deterministic}"
        )
        random.seed(cfg.seed)
        np.random.seed(cfg.seed)
        torch.manual_seed(cfg.seed)
        if cfg.deterministic:
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False

    datasets = [build_dataset(cfg.data.train)]

    model = build_model(cfg.model)
    
    if args.sync_bn:
        logger.info("BatchNorm Synced")
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
    
    model.init_weights()
    
    # freeze_layers = ['encoders.lidar']
    # for name, param in model.named_parameters():
    #     for freeze_layer in freeze_layers:
    #         if freeze_layer in name:
    #             param.requires_grad = False

    logger.info(f"Params need to be updated:")
    for name, param in model.named_parameters():
        if param.requires_grad is True:
            logger.info(name)
            
    logger.info(f"Params not need to be updated:")
    for name, param in model.named_parameters():
        if param.requires_grad is False:
            logger.info(name)

    # logger.info(f"Model:\n{model}")
    train_model(
        model,
        datasets,
        cfg,
        distributed=(not args.not_dist),
        validate=(not args.no_validate),
        timestamp=timestamp,
    )


if __name__ == "__main__":
    main()
