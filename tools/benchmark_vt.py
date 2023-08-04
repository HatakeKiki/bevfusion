# Copyright (c) OpenMMLab. All rights reserved.
import argparse
import time

import numpy as np
import torch
from mmcv import Config
from mmcv.parallel import MMDataParallel
from mmcv.runner import load_checkpoint, wrap_fp16_model

from mmdet3d.datasets import build_dataloader, build_dataset
from mmdet3d.models import build_fusion_model
from mmdet3d.utils import recursive_eval
from torchpack.utils.config import configs

from mmdet3d.ops.bev_pool_v2.bev_pool import bev_pool_v2
from mmdet3d.ops import bev_pool


def parse_args():
    parser = argparse.ArgumentParser(description='MMDet benchmark a model')
    parser.add_argument('config', help='test config file path')
    parser.add_argument('checkpoint', help='checkpoint file')
    parser.add_argument("--samples", default=2000, help="samples to benchmark")
    parser.add_argument("--log-interval", default=50, help="interval of logging")
    parser.add_argument("--fp16", action="store_true")
    parser.add_argument("--bevpool_v2", action="store_true")
    parser.add_argument(
        '--mem-only',
        action='store_true',
        help='Conduct the memory analysis only')
    parser.add_argument(
        '--no-acceleration',
        action='store_true',
        help='Omit the pre-computation acceleration')
    args = parser.parse_args()
    return args


def main():
    args = parse_args()

    args.bevpool_v2 = True
    configs.load(args.config, recursive=True)
    cfg = Config(recursive_eval(configs), filename=args.config)
    # set cudnn_benchmark
    if cfg.get("cudnn_benchmark", False):
        torch.backends.cudnn.benchmark = True
    cfg.model.pretrained = None
    cfg.data.test.test_mode = True

    # build the dataloader
    # TODO: support multiple images per gpu (only minor changes are needed)
    dataset = build_dataset(cfg.data.test)
    data_loader = build_dataloader(
        dataset,
        samples_per_gpu=1,
        workers_per_gpu=cfg.data.workers_per_gpu,
        dist=False,
        shuffle=False,
    )

    # build the model and load checkpoint
    cfg.model.train_cfg = None
    model = build_fusion_model(cfg.model, test_cfg=cfg.get("test_cfg"))
    if args.fp16:
        wrap_fp16_model(model)
    load_checkpoint(model, args.checkpoint, map_location="cpu")

    model = MMDataParallel(model, device_ids=[0])

    model.eval()

    # the first several iterations may be very slow so skip them
    num_warmup = 100
    pure_inf_time = 0
    view_transformer = model.module.encoders['camera']['vtransform']

    # benchmark with several samples and take the average
    for i, data in enumerate(data_loader):

        with torch.no_grad():
            data['img_metas'] = data.pop('metas')
            del data['metas_tta']

            for key in data.keys():
                if isinstance(data[key].data[0], torch.Tensor):
                    data[key] = data[key].data[0].cuda()
                elif key == 'img_metas':
                    continue
                else:
                    assert key == 'points'
                    points = data[key].data[0]
                    data[key] = []
                    for point in points:
                        data[key].append(point.cuda())
                
            data['img'], data['out_bool'] = model.module.extract_camera_features(**data, benchmark_vt=True)
            data['out_bool'] = None

        del data['mask']
        data['cam_intrinsic'] = data.pop('camera_intrinsics')
        data['sensor2ego'] = data.pop('camera2ego')
        data['metas'] = data.pop('img_metas')
            
        if i == 0:
            precomputed_memory_allocated = 0.
            # start_mem_allocated = torch.cuda.memory_allocated()
            img, depth, geom = view_transformer(**data, benchmark_vt=True)
            
            if args.bevpool_v2:
                start_mem_allocated = torch.cuda.memory_allocated()
                feats, depth = view_transformer.get_cam_feats_depth(img, depth)
                x = view_transformer.voxel_pooling_v2(geom, depth, feats, benchmark_vt=True, out_bool=data['out_bool'])
                depth, feat, ranks_depth, ranks_feat, ranks_bev,bev_feat_shape, interval_starts, interval_lengths = x
                bev_pool_v2(depth, feat, ranks_depth, ranks_feat, ranks_bev,
                                        bev_feat_shape, interval_starts,
                                        interval_lengths)
            else:
                start_mem_allocated = torch.cuda.memory_allocated()
                x, depth = view_transformer.get_cam_feats(img, depth)
                x = view_transformer.bev_pool(geom, x, benchmark_vt=True, out_bool=data['out_bool'])
                x, geom_feats, B, nx = x
                bev_pool(x, geom_feats, B, nx[2], nx[0], nx[1])
                
            
            
            # x, depth = view_transformer.get_cam_feats(img, depth)
            # x = view_transformer.bev_pool(geom, x, out_bool=data['out_bool'])
            
            
            end_mem_allocated = torch.cuda.memory_allocated()
            precomputed_memory_allocated = \
                end_mem_allocated - start_mem_allocated
            ref_max_mem_allocated = torch.cuda.max_memory_allocated()
            # occupy the memory
            size = (ref_max_mem_allocated - end_mem_allocated)

            print('Memory analysis: \n'
                  'precomputed_memory_allocated : %d B / %.01f MB \n' %
                  (precomputed_memory_allocated,
                   precomputed_memory_allocated / 1024 / 1024))

            if args.mem_only:
                return

        torch.cuda.synchronize()
        
        with torch.no_grad():
            img, depth, geom = view_transformer(**data, benchmark_vt=True)
            if args.bevpool_v2:
                feats, depth = view_transformer.get_cam_feats_depth(img, depth)
                x = view_transformer.voxel_pooling_v2(geom, depth, feats, benchmark_vt=True, out_bool=data['out_bool'])
                depth, feat, ranks_depth, ranks_feat, ranks_bev,bev_feat_shape, interval_starts, interval_lengths = x
                
                start_time = time.perf_counter()
                bev_pool_v2(depth, feat, ranks_depth, ranks_feat, ranks_bev,
                                        bev_feat_shape, interval_starts,
                                        interval_lengths)
            else:
                x, depth = view_transformer.get_cam_feats(img, depth)
                x = view_transformer.bev_pool(geom, x, benchmark_vt=True, out_bool=data['out_bool'])
                x, geom_feats, B, nx = x
                
                start_time = time.perf_counter()
                bev_pool(x, geom_feats, B, nx[2], nx[0], nx[1])
                
            

            
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - start_time

        if i >= num_warmup:
            pure_inf_time += elapsed
            if (i + 1) % args.log_interval == 0:
                fps = (i + 1 - num_warmup) / pure_inf_time
                print(f'Done image [{i + 1:<3}/ {args.samples}], '
                      f'fps: {fps:.1f} img / s')

        if (i + 1) == args.samples:
            pure_inf_time += elapsed
            fps = (i + 1 - num_warmup) / pure_inf_time
            print(f'Overall fps: {fps:.1f} img / s')
            return fps


if __name__ == '__main__':
    repeat_times = 1
    fps_list = []
    for _ in range(repeat_times):
        fps = main()
        time.sleep(5)
        fps_list.append(fps)
    fps_list = np.array(fps_list, dtype=np.float32)
    print(f'Mean Overall fps: {fps_list.mean():.4f} +'
          f' {np.sqrt(fps_list.var()):.4f} img / s')