seed=0
deterministic=False
max_epochs=20

load_from=None
resume_from=None
cudnn_benchmark=None

fp16=dict(loss_scale=dict(growth_interval=2000))

checkpoint_config=dict(
  interval=1,
  max_keep_ckpts=1,
)

log_config=dict(
    interval=50,
    hooks=dict(
        type='TextLoggerHook',
        type='TensorboardLoggerHook',
    )
)

runner=dict(
  type='CustomEpochBasedRunner',
  max_epochs=max_epochs,
)

dataset_type='NuScenesDataset'
dataset_root='data/nuscenes/'
gt_paste_stop_epoch=-1
reduce_beams=32
load_dim=5
use_dim=5
load_augmented=None

point_cloud_range=[-51.2, -51.2, -5.0, 51.2, 51.2, 3.0]
voxel_size=[0.1, 0.1, 0.2]
image_size=[256, 704]

augment2d=dict(
  resize=[[0.38, 0.55], [0.48, 0.48]],
  rotate=[-5.4, 5.4],
  gridmask=dict(
    prob=0.0,
    fixed_prob=True,
  )
)

augment3d=dict(
  scale=[0.9, 1.1],
  rotate=[-0.78539816, 0.78539816],
  translate=0.5,
)

object_classes=['car', 'truck', 'construction_vehicle', 'bus', 'trailer', 
                'barrier', 'motorcycle', 'bicycle', 'pedestrian', 'traffic_cone']

input_modality=dict(
  use_lidar=True,
  use_camera=True,
  use_radar=False,
  use_map=False,
  use_external=False,
)


train_pipeline = [
    dict(
        type='LoadMultiViewImageFromFiles',
        to_float32=True,
        ),
    dict(
        type='LoadPointsFromFile',
        coord_type='LIDAR',
        load_dim=load_dim,
        use_dim=use_dim,
        reduce_beams=reduce_beams,
        load_augmented=load_augmented,
    ),
    dict(
        type='LoadPointsFromMultiSweeps',
        sweeps_num=9,
        load_dim=load_dim,
        use_dim=use_dim,
        reduce_beams=reduce_beams,
        pad_empty_sweeps=True,
        remove_close=True,
        load_augmented=load_augmented,
    ),
    dict(
        type='ImageAug3D',
        final_dim=image_size,
        resize_lim=augment2d.resize[0],
        bot_pct_lim=[0.0, 0.0],
        rot_lim=augment2d.rotate,
        rand_flip=True,
        is_train=True,
    ),
    dict(
        type='GlobalRotScaleTrans',
        resize_lim=augment3d.scale,
        rot_lim=augment3d.rotate,
        trans_lim=augment3d.translate,
        is_train=True,
    ),
    dict(
        type='RandomFlip3D',
    ),
    dict(
        type='PointsRangeFilter',
        point_cloud_range=point_cloud_range,
    ),
    dict(
        type='ImageNormalize',
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
    ),
    dict(
        type='Collect3D',
        keys=['img', 'points'],
        meta_keys=['camera_intrinsics', 'camera2ego',
            'lidar2ego', 'camera2lidar', 'lidar2camera',
            'lidar2image', 'img_aug_matrix', 'lidar_aug_matrix',
        ],
    ),
]

model=dict(
  type='SimBEV',
  img_encoder=dict(),
  pts_encoder=dict(),
  matcher=dict(),
  
    
)