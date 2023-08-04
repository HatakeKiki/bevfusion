seed=0
deterministic=False
max_epochs=20

load_from=None
resume_from=None
cudnn_benchmark=False

fp16=dict(loss_scale=dict(growth_interval=2000))

checkpoint_config=dict(
  interval=1,
  max_keep_ckpts=1,
)

log_config=dict(
    interval=50,
    hooks=[dict(type='TextLoggerHook'), 
           dict(type='TensorboardLoggerHook')]
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

point_cloud_range=[-54.0, -54.0, -5.0, 54.0, 54.0, 3.0]
voxel_size=[0.1, 0.1, 0.2]
image_size=[256, 704]
sparse_shape=[1440, 1440, 41]

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


train_pipeline=[
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
      resize_lim=augment2d['resize'][0],
      bot_pct_lim=[0.0, 0.0],
      rot_lim=augment2d['rotate'],
      rand_flip=True,
      is_train=True,
    ),
    dict(
      type='GlobalRotScaleTrans',
      resize_lim=augment3d['scale'],
      rot_lim=augment3d['rotate'],
      trans_lim=augment3d['translate'],
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
      type="DefaultFormatBundle3D",
      classes=object_classes,
    ),
    dict(
      type='Collect3D',
      keys=['img', 'points', 'points_single'],
      meta_keys=['lidar2image', 
                 'img_aug_matrix', 
                 'lidar_aug_matrix',
      ],
    ),
]


data=dict(
  samples_per_gpu=2,
  workers_per_gpu=4,
  train=dict(
    type='CBGSDataset',
    dataset=dict(
      type=dataset_type,
      dataset_root=dataset_root,
      ann_file=dataset_root + "nuscenes_infos_train.pkl",
      pipeline=train_pipeline,
      object_classes=object_classes,
      map_classes=None,
      modality=input_modality,
      test_mode=False,
      use_valid_flag=True,
      box_type_3d='LiDAR',
    )
  ),
  # val=dict(),
  # test=dict(),
)

evaluation=dict(
  interval=-1,
  pipeline=train_pipeline,
)
  
  
model=dict(
  type='SimBEV',
  img_backbone=dict(
    type='SwinTransformer',
    embed_dims=96,
    depths=[2, 2, 6, 2],
    num_heads=[3, 6, 12, 24],
    window_size=7,
    mlp_ratio=4,
    qkv_bias=True,
    qk_scale=None,
    drop_rate=0.,
    attn_drop_rate=0.,
    drop_path_rate=0.2,
    patch_norm=True,
    out_indices=[1, 2, 3],
    with_cp=False,
    convert_weights=True,
    init_cfg=dict(
      type='Pretrained',
      checkpoint='pretrained/swin_tiny_patch4_window7_224.pth'),
    ),
  img_neck=dict(
    type='GeneralizedLSSFPN',
    in_channels=[192, 384, 768],
    out_channels=256,
    start_level=0,
    num_outs=3,
    norm_cfg=dict(
      type='BN2d',
      requires_grad=True,
    ),
    act_cfg=dict(
      type='ReLU',
      inplace=True,
    ),
    upsample_cfg=dict(
      mode='bilinear',
      align_corners=False,
    )
  ),
  pts_voxelize=dict(
    max_num_points=10,
    max_voxels=[120000, 160000],
    point_cloud_range=point_cloud_range,
    voxel_size=[0.075, 0.075, 0.2],
    ),
  pts_backbone=dict(
    type='SparseEncoder',
    in_channels=5,
    sparse_shape=sparse_shape,
    output_channels=128,
    order=['conv', 'norm', 'act'],
    encoder_channels=[
      [16, 16, 32],
      [32, 32, 64],
      [64, 64, 128],
      [128, 128],
    ],
    encoder_paddings=[
      [0, 0, 1],
      [0, 0, 1],
      [0, 0, [1, 1, 0]],
      [0, 0],
    ],
    block_type='basicblock',
  ),
  pts_mlp=dict(
    in_dim=256,
    hidden_dim=1024,
    out_dim=256,
    ),
  img_mlp=dict(
    in_dim=256,
    hidden_dim=1024,
    out_dim=256,
    ),
  T=0.07,
  pts_sampler=dict(
    num_point=[2048],
    fps_mod_list=["D-FPS"],
    fps_sample_range_list=[-1],
  )
)


optimizer_config=dict(
  grad_clip=dict(
    max_norm=35,
    norm_type=2,
  )
)

lr_config=dict(
  policy='cyclic',
  target_ratio=5.0,
  cyclic_times=1,
  step_ratio_up=0.4,
)

momentum_config=dict(
  policy='cyclic',
  cyclic_times=1,
  step_ratio_up=0.4,
)
  
optimizer=dict(
  type='AdamW',
  lr=0.0001,
  weight_decay=0.01,
  paramwise_cfg=dict(
    custom_keys={
      'absolute_pos_embed': dict(decay_mult=0),
      'relative_position_bias_table': dict(decay_mult=0),
      'img_backbone': dict(lr_mult=0.1),
      'pts_backbone': dict(lr_mult=0.0),
    }
  )
)
        