seed=0
deterministic=False
max_epochs=6

resume_from=None
cudnn_benchmark=False

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
image_size=[256, 704]

# augment3d=dict(
#   scale=[1.0, 1.0],
#   rotate=[0, 0],
#   translate=0.0,
# )


# augment3d=dict(
#   scale=[0.9, 0.9],
#   rotate=[-0.78539816, -0.78539816],
#   translate=0,
# )

# augment2d=dict(
#   resize=[[0.38, 0.38], [0.48, 0.48]],
#   rotate=[-5.4, -5.4],
#   gridmask=dict(
#     prob=0.0,
#     fixed_prob=True,
#   )
# )

augment3d=dict(
  scale=[0.9, 1.1],
  rotate=[-0.78539816, 0.78539816],
  translate=0.5,
)

augment2d=dict(
  resize=[[0.38, 0.55], [0.48, 0.48]],
  rotate=[-5.4, 5.4],
  gridmask=dict(
    prob=0.0,
    fixed_prob=True,
  )
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
      type='LoadAnnotations3D',
      with_bbox_3d=True,
      with_label_3d=True,
      with_attr_label=False,
    ),
    dict(
      type='ImageAug3D',
      final_dim=image_size,
      resize_lim=augment2d['resize'][0],
      bot_pct_lim=[0.0, 0.0],
      rot_lim=augment2d['rotate'],
      rand_flip=True, #
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
      type='ObjectRangeFilter',
      point_cloud_range=point_cloud_range,
    ),
    dict(
      type='ObjectNameFilter',
      classes=object_classes,
    ),
    dict(
      type='ImageNormalize',
      mean=[0.485, 0.456, 0.406],
      std=[0.229, 0.224, 0.225],
    ),
    dict(
      type='PointShuffle',
    ),
    dict(
      type="DefaultFormatBundle3D",
      classes=object_classes,
    ),
    dict(
      type='Collect3D',
      keys=['img', 'points', 'gt_bboxes_3d', 'gt_labels_3d', 'points_single'],
      meta_keys=['camera_intrinsics',
                 'camera2ego',
                 'lidar2ego',
                 'camera2lidar',
                 'lidar2camera',
                 'lidar2image',
                 'img_aug_matrix',
                 'lidar_aug_matrix'
      ],
    ),
]

test_pipeline=[
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
      type='LoadAnnotations3D',
      with_bbox_3d=True,
      with_label_3d=True,
      with_attr_label=False,
    ),
    dict(
      type='ImageAug3D',
      final_dim=image_size,
      resize_lim=augment2d['resize'][1],
      bot_pct_lim=[0.0, 0.0],
      rot_lim=[0.0, 0.0],
      rand_flip=False,
      is_train=False,
    ),
    dict(
      type='GlobalRotScaleTrans',
      resize_lim=[1.0, 1.0],
      rot_lim=[0.0, 0.0],
      trans_lim=0.0,
      is_train=False,
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
      keys=['img', 'points', 'gt_bboxes_3d', 'gt_labels_3d', 'points_single'],
      meta_keys=['camera_intrinsics',
                 'camera2ego',
                 'lidar2ego',
                 'camera2lidar',
                 'lidar2camera',
                 'lidar2image',
                 'img_aug_matrix',
                 'lidar_aug_matrix'
      ],
    ),
]

data=dict(
  samples_per_gpu=4,
  workers_per_gpu=4,
  train=dict(
    type='CBGSDataset',
    dataset=dict(
      type=dataset_type,
      dataset_root=dataset_root,
      ann_file=dataset_root + "nuscenes_infos_val.pkl", #
      pipeline=train_pipeline,
      object_classes=object_classes,
      map_classes=None,
      modality=input_modality,
      test_mode=False,
      use_valid_flag=True,
      box_type_3d='LiDAR',
    )
  ),
  val=dict(
    type=dataset_type,
    dataset_root=dataset_root,
    ann_file=dataset_root + "nuscenes_infos_val.pkl",
    pipeline=test_pipeline,
    object_classes=object_classes,
    map_classes=None,
    modality=input_modality,
    test_mode=False,
    box_type_3d='LiDAR',
  ),
  test=dict(
    type=dataset_type,
    dataset_root=dataset_root,
    ann_file=dataset_root + "nuscenes_infos_val.pkl",
    pipeline=test_pipeline,
    object_classes=object_classes,
    map_classes=None,
    modality=input_modality,
    test_mode=True,
    box_type_3d='LiDAR',
  )
)

evaluation=dict(
  interval=1,
  pipeline=test_pipeline,
)
  
  
model=dict(
  type='BEVAttn',
  mult_feat=False, #
  encoders=dict(
    camera=dict(
      backbone=dict(
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
        out_indices=[1, 2, 3], #
        with_cp=False,
        convert_weights=True,
        frozen_stages=3, #
        init_cfg=dict(
          type='Pretrained',
          checkpoint='pretrained/swint-nuimages-pretrained.pth',
          ),
      ),
      neck=dict(
        type='GeneralizedLSSFPN',
        in_channels=[192, 384, 768],
        out_channels=256,
        start_level=0,
        num_outs=1, #
        norm_cfg=dict(type='BN2d', requires_grad=True),
        act_cfg=dict(type='ReLU', inplace=True),
        upsample_cfg=dict(mode='bilinear', align_corners=False)),
      vtransform=dict(
        type='DepthAttnTransform',
        in_channels=256,
        out_channels=256,
        image_size=image_size,
        feature_size=[image_size[0] // 8, image_size[1] // 8], #
        img_downsample_factor=[1], #
        xbound=[-54, 54, 0.3],
        ybound=[-54, 54, 0.3],
        zbound=[-10.0, 10.0, 20.0],
        dbound=[1.0, 60.0, 0.25],
        depth_attn_layers=1,
        linear_layer=dict(
          input_channel=128,
          output_channel=256,
        ),
        depth_attn_cfg=dict(
          type='DepthAttnLayer',
          with_res=False,
          embed_dim=256,
          num_heads=8,
          ffn_cfgs=dict(
            type='FFN',
            embed_dims=256,
            feedforward_channels=512,
            num_fcs=2,
            ffn_drop=0.1,
            act_cfg=dict(type='ReLU', inplace=True)
          ),
          norm_cfg=dict(type='LN')
        ),
        with_bev_embedding=True,
        positional_encoding=dict(
          type='LearnedDepthPositionalEncoding',
          input_channel=1,
          num_feats=256,
        ),
        downsample=1,
        ),
      ),
  
    lidar=dict(
      voxelize=dict(
        type='DynamicPillarVFE',
        num_point_features=5,
        voxel_size=[0.3, 0.3, 8.0],
        grid_size=[360, 360, 1],
        point_cloud_range=[-54.0, -54.0, -5.0, 54.0, 54.0, 3.0],
        with_distance=False,
        use_absolute_xyz=True,
        use_norm=True,
        num_filters=[128, 128],
        ),
      backbone=dict(
        type='DSVT',
        input_layers=dict(
          sparse_shape=[360, 360, 1],
          downsample_stride=[],
          d_model=[128],
          set_info=[[90, 4]],
          window_shape=[[30, 30, 1]],
          hybrid_factor=[1, 1, 1], # x, y, z
          shifts_list=[[[0, 0, 0], [15, 15, 0]]],
          normalize_pos=False,
        ),
        block_name=['DSVTBlock'],
        set_info=[[90, 4]],
        d_model=[128],
        nhead=[8],
        dim_feedforward=[256],
        dropout=0.0,
        activation='gelu',
        output_shape=[360, 360],
        conv_out_channel=128,
    ),
      map_to_bev_module=dict(
        type='PointPillarScatter3d',
        input_shape=[360, 360, 1],
        num_bev_features=128,
      ),
    ),
  ),
  fuser=dict(
    type='ConvFuser',
    in_channels=[128, 256],
    out_channels=128
  ),
  # fuser = dict(
  #   type='SparsePool',
  #   input_layers=dict(
  #       sparse_shape=[360, 360, 1],
  #       downsample_stride=[[1, 1, 3]],
  #       d_model=[128, 128],
  #       set_info=[[90, 4], [90, 4]],
  #       window_shape=[[30, 30, 3], [30, 30, 1]],
  #       hybrid_factor=[1, 1, 1], # x, y, z
  #       shifts_list=[[[0, 0, 0], [15, 15, 0]]],
  #       normalize_pos=False,
  #       with_pos=False,
  #   ),
  #   d_model=[128, 128],
  #   pool_volume = 3,
  #   map_to_bev_module=dict(
  #     input_shape=[360, 360, 1],
  #     num_bev_features=128,
  #   ),
  # ),
  decoder=dict(
    backbone=dict(
      type='BaseBEVResBackbone',
      input_channels=128,
      layer_nums=[ 1, 2, 2 ],
      layer_strides=[ 1, 2, 2 ],
      num_filters=[ 128, 128, 256 ],
      upsample_strides=[ 0.5, 1, 2 ],
      num_upsample_filters=[ 128, 128, 128 ],
    ),
  ),
  dense_head=dict(
    type='DSVTHead',
    use_bias_before_norm=False,
    num_proposals=200,
    hidden_channel=128,
    num_classes=10,
    num_heads=8,
    nms_kernel_size=3,
    ffn_channel=256,
    dropout=0.0,
    bn_momentum=0.1,
    activation='relu',
    num_hm_conv=2,
    voxel_size=[0.3, 0.3, 8.0],
    grid_size=[360, 360, 1],
    point_cloud_range=[-54.0, -54.0, -5.0, 54.0, 54.0, 3.0],
    num_class=10,
    input_channels=384,
    predict_boxes_when_training=False,

    separate_head_cfg=dict(
      head_order=['center', 'height', 'dim', 'rot', 'vel'],
      head_dict={
            'center': {'out_channels': 2, 'num_conv': 2},
            'height': {'out_channels': 1, 'num_conv': 2},
            'dim': {'out_channels': 3, 'num_conv': 2},
            'rot': {'out_channels': 2, 'num_conv': 2},
            'vel': {'out_channels': 2, 'num_conv': 2},
            'iou': {'out_channels': 1, 'num_conv': 2},
      },
    ),
    target_assigner_config=dict(
      feature_map_stride=2,
      dataset='nuScenes',
      gaussian_overlap=0.1,
      min_radius=2,
      hungarian_assigner=dict(
          cls_cost={'gamma': 2.0, 'alpha': 0.25, 'weight': 0.15},
          reg_cost={'weight': 0.25},
          iou_cost={'weight': 0.25},
      ),
    ),
    loss_config=dict(
      loss_weights={
          'cls_weight': 1.0,
          'bbox_weight': 0.25,
          'hm_weight': 1.0,
          'loss_iou_rescore_weight': 0.5,
          'code_weights': [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.2, 0.2],
      },
      loss_cls=dict(
        use_sigmoid=True,
        gamma=2.0,
        alpha=0.25,
      ),
    ),
    post_processing=dict(
      score_thresh=0.0,
      post_center_range=[-61.2, -61.2, -10.0, 61.2, 61.2, 10.0],
      use_iou_to_rectify_score=False,
      iou_rectifier=[0.5],
      nms_config=dict(
        nms_type='nms_gpu',
        nms_thresh=0.2,
        nms_pre_maxsize=1000,
        nms_post_maxsize=100,
        score_thres=0.0,
      ),
    )
  )
)


## TODO: renew training config
load_from='pretrained/dsvt_bev_nus_nuim.pth'

fp16=dict(
  loss_scale=dict(
    growth_interval=2000,
    # init_scale=4
    ), 
)

optimizer_config=dict(
  grad_clip=dict(
    max_norm=35,
    norm_type=2,
  )
)

lr_config=dict(
  policy='CosineAnnealing',
  warmup='linear',
  warmup_iters=1000,
  warmup_ratio=0.3333333,
  min_lr_ratio=1.0e-3
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
)