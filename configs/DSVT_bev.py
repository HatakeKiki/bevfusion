seed=0
deterministic=False
max_epochs=20

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
gt_paste_stop_epoch=15
reduce_beams=32
load_dim=5
use_dim=5
load_augmented=None

point_cloud_range=[-54.0, -54.0, -5.0, 54.0, 54.0, 3.0]

object_classes=['car', 'truck', 'construction_vehicle', 'bus', 'trailer', 
                'barrier', 'motorcycle', 'bicycle', 'pedestrian', 'traffic_cone']

input_modality=dict(
  use_lidar=True,
  use_camera=False,
  use_radar=False,
  use_map=False,
  use_external=False,
)

augment3d=dict(
  scale=[0.9, 1.1],
  rotate=[-0.78539816, 0.78539816],
  translate=0.5,
)

train_pipeline=[
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
      type='ObjectPaste',
      stop_epoch=gt_paste_stop_epoch,
      db_sampler=dict(
        dataset_root=dataset_root,
        info_path=dataset_root + "nuscenes_dbinfos_train.pkl",
        rate=1.0,
        prepare=dict(
          filter_by_difficulty=[-1],
          filter_by_min_points=dict(
            car=5,
            truck=5,
            bus=5,
            trailer=5,
            construction_vehicle=5,
            traffic_cone=5,
            barrier=5,
            motorcycle=5,
            bicycle=5,
            pedestrian=5,
          ),
        ),
        classes=object_classes,
        sample_groups=dict(
          car=2,
          truck=3,
          construction_vehicle=7,
          bus=4,
          trailer=6,
          barrier=2,
          motorcycle=6,
          bicycle=6,
          pedestrian=2,
          traffic_cone=2,
        ),
        points_loader=dict(
          type='LoadPointsFromFile',
          coord_type='LIDAR',
          load_dim=load_dim,
          use_dim=use_dim,
          reduce_beams=reduce_beams,
        ),
      ),
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
      type='PointShuffle',
    ),
    dict(
      type="DefaultFormatBundle3D",
      classes=object_classes,
    ),
    dict(
      type='Collect3D',
      keys=['points', 'gt_bboxes_3d', 'gt_labels_3d'],
      meta_keys=['lidar2ego',
                 'lidar_aug_matrix'
      ],
    ),
]

test_pipeline=[
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
      type="DefaultFormatBundle3D",
      classes=object_classes,
    ),
    dict(
      type='Collect3D',
      keys=['points', 'gt_bboxes_3d', 'gt_labels_3d'],
      meta_keys=['lidar2ego',
                 'lidar_aug_matrix'
      ],
    ),
]

data=dict(
  samples_per_gpu=1,
  workers_per_gpu=1,
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
    ann_file=dataset_root + "nuscenes_infos_train_1k.pkl",
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
  encoders=dict(
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
      use_iou_to_rectify_score=True,
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


# ## TODO: renew training config
# load_from='pretrained/dsvt_bev_nus.pth'
load_from=None

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
)
